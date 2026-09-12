import datetime
import hashlib
import json
import logging
import os

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import omegaconf
import numpy as np
import polars as pl
import wandb
from tqdm import tqdm

import drift_linearity
import latent_space
import splits
from latent_gp import LatentConfig

from plnn.dataset import LandscapeSimulationDataset, NumpyLoader
from plnn.models import DeepTimePhiPLNN
from plnn.loss_functions import select_loss_function
from plnn.optimizers import get_optimizer_args, select_optimizer, get_dt_schedule
from plnn.model_training import train_model

logger = logging.getLogger(__name__)

INITIAL_DATE = datetime.datetime(2020, 1, 1)
UNIT_DAYS = 365.25

# NumpyLoader promotes whatever it is given back to float64, so this only
# bounds the memory df_to_data hands over; the solver still runs in the dtype
# the loader produces.
LANDSCAPE_DTYPE = np.float32
# Dtype the loader (and therefore the jitted step) actually sees. evaluate_pairs
# bypasses the loader, so it has to match this or diffrax's internal buffers
# and the inputs disagree.
SOLVE_DTYPE = np.float64


def df_to_data(df):
    """Per-trajectory lists of timestep dicts, as LandscapeSimulationDataset wants.

    Columns are pulled out once per trajectory rather than row by row; at a few
    hundred thousand pairs the per-row dict conversion dominates evaluation.
    """
    out = []
    for p in df.partition_by('filter_value'):
        t0 = p['t0'].to_numpy().astype(LANDSCAPE_DTYPE)
        t1 = p['t1'].to_numpy().astype(LANDSCAPE_DTYPE)
        x0 = p['x0'].to_numpy().astype(LANDSCAPE_DTYPE)
        x1 = p['x1'].to_numpy().astype(LANDSCAPE_DTYPE)
        out.append([{'t0': t0[i], 'x0': x0[i][np.newaxis, :],
                     't1': t1[i], 'x1': x1[i][np.newaxis, :]}
                    for i in range(len(p))])
    return out


def compute_rolling_means(cfg, target_df, dims):
    """Compute rolling means for each filter_value trajectory."""
    return target_df.with_columns([
            pl.col('x0').arr.get(i).alias(f'x0_{i}') for i in dims
        ])\
        .rolling('createtime', period=f'{cfg.rolling_mean_window}d', group_by='filter_value') \
        .agg([pl.col(f'x0_{i}').mean() for i in dims])\
        .with_columns(
            ((pl.col('createtime') - INITIAL_DATE).dt.total_days() / UNIT_DAYS).alias('t'),
        )\
        .sort(['filter_value', 'createtime'])


def build_horizon_pairs(rolling_df, horizon_days, dims, tolerance_frac=0.25):
    """Build (x0, x1) pairs where x1 is approximately horizon_days in the future.

    Estimates the median observation spacing, shifts by the appropriate number
    of rows, then filters to pairs within tolerance of the target horizon.
    """
    dim_cols = [f'x0_{i}' for i in dims]

    spacing_df = rolling_df.with_columns(
        (pl.col('createtime').diff().over('filter_value')).alias('dt')
    ).drop_nulls('dt')
    median_spacing_days = spacing_df['dt'].median().total_seconds() / 86400
    shift_n = max(1, round(horizon_days / median_spacing_days))

    tolerance_days = max(horizon_days * tolerance_frac, 3)

    paired = rolling_df\
        .with_columns(
            [pl.col('t').shift(-shift_n).over('filter_value').alias('t1'),
             pl.col('createtime').shift(-shift_n).over('filter_value').alias('future_createtime')] + \
            [pl.col(c).shift(-shift_n).over('filter_value').alias(f'x1_{i}') for c, i in zip(dim_cols, dims)] + \
            [pl.col(c).shift(shift_n).over('filter_value').alias(f'xm1_{i}') for c, i in zip(dim_cols, dims)]
        )\
        .drop_nulls(['t1'])\
        .with_columns(
            ((pl.col('future_createtime') - pl.col('createtime')).dt.total_days()).alias('actual_gap_days')
        )\
        .filter(
            (pl.col('actual_gap_days') >= horizon_days - tolerance_days) &
            (pl.col('actual_gap_days') <= horizon_days + tolerance_days)
        )\
        .with_columns([
            pl.concat_arr(dim_cols).alias('x0'),
            pl.concat_arr([f'x1_{i}' for i in dims]).alias('x1'),
            pl.concat_arr([f'xm1_{i}' for i in dims]).alias('xm1'),
            pl.col(f'xm1_{dims[0]}').is_not_null().alias('has_prev'),
        ])\
        .rename({'t': 't0'})\
        .select(['t0', 'x0', 't1', 'x1', 'xm1', 'has_prev', 'filter_value',
                 'createtime', 'future_createtime'])

    return paired


def rolling_frame(cfg, target_df, dims):
    """The per-timestep frame `build_horizon_pairs` reads, smoothed if needed.

    A gpfa latent is smooth in time by construction, so it only gets the
    columns the pair builder indexes; a precomputed coord gets the rolling mean
    first.
    """
    if cfg.latents.method != 'gpfa':
        return compute_rolling_means(cfg, target_df, dims)
    return target_df.with_columns(
        [pl.col('x0').arr.get(i).alias(f'x0_{i}') for i in dims]
        + [((pl.col('createtime') - INITIAL_DATE).dt.total_days() / UNIT_DAYS).alias('t')])


def load_target_df(cfg):
    """Load coords and apply early filtering: platform + non-empty filter_value, sorted, renamed."""
    target_path = os.path.join(cfg.trend_path, f'{cfg.dim_reduction_method}_coords.parquet.zstd')
    target_df = pl.read_parquet(target_path)
    coord_col = [c for c in target_df.columns if c.startswith('coord_')][0]

    target_df = latent_space.keep_platform(cfg, target_df).filter(pl.col('filter_value') != '')
    target_df = target_df.select(['createtime', 'filter_value', coord_col])\
        .sort(['filter_value', 'createtime'])\
        .rename({coord_col: 'x0'})
    return target_df


def build_training_pairs(cfg, target_df, smooth=True, max_step_days=10):
    """Build training-time 1-step pairs (rolling mean + 1-step shift + timestep<10d filter).

    smooth=False skips the rolling mean, for latents that are already smooth in
    time — smoothing them again would only widen the window over which x0 and
    x1 share data.

    max_step_days drops pairs that straddle a gap; it has to exceed the grid
    spacing of whatever produced target_df.

    No shuffle here — apply_split shuffles for split_type='random'.
    """
    n_dims = cfg.n_dims
    wide = target_df.with_columns([
            pl.col('x0').arr.get(i).alias(f'x0_{i}') for i in range(n_dims)
        ])
    if smooth:
        wide = wide.rolling('createtime', period=f'{cfg.rolling_mean_window}d',
                            group_by='filter_value') \
            .agg([pl.col(f'x0_{i}').mean() for i in range(n_dims)])
    paired = wide\
        .with_columns(((pl.col('createtime') - INITIAL_DATE).dt.total_days() / UNIT_DAYS).alias('t0'))\
        .sort(['filter_value', 't0'])\
        .with_columns(
            [pl.col('t0').shift(-1).over('filter_value').alias('t1'),
             pl.col('createtime').shift(-1).over('filter_value').alias('next_createtime')] + \
            [pl.col(f'x0_{i}').shift(-1).over('filter_value').alias(f'x1_{i}') for i in range(n_dims)] + \
            [pl.col(f'x0_{i}').shift(1).over('filter_value').alias(f'xm1_{i}') for i in range(n_dims)]
        )\
        .drop_nulls([f'x0_{i}' for i in range(n_dims)] + [f'x1_{i}' for i in range(n_dims)])\
        .with_columns([
            pl.concat_arr([f'x0_{i}' for i in range(n_dims)]).alias('x0'),
            pl.concat_arr([f'x1_{i}' for i in range(n_dims)]).alias('x1'),
            pl.concat_arr([f'xm1_{i}' for i in range(n_dims)]).alias('xm1'),
            pl.col('xm1_0').is_not_null().alias('has_prev'),
        ])\
        .select(['t0', 'x0', 't1', 'x1', 'xm1', 'has_prev', 'filter_value',
                 'createtime', 'next_createtime'])\
        .with_columns(
            (pl.col('next_createtime') - pl.col('createtime')).alias('timestep'),
        )\
        .filter(pl.col('timestep') < pl.duration(days=max_step_days))\
        .drop(['timestep'])

    return paired


def load_latent_df(cfg, spec):
    """Trajectories in latent space, fitted inside the split boundary.

    The latent-GP factor model is fitted here rather than read from disk so
    that its global parameters only ever see training trajectories inside the
    training period; precomputed coords cannot make that guarantee.
    """
    if cfg.latents.method != 'gpfa':
        return load_target_df(cfg)

    df, _, _ = latent_space.load(cfg, spec=spec)
    return latent_space.keep_platform(cfg, df).filter(pl.col('filter_value') != '')\
        .rename({latent_space.COORD: 'x0'})


# Everything that changes the trained landscape. Eval-only settings are left
# out so that re-scoring a model at a new horizon does not re-key its directory.
LANDSCAPE_FIELDS = (
    'platform', 'rolling_mean_window', 'batch_size', 'num_epochs', 'min_epochs',
    'patience', 'train_fraction', 'sigma', 'confine', 'confinement_factor',
    'dt', 'vbt_tol', 'dt_min', 'dt_max', 'solver', 'model_do_sample',
    'phi_hidden_dims', 'phi_hidden_acts', 'phi_final_act',
    'phi_layer_normalize', 'phi_layer_dropout', 'init_phi_weights_method',
    'init_phi_weights_args', 'init_phi_bias_method', 'init_phi_bias_args',
    'nepochs_warmup', 'nepochs_decay', 'optimizer', 'momentum', 'weight_decay',
    'clip', 'lr_schedule', 'learning_rate', 'final_learning_rate',
    'peak_learning_rate', 'warmup_cosine_decay_exponent', 'fix_noise',
    'loss_fn_key', 'loss_fn_kernel', 'loss_fn_bw', 'cont_path',
)


def _normalise(v):
    """Numbers to float, so hydra handing back 100 where the default is 100.0
    does not key a second directory for the same configuration."""
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, (list, tuple)):
        return [_normalise(x) for x in v]
    if isinstance(v, dict):
        return {k: _normalise(x) for k, x in sorted(v.items())}
    return str(v)


def landscape_tag(cfg):
    """Key for the trained landscape, given the representation it sits on."""
    vals = {k: cfg.get(k) for k in LANDSCAPE_FIELDS}
    # not a LatentConfig field -- it picks which cached column the landscape is
    # trained on, so it changes the model without changing the fit
    vals['causal_state'] = cfg.latents.get('causal_state')
    body = json.dumps(_normalise(omegaconf.OmegaConf.to_object(
        omegaconf.OmegaConf.create(vals))), sort_keys=True)
    return hashlib.blake2b(body.encode(), digest_size=6).hexdigest()


def run_dir(cfg):
    """Output directory for one trained landscape, under the fit it was trained on.

    Nested rather than flat because a representation is shared by every trial
    that only retunes the landscape, and a fit costs far more than a landscape.
    """
    return os.path.join(latent_space.latent_dir(cfg), landscape_tag(cfg))


def write_run_record(cfg, dir_path, obj_key, objective):
    """Name the wandb run that wrote this checkpoint, so a sweep result can be
    traced back to a model without re-running the trial."""
    record = {
        'run_id': wandb.run.id,
        'run_name': wandb.run.name,
        'sweep_id': wandb.run.sweep_id,
        'objective_key': obj_key,
        'objective': objective,
        'finished': datetime.datetime.now().isoformat(timespec='seconds'),
        'config': omegaconf.OmegaConf.to_object(cfg),
    }
    with open(os.path.join(dir_path, 'run.json'), 'w') as fh:
        json.dump(record, fh, indent=1, default=str)


def compute_training_split(cfg, target_df=None):
    """Replicate training's train/val split metadata.

    Returns:
        (val_filter_values, cutoff_time):
          - val_filter_values: list[str] held out for val (split_type='filter_value'), else None
          - cutoff_time: datetime cutoff (split_type='time'), else None
        For split_type='random', both are None — apply_split handles random splits inline.

    If target_df is None, rebuilds it from cfg via load_target_df + build_training_pairs.
    Pass an existing target_df (e.g. from training) to avoid redundant work.
    """
    if cfg.split_type in ('random', 'nested'):
        return None, None

    if target_df is None:
        target_df = build_training_pairs(cfg, load_target_df(cfg))

    if cfg.split_type == 'filter_value':
        filter_values = target_df['filter_value'].unique().shuffle(seed=42).to_list()
        num_train = int(len(filter_values) * cfg.train_fraction)
        return filter_values[num_train:], None
    elif cfg.split_type == 'time':
        sorted_df = target_df.sort('createtime')
        cutoff_idx = int(len(sorted_df) * cfg.train_fraction)
        return None, sorted_df['createtime'].item(cutoff_idx)
    else:
        raise ValueError(f"Unknown split_type: {cfg.split_type}. Must be 'random', 'filter_value', or 'time'")


def apply_split(df, split_type, train_fraction, val_filter_values=None,
                cutoff_time=None, spec=None, scenario='val_out', key=None):
    """Split df into train/val using metadata from compute_training_split.

    For 'random', shuffles df with seed=42 then takes head/tail.
    For 'filter_value' / 'time', filters df by val_filter_values / cutoff_time.
    For 'nested', `spec` and `scenario` select one cell of the trajectory x time
    design; the returned 'val' half is that cell, and `key` is the split key
    (see splits.assign_trajectory_split).
    """
    if split_type == 'nested':
        if spec is None:
            raise ValueError("split_type='nested' requires a SplitSpec")
        time_col = ('future_createtime' if 'future_createtime' in df.columns
                    else 'next_createtime')
        labelled = splits.label_pairs(df, spec, time_col=time_col, key=key)
        traj, time = scenario.rsplit('_', 1)
        return splits.training_rows(labelled), splits.select(labelled, traj, time)
    if split_type == 'random':
        df = df.sample(fraction=1.0, shuffle=True, seed=42)
        n_train = int(len(df) * train_fraction)
        return df.head(n_train), df.tail(len(df) - n_train)
    elif split_type == 'filter_value':
        if val_filter_values is None:
            raise ValueError("val_filter_values required for split_type='filter_value'")
        train_df = df.filter(~pl.col('filter_value').is_in(val_filter_values))
        val_df = df.filter(pl.col('filter_value').is_in(val_filter_values))
        return train_df, val_df
    elif split_type == 'time':
        if cutoff_time is None:
            raise ValueError("cutoff_time required for split_type='time'")
        train_df = df.filter(pl.col('createtime') < cutoff_time)
        val_df = df.filter(pl.col('createtime') >= cutoff_time)
        return train_df, val_df
    else:
        raise ValueError(f"Unknown split_type: {split_type}. Must be 'random', 'filter_value', or 'time'")


def load_seed_metadata(cfg):
    """Load seed metadata (MainType, Party) from the stance data.

    Keyed by whatever a trajectory is, so a handle-keyed run's breakdowns join
    rather than coming back all null.
    """
    traj_col = LatentConfig.from_cfg(cfg).traj_col \
        if cfg.latents.method == 'gpfa' else 'SeedName'
    dir_path = cfg.base_stance_path
    file_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.parquet.zstd')]
    df = pl.read_parquet(file_paths, columns=['seed'])
    return df.select([
        pl.col('seed').struct.field(traj_col).alias('SeedName'),
        pl.col('seed').struct.field('MainType'),
        pl.col('seed').struct.field('SubType'),
        pl.col('seed').struct.field('Party'),
    ]).unique('SeedName')


def _incremental_gain(d2, p2, m2, c, c_dm, c_pm):
    """Correlations and what the model adds over momentum, from six moments.

    Shared by the pooled metrics and the per-dimension ones, so the two are the
    same estimator applied to different moments rather than two definitions
    that could drift apart.
    """
    rho = c / np.sqrt(d2 * p2) if d2 > 0 and p2 > 0 else 0.0
    rho_dm = c_dm / np.sqrt(d2 * m2) if d2 > 0 and m2 > 0 else 0.0
    rho_pm = c_pm / np.sqrt(p2 * m2) if p2 > 0 and m2 > 0 else 0.0
    denom = 1.0 - rho_pm ** 2
    combined = ((rho ** 2 + rho_dm ** 2 - 2 * rho * rho_dm * rho_pm) / denom
                if denom > 1e-6 else max(rho ** 2, rho_dm ** 2))
    return float(rho), float(rho_dm), float(rho_pm), float(np.clip(combined, 0, 1))


def per_dimension_metrics(dim_moments, floor=1e-6):
    """The same gain computed within each moving dimension, then averaged.

    Pooled rho weights a dimension by its share of the displacement variance,
    so a configuration whose motion lives in one dimension is scored on an
    easier problem than one spread over six, and n_fast stops being a fair
    comparison. Averaging per-dimension rho removes the weighting; the price is
    that it no longer asks whether the joint direction is right.

    Dimensions that never move are excluded rather than scored, so they neither
    help nor hurt -- spurious_drift_ratio is what keeps that visible.
    """
    d2, p2, m2, c, c_dm, c_pm = (np.asarray(np.mean(a, axis=0), dtype=float).ravel()
                                 for a in dim_moments)
    if d2.size == 0 or not np.isfinite(d2).any() or d2.max() <= 0:
        return {}
    moving = d2 > floor * d2.max()
    if not moving.any():
        return {}

    rows = [_incremental_gain(d2[k], p2[k], m2[k], c[k], c_dm[k], c_pm[k])
            for k in np.flatnonzero(moving)]
    rho = float(np.mean([r[0] for r in rows]))
    rho_dm = float(np.mean([r[1] for r in rows]))
    gain = float(np.mean([r[3] - r[1] ** 2 for r in rows]))
    frozen = p2[~moving].sum()
    return {
        'direction_rho_per_dim': rho,
        'momentum_ceiling_per_dim': float(np.mean([r[1] ** 2 for r in rows])),
        'ceiling_gain_per_dim': gain,
        'n_moving_dims': int(moving.sum()),
        # predicted motion in directions that never move; excluded from the
        # average above, so it would otherwise be a free pass
        'spurious_drift_ratio': float(frozen / d2[moving].sum()),
    }


def compute_metrics(model_losses, baseline_losses, pred_losses=None, dots=None,
                    mom_losses=None, dots_dm=None, dots_pm=None,
                    dim_moments=None):
    """Evaluation metrics from per-sample losses, mean and median based.

    The mean ratio is dominated by the minority of pairs with a near-zero
    baseline, so it can be negative while the model beats the baseline on most
    pairs. median_skill and frac_better say what happens typically; skill_score
    says what happens to the total.

    Given the displacement statistics, skill decomposes exactly. Writing
    d = x1 - x0 for the observed motion and p = xhat - x0 for the predicted,

        skill = 2 * rho * R - R^2

    with R = sqrt(E|p|^2 / E|d|^2) the amplitude ratio and rho the correlation
    between predicted and observed motion. Skill therefore goes to zero as
    p -> 0 whatever the model has learned, which is what makes a collapsed
    model indistinguishable from a merely unhelpful one. rho does not: it is
    scale free, and skill_ceiling = rho^2 is the best skill any rescaling of
    this drift field could reach, attained at R = rho. A model with real
    direction and the wrong amplitude is a tuning problem; rho near zero means
    there is no direction to find.
    """
    model_mse = float(np.mean(model_losses))
    baseline_mse = float(np.mean(baseline_losses))
    model_med = float(np.median(model_losses))
    baseline_med = float(np.median(baseline_losses))
    out = {
        'model_mse': model_mse,
        'baseline_mse': baseline_mse,
        'skill_score': 1.0 - model_mse / baseline_mse if baseline_mse > 0 else 0.0,
        'model_median': model_med,
        'baseline_median': baseline_med,
        'median_skill': 1.0 - model_med / baseline_med if baseline_med > 0 else 0.0,
        'frac_better': float(np.mean(model_losses < baseline_losses)),
        'n': len(model_losses),
    }
    if pred_losses is None or dots is None:
        return out

    d2 = float(np.mean(baseline_losses))
    p2 = float(np.mean(pred_losses))
    c = float(np.mean(dots))
    rho = c / np.sqrt(d2 * p2) if d2 > 0 and p2 > 0 else 0.0
    ratio = np.sqrt(p2 / d2) if d2 > 0 else 0.0
    if dim_moments is not None:
        out.update(per_dimension_metrics(dim_moments))
    out.update({
        'displacement_ratio': float(ratio),
        'direction_rho': float(rho),
        'skill_ceiling': float(rho ** 2),
        # 1.0 means the drift is already scaled the way the correlation warrants
        'amplitude_vs_optimal': float(ratio / rho) if rho > 0 else float('inf'),
    })
    if mom_losses is None or dots_dm is None or dots_pm is None:
        return out

    # Momentum predicts the reverse of recent motion. On these latents it
    # reaches |rho| ~ 0.25 on its own, so a model scored only against
    # no-movement can look predictive while adding nothing a single line of
    # numpy does not already give.
    m2, c_dm, c_pm = (float(np.mean(x)) for x in (mom_losses, dots_dm, dots_pm))
    # Guarding these with `> 0` would read False for NaN and silently report a
    # momentum baseline of exactly zero, which is indistinguishable from one
    # that genuinely explains nothing.
    if not all(np.isfinite(v) for v in (m2, c_dm, c_pm)):
        raise ValueError('non-finite momentum moments: pairs without a '
                         'predecessor reached the scorer')
    # variance of the observed motion explained by the two predictors together
    _, rho_dm, rho_pm, combined = _incremental_gain(d2, p2, m2, c, c_dm, c_pm)
    out.update({
        'momentum_rho': rho_dm,
        'momentum_ceiling': rho_dm ** 2,
        'model_vs_momentum_rho': rho_pm,
        'combined_ceiling': combined,
        # what the landscape adds over the one-liner; the sweep optimises this
        'ceiling_gain': combined - rho_dm ** 2,
    })
    return out


def scenario_subsets(cell, n_dims, seed_df, breakdowns):
    """The (name, rows) pairs to score for one scenario cell."""
    cell = cell.with_columns(pl.col('filter_value').cast(pl.String)) \
        .join(seed_df, left_on='filter_value', right_on='SeedName', how='left') \
        .with_columns([pl.col('x0').arr.get(i).alias(f'x0_{i}') for i in range(n_dims)])
    subsets = [('overall', cell)]
    if not breakdowns:
        return subsets

    for i in range(n_dims):
        col = f'x0_{i}'
        mean_val = cell[col].mean()
        subsets.append((f'dim{i}_above_mean', cell.filter(pl.col(col) >= mean_val)))
        subsets.append((f'dim{i}_below_mean', cell.filter(pl.col(col) < mean_val)))

    for col in ['MainType', 'SubType', 'Party']:
        if col not in cell.columns:
            continue
        for val in cell.drop_nulls(col).filter(pl.col(col) != '')[col].unique().sort().to_list():
            subsets.append((f'{col}_{val}', cell.filter(pl.col(col) == val)))
    return subsets


# Evaluation cost grows sublinearly in batch size -- 8x the pairs costs about
# 2x the time -- so scoring a scenario in one large batch is far cheaper than
# in many small ones. Grouping pairs by their (t0, t1) makes it worse, not
# better, for the same reason: it produces many tiny solves.
#
# The solver's step sizes cannot be loosened after the fact: dt0, dt_min,
# dt_max and vbt_tol are baked in when the model is constructed, and
# overwriting the attributes leaves the solve bit-identical.


def evaluate_pairs(model, frame, key, batch_size):
    """Score (x0, x1) pairs in the largest batches the caller allows."""
    model = eqx.tree_inference(model, True)
    names = ('model', 'base', 'pred', 'dot', 'mom', 'dot_dm', 'dot_pm')
    dim_names = ('d2', 'p2', 'm2', 'dp', 'dm', 'pm')
    parts = {k: [] for k in names + dim_names}
    x0 = np.stack(frame['x0'].to_numpy())[:, None, :]
    x1 = np.stack(frame['x1'].to_numpy())[:, None, :]
    xm1 = np.stack(frame['xm1'].to_numpy())[:, None, :]
    t0 = frame['t0'].to_numpy()
    t1 = frame['t1'].to_numpy()
    for start in tqdm(range(0, len(frame), batch_size), desc="    Batches"):
        sl = slice(start, start + batch_size)
        key, subkey = jax.random.split(key)
        out = _eval_batch(model, jnp.asarray(t0[sl]), jnp.asarray(t1[sl]),
                          jnp.asarray(x0[sl]), jnp.asarray(x1[sl]), subkey,
                          jnp.asarray(xm1[sl]))
        for name, arr in zip(names + dim_names, out):
            parts[name].append(np.array(arr))
    # one trailing element, so compute_metrics(*evaluate_pairs(...)) still lines up
    return (tuple(np.concatenate(parts[k]) for k in names)
            + (tuple(np.concatenate(parts[k]) for k in dim_names),))


def subsample(cell, cfg):
    """Cap a scenario cell so evaluation cost does not track trajectory count.

    Drawn with a fixed seed so the estimate is a property of the model rather
    than of which pairs were picked.
    """
    cap = cfg.get('eval_max_pairs') or 0
    if cap and len(cell) > cap:
        return cell.sample(n=cap, seed=cfg.split.seed)
    return cell


def evaluate_scenarios(model, labelled, cfg, key, n_dims, seed_df, prefix,
                       breakdown_scenario=None):
    """Score every scenario cell of the nested split and log it to wandb.

    Skill against the no-movement baseline is the headline rather than raw MSE:
    a smoother latent lowers both, so MSE would reward smoothing the latent
    into a constant, while skill is roughly invariant to it.
    """
    train = splits.training_rows(labelled)
    wandb_metrics = {}
    results = {}

    for traj in splits.TRAJ_SPLITS:
        for time in splits.TIME_SPLITS:
            name = splits.scenario_name(traj, time)
            cell = splits.select(labelled, traj, time)
            splits.check_leakage(train, cell, name)
            if len(cell) == 0:
                logger.info(f'  {prefix}/{name}: no pairs, skipping')
                continue
            cell = subsample(cell.filter(pl.col('has_prev')), cfg)

            for sub_name, sub in scenario_subsets(
                    cell, n_dims, seed_df, breakdowns=(name == breakdown_scenario)):
                if len(sub) == 0:
                    continue
                key, subkey = jax.random.split(key)
                metrics = compute_metrics(*evaluate_pairs(
                    model, sub, subkey, cfg.eval_batch_size))
                if sub_name == 'overall':
                    results[name] = metrics
                    logger.info(
                        f"  {prefix}/{name}: gain={metrics.get('ceiling_gain', 0):.5f} "
                        f"ceiling={metrics['skill_ceiling']:.5f} "
                        f"mom={metrics.get('momentum_ceiling', 0):.5f} "
                        f"rho={metrics['direction_rho']:+.4f} "
                        f"R={metrics['displacement_ratio']:.5f} "
                        f"skill={metrics['skill_score']:+.5f} "
                        f"median_skill={metrics['median_skill']:+.5f} "
                        f"frac_better={metrics['frac_better']:.3f} n={metrics['n']}")
                for k, v in metrics.items():
                    tag = f'{prefix}/{name}' if sub_name == 'overall' \
                        else f'{prefix}/{name}/{sub_name}'
                    wandb_metrics[f'{tag}/{k}'] = v

    for k, v in wandb_metrics.items():
        wandb.run.summary[k] = v
    return results


def write_scenario_metrics(results, cfg, dir_path, prefix):
    """Persist scenario scores next to the checkpoint.

    The rolling-holdout comparison needs these across runs, and reading them
    back from disk keeps it independent of whether wandb was reachable.
    """
    # latent_tag identifies the configuration exactly, so a rolling scan can
    # tell its own folds apart from every other run left under out/
    rows = [dict(scenario=name, prefix=prefix,
                 holdout_days=cfg.split.holdout_days,
                 origin_offset_days=cfg.split.origin_offset_days,
                 n_dims=cfg.n_dims, n_fast=cfg.latents.n_fast,
                 fast_tau=cfg.latents.fast_tau, fast_kind=cfg.latents.fast_kind,
                 slow_kind=cfg.latents.slow_kind,
                 latent_tag=LatentConfig.from_cfg(cfg).tag, **m)
            for name, m in results.items()]
    if not rows:
        return
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, f'scenario_metrics_{prefix}.parquet.zstd')
    pl.from_dicts(rows).write_parquet(path, compression='zstd')
    logger.info(f'wrote {path}')


def compute_per_sample_mse(y_pred, y_true):
    """Compute per-sample MSE (squared L2 distance)."""
    return jnp.sum(jnp.square(y_pred - y_true), axis=(-2, -1))


@eqx.filter_jit
def _eval_batch(model, t0, t1, y0, y1, key, ym1=None):
    """JIT-compiled evaluation of a single batch.

    Cross-moments are accumulated directly rather than recovered from squared
    norms: in float32 those differ by far less than their own magnitude, so the
    subtraction would keep almost none of its precision.

    m = -(y0 - ym1) is the momentum rule, which predicts the reverse of the
    trajectory's own recent motion. Its moments travel alongside the model's so
    that compute_metrics can ask what the landscape adds beyond it.
    """
    y_pred = model(t0, t1, y0, key)
    p = y_pred - y0
    d = y1 - y0
    m = jnp.zeros_like(p) if ym1 is None else -(y0 - ym1)
    axes = (-2, -1)
    # the same six moments again, keeping the dimension axis: per_dimension_metrics
    # needs them within a dimension, and the pooled sums cannot be split back up
    per = lambda a: jnp.sum(a, axis=-2)
    return (compute_per_sample_mse(y_pred, y1),
            compute_per_sample_mse(y0, y1),
            jnp.sum(jnp.square(p), axis=axes),
            jnp.sum(d * p, axis=axes),
            jnp.sum(jnp.square(m), axis=axes),
            jnp.sum(d * m, axis=axes),
            jnp.sum(p * m, axis=axes),
            per(jnp.square(d)), per(jnp.square(p)), per(jnp.square(m)),
            per(d * p), per(d * m), per(p * m))


def evaluate_dataloader(model, dataloader, key, with_displacement=False):
    """Evaluate model and no-movement baseline on a dataloader.

    Returns arrays of per-sample MSE for the model and for the baseline, and
    with_displacement also the predicted squared displacement and its dot
    product with the observed one, which compute_metrics turns into rho and R.
    """
    inference_model = eqx.tree_inference(model, True)
    model_losses = []
    baseline_losses = []
    pred_losses = []
    dots = []

    for data in tqdm(dataloader, desc="    Batches"):
        inputs, y1 = data
        t0, y0, t1 = inputs

        key, subkey = jax.random.split(key)
        model_mse, baseline_mse, pred_mse, dot = _eval_batch(
            inference_model, t0, t1, y0, y1, subkey)[:4]

        model_losses.append(np.array(model_mse))
        baseline_losses.append(np.array(baseline_mse))
        if with_displacement:
            pred_losses.append(np.array(pred_mse))
            dots.append(np.array(dot))

    out = (np.concatenate(model_losses), np.concatenate(baseline_losses))
    if with_displacement:
        out = out + (np.concatenate(pred_losses), np.concatenate(dots))
    return out

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    project_name = 'potential_landscape_training'
    wandb_config = omegaconf.OmegaConf.to_object(cfg)
    wandb.init(project=project_name, config=wandb_config)

    n_dims = cfg.n_dims
    spec = splits.SplitSpec.from_cfg(cfg)
    dir_path = run_dir(cfg)
    logger.info(f'split {spec}  ->  {dir_path}')

    if cfg.latents.method != 'gpfa':
        logger.warning(
            'precomputed coords were fitted over all trajectories and all time, '
            'so held-out scenarios are only held out of the landscape model, '
            'not of the representation')

    target_df = load_latent_df(cfg, spec)

    # Logged, not optimised: whether a fit the forecast likes is also one whose
    # dimensions can be named is the trade-off the sweep is there to expose.
    for k, v in latent_space.dimension_quality(cfg).items():
        wandb.run.summary[f'dim_quality/{k}'] = v
        logger.info(f'dim_quality/{k} = {v:.4f}')

    smooth = cfg.latents.method != 'gpfa'
    rolling_df = rolling_frame(cfg, target_df, list(range(n_dims)))

    grid_days = cfg.latents.interp_days or 2 * cfg.latents.bin_factor
    pairs = build_training_pairs(
        cfg, target_df, smooth=smooth,
        max_step_days=10 if smooth else 1.5 * grid_days)
    split_key = latent_space.split_key(cfg)
    labelled = splits.label_pairs(pairs, spec, time_col='next_createtime',
                                  key=split_key)
    logger.info('pair counts by scenario:\n' + str(splits.summarise(labelled)))

    train_df = splits.training_rows(labelled)
    # Early stopping uses every validation trajectory, in and out of time: the
    # out-of-time cell alone is too small to stop on at short holdout windows.
    val_df = splits.select(labelled, 'val', splits.TIME_SPLITS)
    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError(f'empty train ({len(train_df)}) or val ({len(val_df)}) set')

    train_dataset = LandscapeSimulationDataset(data=df_to_data(train_df))
    valid_dataset = LandscapeSimulationDataset(data=df_to_data(val_df))

    batch_size = cfg.batch_size
    train_dataloader = NumpyLoader(
        train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)
    valid_dataloader = NumpyLoader(
        valid_dataset, batch_size=min(batch_size, len(valid_dataset)), shuffle=False)

    scheduler_kwargs = {
        'dt': cfg.dt,
        'dt_schedule_bounds': [0, 1],
        'dt_schedule_scales': [1.0, 0.1]
    }
    dt_schedule = get_dt_schedule('stepped', scheduler_kwargs)

    rng = np.random.default_rng(seed=42)
    key = jax.random.PRNGKey(int(rng.integers(2**32)))
    key, modelkey, initkey, trainkey = jax.random.split(key, 4)

    # Confinement is set from the training states only; the holdout must not
    # get to widen the box it is scored inside.
    confinement_threshold = np.max(np.linalg.norm(train_df['x0'].to_numpy(), axis=1)) * 1.1

    dtype = jnp.float32
    args_make = {
        'ndims' : n_dims,
        'nparams' : n_dims,
        'ncells' : len(train_dataset),
        'sigma_init' : cfg.sigma,
        'confine' : cfg.confine,
        'confinement_factor' : cfg.confinement_factor,
        'confinement_threshold' : confinement_threshold,
        'dt0' : cfg.dt,
        'vbt_tol' : cfg.vbt_tol,
        'dt_min' : cfg.dt_min,
        'dt_max' : cfg.dt_max,
        'solver' : cfg.solver,
        'sample_cells' : cfg.model_do_sample,
        'include_phi_bias' : False,
        'phi_hidden_dims' : list(cfg.phi_hidden_dims),
        'phi_hidden_acts' : cfg.phi_hidden_acts,
        'phi_final_act' : cfg.phi_final_act,
        'phi_layer_normalize' : cfg.phi_layer_normalize,
        'phi_layer_dropout' : cfg.phi_layer_dropout,
    }
    args_init = {
        'init_phi_weights_method' : cfg.init_phi_weights_method,
        'init_phi_weights_args' : cfg.init_phi_weights_args,
        'init_phi_bias_method' : cfg.init_phi_bias_method,
        'init_phi_bias_args' : cfg.init_phi_bias_args,
    }

    if cfg.cont_path:
        logger.info(f'Loading model from {cfg.cont_path}...')
        model, hyperparams = DeepTimePhiPLNN.load(cfg.cont_path, dtype=dtype)
        if cfg.dt > 0:
            logger.info(f"Overwriting loaded model's dt0. Was {model.dt0}. Now {cfg.dt}.")
            model = eqx.tree_at(lambda m: m.dt0, model, cfg.dt)
            hyperparams['dt0'] = cfg.dt
    else:
        logger.info('Constructing new model...')
        model, hyperparams = DeepTimePhiPLNN.make_model(key=modelkey, dtype=dtype, **args_make)
        model = model.initialize(initkey, dtype=dtype, **args_init)

    loss_fn = select_loss_function(cfg.loss_fn_key, kernel=cfg.loss_fn_kernel,
                                   bw_range=cfg.loss_fn_bw)
    optimizer = select_optimizer(
        'rms', get_optimizer_args(cfg, cfg.num_epochs),
        batch_size=batch_size, dataset_size=len(train_dataset))

    plotting_opts = {
        'equal_axes': True,
        'plot_radius': cfg.plot_radius,
        'plot_losses': True,
        'plot_sigma_hist': True,
        'sigma_true': None,
    }

    model = train_model(
        model, loss_fn, optimizer, train_dataloader, valid_dataloader,
        key=trainkey, num_epochs=cfg.num_epochs, batch_size=batch_size,
        min_epochs=cfg.min_epochs, patience=cfg.patience, dt_schedule=dt_schedule,
        hyperparams=hyperparams, plotting_opts=plotting_opts,
        reduce_dt_on_nan=True, reduce_cf_on_nan=True,
        logprint=logger.info, outdir=dir_path,
    )

    # train_model returns the best epoch but only checkpoints improvements, so
    # after an early stop the newest file on disk is the epoch that triggered
    # it. Every downstream script loads the newest, so write the scored model
    # last and they get the one these metrics describe.
    model.save(os.path.join(dir_path, 'states', 'model_best.pth'), hyperparams)

    # A field one linear map reproduces has learned no landscape: momentum
    # cannot predict reversion, so a uniform pull inward scores against it
    # without explaining anything. Logged, not optimised.
    held = splits.select(labelled, 'val', 'out')
    for k, v in drift_linearity.report(
            model,
            held['t0'].to_numpy().astype(np.float32),
            held['x0'].to_numpy().astype(np.float32)).items():
        wandb.run.summary[k] = v
        logger.info(f'{k} = {v:.4f}')

    seed_df = load_seed_metadata(cfg)
    key, evalkey = jax.random.split(key)
    scored = {}
    scored['step'] = evaluate_scenarios(
        model, labelled, cfg, evalkey, n_dims, seed_df, prefix='step',
        breakdown_scenario=cfg.breakdown_scenario)
    write_scenario_metrics(scored['step'], cfg, dir_path, 'step')

    for horizon_days in cfg.eval_horizons:
        key, evalkey = jax.random.split(key)
        paired = build_horizon_pairs(rolling_df, horizon_days, list(range(n_dims)))
        if len(paired) == 0:
            logger.info(f'{horizon_days}d: no pairs found, skipping')
            continue
        prefix = f'horizon_{horizon_days}d'
        scored[prefix] = evaluate_scenarios(
            model, splits.label_pairs(paired, spec, time_col='future_createtime',
                                      key=split_key),
            cfg, evalkey, n_dims, seed_df, prefix=prefix)
        write_scenario_metrics(scored[prefix], cfg, dir_path, prefix)

    # Sweeps select on val_out and report test_out once; selecting on test_out
    # would spend the only estimate that is clean in both dimensions.
    obj_key = f'{cfg.objective_prefix}/{cfg.objective_scenario}/{cfg.objective_metric}'
    objective = scored.get(cfg.objective_prefix, {}) \
        .get(cfg.objective_scenario, {}).get(cfg.objective_metric)
    if objective is None:
        logger.warning(f'objective {obj_key} unavailable; '
                       'the sweep has nothing to optimise')
    else:
        wandb.run.summary['objective'] = objective
        logger.info(f'objective ({obj_key}) = {objective:.5f}')

    write_run_record(cfg, dir_path, obj_key, objective)
    wandb.finish()


if __name__ == '__main__':
    main()