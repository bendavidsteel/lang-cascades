import hashlib
import json
import os
import warnings

import hydra
import numpy as np
import polars as pl
from omegaconf import OmegaConf
from tqdm import tqdm

import latent_space
import splits
from multiple_testing import benjamini_hochberg, significance_stars

# Heavy imports (jax, plnn, matplotlib) are deferred into main() so that
# import-time costs are only paid for the ETS/landscape branches that need
# them.


HORIZON_DAYS = [7, 14, 30, 60, 120, 240, 360, 720]
# Cells of the nested trajectory x time split, and what each one demonstrates.
# The landscape is always fitted on train_in (nn_potential labels its pairs with
# splits.label_pairs whatever cfg.split_type says), so these are the three
# distinct ways of being held out from it. train_in itself is not reported: it
# is the fit, not a result.
SCENARIOS = (
    ('test_in', 'Impute', 'unseen trajectories, seen time'),
    ('train_out', 'Forecast', 'seen trajectories, unseen time'),
    ('test_out', 'Zero-shot forecast', 'unseen trajectories, unseen time'),
)
# Minimum history (incl. t0) for a pair to enter the sample, as a floor and as
# a multiple of the forecast's own row-shift: a local method is never asked to
# extrapolate further than it has been allowed to observe. The floor is what
# ETS damped-trend / Theta-2 need to identify their parameters at all. Every
# method is scored on the pool this defines, so raising it costs sample size
# rather than comparability.
MIN_HISTORY = 5
HISTORY_HORIZON_RATIO = 1.0
# Half-width of the window a pair's actual gap must fall in, as a fraction of
# the requested horizon. Shared with build_horizon_pairs so the pool the
# baselines see is the pool the landscape is scored on.
TOLERANCE_FRAC = 0.25
# Target sample size per horizon. Pairs are drawn uniformly from the whole
# scenario cell (same population the landscape model is evaluated on).
PAIRS_PER_HORIZON = 30_000
ETS_SAMPLE_SEED = 42
# Gardner-McKenzie damping factor for Theta-2's trend extrapolation. Standard
# M-competition default; smaller = more aggressive damping (1.0 = no damping).
THETA_DAMPING_PHI = 0.98

# Caches store per-pair losses (one row per evaluated pair) rather than
# pre-computed summary stats, so plotting can use robust quartile summaries
# without re-running the (expensive) fits. Parquet + zstd because the per-pair
# row count grows quickly with #trajectories × #pairs/trajectory × #horizons.
#
# Every row also carries `fingerprint` (of the data, split, model and sampling
# that produced it) and `actual_days` (the horizon the row shift really spans).
# Rows whose fingerprint does not match the current run are ignored, so a cell
# recomputed under one configuration can never be plotted beside one left over
# from another.
COMMON_PAIR_COLS = {'horizon', 'fingerprint', 'actual_days'}
# Bump when the loss or the pair selection changes in a way that makes existing
# rows incomparable but leaves the fingerprint inputs untouched.
CACHE_FORMAT_VERSION = 2
# cfg entries that determine the trajectories, the representation and the
# split, and so which pairs exist and what they contain.
FINGERPRINT_CFG_FIELDS = ('trend_path', 'dim_reduction_method', 'platform',
                          'n_dims', 'rolling_mean_window', 'latents', 'split')

MODEL_PAIR_COLS = {'model_loss', 'baseline_loss'} | COMMON_PAIR_COLS
# ETS cache: Holt's damped-trend exponential-smoothing losses (ETS(A,Ad,N)) on
# a subsample of the smoothed trajectories the landscape model is
# trained/evaluated on, so the two methods predict the same target. Forecasts
# run `shift_n` rows ahead, the one global row-shift build_horizon_pairs uses.
# No-movement baseline is computed on the *same* subsample (so the ETS ratio is
# fair). The model cache holds the landscape scored on the whole cell instead,
# against its own no-movement baseline on that cell.
# We use ETS instead of ARIMA because the landscape model's training target is
# heavily smoothed (a wide rolling mean, or a GP latent), which makes ARIMA
# fits ill-conditioned (AR root → 1, recursive forecasts explode). Holt's
# damped-trend method is bounded by construction.
ETS_PAIR_COLS = {'ets_loss', 'ets_baseline_loss'} | COMMON_PAIR_COLS
# Theta cache: same pair sampling as ETS, but the per-pair forecast comes from
# the Theta-2 method (Assimakopoulos & Nikolopoulos 2000) — average of OLS
# linear-trend-on-time and simple exponential smoothing. Robust to
# near-constant series where ARIMA's AR-root estimation degenerates: the
# trend slope just goes to zero. We guard the (rare) exactly-constant case
# explicitly because statsmodels' MLE for SES diverges with zero variance.
THETA_PAIR_COLS = {'theta_loss', 'theta_baseline_loss'} | COMMON_PAIR_COLS
# Reversion caches: the two baselines that can do what no-movement, ETS and
# Theta cannot, which is revert. No-movement, Holt's damped *trend* and Theta's
# SES half all forecast the level as persistence, so any mean-reverting
# predictor beats them by a margin that widens with the horizon -- including a
# confinement term that was specified rather than learned. 'mean' predicts the
# training mean and 'ar1' shrinks toward it, both fitted on train_in only, so
# together they are the null for "the landscape learned a confining potential".
MEAN_PAIR_COLS = {'mean_loss', 'mean_baseline_loss'} | COMMON_PAIR_COLS
AR1_PAIR_COLS = {'ar1_loss', 'ar1_baseline_loss'} | COMMON_PAIR_COLS

# (cache name, filename stem, required columns, loss column, baseline column).
# One file per (cache, scenario) so a scenario can be recomputed on its own.
CACHES = (
    ('model', '', MODEL_PAIR_COLS, 'model_loss', 'baseline_loss'),
    ('model_shared', '_model_shared', MODEL_PAIR_COLS, 'model_loss', 'baseline_loss'),
    ('ets', '_ets', ETS_PAIR_COLS, 'ets_loss', 'ets_baseline_loss'),
    ('theta', '_theta', THETA_PAIR_COLS, 'theta_loss', 'theta_baseline_loss'),
    ('mean', '_mean', MEAN_PAIR_COLS, 'mean_loss', 'mean_baseline_loss'),
    ('ar1', '_ar1', AR1_PAIR_COLS, 'ar1_loss', 'ar1_baseline_loss'),
)
# How each cache is drawn in the panels, in legend order.
CACHE_STYLE = {
    'model': ('Potential landscape', 'o-'),
    'model_shared': ('Potential landscape', 'o-'),
    'ets': ('Holt damped trend', 's--'),
    'theta': ('Damped Theta', '^-.'),
    'mean': ('Training mean', 'v--'),
    'ar1': ('AR(1) to mean', 'P-.'),
}

# Plot aggregator. Per-pair variants compute model_loss_i / baseline_loss_i
# first, then aggregate; aggregate variants aggregate model_loss and baseline_loss
# separately, then divide. Per-pair is dominated by pairs with baseline ≈ 0
# (no-movement) — fine for median (robust) but blows up for mean (weighted by
# 1/baseline_i). Aggregate variants are robust to that pathology.
#   'median_per_pair' = median(model_i / baseline_i) with IQR bars
#   'mean_per_pair'   = mean(model_i / baseline_i) with bootstrap percentile CI
#   'mean_ratio'      = mean(model) / mean(baseline) with bootstrap percentile CI
#   'median_ratio'    = median(model) / median(baseline) with Q1/Q3 data-quantile bars
#   'absolute_mean'   = mean(loss) per method in raw MSE units, with the
#                       no-movement baseline plotted as its own line (log y).
#                       Uses model_shared cache so all methods compare against
#                       the same baseline pair set.
#   'absolute_median' = median(loss) per method in raw MSE units with Q1/Q3
#                       IQR bars; baseline plotted separately as above.
PLOT_AGGREGATOR = 'median_ratio'
ABSOLUTE_AGGREGATORS = ('absolute_mean', 'absolute_median')
# Which pairs the landscape is scored on. 'shared' is exactly the pairs the
# local methods could be fitted on, so every line in a panel comes from one pair
# set and the comparison is a true head-to-head; 'full' is every pair in the
# cell, which is the population estimate but not comparable to the baselines. Absolute mode
# needs 'shared' for the three no-movement lines to coincide.
LANDSCAPE_POOL = 'shared'
LANDSCAPE_CACHE = 'model' if (LANDSCAPE_POOL == 'full'
                              and PLOT_AGGREGATOR not in ABSOLUTE_AGGREGATORS) \
    else 'model_shared'
# The other landscape cache is neither computed nor plotted -- scoring a cell
# twice costs a forward pass per pair for a number nothing reads.
#
# Two comparisons, one of each kind: Theta extrapolates a damped trend, AR(1)
# reverts toward the training mean. Holt's damped trend is the same idea as
# Theta's and the training mean is the no-movement line with a worse constant,
# so a panel carrying all four spends its ink on the distinction it is not
# making. Their caches stay on disk; naming one here plots it again.
ACTIVE_CACHES = tuple(c for c in CACHES
                      if c[0] in (LANDSCAPE_CACHE, 'theta', 'ar1'))
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 0
BOOTSTRAP_CI_LO = 0.025
BOOTSTRAP_CI_HI = 0.975
# Horizon the significance test is reported at, fixed in advance. Testing at
# the horizon that happens to minimise the curve would condition the p-value on
# picking the most favourable of HORIZON_DAYS, so every horizon is reported and
# this one is the headline.
REPORT_HORIZON_DAYS = 360


def _digest(obj):
    return hashlib.blake2b(
        json.dumps(obj, sort_keys=True, default=str).encode(),
        digest_size=8).hexdigest()


def _file_digest(path):
    """Content hash of a checkpoint, so a retrain at the same path invalidates
    the rows it produced. Checkpoints are small enough to read whole."""
    if not path or not os.path.exists(path):
        return None
    with open(path, 'rb') as fh:
        return hashlib.blake2b(fh.read(), digest_size=8).hexdigest()


def cache_file(fig_path, stem, scenario, spec):
    """Where one (method, scenario) cache lives for this split.

    The origin is part of the name, and only when it is rolled back -- the same
    rule SplitSpec.tag follows, so the fixed-holdout caches keep the names they
    were written under while a fold gets its own file instead of overwriting
    them.
    """
    origin = f'_o{spec.origin_offset_days}' if spec.origin_offset_days else ''
    return os.path.join(
        fig_path, f'nn_potential_horizon_skill{stem}_{scenario}{origin}.parquet.zstd')


def cache_fingerprints(cfg, spec, state_path):
    """One fingerprint per cache name, covering everything that decides its rows.

    The pair population is shared, so it enters every fingerprint; the model
    checkpoint enters only the landscape's, and Theta's damping only Theta's.
    """
    shared = {
        'version': CACHE_FORMAT_VERSION,
        'cfg': {f: OmegaConf.to_container(cfg[f], resolve=True)
                if OmegaConf.is_config(cfg[f]) else cfg[f]
                for f in FINGERPRINT_CFG_FIELDS},
        'split': spec.tag,
        'min_history': MIN_HISTORY,
        'history_horizon_ratio': HISTORY_HORIZON_RATIO,
        'tolerance_frac': TOLERANCE_FRAC,
        'pairs_per_horizon': PAIRS_PER_HORIZON,
        'sample_seed': ETS_SAMPLE_SEED,
    }
    extra = {
        'model': {'state': _file_digest(state_path)},
        'model_shared': {'state': _file_digest(state_path)},
        'ets': {},
        'theta': {'phi': THETA_DAMPING_PHI},
        'mean': {},
        'ar1': {},
    }
    return {name: _digest({**shared, **extra[name]}) for name, *_ in CACHES}


def _load_pair_cache(path, required_cols, fingerprint):
    """Load a per-pair parquet cache. Returns {horizon: {col: np.ndarray}},
    or {} if the file is missing or doesn't satisfy required_cols. Rows written
    under a different fingerprint are dropped rather than mixed in.
    """
    if not os.path.exists(path):
        return {}
    df = pl.read_parquet(path)
    if not required_cols.issubset(df.columns):
        print(f"Cache {path} missing columns {required_cols - set(df.columns)}; ignoring.", flush=True)
        return {}
    stale = df.filter(pl.col('fingerprint') != fingerprint)
    if len(stale):
        horizons = sorted(stale['horizon'].unique().to_list())
        print(f"Cache {path}: dropping {len(stale)} rows from another "
              f"configuration at horizons {horizons}.", flush=True)
        df = df.filter(pl.col('fingerprint') == fingerprint)
    by_h = {}
    for grp_key, g in df.group_by('horizon', maintain_order=True):
        h = grp_key[0] if isinstance(grp_key, tuple) else grp_key
        by_h[int(h)] = {c: g[c].to_numpy() for c in g.columns if c != 'horizon'}
    return by_h


def _save_pair_cache(path, by_horizon):
    if not by_horizon:
        return
    frames = []
    for h in sorted(by_horizon):
        d = by_horizon[h]
        n = len(next(iter(d.values())))
        frames.append(pl.DataFrame({'horizon': np.full(n, h, dtype=np.int64), **d}))
    pl.concat(frames).write_parquet(path, compression='zstd')


def median_spacing_days(rolling_df):
    """Median gap between consecutive observations of a trajectory."""
    spacing_df = rolling_df.with_columns(
        (pl.col('createtime').diff().over('filter_value')).alias('dt')
    ).drop_nulls('dt')
    return spacing_df['dt'].median().total_seconds() / 86400


def horizon_shift(horizon_days, spacing_days):
    """Rows to shift for a horizon — build_horizon_pairs' derivation, kept here
    so idx_t1 - idx_t0 is exactly this in each trajectory's row ordering."""
    return max(1, int(round(horizon_days / spacing_days)))


def min_history(shift_n):
    return max(MIN_HISTORY, int(np.ceil(HISTORY_HORIZON_RATIO * shift_n)))


def usable_horizons(horizons, spacing_days, tolerance_frac=TOLERANCE_FRAC):
    """Horizons a whole number of rows can land within tolerance of.

    A requested horizon below the observation spacing rounds to a shift whose
    span falls outside its own tolerance window, so it yields no pairs at all;
    report it once rather than rediscovering it per scenario. The kept
    horizons are requested values, not realised ones -- `actual_days` carries
    what each row shift really spans.
    """
    keep = []
    for h in horizons:
        realised = horizon_shift(h, spacing_days) * spacing_days
        if abs(realised - h) <= max(h * tolerance_frac, 3):
            keep.append(h)
        else:
            print(f"Skipping {h}d: the nearest whole row shift spans "
                  f"{realised:.1f}d at {spacing_days:.1f}d spacing, outside "
                  f"its tolerance window.", flush=True)
    return keep


def scenario_pairs(paired, spec, scenario, split_key):
    """The pairs in one cell of the nested trajectory x time split.

    Not routed through `apply_split`/`cfg.split_type`: the landscape is fitted
    on train_in of this split whatever split_type says, so anything else would
    score it against a held-out set it was never held out from.
    """
    labelled = splits.label_pairs(paired, spec, time_col='future_createtime',
                                  key=split_key)
    traj, time = scenario.rsplit('_', 1)
    cell = splits.select(labelled, traj, time)
    splits.check_leakage(splits.training_rows(labelled), cell, scenario)
    return cell


def _store(cache, path, horizon, label, losses, baseline_losses,
           loss_key, baseline_key, fingerprint, actual_days):
    """Log one horizon's per-pair losses and append them to the cache on disk."""
    losses = np.asarray(losses, dtype=np.float64)
    baseline_losses = np.asarray(baseline_losses, dtype=np.float64)
    if len(losses) == 0:
        print(f"  [{label}] no eligible pairs at {horizon}d, skipping.", flush=True)
        return
    mse = float(np.mean(losses))
    base_mse = float(np.mean(baseline_losses))
    ratio = mse / base_mse if base_mse > 0 else float('nan')
    print(
        f"  [{label}] n={len(losses)} mse={mse:.6f} baseline_mse={base_mse:.6f} "
        f"ratio={ratio:.4f} frac_better={float(np.mean(losses < baseline_losses)):.3f}",
        flush=True,
    )
    cache[horizon] = {
        loss_key: losses,
        baseline_key: baseline_losses,
        'fingerprint': np.full(len(losses), fingerprint, dtype=object),
        'actual_days': np.full(len(losses), actual_days, dtype=np.float64),
    }
    _save_pair_cache(path, cache)


def _select_horizon_pairs(rolling_df, horizon_days, dim_cols, spec, scenario,
                          split_key, n_pairs, seed, tolerance_frac, dims):
    """Random sample of a scenario's pairs (with attached histories) for ts
    evaluation.

    Pulls from the same pool the landscape model is evaluated on
    (`build_horizon_pairs` + `scenario_pairs`), then:
      1. Joins each pair to its row index in the trajectory's smoothed series
         (so callers can slice history up to and including t0).
      2. Drops pairs whose history at t0 has fewer than `min_history(shift_n)`
         points.
      3. Uniformly subsamples to ≤ `n_pairs` pairs.
      4. Sorts by (filter_value, idx_t0). Every method then walks the pairs in
         one order, and `df_to_data`'s partition_by preserves it, so the cached
         per-pair losses line up row for row across caches.

    Forecast horizon for ts methods is `shift_n` rows, the global row-shift
    used by `build_horizon_pairs` (= round(horizon_days / median spacing)).

    Returns (pair_specs, traj_arrays).
    """
    from nn_potential import build_horizon_pairs

    paired = build_horizon_pairs(rolling_df, horizon_days, dims, tolerance_frac)
    if len(paired) == 0:
        return [], {}
    cell_paired = scenario_pairs(paired, spec, scenario, split_key)
    if len(cell_paired) == 0:
        return [], {}

    shift_n = horizon_shift(horizon_days, median_spacing_days(rolling_df))

    rolling_df = rolling_df.sort(['filter_value', 'createtime'])
    rolling_with_idx = rolling_df.with_columns(
        pl.int_range(pl.len()).over('filter_value').alias('idx_t0')
    )
    cell_with_idx = cell_paired.join(
        rolling_with_idx.select(['filter_value', 'createtime', 'idx_t0']),
        on=['filter_value', 'createtime'], how='inner',
    ).filter(pl.col('idx_t0') >= min_history(shift_n) - 1)

    n_avail = len(cell_with_idx)
    if n_avail == 0:
        return [], {}
    if n_avail > n_pairs:
        rng = np.random.default_rng(seed)
        chosen_idx = np.sort(rng.choice(n_avail, size=n_pairs, replace=False))
        cell_with_idx = cell_with_idx[chosen_idx]
    cell_with_idx = cell_with_idx.sort(['filter_value', 'idx_t0'])

    # Build per-trajectory value arrays only for trajectories that contributed
    # a sampled pair — keeps memory and traversal cost proportional to N.
    used_fvs = set(cell_with_idx['filter_value'].unique().to_list())
    traj_arrays = {}
    for grp_key, g in rolling_df.filter(pl.col('filter_value').is_in(list(used_fvs))) \
            .group_by('filter_value', maintain_order=True):
        fv = grp_key[0] if isinstance(grp_key, tuple) else grp_key
        traj_arrays[fv] = g.select(dim_cols).to_numpy()

    pair_specs = []
    for row in cell_with_idx.iter_rows(named=True):
        fv = row['filter_value']
        idx = int(row['idx_t0'])
        x0 = np.asarray(row['x0'], dtype=np.float64)
        x1 = np.asarray(row['x1'], dtype=np.float64)
        spec = {
            'filter_value': fv,
            'idx_t0': idx,
            'shift_n': shift_n,
            'x0': x0,
            'x1': x1,
            't0': float(row['t0']),
            't1': float(row['t1']),
        }
        pair_specs.append(spec)

    return pair_specs, traj_arrays


def _evaluate_pairs_with_method(pair_specs, traj_arrays, n_dims, horizon_days,
                                fit_forecast_1d, method_name):
    """For each pair, fit fit_forecast_1d on history up to and including t0
    and compare the shift_n-step forecast to the smoothed x1 observation.
    Method is fit per-dimension. Baseline is no-movement on the same pairs.

    fit_forecast_1d(hist_d, shift_n) -> forecast value at step shift_n.
    Returns (method_losses, baseline_losses) numpy arrays in matching order.
    """
    if len(pair_specs) == 0:
        return np.array([]), np.array([])

    shifts = [s['shift_n'] for s in pair_specs]
    n_traj = len({s['filter_value'] for s in pair_specs})
    print(
        f"    {method_name} {horizon_days}d: {len(pair_specs)} pairs across {n_traj} trajectories, "
        f"shift_n range [{min(shifts)}, {max(shifts)}], median {int(np.median(shifts))}",
        flush=True,
    )

    # Per-trajectory full historical range — magnitude guard. ETS(A,Ad,N) and
    # Theta are bounded in expectation by construction, but we keep a cheap
    # sanity check in case a degenerate fit produces a wildly out-of-range
    # forecast.
    traj_range = {fv: np.maximum(np.ptp(arr, axis=0), 1e-6) for fv, arr in traj_arrays.items()}
    EXPLOSIVE_FACTOR = 10.0

    method_losses = []
    baseline_losses = []
    n_fit_failed = 0
    n_explosive = 0
    first_error_repr = None

    pbar = tqdm(total=len(pair_specs), desc=f"    {method_name} {horizon_days}d")
    try:
        for spec in pair_specs:
            idx = spec['idx_t0']
            shift_n = spec['shift_n']
            x0 = spec['x0']
            x1 = spec['x1']
            fv = spec['filter_value']
            history = traj_arrays[fv][: idx + 1]
            ranges = traj_range[fv]

            forecast = np.empty(n_dims)
            for d in range(n_dims):
                hist_d = np.ascontiguousarray(history[:, d], dtype=np.float64)
                last_d = float(hist_d[-1])
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        val = fit_forecast_1d(hist_d, shift_n)
                except Exception as e:
                    n_fit_failed += 1
                    if first_error_repr is None:
                        first_error_repr = repr(e)
                    forecast[d] = last_d
                    continue

                if not np.isfinite(val) \
                        or abs(val - last_d) > EXPLOSIVE_FACTOR * float(ranges[d]):
                    n_explosive += 1
                    forecast[d] = last_d
                else:
                    forecast[d] = val

            loss = float(np.sum((forecast - x1) ** 2))
            if not np.isfinite(loss):
                loss = float(np.sum((x0 - x1) ** 2))
            method_losses.append(loss)
            baseline_losses.append(float(np.sum((x0 - x1) ** 2)))
            pbar.update(1)
    finally:
        pbar.close()

    if first_error_repr:
        print(f"    {method_name} first fit error: {first_error_repr}", flush=True)

    if n_fit_failed or n_explosive:
        print(
            f"    {method_name} fallbacks: {n_fit_failed} fit failures, "
            f"{n_explosive} explosive (>{EXPLOSIVE_FACTOR:g}x trajectory range) — "
            "substituted no-movement",
            flush=True,
        )

    return np.array(method_losses), np.array(baseline_losses)


def compute_ets_losses(rolling_df, horizon_days, dims, spec, scenario, split_key,
                       n_pairs=PAIRS_PER_HORIZON,
                       seed=ETS_SAMPLE_SEED,
                       tolerance_frac=TOLERANCE_FRAC):
    """Holt's damped-trend exponential smoothing on the smoothed trajectories.

    Uses the same smoothed series the landscape model is trained/evaluated on
    so both methods are predicting the same target — keeps the head-to-head
    fair. We use ETS instead of ARIMA because the heavily smoothed target
    makes ARIMA fits ill-conditioned (AR root → 1, recursive forecasts
    explode); damped-trend is bounded by construction.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    dim_cols = [f'x0_{i}' for i in dims]
    pair_specs, traj_arrays = _select_horizon_pairs(
        rolling_df, horizon_days, dim_cols, spec, scenario, split_key,
        n_pairs, seed, tolerance_frac, dims,
    )

    def _fit_forecast(hist_d, shift_n):
        fit = ExponentialSmoothing(
            hist_d,
            trend='add',
            damped_trend=True,
            initialization_method='estimated',
        ).fit()
        f = fit.forecast(steps=shift_n)
        return float(f[-1]) if len(f) else float('nan')

    return _evaluate_pairs_with_method(
        pair_specs, traj_arrays, len(dims), horizon_days,
        _fit_forecast, 'ETS',
    )


def compute_theta_losses(rolling_df, horizon_days, dims, spec, scenario, split_key,
                         n_pairs=PAIRS_PER_HORIZON,
                         seed=ETS_SAMPLE_SEED,
                         tolerance_frac=TOLERANCE_FRAC):
    """Damped Theta-2: average of an OLS linear-trend forecast (with damped
    extrapolation) and a simple-exponential-smoothing forecast.

    Standard Theta-2 (Assimakopoulos & Nikolopoulos 2000) extrapolates the OLS
    slope linearly as b·h. On heavily-smoothed inputs that overshoots —
    Theta-2 underperforms no-movement at every horizon. We
    apply Gardner-McKenzie geometric damping to the trend term: b·h becomes
    b·φ·(1-φ^h)/(1-φ), so the trend contribution is bounded as h grows. φ=0.98
    is the standard M-competition default. Constant-history fallback retained
    because statsmodels' SES MLE diverges with zero variance.
    """
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing

    dim_cols = [f'x0_{i}' for i in dims]
    pair_specs, traj_arrays = _select_horizon_pairs(
        rolling_df, horizon_days, dim_cols, spec, scenario, split_key,
        n_pairs, seed, tolerance_frac, dims,
    )

    def _fit_forecast(hist_d, shift_n):
        n = len(hist_d)
        if n < 3 or float(np.std(hist_d)) == 0.0:
            return float(hist_d[-1])
        # OLS slope on (t, y) with t = 0..n-1
        t = np.arange(n, dtype=np.float64)
        t_mean, y_mean = t.mean(), hist_d.mean()
        cov = float(np.sum((t - t_mean) * (hist_d - y_mean)))
        var = float(np.sum((t - t_mean) ** 2))
        if var <= 0:
            return float(hist_d[-1])
        b = cov / var
        # Damped trend extrapolation from the last point
        damped_h = float(shift_n) if THETA_DAMPING_PHI == 1.0 \
            else THETA_DAMPING_PHI * (1.0 - THETA_DAMPING_PHI**shift_n) \
                / (1.0 - THETA_DAMPING_PHI)
        trend_fc = float(hist_d[-1]) + b * damped_h
        # SES forecast (constant beyond last observation)
        ses_fit = SimpleExpSmoothing(hist_d, initialization_method='estimated').fit()
        ses_fc = float(np.asarray(ses_fit.forecast(steps=shift_n))[-1])
        return 0.5 * (trend_fc + ses_fc)

    return _evaluate_pairs_with_method(
        pair_specs, traj_arrays, len(dims), horizon_days,
        _fit_forecast, 'Damped Theta',
    )


def fit_reversion(rolling_df, horizon_days, dims, spec, split_key,
                  tolerance_frac=TOLERANCE_FRAC):
    """Per-dimension (mu, alpha) for reversion toward the training mean.

    mu is the mean of x0 and alpha the least-squares coefficient of
    x1 - mu on x0 - mu, both over train_in at this horizon -- the cell the
    landscape was fitted on, so neither baseline sees anything the model
    didn't. alpha is fitted per dimension because the latent dimensions revert
    on different timescales, which is the strongest form of the null.

    Returns (mu, alpha) or None if train_in is empty.
    """
    from nn_potential import build_horizon_pairs

    paired = build_horizon_pairs(rolling_df, horizon_days, dims, tolerance_frac)
    if len(paired) == 0:
        return None
    labelled = splits.label_pairs(paired, spec, time_col='future_createtime',
                                  key=split_key)
    train = splits.training_rows(labelled)
    if len(train) == 0:
        return None
    x0 = train['x0'].to_numpy().astype(np.float64)
    x1 = train['x1'].to_numpy().astype(np.float64)
    mu = x0.mean(axis=0)
    centred = x0 - mu
    var = np.sum(centred ** 2, axis=0)
    alpha = np.where(var > 0, np.sum(centred * (x1 - mu), axis=0)
                     / np.maximum(var, 1e-30), 0.0)
    print(f"    reversion fit on train_in: n={len(train)} "
          f"mu={np.array2string(mu, precision=3)} "
          f"alpha={np.array2string(alpha, precision=3)}", flush=True)
    return mu, alpha


def compute_reversion_losses(rolling_df, horizon_days, dims, spec, scenario,
                             split_key, kind,
                             n_pairs=PAIRS_PER_HORIZON,
                             seed=ETS_SAMPLE_SEED,
                             tolerance_frac=TOLERANCE_FRAC):
    """Climatology (kind='mean') or AR(1) shrinkage toward it (kind='ar1').

    Both are closed-form given `fit_reversion`, and both are scored on the
    shared pool, so they sit in the same head-to-head as ETS and Theta.
    """
    dim_cols = [f'x0_{i}' for i in dims]
    fit = fit_reversion(rolling_df, horizon_days, dims, spec, split_key,
                        tolerance_frac)
    pair_specs, _ = _select_horizon_pairs(
        rolling_df, horizon_days, dim_cols, spec, scenario, split_key,
        n_pairs, seed, tolerance_frac, dims,
    )
    if fit is None or len(pair_specs) == 0:
        return np.array([]), np.array([])

    mu, alpha = fit
    shrink = alpha if kind == 'ar1' else np.zeros_like(alpha)
    x0 = np.stack([s['x0'] for s in pair_specs])
    x1 = np.stack([s['x1'] for s in pair_specs])
    forecast = mu + shrink * (x0 - mu)
    print(f"    {kind} {horizon_days}d: {len(pair_specs)} pairs across "
          f"{len({s['filter_value'] for s in pair_specs})} trajectories",
          flush=True)
    return (np.sum((forecast - x1) ** 2, axis=1),
            np.sum((x0 - x1) ** 2, axis=1))


def compute_landscape_shared_losses(rolling_df, horizon_days, dims, spec, scenario,
                                    split_key, model, key, eval_batch_size,
                                    n_pairs=PAIRS_PER_HORIZON,
                                    seed=ETS_SAMPLE_SEED,
                                    tolerance_frac=TOLERANCE_FRAC):
    """Evaluate the landscape model on the same per-trajectory pair sample the
    baselines get in this scenario. `_select_horizon_pairs` is deterministic at
    seed=42, so a cache written by a separate run holds the same pairs in the
    same order — the fingerprint is what rules out its having been a different
    run over different data.
    """
    from plnn.dataset import LandscapeSimulationDataset, NumpyLoader
    from nn_potential import df_to_data, evaluate_dataloader

    dim_cols = [f'x0_{i}' for i in dims]
    pair_specs, _ = _select_horizon_pairs(
        rolling_df, horizon_days, dim_cols, spec, scenario, split_key,
        n_pairs, seed, tolerance_frac, dims,
    )
    if len(pair_specs) == 0:
        return np.array([]), np.array([])

    paired_df = pl.DataFrame({
        't0': [s['t0'] for s in pair_specs],
        'x0': [s['x0'].astype(np.float32) for s in pair_specs],
        't1': [s['t1'] for s in pair_specs],
        'x1': [s['x1'].astype(np.float32) for s in pair_specs],
        'filter_value': [s['filter_value'] for s in pair_specs],
    })
    val_data = df_to_data(paired_df)
    val_dataset = LandscapeSimulationDataset(data=val_data)
    val_dataloader = NumpyLoader(
        val_dataset,
        batch_size=min(eval_batch_size, len(val_dataset)),
        shuffle=False,
    )
    print(
        f"    landscape(shared) {horizon_days}d: {len(pair_specs)} pairs across "
        f"{len({s['filter_value'] for s in pair_specs})} trajectories",
        flush=True,
    )
    model_losses, baseline_losses = evaluate_dataloader(model, val_dataloader, key)
    return np.asarray(model_losses, dtype=np.float64), \
        np.asarray(baseline_losses, dtype=np.float64)


def write_significance(significance, rivals, out_dir, rolling):
    """The head-to-head tests beside the figure, as a table the paper can use.

    A cell is how often the landscape beat that rival: `frac_better`, and for a
    rolling design how many folds cleared the correction, since five folds over
    shared trajectories are not five independent tests.
    """
    scenarios = [s for s, _, _ in SCENARIOS]
    horizons = sorted({r['horizon'] for rows in significance.values() for r in rows})
    lines = [f"\\begin{{tabular}}{{ll{'r' * len(rivals)}}}", '\\toprule',
             'Cell & Horizon & ' + ' & '.join(rivals) + ' \\\\', '\\midrule']
    for scenario, title, _ in SCENARIOS:
        for horizon in horizons:
            cells = []
            for rival in rivals:
                rows = [r for key, v in significance.items() if key[0] == scenario
                        for r in v if r['horizon'] == horizon and r['rival'] == rival]
                if not rows:
                    cells.append('--')
                    continue
                frac = np.mean([r['frac_better'] for r in rows])
                if rolling:
                    beat = sum(1 for r in rows if r['q'] < 0.05)
                    cells.append(f'{frac:.2f} ({beat}/{len(rows)})')
                else:
                    cells.append(f"{frac:.2f}{significance_stars(rows[0]['q'])}")
            lines.append(f'{title} & {horizon}d & ' + ' & '.join(cells) + ' \\\\')
        lines.append('\\midrule')
    lines[-1] = '\\bottomrule'
    lines.append('\\end{tabular}')

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'horizon_significance.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Wrote {path}', flush=True)


def head_to_head(model_cache, rivals, horizons):
    """One-sided Wilcoxon of the landscape against each rival, per horizon.

    `rivals` maps a label to per-pair losses keyed by horizon. Every method is
    scored on one shared pair set in one order, so the pairing is by position;
    a length that disagrees means the caches came from different pools and the
    row is dropped rather than silently mispaired.

    Corrected together: the horizons and rivals of one panel are one family.
    """
    from scipy import stats

    rows = []
    for h in horizons:
        if h not in model_cache:
            continue
        losses = np.asarray(model_cache[h]['model_loss'], dtype=np.float64)
        for label, by_horizon in rivals.items():
            if by_horizon is None:
                rival = np.asarray(model_cache[h]['baseline_loss'], dtype=np.float64)
            elif h not in by_horizon:
                continue
            else:
                key = next(k for k in by_horizon[h] if k.endswith('_loss')
                           and not k.endswith('_baseline_loss'))
                rival = np.asarray(by_horizon[h][key], dtype=np.float64)
            if len(rival) != len(losses):
                print(f"  {label} at {h}d: {len(rival)} pairs against the "
                      f"landscape's {len(losses)}; not one pool, skipping",
                      flush=True)
                continue
            res = stats.wilcoxon(losses, rival, alternative='less')
            rows.append({
                'horizon': h, 'rival': label, 'n': len(losses),
                'p': float(res.pvalue),
                'frac_better': float(np.mean(losses < rival)),
                'median_diff': float(np.median(losses - rival)),
            })
    q, _ = benjamini_hochberg(np.array([r['p'] for r in rows]))
    for row, adjusted in zip(rows, q):
        row['q'] = float(adjusted)
    return rows


def fold_caches(cfg, offsets, fig_path, run_dir, select_state):
    """Every fold's pair caches, keyed (offset, method, scenario).

    Each fold was scored against its own model over its own split, so its
    fingerprint is rebuilt from that fold's config rather than assumed.
    """
    import copy

    out = {}
    for offset in offsets:
        cfg_o = copy.deepcopy(cfg)
        cfg_o.split.origin_offset_days = int(offset)
        spec_o = splits.SplitSpec.from_cfg(cfg_o)
        state_o = select_state(run_dir(cfg_o))
        if state_o is None:
            raise SystemExit(
                f'no checkpoint for origin -{offset}d under {run_dir(cfg_o)}. '
                'rolling_holdout.py trains the folds; run it first.')
        fp = cache_fingerprints(cfg_o, spec_o, state_o)
        for name, stem, cols, _, _ in ACTIVE_CACHES:
            for scenario, _, _ in SCENARIOS:
                path = cache_file(fig_path, stem, scenario, spec_o)
                by_horizon = _load_pair_cache(path, cols, fp[name])
                if not by_horizon:
                    raise SystemExit(
                        f'no {name} cache for {scenario} at origin -{offset}d '
                        f'({path}). Score that fold first:\n  python '
                        f'eval_horizons.py <overrides> '
                        f'split.origin_offset_days={offset}')
                out[(offset, name, scenario)] = by_horizon
    return out


def across_folds(per_fold, aggregate, loss_key, baseline_key):
    """One fold-averaged curve, with the spread across folds as the error bar.

    Each fold contributes its own summary of its own pairs; the bar is the
    range over folds, which is what a rolling holdout has to show -- not the
    spread within any one of them.
    """
    curves = {}
    for by_horizon in per_fold:
        hs, point, _ = aggregate(by_horizon, loss_key, baseline_key)
        for h, p in zip(hs, point):
            curves.setdefault(h, []).append(p)
    # only horizons every fold reached: a mean over a different set of folds at
    # each horizon is not a curve
    hs = sorted(h for h, v in curves.items() if len(v) == len(per_fold))
    point = np.array([np.mean(curves[h]) for h in hs])
    lo = np.array([np.min(curves[h]) for h in hs])
    hi = np.array([np.max(curves[h]) for h in hs])
    return hs, point, np.vstack([np.maximum(point - lo, 0.0),
                                 np.maximum(hi - point, 0.0)])


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    from plnn.dataset import LandscapeSimulationDataset, NumpyLoader
    from plnn.models import DeepTimePhiPLNN

    from nn_potential import df_to_data, rolling_frame, build_horizon_pairs, \
        evaluate_dataloader, run_dir
    from sweep_runs import state_path as select_state

    n_dims = cfg.n_dims
    dims = list(range(n_dims))
    trend_name = os.path.basename(cfg.trend_path.rstrip('/'))

    # same layout the trainer writes, including the latent and split tags
    dir_path = run_dir(cfg)
    if cfg.latents.method != 'gpfa' and cfg.rolling_mean_window != 100:
        dir_path = f"{dir_path}_rm{cfg.rolling_mean_window}"

    fig_path = f'./figs/{trend_name}'
    os.makedirs(fig_path, exist_ok=True)

    # The same nested split the trainer labelled its pairs with, so the
    # scenario cells here are the ones the model was held out from.
    spec = splits.SplitSpec.from_cfg(cfg)
    state_path = select_state(dir_path)
    fingerprint = cache_fingerprints(cfg, spec, state_path)

    # One cache file per (method, scenario); cached[(method, scenario)] is
    # {horizon: {loss column: per-pair array}}.
    cache_path, cached = {}, {}
    for name, stem, cols, _, _ in ACTIVE_CACHES:
        for scenario, _, _ in SCENARIOS:
            path = cache_file(fig_path, stem, scenario, spec)
            cache_path[(name, scenario)] = path
            cached[(name, scenario)] = _load_pair_cache(path, cols, fingerprint[name])

    todo = {k: [h for h in HORIZON_DAYS if h not in v] for k, v in cached.items()}
    for (name, scenario), horizons in sorted(todo.items()):
        if horizons:
            print(f"Need {name} at {scenario} for {horizons}", flush=True)

    if any(todo.values()):
        print("Loading data...", flush=True)
        target_df, _, _ = latent_space.load(cfg, smooth=False)
        target_df = target_df.rename({latent_space.COORD: 'x0'})

        if cfg.platform != 'all':
            target_df = target_df.filter(
                pl.col('filter_value').cast(pl.String) \
                    .str.to_lowercase() \
                    .str.contains(f'-{cfg.platform}-')
            )

        target_df = target_df.filter(pl.col('filter_value') != '') \
            .select(['createtime', 'filter_value', 'x0']) \
            .sort(['filter_value', 'createtime'])

        rolling_df = rolling_frame(cfg, target_df, dims)
        print(f"Timestep rows: {len(rolling_df)}", flush=True)

        split_key = latent_space.split_key(cfg)

        # A whole number of rows is what a horizon actually becomes, so the
        # span a cell is scored over is a multiple of the observation spacing
        # rather than the requested horizon. Horizons no row shift can reach
        # are dropped, and the rest record what they really span.
        spacing = median_spacing_days(rolling_df)
        print(f"Median observation spacing: {spacing:.2f}d", flush=True)
        horizons = usable_horizons(HORIZON_DAYS, spacing)
        actual_days = {h: horizon_shift(h, spacing) * spacing for h in horizons}
        todo = {k: [h for h in v if h in horizons] for k, v in todo.items()}

        # Lazily load the landscape model only if some cell needs model
        # results. Every baseline shares rolling_df so they predict the same
        # smoothed target as the landscape model.
        model = None
        key = None
        if any(todo[(LANDSCAPE_CACHE, s)] for s, _, _ in SCENARIOS):
            dtype = jnp.float32
            print(f"Loading model from: {state_path}", flush=True)
            model, _ = DeepTimePhiPLNN.load(state_path, dtype=dtype)

            seed = 42
            rng = np.random.default_rng(seed=seed)
            key = jax.random.PRNGKey(int(rng.integers(2**32)))

        for scenario, title, subtitle in SCENARIOS:
            horizons_needed = sorted(
                {h for name, _, _, _, _ in ACTIVE_CACHES for h in todo[(name, scenario)]})
            if not horizons_needed:
                continue
            print(f"\n=== {scenario}: {title} ({subtitle}) ===", flush=True)

            def store(name, horizon, losses, baselines, loss_key, baseline_key):
                _store(cached[(name, scenario)], cache_path[(name, scenario)],
                       horizon, name, losses, baselines, loss_key, baseline_key,
                       fingerprint[name], actual_days[horizon])

            for horizon in horizons_needed:
                print(f"\n--- {scenario} @ {horizon}d "
                      f"(spans {actual_days[horizon]:.1f}d) ---", flush=True)

                if horizon in todo[(LANDSCAPE_CACHE, scenario)]:
                    key, subkey = jax.random.split(key)
                    if LANDSCAPE_CACHE == 'model_shared':
                        losses, baselines = compute_landscape_shared_losses(
                            rolling_df, horizon, dims, spec, scenario, split_key,
                            model, subkey, cfg.eval_batch_size)
                    else:
                        paired_df = build_horizon_pairs(rolling_df, horizon, dims,
                                                        TOLERANCE_FRAC)
                        cell_df = scenario_pairs(paired_df, spec, scenario, split_key) \
                            if len(paired_df) else paired_df
                        print(f"  Pairs: {len(paired_df)} total, "
                              f"{len(cell_df)} in {scenario}", flush=True)
                        losses, baselines = np.array([]), np.array([])
                        if len(cell_df):
                            dataloader = NumpyLoader(
                                LandscapeSimulationDataset(data=df_to_data(cell_df)),
                                batch_size=min(cfg.eval_batch_size, len(cell_df)),
                                shuffle=False,
                            )
                            losses, baselines = evaluate_dataloader(
                                model, dataloader, subkey)
                    store(LANDSCAPE_CACHE, horizon, losses, baselines,
                          'model_loss', 'baseline_loss')

                # .get, not [], for every optional method: a method left out
                # of ACTIVE_CACHES has no todo list to consult
                if horizon in todo.get(('ets', scenario), ()):
                    store('ets', horizon, *compute_ets_losses(
                        rolling_df, horizon, dims, spec, scenario, split_key),
                        loss_key='ets_loss', baseline_key='ets_baseline_loss')

                if horizon in todo.get(('theta', scenario), ()):
                    store('theta', horizon, *compute_theta_losses(
                        rolling_df, horizon, dims, spec, scenario, split_key),
                        loss_key='theta_loss', baseline_key='theta_baseline_loss')

                for kind in ('mean', 'ar1'):
                    if horizon in todo.get((kind, scenario), ()):
                        store(kind, horizon, *compute_reversion_losses(
                            rolling_df, horizon, dims, spec, scenario, split_key,
                            kind),
                            loss_key=f'{kind}_loss',
                            baseline_key=f'{kind}_baseline_loss')
    else:
        print("All horizons cached for every method and scenario; skipping computation.",
              flush=True)

    if not any(cached.values()):
        print("No horizons produced results; nothing to plot.", flush=True)
        return

    # One panel per scenario, each normalised by its own no-movement baseline
    # so y=1 means "no-movement" everywhere. Median of per-pair ratios with
    # Q1-Q3 error bars — robust to the heavy right tail of squared errors.
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(4.1 * len(SCENARIOS), 3.3),
                             sharey=True)

    def _median_iqr(by_horizon, loss_key, baseline_key):
        # Per-pair ratio model_loss_i / baseline_loss_i, then median + IQR. An
        # earlier version divided by mean(baseline) across pairs, which biased
        # the median low by ~mean(baseline)/median(baseline) (~3× for squared
        # errors' heavy tail) — making the model look better than it is.
        hs = sorted(by_horizon)
        medians, q1s, q3s = [], [], []
        for h in hs:
            losses = np.asarray(by_horizon[h][loss_key], dtype=np.float64)
            baseline = np.asarray(by_horizon[h][baseline_key], dtype=np.float64)
            if len(losses) == 0:
                medians.append(np.nan); q1s.append(np.nan); q3s.append(np.nan)
                continue
            ratios = losses / np.maximum(baseline, 1e-12)
            medians.append(float(np.median(ratios)))
            q1s.append(float(np.quantile(ratios, 0.25)))
            q3s.append(float(np.quantile(ratios, 0.75)))
        medians = np.asarray(medians)
        yerr = np.vstack([medians - np.asarray(q1s), np.asarray(q3s) - medians])
        return hs, medians, yerr

    def _mean_per_pair_bootstrap(by_horizon, loss_key, baseline_key):
        # Per-pair ratio model_loss_i / baseline_loss_i, then mean with
        # bootstrap percentile CI on that mean. Same per-pair ratio as
        # `_median_iqr`; differs only in the aggregator (mean vs median) and
        # in the error bars (bootstrap CI on the mean vs descriptive IQR).
        hs = sorted(by_horizon)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        means, lows, highs = [], [], []
        for h in hs:
            losses = np.asarray(by_horizon[h][loss_key], dtype=np.float64)
            baseline = np.asarray(by_horizon[h][baseline_key], dtype=np.float64)
            n = len(losses)
            if n == 0:
                means.append(np.nan); lows.append(np.nan); highs.append(np.nan)
                continue
            ratios = losses / np.maximum(baseline, 1e-12)
            point = float(np.mean(ratios))
            idx = rng.integers(0, n, size=(BOOTSTRAP_N, n))
            boot_means = ratios[idx].mean(axis=1)
            means.append(point)
            lows.append(float(np.quantile(boot_means, BOOTSTRAP_CI_LO)))
            highs.append(float(np.quantile(boot_means, BOOTSTRAP_CI_HI)))
        means = np.asarray(means)
        yerr = np.vstack([
            np.maximum(means - np.asarray(lows), 0.0),
            np.maximum(np.asarray(highs) - means, 0.0),
        ])
        return hs, means, yerr

    def _ratio_of_aggregates_bootstrap(by_horizon, loss_key, baseline_key, agg):
        # Aggregate-then-divide: agg(losses) / agg(baseline) with bootstrap
        # percentile CI. `agg` is np.mean or np.median. Pairs are resampled
        # jointly so numerator/denominator covary. Robust to near-zero baseline
        # pairs (which dominate per-pair-ratio aggregators).
        hs = sorted(by_horizon)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        points, lows, highs = [], [], []
        for h in hs:
            losses = np.asarray(by_horizon[h][loss_key], dtype=np.float64)
            baseline = np.asarray(by_horizon[h][baseline_key], dtype=np.float64)
            n = len(losses)
            base_agg = float(agg(baseline)) if n else 0.0
            if n == 0 or base_agg <= 0:
                points.append(np.nan); lows.append(np.nan); highs.append(np.nan)
                continue
            point = float(agg(losses)) / base_agg
            idx = rng.integers(0, n, size=(BOOTSTRAP_N, n))
            boot_num = agg(losses[idx], axis=1)
            boot_den = agg(baseline[idx], axis=1)
            boot_ratio = boot_num / np.maximum(boot_den, 1e-12)
            points.append(point)
            lows.append(float(np.quantile(boot_ratio, BOOTSTRAP_CI_LO)))
            highs.append(float(np.quantile(boot_ratio, BOOTSTRAP_CI_HI)))
        points = np.asarray(points)
        yerr = np.vstack([
            np.maximum(points - np.asarray(lows), 0.0),
            np.maximum(np.asarray(highs) - points, 0.0),
        ])
        return hs, points, yerr

    def _mean_ratio_bootstrap(by_horizon, loss_key, baseline_key):
        return _ratio_of_aggregates_bootstrap(by_horizon, loss_key, baseline_key, np.mean)

    def _median_ratio_quantile(by_horizon, loss_key, baseline_key):
        # median(loss) / median(baseline) with Q1/Q3 error bars taken from
        # the *data* (not bootstrap): Q1(loss)/median(baseline) and
        # Q3(loss)/median(baseline). Spread of the loss distribution
        # rescaled into ratio units — analogous to `_median_iqr` but with
        # aggregate-then-divide normalization.
        hs = sorted(by_horizon)
        points, lows, highs = [], [], []
        for h in hs:
            losses = np.asarray(by_horizon[h][loss_key], dtype=np.float64)
            baseline = np.asarray(by_horizon[h][baseline_key], dtype=np.float64)
            n = len(losses)
            base_med = float(np.median(baseline)) if n else 0.0
            if n == 0 or base_med <= 0:
                points.append(np.nan); lows.append(np.nan); highs.append(np.nan)
                continue
            points.append(float(np.median(losses)) / base_med)
            lows.append(float(np.quantile(losses, 0.25)) / base_med)
            highs.append(float(np.quantile(losses, 0.75)) / base_med)
        points = np.asarray(points)
        yerr = np.vstack([
            np.maximum(points - np.asarray(lows), 0.0),
            np.maximum(np.asarray(highs) - points, 0.0),
        ])
        return hs, points, yerr

    def _absolute_mean(by_horizon, loss_key, baseline_key):
        # mean(loss_i) in raw MSE units with bootstrap percentile CI on the mean.
        # baseline_key is unused; the no-movement baseline is plotted separately.
        del baseline_key
        hs = sorted(by_horizon)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        points, lows, highs = [], [], []
        for h in hs:
            losses = np.asarray(by_horizon[h][loss_key], dtype=np.float64)
            n = len(losses)
            if n == 0:
                points.append(np.nan); lows.append(np.nan); highs.append(np.nan)
                continue
            points.append(float(np.mean(losses)))
            idx = rng.integers(0, n, size=(BOOTSTRAP_N, n))
            boot_means = losses[idx].mean(axis=1)
            lows.append(float(np.quantile(boot_means, BOOTSTRAP_CI_LO)))
            highs.append(float(np.quantile(boot_means, BOOTSTRAP_CI_HI)))
        points = np.asarray(points)
        yerr = np.vstack([
            np.maximum(points - np.asarray(lows), 0.0),
            np.maximum(np.asarray(highs) - points, 0.0),
        ])
        return hs, points, yerr

    def _absolute_median(by_horizon, loss_key, baseline_key):
        # median(loss_i) in raw MSE units with Q1/Q3 IQR bars from the data.
        # baseline_key is unused; the no-movement baseline is plotted separately.
        del baseline_key
        hs = sorted(by_horizon)
        points, lows, highs = [], [], []
        for h in hs:
            losses = np.asarray(by_horizon[h][loss_key], dtype=np.float64)
            n = len(losses)
            if n == 0:
                points.append(np.nan); lows.append(np.nan); highs.append(np.nan)
                continue
            points.append(float(np.median(losses)))
            lows.append(float(np.quantile(losses, 0.25)))
            highs.append(float(np.quantile(losses, 0.75)))
        points = np.asarray(points)
        yerr = np.vstack([
            np.maximum(points - np.asarray(lows), 0.0),
            np.maximum(np.asarray(highs) - points, 0.0),
        ])
        return hs, points, yerr

    ci_pct = int(round((BOOTSTRAP_CI_HI - BOOTSTRAP_CI_LO) * 100))
    absolute_mode = PLOT_AGGREGATOR in ABSOLUTE_AGGREGATORS
    if PLOT_AGGREGATOR == 'median_per_pair':
        _aggregate = _median_iqr
        ylabel = 'Per-pair loss / No-movement loss (median, IQR)'
    elif PLOT_AGGREGATOR == 'mean_per_pair':
        _aggregate = _mean_per_pair_bootstrap
        ylabel = f'Per-pair loss / No-movement loss (mean, {ci_pct}% bootstrap CI)'
    elif PLOT_AGGREGATOR == 'mean_ratio':
        _aggregate = _mean_ratio_bootstrap
        ylabel = f'Mean loss / Mean no-movement loss ({ci_pct}% bootstrap CI)'
    elif PLOT_AGGREGATOR == 'median_ratio':
        _aggregate = _median_ratio_quantile
        ylabel = 'Median MSE / Median no-movement MSE'
    elif PLOT_AGGREGATOR == 'absolute_mean':
        _aggregate = _absolute_mean
        ylabel = f'Mean squared error ({ci_pct}% bootstrap CI)'
    elif PLOT_AGGREGATOR == 'absolute_median':
        _aggregate = _absolute_median
        ylabel = 'Median squared error (Q1/Q3 IQR)'
    else:
        raise ValueError(f"Unknown PLOT_AGGREGATOR: {PLOT_AGGREGATOR}")

    from scipy import stats

    # Rolling holdout: one curve per method averaged over the folds, with the
    # spread across origins as the error bar. Empty means the single fixed
    # holdout this run was scored on.
    rolling = [int(o) for o in (cfg.get('rolling_offsets') or [])]
    significance = {}
    folds = (fold_caches(cfg, rolling, fig_path, run_dir, select_state)
             if rolling else {})

    def _spans(by_horizon, hs):
        """What each requested horizon's row shift really spans, for the x axis."""
        return [float(by_horizon[h]['actual_days'][0]) for h in hs]

    for ax, (scenario, title, subtitle) in zip(np.atleast_1d(axes), SCENARIOS):
        model_cache = (folds[(rolling[0], LANDSCAPE_CACHE, scenario)]
                       if rolling else cached[(LANDSCAPE_CACHE, scenario)])

        h_model = med_model = None
        for name, _, _, loss_key, baseline_key in ACTIVE_CACHES:
            if rolling:
                per_fold = [folds[(o, name, scenario)] for o in rolling]
                by_horizon = per_fold[0]
            else:
                by_horizon = cached[(name, scenario)]
            if not by_horizon:
                continue
            label, fmt = CACHE_STYLE[name]
            if rolling:
                hs, point, err = across_folds(per_fold, _aggregate,
                                              loss_key, baseline_key)
            else:
                hs, point, err = _aggregate(by_horizon, loss_key, baseline_key)
            ax.errorbar(_spans(by_horizon, hs), point, yerr=err,
                        fmt=fmt, capsize=3, label=label)
            if name == LANDSCAPE_CACHE:
                h_model, med_model = hs, point

        if absolute_mode:
            # Every method is scored on the same pair set, so any cache draws
            # the same no-movement line; the landscape's differs from the rest
            # only by its float32 cast of x0/x1.
            baseline = next(((cached[(n, scenario)], bk)
                             for n, _, _, _, bk in ACTIVE_CACHES
                             if cached[(n, scenario)]), None)
            if baseline is not None:
                by_horizon, baseline_key = baseline
                hs, point, err = _aggregate(by_horizon, baseline_key, None)
                ax.errorbar(_spans(by_horizon, hs), point, yerr=err,
                            fmt='D:', capsize=3, color='k', label='No-movement')
            ax.set_yscale('log')
        else:
            ax.axhline(1.0, color='k', linestyle=':', linewidth=1, label='No-movement')
            ax.set_ylim(top=1.5)
        ax.set_xscale('log')
        ax.set_xlabel('Prediction horizon (days)')
        ax.set_title(f'{title}\n({subtitle})', fontsize=9)
        ax.grid(True, which='both', alpha=0.3)

        if med_model is None or not np.any(np.isfinite(med_model)):
            print(f"{scenario}: no landscape results to summarise.", flush=True)
            continue
        print(
            f"{scenario} ({title}): largest improvement over baseline of "
            f"{1.0 - np.nanmin(med_model):.2%} at "
            f"{h_model[int(np.nanargmin(med_model))]}d (descriptive -- the "
            f"horizon is chosen by the curve, so it carries no p-value)",
            flush=True,
        )

        # Paired one-sided Wilcoxon signed-rank test per horizon: H1 is that
        # model_loss < baseline_loss per pair. Non-parametric because the
        # squared-error distribution is heavily right-tailed. Every horizon is
        # reported so the headline at REPORT_HORIZON_DAYS is not a selection.
        # Against the no-movement baseline and against every method drawn
        # beside it, one-sided and paired over the shared pool. Per fold when
        # rolling: the folds share trajectories and differ only in where the
        # origin sits, so pooling their pairs would overstate n.
        report_folds = (list(rolling) if rolling else [None])
        for offset in report_folds:
            cache = (folds[(offset, LANDSCAPE_CACHE, scenario)]
                     if rolling else model_cache)
            against = {'No-movement': None}
            for name, _, _, _, _ in ACTIVE_CACHES:
                if name == LANDSCAPE_CACHE:
                    continue
                against[CACHE_STYLE[name][0]] = (folds[(offset, name, scenario)]
                                                 if rolling
                                                 else cached[(name, scenario)])
            rows = head_to_head(cache, against, h_model)
            significance[(scenario, offset)] = rows
            prefix = f'  origin -{offset}d ' if rolling else '  '
            for row in rows:
                headline = (' <- reported' if row['horizon'] == REPORT_HORIZON_DAYS
                            and row['rival'] == 'No-movement' else '')
                print(
                    f"{prefix}landscape vs {row['rival']} at {row['horizon']}d: "
                    f"n={row['n']} p={row['p']:.3g} q={row['q']:.3g} "
                    f"{significance_stars(row['q'])} "
                    f"median(model-rival)={row['median_diff']:.6g} "
                    f"frac_better={row['frac_better']:.3f}{headline}",
                    flush=True,
                )

    axes_list = list(np.atleast_1d(axes))
    axes_list[0].set_ylabel(
        f'{ylabel}\n(mean of {len(rolling)} origins, range)' if rolling else ylabel)

    # One legend under all three panels: in-axes it lands on the curves, which
    # converge on the no-movement line at the right of every panel. Matplotlib
    # returns the axhline (a Line2D) ahead of the errorbar containers, so the
    # no-movement reference is moved to the end rather than leading the legend.
    handles, labels = axes_list[0].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i] == 'No-movement')
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               fontsize=8, ncol=len(labels), loc='lower center', frameon=False)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    rival_names = ['No-movement'] + [CACHE_STYLE[n][0] for n, _, _, _, _ in ACTIVE_CACHES
                                     if n != LANDSCAPE_CACHE]
    write_significance(significance, rival_names,
                       cfg.get('out_dir', './out'), rolling)

    # the origin in the name, like the caches: a fold's own figure must not
    # land on top of the fixed holdout's
    origin = f'_o{spec.origin_offset_days}' if spec.origin_offset_days else ''
    stem = '_rolling' if rolling else origin
    fig_file = os.path.join(fig_path, f'nn_potential_horizon_skill{stem}.png')
    fig.savefig(fig_file, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {fig_file}", flush=True)


if __name__ == '__main__':
    main()
