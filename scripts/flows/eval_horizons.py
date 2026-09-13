import os
import warnings

import hydra
import numpy as np
import polars as pl
from tqdm import tqdm

import latent_space
import splits

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
    ('test_in', 'Interpolation', 'unseen trajectories, seen time'),
    ('train_out', 'Forecast', 'seen trajectories, unseen time'),
    ('test_out', 'Zero-shot forecast', 'unseen trajectories, unseen time'),
)
# Minimum rolling-mean history (incl. t0) for a pair to enter the time-series
# sample. Kept low so we don't bias toward long-running trajectories — the
# landscape model has no such requirement, and 5 points is enough for ETS
# damped-trend / Theta-2 to identify their parameters.
MIN_HISTORY = 5
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
MODEL_PAIR_COLS = {'horizon', 'model_loss', 'baseline_loss'}
# ETS cache: Holt's damped-trend exponential-smoothing losses (ETS(A,Ad,N)) on
# a subsample of the rolling-mean trajectories — the same smoothing the
# landscape model is trained/evaluated on, so the two methods predict the same
# target. Uses a per-trajectory shift_n derived from each trajectory's own
# median spacing. No-movement baseline is computed on the *same* subsample (so
# the ETS ratio is fair). The model cache holds the landscape scored on the
# whole cell instead, against its own no-movement baseline on that cell.
# We use ETS instead of ARIMA because the landscape model's training target is
# a heavily smoothed series (large rolling-mean window), which makes ARIMA
# fits ill-conditioned (AR root → 1, recursive forecasts explode). Holt's
# damped-trend method is bounded by construction.
ETS_PAIR_COLS = {'horizon', 'ets_loss', 'ets_baseline_loss'}
# Theta cache: same pair sampling as ETS, but the per-pair forecast comes from
# the Theta-2 method (Assimakopoulos & Nikolopoulos 2000) — average of OLS
# linear-trend-on-time and simple exponential smoothing. Robust to
# near-constant series where ARIMA's AR-root estimation degenerates: the
# trend slope just goes to zero. We guard the (rare) exactly-constant case
# explicitly because statsmodels' MLE for SES diverges with zero variance.
THETA_PAIR_COLS = {'horizon', 'theta_loss', 'theta_baseline_loss'}

# (cache name, filename stem, required columns, loss column, baseline column).
# One file per (cache, scenario) so a scenario can be recomputed on its own.
CACHES = (
    ('model', '', MODEL_PAIR_COLS, 'model_loss', 'baseline_loss'),
    ('model_shared', '_model_shared', MODEL_PAIR_COLS, 'model_loss', 'baseline_loss'),
    ('ets', '_ets', ETS_PAIR_COLS, 'ets_loss', 'ets_baseline_loss'),
    ('theta', '_theta', THETA_PAIR_COLS, 'theta_loss', 'theta_baseline_loss'),
)

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
# Which pairs the landscape is scored on. 'shared' is exactly the pairs ETS and
# Theta could be fitted on, so every line in a panel comes from one pair set and
# the comparison is a true head-to-head; 'full' is every pair in the cell, which
# is the population estimate but not comparable to the baselines. Absolute mode
# needs 'shared' for the three no-movement lines to coincide.
LANDSCAPE_POOL = 'shared'
LANDSCAPE_CACHE = 'model' if (LANDSCAPE_POOL == 'full'
                              and PLOT_AGGREGATOR not in ABSOLUTE_AGGREGATORS) \
    else 'model_shared'
# The other landscape cache is neither computed nor plotted -- scoring a cell
# twice costs a forward pass per pair for a number nothing reads.
ACTIVE_CACHES = tuple(c for c in CACHES
                      if c[0] in (LANDSCAPE_CACHE, 'ets', 'theta'))
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 0
BOOTSTRAP_CI_LO = 0.025
BOOTSTRAP_CI_HI = 0.975


def _load_pair_cache(path, required_cols):
    """Load a per-pair parquet cache. Returns {horizon: {col: np.ndarray}},
    or {} if the file is missing or doesn't satisfy required_cols.
    """
    if not os.path.exists(path):
        return {}
    df = pl.read_parquet(path)
    if not required_cols.issubset(df.columns):
        print(f"Cache {path} missing columns {required_cols - set(df.columns)}; ignoring.", flush=True)
        return {}
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
           loss_key, baseline_key):
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
    cache[horizon] = {loss_key: losses, baseline_key: baseline_losses}
    _save_pair_cache(path, cache)


def _select_horizon_pairs(rolling_df, horizon_days, dim_cols, spec, scenario,
                          split_key, n_pairs, seed, tolerance_frac, dims):
    """Random sample of a scenario's pairs (with attached histories) for ts
    evaluation.

    Pulls from the same pool the landscape model is evaluated on
    (`build_horizon_pairs` + `scenario_pairs`), then:
      1. Joins each pair to its row index in the trajectory's rolling-mean
         series (so callers can slice history up to and including t0).
      2. Drops pairs whose history at t0 has fewer than `MIN_HISTORY` points.
      3. Uniformly subsamples to ≤ `n_pairs` pairs.

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

    # Global shift_n — same derivation as build_horizon_pairs, kept in sync so
    # idx_t1 - idx_t0 is exactly shift_n in each trajectory's row ordering.
    spacing_df = rolling_df.with_columns(
        (pl.col('createtime').diff().over('filter_value')).alias('dt')
    ).drop_nulls('dt')
    median_spacing_days = spacing_df['dt'].median().total_seconds() / 86400
    shift_n = max(1, int(round(horizon_days / median_spacing_days)))

    rolling_df = rolling_df.sort(['filter_value', 'createtime'])
    rolling_with_idx = rolling_df.with_columns(
        pl.int_range(pl.len()).over('filter_value').alias('idx_t0')
    )
    cell_with_idx = cell_paired.join(
        rolling_with_idx.select(['filter_value', 'createtime', 'idx_t0']),
        on=['filter_value', 'createtime'], how='inner',
    ).filter(pl.col('idx_t0') >= MIN_HISTORY - 1)

    n_avail = len(cell_with_idx)
    if n_avail == 0:
        return [], {}
    if n_avail > n_pairs:
        rng = np.random.default_rng(seed)
        chosen_idx = np.sort(rng.choice(n_avail, size=n_pairs, replace=False))
        cell_with_idx = cell_with_idx[chosen_idx]

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
    and compare the shift_n-step forecast to the rolling-mean x1 observation.
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
                       tolerance_frac=0.25):
    """Holt's damped-trend exponential smoothing on rolling-mean trajectories.

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
                         tolerance_frac=0.25):
    """Damped Theta-2: average of an OLS linear-trend forecast (with damped
    extrapolation) and a simple-exponential-smoothing forecast.

    Standard Theta-2 (Assimakopoulos & Nikolopoulos 2000) extrapolates the OLS
    slope linearly as b·h. On heavily-smoothed inputs (rolling-mean targets),
    that overshoots — Theta-2 underperforms no-movement at every horizon. We
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


def compute_landscape_shared_losses(rolling_df, horizon_days, dims, spec, scenario,
                                    split_key, model, key, eval_batch_size,
                                    n_pairs=PAIRS_PER_HORIZON,
                                    seed=ETS_SAMPLE_SEED,
                                    tolerance_frac=0.25):
    """Evaluate the landscape model on the same per-trajectory pair sample
    used for ETS/Theta in this scenario. `_select_horizon_pairs` is
    deterministic at seed=42 so the resulting set is identical to what's
    already cached for ETS/Theta — this is the strict head-to-head population.
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

    # One cache file per (method, scenario); cached[(method, scenario)] is
    # {horizon: {loss column: per-pair array}}.
    cache_path, cached = {}, {}
    for name, stem, cols, _, _ in ACTIVE_CACHES:
        for scenario, _, _ in SCENARIOS:
            path = os.path.join(
                fig_path, f'nn_potential_horizon_skill{stem}_{scenario}.parquet.zstd')
            cache_path[(name, scenario)] = path
            cached[(name, scenario)] = _load_pair_cache(path, cols)

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

        # The same nested split the trainer labelled its pairs with, so the
        # scenario cells here are the ones the model was held out from.
        spec = splits.SplitSpec.from_cfg(cfg)
        split_key = latent_space.split_key(cfg)

        # Lazily load the landscape model only if some cell needs model
        # results. ETS/Theta share rolling_df so they predict the same smoothed
        # target as the landscape model.
        model = None
        key = None
        if any(todo[(LANDSCAPE_CACHE, s)] for s, _, _ in SCENARIOS):
            dtype = jnp.float32
            state_path = select_state(dir_path)
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

            for horizon in horizons_needed:
                print(f"\n--- {scenario} @ {horizon}d ---", flush=True)

                if horizon in todo[(LANDSCAPE_CACHE, scenario)]:
                    key, subkey = jax.random.split(key)
                    if LANDSCAPE_CACHE == 'model_shared':
                        losses, baselines = compute_landscape_shared_losses(
                            rolling_df, horizon, dims, spec, scenario, split_key,
                            model, subkey, cfg.eval_batch_size)
                    else:
                        paired_df = build_horizon_pairs(rolling_df, horizon, dims)
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
                    _store(cached[(LANDSCAPE_CACHE, scenario)],
                           cache_path[(LANDSCAPE_CACHE, scenario)],
                           horizon, LANDSCAPE_CACHE, losses, baselines,
                           'model_loss', 'baseline_loss')

                if horizon in todo[('ets', scenario)]:
                    _store(
                        cached[('ets', scenario)], cache_path[('ets', scenario)],
                        horizon, 'ets',
                        *compute_ets_losses(
                            rolling_df, horizon, dims, spec, scenario, split_key),
                        loss_key='ets_loss', baseline_key='ets_baseline_loss',
                    )

                if horizon in todo[('theta', scenario)]:
                    _store(
                        cached[('theta', scenario)], cache_path[('theta', scenario)],
                        horizon, 'theta',
                        *compute_theta_losses(
                            rolling_df, horizon, dims, spec, scenario, split_key),
                        loss_key='theta_loss', baseline_key='theta_baseline_loss',
                    )
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

    for ax, (scenario, title, subtitle) in zip(np.atleast_1d(axes), SCENARIOS):
        model_cache = cached[(LANDSCAPE_CACHE, scenario)]
        ets_cache = cached[('ets', scenario)]
        theta_cache = cached[('theta', scenario)]

        h_model = med_model = None
        if model_cache:
            h_model, med_model, err_model = _aggregate(
                model_cache, 'model_loss', 'baseline_loss')
            ax.errorbar(h_model, med_model, yerr=err_model,
                        fmt='o-', capsize=3, label='Potential landscape')
        if ets_cache:
            h_ets, med_ets, err_ets = _aggregate(
                ets_cache, 'ets_loss', 'ets_baseline_loss')
            ax.errorbar(h_ets, med_ets, yerr=err_ets,
                        fmt='s--', capsize=3, label='Holt damped trend')
        if theta_cache:
            h_theta, med_theta, err_theta = _aggregate(
                theta_cache, 'theta_loss', 'theta_baseline_loss')
            ax.errorbar(h_theta, med_theta, yerr=err_theta,
                        fmt='^-.', capsize=3, label='Damped Theta')

        if absolute_mode:
            # model_shared/ETS/theta baselines are bit-identical by
            # construction (deterministic seed=42 pair selection), so any of
            # them draws the same no-movement line.
            baseline_cache = ets_cache or theta_cache or model_cache
            baseline_key = 'ets_baseline_loss' if ets_cache \
                else ('theta_baseline_loss' if theta_cache else 'baseline_loss')
            if baseline_cache:
                h_b, med_b, err_b = _aggregate(baseline_cache, baseline_key, None)
                ax.errorbar(h_b, med_b, yerr=err_b,
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
        best_horizon = h_model[int(np.nanargmin(med_model))]
        print(
            f"{scenario} ({title}): maximum improvement over baseline of "
            f"{1.0 - np.nanmin(med_model):.2%} at {best_horizon}d",
            flush=True,
        )

        # Paired one-sided Wilcoxon signed-rank test at the best horizon: H1 is
        # that model_loss < baseline_loss per pair. Non-parametric because the
        # squared-error distribution is heavily right-tailed.
        best_losses = np.asarray(
            model_cache[best_horizon]['model_loss'], dtype=np.float64)
        best_baseline = np.asarray(
            model_cache[best_horizon]['baseline_loss'], dtype=np.float64)
        wilcoxon_res = stats.wilcoxon(best_losses, best_baseline, alternative='less')
        print(
            f"  Wilcoxon signed-rank (model < baseline) at {best_horizon}d: "
            f"n={len(best_losses)} statistic={wilcoxon_res.statistic:.3g} "
            f"p={wilcoxon_res.pvalue:.3g} "
            f"median(model-baseline)={float(np.median(best_losses - best_baseline)):.6g} "
            f"frac_better={float(np.mean(best_losses < best_baseline)):.3f}",
            flush=True,
        )

    axes_list = list(np.atleast_1d(axes))
    axes_list[0].set_ylabel(ylabel)

    # One legend under all three panels: in-axes it lands on the curves, which
    # converge on the no-movement line at the right of every panel. Matplotlib
    # returns the axhline (a Line2D) ahead of the errorbar containers, so the
    # no-movement reference is moved to the end rather than leading the legend.
    handles, labels = axes_list[0].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i] == 'No-movement')
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               fontsize=8, ncol=len(labels), loc='lower center', frameon=False)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig_file = os.path.join(fig_path, 'nn_potential_horizon_skill.png')
    fig.savefig(fig_file, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {fig_file}", flush=True)


if __name__ == '__main__':
    main()
