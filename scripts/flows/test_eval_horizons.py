"""Checks for the horizon-skill evaluation's scenario path.

Synthetic OU trajectories, so it needs neither the real data nor a GPU: the
landscape model itself is not exercised, only the pair selection every method
is scored on and the time-series baselines fitted to it.

Run as: python test_eval_horizons.py
"""

import datetime
import os
import sys
import types

import numpy as np
import polars as pl

# plnn is only needed to train or to load a checkpoint; stub it so nn_potential
# imports, the same way test_nested_split_pipeline does.
for name in ('plnn', 'plnn.dataset', 'plnn.models', 'plnn.loss_functions',
             'plnn.optimizers', 'plnn.model_training', 'wandb', 'hydra'):
    sys.modules.setdefault(name, types.ModuleType(name))
for name, attrs in (
        ('plnn.dataset', ['LandscapeSimulationDataset', 'NumpyLoader']),
        ('plnn.models', ['DeepTimePhiPLNN']),
        ('plnn.loss_functions', ['select_loss_function']),
        ('plnn.optimizers', ['get_optimizer_args', 'select_optimizer', 'get_dt_schedule']),
        ('plnn.model_training', ['train_model'])):
    for a in attrs:
        setattr(sys.modules[name], a, object)
sys.modules['hydra'].main = lambda **kw: (lambda f: f)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_horizons as eh                       # noqa: E402
import splits                                    # noqa: E402
from nn_potential import build_horizon_pairs     # noqa: E402

DIMS = [0, 1]
HORIZON = 60
SPEC = splits.SplitSpec(holdout_days=365, origin_offset_days=0,
                        train_frac=0.70, val_frac=0.10, seed=42)


def make_rolling(n_traj=60, n_steps=400, start=datetime.datetime(2022, 1, 1)):
    """Mean-reverting walks on a 2-day grid, the shape rolling_frame returns."""
    rng = np.random.default_rng(0)
    rows = []
    for j in range(n_traj):
        x = rng.normal(size=len(DIMS))
        for i in range(n_steps):
            x = 0.98 * x + 0.05 * rng.normal(size=len(DIMS))
            rows.append({'filter_value': f'seed{j:03d}',
                         'createtime': start + datetime.timedelta(days=2 * i),
                         't': float(2 * i),
                         **{f'x0_{d}': float(x[d]) for d in DIMS}})
    return pl.DataFrame(rows).sort(['filter_value', 'createtime'])


def cell(rolling_df, scenario):
    return eh.scenario_pairs(
        build_horizon_pairs(rolling_df, HORIZON, DIMS), SPEC, scenario, None)


def test_reported_scenarios_are_the_three_held_out_cells():
    """The figure reports interpolation, forecast and zero-shot forecast."""
    assert [s for s, _, _ in eh.SCENARIOS] == ['test_in', 'train_out', 'test_out']
    for scenario, _, _ in eh.SCENARIOS:
        traj, time = scenario.rsplit('_', 1)
        assert traj in splits.TRAJ_SPLITS and time in splits.TIME_SPLITS
        assert (traj, time) != ('train', 'in'), 'train_in is the fit, not a result'
    print('scenarios ' + ', '.join(s for s, _, _ in eh.SCENARIOS))


def test_cells_are_populated_and_disjoint():
    """Each reported cell has pairs, and shares none with the others or the fit."""
    rolling_df = make_rolling()
    key = lambda df: set(zip(df['filter_value'].to_list(), df['t0'].to_list()))

    train = key(cell(rolling_df, 'train_in'))
    reported = {}
    for scenario, _, _ in eh.SCENARIOS:
        got = cell(rolling_df, scenario)
        assert len(got) > 0, f'{scenario} is empty'
        reported[scenario] = key(got)
        print(f'  {scenario}: {len(got)} pairs, '
              f'{got["filter_value"].n_unique()} trajectories')

    for a, pairs_a in reported.items():
        assert not (pairs_a & train), f'{a} overlaps the training cell'
        for b, pairs_b in reported.items():
            if a != b:
                assert not (pairs_a & pairs_b), f'{a} overlaps {b}'
    print('reported cells are disjoint from each other and from train_in')


def test_leakage_check_fires_on_the_wrong_split():
    """scenario_pairs must reject a cell that is not held out as claimed."""
    rolling_df = make_rolling()
    paired = build_horizon_pairs(rolling_df, HORIZON, DIMS)
    labelled = splits.label_pairs(paired, SPEC, time_col='future_createtime')
    # train_in rows labelled as if they were the zero-shot cell
    leaky = splits.select(labelled, 'train', 'in').with_columns(
        pl.lit('test').alias('traj_split'), pl.lit('out').alias('time_split'))
    try:
        splits.check_leakage(splits.training_rows(labelled), leaky, 'test_out')
    except AssertionError:
        print('leakage check fires on a train_in cell relabelled test_out')
    else:
        raise AssertionError('leakage check did not fire')


def test_baselines_share_one_pair_set_per_scenario():
    """ETS and Theta are scored on the same pairs, so their no-movement
    baselines -- and any landscape line drawn from the shared pool -- coincide."""
    rolling_df = make_rolling()
    for scenario, _, _ in eh.SCENARIOS:
        ets, ets_base = eh.compute_ets_losses(
            rolling_df, HORIZON, DIMS, SPEC, scenario, None, n_pairs=40)
        theta, theta_base = eh.compute_theta_losses(
            rolling_df, HORIZON, DIMS, SPEC, scenario, None, n_pairs=40)
        assert len(ets) == len(theta) > 0, scenario
        assert np.array_equal(ets_base, theta_base), scenario
        assert np.all(np.isfinite(ets)) and np.all(np.isfinite(theta)), scenario
        print(f'  {scenario}: n={len(ets)} ets={ets.mean():.5f} '
              f'theta={theta.mean():.5f} no-movement={ets_base.mean():.5f}')


def test_sample_is_deterministic():
    """The shared pool is reproducible, which is what lets the landscape and the
    time-series caches be written by separate runs and still line up."""
    rolling_df = make_rolling()
    dim_cols = [f'x0_{d}' for d in DIMS]
    first, second = (eh._select_horizon_pairs(
        rolling_df, HORIZON, dim_cols, SPEC, 'test_out', None,
        50, eh.ETS_SAMPLE_SEED, eh.TOLERANCE_FRAC, DIMS)[0] for _ in range(2))
    assert [(s['filter_value'], s['idx_t0']) for s in first] \
        == [(s['filter_value'], s['idx_t0']) for s in second]
    assert first == sorted(first, key=lambda s: (s['filter_value'], s['idx_t0'])), \
        'the pool must come back in one canonical order for the caches to align'
    print(f'pair sample of {len(first)} is stable across calls')


def test_history_floor_scales_with_the_horizon():
    """No pair is asked to extrapolate further than it has been allowed to see."""
    rolling_df = make_rolling()
    dim_cols = [f'x0_{d}' for d in DIMS]
    spacing = eh.median_spacing_days(rolling_df)
    for horizon in (14, 60, 240):
        shift_n = eh.horizon_shift(horizon, spacing)
        assert eh.min_history(shift_n) >= shift_n
        specs, _ = eh._select_horizon_pairs(
            rolling_df, horizon, dim_cols, SPEC, 'test_out', None,
            50, eh.ETS_SAMPLE_SEED, eh.TOLERANCE_FRAC, DIMS)
        assert specs, f'{horizon}d left no pairs'
        assert min(s['idx_t0'] for s in specs) >= shift_n - 1, horizon
        print(f'  {horizon}d: shift_n={shift_n}, history >= '
              f'{eh.min_history(shift_n)} points')


def test_unreachable_horizons_are_dropped():
    """A horizon below the observation spacing rounds to a shift outside its
    own tolerance window, so it can never produce a pair."""
    assert eh.usable_horizons([7, 240], spacing_days=16.0) == [240]
    assert eh.usable_horizons([7, 240], spacing_days=2.0) == [7, 240]
    print('7d is dropped at 16d spacing and kept at 2d spacing')


def test_reversion_baselines_beat_persistence_on_an_ou_process():
    """The mean and AR(1) baselines revert, which is what ETS and Theta cannot
    do -- on a known OU process the shrinkage must beat no-movement, and the
    unconditional mean must not."""
    rolling_df = make_rolling()
    mu, alpha = eh.fit_reversion(rolling_df, HORIZON, DIMS, SPEC, None)
    assert np.all((alpha > 0) & (alpha < 1)), alpha

    ets, no_movement = eh.compute_ets_losses(
        rolling_df, HORIZON, DIMS, SPEC, 'test_out', None, n_pairs=40)
    got = {}
    for kind in ('mean', 'ar1'):
        losses, baseline = eh.compute_reversion_losses(
            rolling_df, HORIZON, DIMS, SPEC, 'test_out', None, kind, n_pairs=40)
        assert np.array_equal(baseline, no_movement), kind
        got[kind] = losses.mean()
    assert got['ar1'] < no_movement.mean(), got
    assert got['ar1'] < got['mean'], got
    assert got['mean'] > no_movement.mean(), got
    print(f"  alpha={np.array2string(alpha, precision=3)} ar1={got['ar1']:.5f} "
          f"mean={got['mean']:.5f} no-movement={no_movement.mean():.5f} "
          f"ets={ets.mean():.5f}")


def test_stale_cache_rows_are_ignored():
    """Rows from another configuration must not be plotted beside fresh ones."""
    import tempfile
    cache = {}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'pairs.parquet.zstd')
        eh._store(cache, path, 60, 'ets', [1.0, 2.0], [3.0, 4.0],
                  'ets_loss', 'ets_baseline_loss', 'fp-old', 64.0)
        assert set(eh._load_pair_cache(path, eh.ETS_PAIR_COLS, 'fp-old')) == {60}
        assert eh._load_pair_cache(path, eh.ETS_PAIR_COLS, 'fp-new') == {}
    print('a cache written under one fingerprint is invisible under another')


if __name__ == '__main__':
    for fn in (test_reported_scenarios_are_the_three_held_out_cells,
               test_cells_are_populated_and_disjoint,
               test_leakage_check_fires_on_the_wrong_split,
               test_baselines_share_one_pair_set_per_scenario,
               test_sample_is_deterministic,
               test_history_floor_scales_with_the_horizon,
               test_unreachable_horizons_are_dropped,
               test_reversion_baselines_beat_persistence_on_an_ou_process,
               test_stale_cache_rows_are_ignored):
        fn()
    print('\nall horizon-evaluation checks passed')
