"""Synthetic-data tests for the stationarity statistics.

Each test builds a panel whose answer is known by construction -- a random
walk, an AR(1), a translating cloud -- and checks the statistic recovers it.
The one that matters most is `test_cips_defactors_a_common_walk`: a panel of
stationary series riding a shared random walk is exactly the shape our
trajectories have, and it is the case a per-series test gets wrong.
"""

import datetime

import numpy as np
import polars as pl
import pytest

import stationarity as st
from latent_gp import variogram


def ar1_panel(T, M, phi=0.6, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    e = rng.normal(scale=scale, size=(T, M))
    z = np.zeros((T, M))
    for t in range(1, T):
        z[t] = phi * z[t - 1] + e[t]
    return z


def walk_panel(T, M, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(scale=scale, size=(T, M)), axis=0)


# --- unit root machinery ---------------------------------------------------

def test_adf_t_separates_walk_from_ar1():
    ar = ar1_panel(300, 1, phi=0.3, seed=1)[:, 0]
    walk = walk_panel(300, 1, seed=1)[:, 0]
    assert st.adf_t(ar) < -3.0        # stationary: strongly negative
    assert st.adf_t(walk) > -2.5      # unit root: near zero


def test_kpss_stat_matches_statsmodels():
    import warnings
    from statsmodels.tsa.stattools import kpss as sm_kpss
    warnings.simplefilter('ignore')      # interpolation + return-type notices
    z = ar1_panel(200, 3, phi=0.5, seed=2)
    T = z.shape[0]
    nlags = int(np.ceil(12 * (T / 100.0) ** 0.25))
    ours = st.kpss_stat(z, 'c', nlags=nlags)
    for m in range(z.shape[1]):
        theirs = sm_kpss(z[:, m], regression='c', nlags=nlags)[0]
        assert ours[m] == pytest.approx(theirs, rel=1e-6)


def test_kpss_stat_separates_a_walk_from_an_ar1():
    crit = st.KPSS_CRIT['c'][0.05]
    assert st.kpss_stat(walk_panel(200, 8, seed=3)).mean() > crit
    assert st.kpss_stat(ar1_panel(200, 8, seed=3)).mean() < crit


def test_kpss_trend_regression_forgives_a_linear_trend():
    z = ar1_panel(200, 6, phi=0.3, seed=4) + np.linspace(0, 8, 200)[:, None]
    assert st.kpss_stat(z, 'c').mean() > st.KPSS_CRIT['c'][0.05]
    assert st.kpss_stat(z, 'ct').mean() < st.KPSS_CRIT['ct'][0.05]


def test_cips_defactors_a_common_walk():
    """Stationary units on a shared random walk: the case Fisher gets wrong.

    Every series has a unit root marginally, because the common factor does.
    The per-series ADF therefore cannot reject, while the cross-sectionally
    augmented statistic sees through the factor to the stationary idiosyncratic
    part.
    """
    T, M = 300, 40
    z = ar1_panel(T, M, phi=0.4, seed=5) + walk_panel(T, 1, seed=6) * 3.0
    plain = np.mean([st.adf_t(z[:, m]) for m in range(M)])
    augmented, _ = st.cips(z)
    assert plain > -2.5           # per-series: the factor hides the signal
    assert augmented < -4.0       # defactored: recovers the AR(1)
    assert augmented < plain - 1.5


# --- bootstrap -------------------------------------------------------------

def test_block_resample_keeps_units_aligned():
    rng = np.random.default_rng(7)
    X = np.arange(60)[:, None] * np.ones((1, 4))
    out = st.block_resample(X, 5, rng)
    assert out.shape == X.shape
    # the same time index for every unit, so every row stays constant across units
    assert np.allclose(out, out[:, :1])


def test_unit_root_surrogate_is_a_driftless_walk():
    z = walk_panel(200, 10, seed=8) + np.linspace(0, 20, 200)[:, None]
    s = st.unit_root_surrogate(z, st.default_block(200), np.random.default_rng(9))
    assert s.shape == z.shape
    # the drift the original carries is removed, the walk is not
    assert abs(np.diff(s, axis=0).mean()) < 0.05
    assert st.kpss_stat(s).mean() > st.KPSS_CRIT['c'][0.05]


def test_stationary_surrogate_is_stationary():
    z = walk_panel(300, 10, seed=10)
    s = st.stationary_surrogate(z, st.default_block(300), np.random.default_rng(11))
    assert st.kpss_stat(z).mean() > st.KPSS_CRIT['c'][0.05]
    assert st.kpss_stat(s).mean() < st.KPSS_CRIT['c'][0.05]


def test_bootstrap_p_is_never_zero_and_bounded():
    assert st.bootstrap_p(-10.0, np.zeros(99), 'lower') == pytest.approx(1 / 100)
    assert st.bootstrap_p(10.0, np.zeros(99), 'lower') == pytest.approx(1.0)
    assert np.isnan(st.bootstrap_p(np.nan, np.zeros(9), 'lower'))


def test_panel_unit_root_rejects_stationary_panel_only():
    quiet = lambda *a, **k: None
    ar = st.panel_unit_root(ar1_panel(200, 25, phi=0.3, seed=12),
                            n_boot=99, log=quiet)
    walk = st.panel_unit_root(walk_panel(200, 25, seed=13), n_boot=99, log=quiet)
    assert ar['rejects_unit_root']
    assert not walk['rejects_unit_root']


def test_panel_kpss_rejects_a_walk_only():
    quiet = lambda *a, **k: None
    ar = st.panel_kpss(ar1_panel(200, 25, phi=0.3, seed=14), n_boot=99, log=quiet)
    walk = st.panel_kpss(walk_panel(200, 25, seed=15), n_boot=99, log=quiet)
    assert not ar['rejects_stationarity']
    assert walk['rejects_stationarity']


# --- displacement ----------------------------------------------------------

def test_msd_recovers_diffusive_scaling():
    z = walk_panel(400, 30, seed=16)[:, :, None]
    r = st.msd(z, dt_days=16.0, log=lambda *a, **k: None)
    assert r['alpha'] == pytest.approx(1.0, abs=0.1)
    assert not r['saturates']


def test_msd_recovers_ballistic_drift():
    T = 400
    z = (np.linspace(0, 40, T)[:, None] + walk_panel(T, 30, seed=17) * 0.05)[:, :, None]
    r = st.msd(z, dt_days=16.0, log=lambda *a, **k: None)
    assert r['alpha'] == pytest.approx(2.0, abs=0.1)
    assert r['v'] > r['D']


def test_msd_saturates_for_an_ornstein_uhlenbeck_process():
    z = ar1_panel(600, 40, phi=0.9, seed=18)[:, :, None]
    r = st.msd(z, dt_days=16.0, max_lag=250, log=lambda *a, **k: None)
    assert r['saturates']
    assert r['plateau_ratio'] == pytest.approx(1.0, abs=0.15)
    assert r['alpha'] < 0.5


def test_msd_does_not_call_a_random_walk_saturated():
    r = st.msd(walk_panel(600, 40, seed=25)[:, :, None], dt_days=16.0,
               max_lag=250, log=lambda *a, **k: None)
    assert not r['saturates']
    assert r['tail_growth'] > 0.1


def test_msd_lags_are_in_days():
    z = walk_panel(100, 5, seed=19)[:, :, None]
    r = st.msd(z, dt_days=16.0, log=lambda *a, **k: None)
    assert r['lags_days'][0] == pytest.approx(16.0)
    assert r['lags_days'][-1] == pytest.approx(50 * 16.0)


def test_msd_diffusion_and_drift_stay_non_negative():
    """nnls, not lstsq with a clip: a clipped negative coefficient would leave
    the reported curve inconsistent with the reported D and |v|."""
    z = ar1_panel(300, 20, phi=0.95, seed=20)[:, :, None]
    r = st.msd(z, dt_days=16.0, log=lambda *a, **k: None)
    assert r['D'] >= 0 and r['v'] >= 0


# --- panel construction ----------------------------------------------------

def _frame(spans, n_dims=2, dt_days=16.0, sd=0.1):
    """A latent-shaped frame from {seed: (first_bin, last_bin)}."""
    t0 = datetime.datetime(2022, 1, 1)
    rows = []
    for seed, (lo, hi) in spans.items():
        for t in range(lo, hi + 1):
            rows.append({
                'createtime': t0 + datetime.timedelta(days=dt_days * t),
                'filter_value': seed,
                'causal_2d': [float(t), float(-t)][:n_dims],
                'sd_2d': [sd] * n_dims,
                'n_posts': 10.0,
            })
    return pl.DataFrame(rows, schema_overrides={
        'causal_2d': pl.Array(pl.Float64, n_dims),
        'sd_2d': pl.Array(pl.Float64, n_dims)})


def test_balanced_panel_maximises_the_rectangle():
    df = _frame({'a': (0, 30), 'b': (0, 30), 'c': (10, 40), 'd': (10, 40)})
    p = st.balanced_panel(df, 'causal_2d', 'sd_2d', 2, 16.0, min_bins=8,
                          min_seeds=2, log=lambda *a, **k: None)
    # all four over the shared 21 bins (84 cells) beats the two long ones (62)
    assert p.Z.shape == (21, 4, 2)
    assert p.seeds == ['a', 'b', 'c', 'd']


def test_balanced_panel_span_floor_beats_the_seed_count():
    """The floor is why it exists: without min_bins the search would take the
    short wide rectangle and leave the panel tests without a span."""
    df = _frame({'a': (0, 30), 'b': (0, 30), 'c': (10, 40), 'd': (10, 40)})
    p = st.balanced_panel(df, 'causal_2d', 'sd_2d', 2, 16.0, min_bins=25,
                          min_seeds=2, log=lambda *a, **k: None)
    assert p.Z.shape == (31, 2, 2)
    assert p.seeds == ['a', 'b']


def test_balanced_panel_raises_when_no_rectangle_clears_the_floors():
    df = _frame({'a': (0, 10), 'b': (20, 30)})
    with pytest.raises(ValueError, match='no seed set covers'):
        st.balanced_panel(df, 'causal_2d', 'sd_2d', 2, 16.0, min_bins=5,
                          min_seeds=2, log=lambda *a, **k: None)


def test_balanced_panel_drops_seeds_with_interior_gaps():
    df = _frame({'a': (0, 20), 'b': (0, 20), 'c': (0, 20)})
    df = df.filter(~((pl.col('filter_value') == 'c')
                     & (pl.col('createtime') == df['createtime'][5])))
    p = st.balanced_panel(df, 'causal_2d', 'sd_2d', 2, 16.0, min_bins=8,
                          min_seeds=2, log=lambda *a, **k: None)
    assert 'c' not in p.seeds and p.Z.shape[1] == 2


def test_panel_weights_follow_posterior_precision():
    p = st.Panel(Z=np.array([[[1.0], [3.0]]]), SD=np.array([[[0.1], [1.0]]]),
                 times=np.array([np.datetime64('2022-01-01')]), seeds=['a', 'b'],
                 dt_days=16.0)
    # the well-determined seed dominates: near 1, not the unweighted 2
    assert p.wmean()[0, 0] == pytest.approx(1.0, abs=0.05)


def test_panel_demeaned_removes_the_common_factor():
    T, M = 50, 6
    common = np.linspace(0, 5, T)[:, None, None]
    Z = np.zeros((T, M, 1)) + common + np.arange(M)[None, :, None]
    p = st.Panel(Z=Z, SD=np.ones_like(Z), times=np.arange(T).astype('datetime64[D]'),
                 seeds=list('abcdef'), dt_days=16.0)
    d = p.demeaned()
    assert np.allclose(d - d[0], 0, atol=1e-9)   # only the constant offsets survive


# --- window statistics and spread ------------------------------------------

def test_window_drift_recovers_a_known_shift():
    T, M = 120, 30
    Z = (ar1_panel(T, M, phi=0.2, seed=21)
         + np.linspace(0, 2.0, T)[:, None])[:, :, None]
    p = st.Panel(Z=Z, SD=np.full_like(Z, 0.1),
                 times=(np.datetime64('2022-01-01')
                        + np.arange(T) * np.timedelta64(16, 'D')),
                 seeds=[str(i) for i in range(M)], dt_days=16.0)
    r = st.window_drift(p, n_windows=6, log=lambda *a, **k: None)
    assert r['mean_drift'][0] == pytest.approx(2.0 * 5 / 6, abs=0.15)
    assert r['max_abs_d'] > 0.8


def test_window_drift_flat_for_a_stationary_panel():
    T, M = 200, 40
    Z = ar1_panel(T, M, phi=0.2, seed=22)[:, :, None]
    p = st.Panel(Z=Z, SD=np.full_like(Z, 0.1),
                 times=(np.datetime64('2022-01-01')
                        + np.arange(T) * np.timedelta64(16, 'D')),
                 seeds=[str(i) for i in range(M)], dt_days=16.0)
    r = st.window_drift(p, n_windows=6, log=lambda *a, **k: None)
    assert r['max_abs_d'] < 0.2


def test_ensemble_spread_separates_translation_from_dispersion():
    T, M = 100, 30
    rng = np.random.default_rng(23)
    base = rng.normal(size=(1, M, 1))
    times = np.datetime64('2022-01-01') + np.arange(T) * np.timedelta64(16, 'D')
    seeds = [str(i) for i in range(M)]
    moving = base + np.linspace(0, 10, T)[:, None, None]      # rigid translation
    fanning = base * np.linspace(1, 4, T)[:, None, None]      # dispersion
    for Z, expect in ((moving, False), (fanning, True)):
        p = st.Panel(Z=Z, SD=np.ones_like(Z), times=times, seeds=seeds, dt_days=16.0)
        assert st.ensemble_spread(p, log=lambda *a, **k: None)['dispersing'] is expect


def test_common_break_finds_a_level_shift():
    T, M = 200, 20
    shift = np.where(np.arange(T) > 120, 4.0, 0.0)[:, None, None]
    Z = ar1_panel(T, M, phi=0.2, seed=24)[:, :, None] + shift
    p = st.Panel(Z=Z, SD=np.ones_like(Z),
                 times=(np.datetime64('2022-01-01')
                        + np.arange(T) * np.timedelta64(16, 'D')),
                 seeds=[str(i) for i in range(M)], dt_days=16.0)
    out = st.common_break(p, [0], log=lambda *a, **k: None)
    assert out[0]['rejects']
    assert out[0]['break_date'] == str(p.times[120])[:10]


# --- variogram verdict -----------------------------------------------------

def _rows(excess):
    return [(16.0 * (i + 1), 100, e, 0.0, e) for i, e in enumerate(excess)]


def test_variogram_verdict_calls_a_plateau_bounded():
    real = _rows([0.0, 0.4, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0])
    null = _rows([0.0] * 8)
    v = variogram.verdict(real, null)
    assert v['saturates']
    assert v['drift_sd'] == pytest.approx(np.sqrt(0.5), rel=1e-6)


def test_variogram_verdict_calls_sustained_growth_a_walk():
    real = _rows(list(np.linspace(0, 1, 8)))
    null = _rows([0.0] * 8)
    assert not variogram.verdict(real, null)['saturates']


# --- end to end ------------------------------------------------------------

def synthetic_frame(T=140, M=24, n_fast=2, n_dims=4, dt_days=16.0, seed=30,
                    with_smoothed=False):
    """A latent frame shaped like `build_latents` output.

    Fast dims drift on a shared factor plus idiosyncratic AR(1) noise; slow
    dims are frozen per seed, as `slow_kind='const'` makes them. Seeds start
    and end at staggered bins, so the balanced-panel search has something to do.
    """
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(scale=0.15, size=(T, n_fast)), axis=0)
    Z = np.zeros((T, M, n_dims))
    Z[:, :, :n_fast] = (common[:, None, :]
                        + np.stack([ar1_panel(T, M, phi=0.5, seed=seed + k)
                                    for k in range(n_fast)], axis=2) * 0.3)
    Z[:, :, n_fast:] = rng.normal(size=(1, M, n_dims - n_fast))
    # The smoothed state of a const-prior dimension is exactly constant; the
    # filtered state is a running estimate of that constant and still carries a
    # decaying transient, which is the artefact `smoothed_col` exists to dodge.
    S = Z.copy()
    S[:, :, n_fast:] = Z[0, :, n_fast:][None, :, :]
    if with_smoothed:
        transient = np.exp(-np.arange(T) / (T / 3.0))[:, None, None]
        Z = Z.copy()
        Z[:, :, n_fast:] = S[:, :, n_fast:] + transient * rng.normal(
            size=(1, M, n_dims - n_fast))
    t0 = datetime.datetime(2022, 1, 1)
    rows = []
    for m in range(M):
        lo, hi = (m % 5) * 3, T - 1 - (m % 4) * 3
        for t in range(lo, hi + 1):
            rows.append({'createtime': t0 + datetime.timedelta(days=dt_days * t),
                         'filter_value': f'seed-{m:03d}',
                         'causal_4d': list(Z[t, m]),
                         'coord_4d': list(S[t, m]),
                         'sd_4d': [0.2] * n_dims,
                         'n_posts': 20.0})
    return pl.DataFrame(rows, schema_overrides={
        'causal_4d': pl.Array(pl.Float64, n_dims),
        'coord_4d': pl.Array(pl.Float64, n_dims),
        'sd_4d': pl.Array(pl.Float64, n_dims)})


@pytest.fixture(scope='module')
def analysed():   # one run, six assertions over it
    quiet = lambda *a, **k: None
    v = variogram.verdict(_rows([0.0, 0.4, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0]),
                          _rows([0.0] * 8))
    return st.analyse(synthetic_frame(), 'causal_4d', 'sd_4d', 16.0, [0, 1], [2, 3],
                      v, n_boot=99, n_windows=4, min_bins=30, min_seeds=10,
                      log=quiet)


@pytest.fixture(scope='module')
def analysed_all_fast():
    """A homogeneous prior: every dimension is fast, so there is no control."""
    quiet = lambda *a, **k: None
    v = variogram.verdict(_rows([0.0, 0.4, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0]),
                          _rows([0.0] * 8))
    return st.analyse(synthetic_frame(), 'causal_4d', 'sd_4d', 16.0,
                      [0, 1, 2, 3], [], v, n_boot=99, n_windows=4, min_bins=30,
                      min_seeds=10, log=quiet)


def test_analyse_skips_the_control_when_no_dimension_is_slow(analysed_all_fast):
    assert 'msd_slow' not in analysed_all_fast
    assert 'slow' not in analysed_all_fast['panel']
    assert analysed_all_fast['panel']['fast']['by_dim']


def test_the_report_runs_without_a_control_block(analysed_all_fast, tmp_path):
    lines = []
    st.summarise(analysed_all_fast, log=lines.append)
    assert any('every dimension is fast' in ln for ln in lines)

    table, _ = st.write_tex(analysed_all_fast, cfg=None, out_dir=str(tmp_path))
    body = open(table).read()
    assert 'slow (control)' not in body
    assert 'no slow block configured' in body


def test_analyse_produces_every_key_the_report_reads(analysed):
    for key in ('blocks', 'dt_days', 'panel_shape', 'variogram', 'msd_raw',
                'msd_demeaned', 'msd_slow', 'panel', 'window',
                'window_unbalanced', 'spread', 'breaks'):
        assert key in analysed
    for key in ('fast', 'slow', 'fast_kpss', 'slow_kpss', 'fast_ct'):
        assert key in analysed['panel']
    assert len(analysed['panel']['fast']['by_dim']) == 2


def test_analyse_reports_the_frozen_block_as_frozen(analysed):
    """The slow block is the control: `slow_kind='const'` cannot move, so a
    test that finds it drifting is measuring something other than the data."""
    assert analysed['msd_slow']['msd'].max() < 1e-6
    assert analysed['panel']['slow']['degenerate']
    assert analysed['panel']['slow_kpss']['degenerate']
    assert analysed['panel']['slow']['n_live'] == 0
    assert analysed['msd_raw']['msd'].max() > analysed['msd_slow']['msd'].max()


def test_analyse_verdict_needs_every_live_dimension_to_agree(analysed):
    """Both fast dims are stationary AR(1) on a shared walk by construction,
    so the defactored test should reject the unit root on both of them."""
    fast = analysed['panel']['fast']
    assert fast['n_live'] == 2 and fast['n_reject'] == 2
    assert fast['rejects_unit_root']


def test_combine_dims_will_not_reject_on_one_dimension_of_two():
    split = [{'rejects_unit_root': True, 'p_value': 0.01, 'cips': -4.0},
             {'rejects_unit_root': False, 'p_value': 0.40, 'cips': -1.0}]
    got = st.combine_dims(split, 'rejects_unit_root', 'cips',
                          lambda v: float(np.mean(v)))
    assert not got['rejects_unit_root']
    assert got['p_value'] == 0.40          # the weakest, not the most flattering
    assert got['n_reject'] == 1 and got['n_live'] == 2


def test_combine_dims_ignores_frozen_dimensions():
    mixed = [{'rejects_unit_root': True, 'p_value': 0.01, 'cips': -4.0},
             {'degenerate': True, 'rejects_unit_root': False,
              'p_value': np.nan, 'cips': np.nan}]
    got = st.combine_dims(mixed, 'rejects_unit_root', 'cips',
                          lambda v: float(np.mean(v)))
    assert got['n_live'] == 1 and got['rejects_unit_root']


def test_control_is_frozen_only_on_the_smoothed_state():
    """The bug the real data exposed.

    A `slow_kind='const'` dimension is exactly constant only in the smoothed
    state. Its filtered state is a running estimate of that constant, so the
    control reads as moving and the test appears to detect non-stationarity
    where by construction there is none.
    """
    quiet = lambda *a, **k: None
    v = variogram.verdict(_rows([0.0, 0.4, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0]),
                          _rows([0.0] * 8))
    df = synthetic_frame(with_smoothed=True)
    kw = dict(n_boot=19, n_windows=4, min_bins=30, min_seeds=10, log=quiet)

    on_filtered = st.analyse(df, 'causal_4d', 'sd_4d', 16.0, [0, 1], [2, 3], v, **kw)
    assert not on_filtered['panel']['slow'].get('degenerate')   # the artefact

    on_smoothed = st.analyse(df, 'causal_4d', 'sd_4d', 16.0, [0, 1], [2, 3], v,
                             smoothed_col='coord_4d', **kw)
    assert on_smoothed['panel']['slow']['degenerate']           # control restored
    assert on_smoothed['panel']['slow']['n_live'] == 0


def test_fast_block_is_reported_on_both_states():
    quiet = lambda *a, **k: None
    v = variogram.verdict(_rows([0.0, 0.4, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0]),
                          _rows([0.0] * 8))
    got = st.analyse(synthetic_frame(with_smoothed=True), 'causal_4d', 'sd_4d',
                     16.0, [0, 1], [2, 3], v, smoothed_col='coord_4d',
                     n_boot=19, n_windows=4, min_bins=30, min_seeds=10, log=quiet)
    assert got['panel']['fast_smoothed']['n_live'] == 2
    assert not got['panel']['fast_smoothed'].get('degenerate')


def test_analyse_separates_balanced_from_unbalanced_drift(analysed):
    assert analysed['window']['max_abs_d'] >= 0
    assert analysed['window_unbalanced']['n_seeds_first'] > 0
    assert analysed['window_unbalanced']['n_seeds_last'] > 0


def test_summarise_and_write_tex_run_over_a_real_result(analysed, tmp_path):
    lines = []
    st.summarise(analysed, log=lines.append)
    assert any('variogram' in ln for ln in lines)
    assert any('latent_gp.sweep' in ln for ln in lines)

    table, macros = st.write_tex(analysed, cfg=None, out_dir=str(tmp_path))
    body = open(table).read()
    assert '\\toprule' in body and '\\label{tab:stationarity}' in body
    assert 'slow (control)' in body
    text = open(macros).read()
    assert text.count('\\newcommand') == 13
    assert '\\statCIPS' in text and '\\statMaxCohenD' in text


def test_write_tex_macros_carry_no_placeholder_when_the_tests_ran(analysed, tmp_path):
    _, macros = st.write_tex(analysed, cfg=None, out_dir=str(tmp_path))
    for line in open(macros):
        assert '{--}' not in line, line


def test_panel_tests_report_a_frozen_block_instead_of_nan():
    """`slow_kind='const'` gives constant series. Reporting that as 'frozen'
    rather than nan is what makes the slow block usable as a control."""
    quiet = lambda *a, **k: None
    frozen = np.tile(np.arange(20.0), (60, 1))     # constant in time per unit
    root = st.panel_unit_root(frozen, n_boot=19, log=quiet)
    kp = st.panel_kpss(frozen, n_boot=19, log=quiet)
    assert root['degenerate'] and not root['rejects_unit_root']
    assert kp['degenerate'] and not kp['rejects_stationarity']


def test_panel_tests_are_not_degenerate_on_live_data():
    quiet = lambda *a, **k: None
    z = ar1_panel(120, 20, seed=31)
    assert not st.panel_unit_root(z, n_boot=19, log=quiet)['degenerate']
    assert not st.panel_kpss(z, n_boot=19, log=quiet)['degenerate']


def test_drop_prior_dominated_keeps_the_measured_seeds():
    """A seed the fit could not pin down sits at the prior mean, which reads
    as stationary for reasons that have nothing to do with the data."""
    df = pl.concat([
        _frame({'sharp-a': (0, 20), 'sharp-b': (0, 20)}, sd=0.2),
        _frame({'vague-c': (0, 20)}, sd=0.95),
    ])
    kept = st.drop_prior_dominated(df, 'sd_2d', [0, 1], 0.8,
                                   log=lambda *a, **k: None)
    assert sorted(kept['filter_value'].unique().to_list()) == ['sharp-a', 'sharp-b']


def test_drop_prior_dominated_raises_when_nothing_survives():
    df = _frame({'a': (0, 20), 'b': (0, 20)}, sd=0.95)
    with pytest.raises(ValueError, match='leaves no panel'):
        st.drop_prior_dominated(df, 'sd_2d', [0, 1], 0.8, log=lambda *a, **k: None)


def test_frozen_block_survives_smoother_round_off():
    """The full-data control failure: a `const` dimension is constant only up
    to the smoother's round-off, which on the full fit reached ~1e-7 of the
    cross-sectional spread — far above any absolute floor, and still frozen."""
    quiet = lambda *a, **k: None
    rng = np.random.default_rng(40)
    levels = rng.normal(scale=1.27, size=(1, 60))        # real between-seed spread
    for noise in (1e-15, 1e-9, 1e-7, 1e-5):
        Z = levels + rng.normal(scale=noise, size=(94, 60))
        got = st.panel_unit_root(Z, n_boot=19, log=quiet)
        assert got['degenerate'], f'round-off {noise:g} read as motion'
        assert st.panel_kpss(Z, n_boot=19, log=quiet)['degenerate']


def test_real_motion_is_never_called_frozen():
    """The other side of the same threshold: within-seed motion at the scale
    the fast dims actually show (~0.5 of the cross-sectional spread)."""
    quiet = lambda *a, **k: None
    rng = np.random.default_rng(41)
    Z = rng.normal(scale=1.27, size=(1, 60)) + ar1_panel(94, 60, phi=0.8, seed=42) * 0.6
    assert not st.panel_unit_root(Z, n_boot=19, log=quiet)['degenerate']
    assert not st.panel_kpss(Z, n_boot=19, log=quiet)['degenerate']


def test_window_drift_reports_a_signed_contraction():
    """A shrinking variance must not reach the prose as growth: the macros the
    paper cites carry the sign, not the magnitude."""
    T, M = 120, 30
    rng = np.random.default_rng(50)
    # variance contracts and the mean falls, the shape the real data shows
    Z = (rng.normal(size=(T, M)) * np.linspace(1.0, 0.6, T)[:, None]
         - np.linspace(0, 1.0, T)[:, None])[:, :, None]
    p = st.Panel(Z=Z, SD=np.full_like(Z, 0.1),
                 times=(np.datetime64('2022-01-01')
                        + np.arange(T) * np.timedelta64(16, 'D')),
                 seeds=[str(i) for i in range(M)], dt_days=16.0)
    r = st.window_drift(p, n_windows=6, log=lambda *a, **k: None)
    assert r['max_d'] < 0 and r['max_var_rel'] < 0
    assert r['max_abs_d'] == pytest.approx(abs(r['max_d']))
    assert r['max_abs_var_rel'] == pytest.approx(abs(r['max_var_rel']))


def test_macros_carry_the_sign_of_the_drift(analysed, tmp_path):
    _, macros = st.write_tex(analysed, cfg=None, out_dir=str(tmp_path))
    text = dict(l.strip().removeprefix('\\newcommand{\\').rstrip('}').split('}{')
                for l in open(macros) if l.strip())
    w = analysed['window']
    assert float(text['statMaxCohenD']) == pytest.approx(w['max_d'], abs=5e-4)
    assert text['statMaxVarPct'][0] in '+-'
