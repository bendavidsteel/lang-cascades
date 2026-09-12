"""Prior components: what each kernel implies for the latent's covariance.

The fit only ever sees a component through the (F, Q, P0) it contributes, so
these check the implied prior covariance rather than the blocks themselves.
"""

import numpy as np
import pytest

import jax.numpy as jnp

from . import core as gpfa
from . import fit

DT = 16.0

# the jitter build_ssm adds to Q accumulates over the horizon, and an additive
# form carries more states than the single component it equals, so two exactly
# equal kernels still differ by a little more than machine epsilon
JITTER_FLOOR = 1e-9


def prior_cov(comps, K=2, T=40, dt=DT):
    """Cov(f_s, f_t) for one latent dimension under a component list."""
    F, Q, P0, S = gpfa.build_ssm(dt, comps, K)
    P = [np.asarray(P0)]
    for _ in range(T - 1):
        P.append(F @ P[-1] @ F.T + Q)
    Fp = [np.eye(F.shape[0])]
    for _ in range(T - 1):
        Fp.append(F @ Fp[-1])
    return np.array([[float((S @ Fp[t - s] @ P[s] @ S.T)[0, 0]) if t >= s else 0.0
                      for s in range(T)] for t in range(T)])


def variance(comps, **kw):
    return np.diag(prior_cov(comps, **kw))


# ---------------------------------------------------------------- degeneracy

def test_two_wieners_are_one_wiener():
    """Summing Wiener components on one dimension buys nothing.

    Both have F = 1, so the sum of the two states is itself a random walk whose
    Q is the sum of theirs -- a single Wiener at the combined timescale.
    """
    two = prior_cov([dict(kind='wiener', tau=640., p0=0.5),
                     dict(kind='wiener', tau=640., p0=0.5)])
    one = prior_cov([dict(kind='wiener', tau=320., p0=1.0)])
    assert np.abs(two - one).max() / np.abs(one).max() < JITTER_FLOOR


def test_const_plus_wiener_is_one_wiener():
    """A frozen level only shifts the Wiener's initial variance."""
    add = prior_cov([dict(kind='const', var=1.0),
                     dict(kind='wiener', tau=640., p0=0.0)])
    one = prior_cov([dict(kind='wiener', tau=640., p0=1.0)])
    assert np.abs(add - one).max() / np.abs(one).max() < JITTER_FLOOR


def test_wiener_plus_matern_is_not_degenerate():
    """A real two-timescale sum needs components of different shape."""
    add = prior_cov([dict(kind='wiener', tau=2560., p0=1.0),
                     dict(kind='matern32', tau=80., var=1.0)])
    one = prior_cov([dict(kind='wiener', tau=2560., p0=2.0)])
    assert np.abs(add - one).max() / np.abs(one).max() > 1e-3


# ------------------------------------------------------------------------ ou

def test_ou_is_stationary():
    v = variance([dict(kind='ou', tau=640., var=1.0)])
    assert np.abs(v - 1.0).max() < 1e-10


def test_wiener_is_not_stationary():
    v = variance([dict(kind='wiener', tau=640., var=1.0)])
    assert v[-1] > 1.5 * v[0]


@pytest.mark.parametrize('tau', [640., 2560.])
def test_ou_matches_wiener_increments_at_short_lag(tau):
    """The tau scaling is chosen so the two differ only by reverting."""
    ou = prior_cov([dict(kind='ou', tau=tau, var=1.0)])
    wi = prior_cov([dict(kind='wiener', tau=tau, var=1.0, p0=1.0)])

    def first_increment(C):
        return C[0, 0] + C[1, 1] - 2 * C[1, 0]

    a, b = first_increment(ou), first_increment(wi)
    assert abs(a - b) / b < 2.0 * DT / tau


def test_ou_reverts_where_wiener_does_not():
    """Past its correlation time the OU forgets where it started; the Wiener does not."""
    def lag_corr(comps, lag=60):
        C = prior_cov(comps, T=lag + 1)
        return C[lag, 0] / np.sqrt(C[0, 0] * C[lag, lag])

    # lag 60 bins at dt=16 is 960 days, three OU correlation times at this tau
    assert lag_corr([dict(kind='ou', tau=160., var=1.0)]) < 0.15
    assert lag_corr([dict(kind='wiener', tau=160., var=1.0, p0=1.0)]) > 0.3


def test_ou_becomes_const_as_tau_grows():
    ou = prior_cov([dict(kind='ou', tau=1e7, var=1.0)])
    const = prior_cov([dict(kind='const', var=1.0)])
    assert np.abs(ou - const).max() < 1e-4


def test_ou_is_rougher_than_matern_at_one_tau():
    """Both revert; only the Matern has differentiable paths, so it steps less."""
    def first_increment(comps):
        C = prior_cov(comps)
        return C[0, 0] + C[1, 1] - 2 * C[1, 0]

    assert (first_increment([dict(kind='ou', tau=640., var=1.0)])
            > first_increment([dict(kind='matern32', tau=640., var=1.0)]))


# ------------------------------------------------------- prior_components

def test_homogeneous_mix_is_shared_across_dimensions():
    """n_fast >= K is how the sweep reaches a homogeneous prior."""
    comps = fit.prior_components(K=6, n_fast=6, fast_tau=80., fast_kind='ou')
    assert comps == [dict(kind='ou', tau=80.0, var=1.0)]
    assert not gpfa.is_heterogeneous(comps)


def test_a_real_mix_is_per_dimension_and_fixes_the_basis():
    comps = fit.prior_components(K=3, n_fast=1, fast_tau=80., fast_kind='ou',
                                 slow_kind='const')
    assert len(comps) == 3 and comps[0][0]['kind'] == 'ou'
    assert comps[1][0]['kind'] == 'const'
    assert gpfa.is_heterogeneous(comps)


def test_both_kinds_reach_the_drifting_dimensions():
    for kind in ('wiener', 'ou'):
        comps = fit.prior_components(K=3, n_fast=2, fast_tau=40., fast_kind=kind,
                                     slow_kind='wiener', slow_tau=2560.)
        assert comps[0][0] == dict(kind=kind, tau=40.0, var=1.0)
        assert comps[2][0] == dict(kind='wiener', tau=2560.0, var=1.0)


def test_slow_kind_takes_an_ou():
    comps = fit.prior_components(K=3, n_fast=1, fast_tau=80., slow_kind='ou',
                                 slow_tau=640.)
    assert comps[1][0] == dict(kind='ou', tau=640.0, var=1.0)


# --------------------------------------------------- loading prior (w_ridge)

def _two_target_cells(n_small, n_large, K=2, T=4):
    """One sparse target and one dense one, over a shared latent."""
    rng = np.random.default_rng(0)
    M = 6
    j, m, t = [], [], []
    for target, n in ((0, n_small), (1, n_large)):
        for _ in range(n):
            j.append(target)
            m.append(rng.integers(M))
            t.append(rng.integers(T))
    j = np.array(j); m = np.array(m); t = np.array(t)
    d = {'J': 2, 'M': M, 'T': T, 'j': jnp.asarray(j),
         'flat': jnp.asarray(m * T + t)}
    Ez = jnp.asarray(rng.normal(size=(M, T, K)))
    Ezz = jnp.zeros((M, T, K, K)) + 1e-3 * jnp.eye(K)
    return d, Ez, Ezz, K


def _solve(w_ridge, n_small=4, n_large=400, prec_scale=1.0):
    d, Ez, Ezz, K = _two_target_cells(n_small, n_large)
    rng = np.random.default_rng(1)
    n_cells = d['j'].shape[0]
    prec = jnp.full(n_cells, prec_scale)
    target = jnp.asarray(rng.normal(size=n_cells))
    return gpfa.m_step(d, Ez, Ezz, prec, target, K, w_ridge=w_ridge)


def test_w_ridge_shrinks_the_sparse_target_more_than_the_dense_one():
    W_off, _ = _solve(None)
    W_on, _ = _solve(50.0)
    sparse = np.linalg.norm(W_on[0]) / max(np.linalg.norm(W_off[0]), 1e-12)
    dense = np.linalg.norm(W_on[1]) / max(np.linalg.norm(W_off[1]), 1e-12)
    assert sparse < dense, (sparse, dense)
    assert dense > 0.8, dense           # a target with the evidence keeps it


def test_a_ridge_proportional_to_the_evidence_would_not_discriminate():
    """Why the prior is an absolute scale: scaling every cell's precision
    together rescales A_j, and a ridge that tracked it would shrink both
    targets by the same factor."""
    W_a, _ = _solve(50.0, prec_scale=1.0)
    W_b, _ = _solve(500.0, prec_scale=10.0)
    ratio_sparse = np.linalg.norm(W_b[0]) / max(np.linalg.norm(W_a[0]), 1e-12)
    ratio_dense = np.linalg.norm(W_b[1]) / max(np.linalg.norm(W_a[1]), 1e-12)
    assert np.isclose(ratio_sparse, ratio_dense, rtol=0.2), \
        (ratio_sparse, ratio_dense)


def test_the_intercept_is_not_shrunk_toward_neutral():
    """W and b are solved jointly, so penalising W moves b -- but it must move
    it toward the target's own mean, not toward zero, which is what penalising
    b would do."""
    d, Ez, Ezz, K = _two_target_cells(4, 400)
    rng = np.random.default_rng(1)
    n_cells = d['j'].shape[0]
    prec = jnp.full(n_cells, 1.0)
    target = jnp.asarray(rng.normal(loc=2.0, size=n_cells))
    _, b = gpfa.m_step(d, Ez, Ezz, prec, target, K, w_ridge=1e6)

    j = np.asarray(d['j'])
    tg = np.asarray(target)
    for target_idx in (0, 1):
        sel = j == target_idx
        weighted_mean = tg[sel].mean()
        assert np.isclose(b[target_idx], weighted_mean, atol=1e-3), \
            (target_idx, float(b[target_idx]), weighted_mean)


def test_default_w_ridge_keeps_the_cache_tag_of_fits_that_predate_it():
    from latent_gp.latents import LatentConfig, W_RIDGE_OFF
    base = dict(cells_path='x.parquet', n_dims=6, n_fast=1, fast_tau=20.0,
                fast_kind='ou', slow_kind='const', bin_factor=8)
    assert LatentConfig(**base).tag == LatentConfig(**base, w_ridge=W_RIDGE_OFF).tag
    assert LatentConfig(**base).tag != LatentConfig(**base, w_ridge=50.0).tag


def test_the_bridge_matches_the_exact_ou_posterior_between_knots():
    """Linear weights are the F -> 1 limit, so the gap is the whole correction:
    the OU bridge sags toward the prior mean and linear interpolation does not.
    """
    from latent_gp.latents import _bridge_weights
    dt, tau = 32.0, 10.0
    F = float(gpfa.component('ou', dt=dt, tau=tau)[0][0, 0])
    w = np.linspace(0.0, 1.0, 9)[:, None]
    lo, hi = _bridge_weights(w, np.array([F]))

    # exact conditional mean under the exponential kernel, knots at 1 and 1
    rho = lambda s: F ** s
    exact = [(rho(x) - rho(2 - x)) / (1 - F ** 2) + (rho(1 - x) - rho(1 + x)) / (1 - F ** 2)
             for x in w[:, 0]]
    assert np.allclose(lo[:, 0] + hi[:, 0], exact, atol=1e-12)
    assert np.isclose(lo[0, 0], 1.0) and np.isclose(hi[0, 0], 0.0)
    assert np.isclose(lo[-1, 0], 0.0) and np.isclose(hi[-1, 0], 1.0)
    mid = lo[4, 0] + hi[4, 0]
    assert mid < 0.8, mid                      # linear would be exactly 1.0


def test_a_non_reverting_prior_still_interpolates_linearly():
    from latent_gp.latents import _bridge_weights
    w = np.linspace(0.0, 1.0, 5)[:, None]
    for kind, kw in (('const', {}), ('wiener', dict(tau=640.0))):
        F = float(gpfa.component(kind, dt=16.0, **kw)[0][0, 0])
        lo, hi = _bridge_weights(w, np.array([F]))
        assert np.allclose(lo[:, 0], 1 - w[:, 0]), kind
        assert np.allclose(hi[:, 0], w[:, 0]), kind


def test_the_causal_column_never_reads_the_bin_ahead():
    """Every interpolated causal row must be a function of the knot behind it
    alone; a bridge would carry the next knot's observations backward in time.
    """
    from latent_gp.latents import _extrapolate
    Ez_c = np.array([[[1.0], [5.0], [9.0]]])          # one seed, three bins
    m_i = np.zeros(5, dtype=int)
    u = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    held = _extrapolate(Ez_c, m_i, u, decay=None)[:, 0]
    assert np.allclose(held[:4], 1.0) and np.isclose(held[4], 5.0)

    F = float(gpfa.component('ou', dt=32.0, tau=10.0)[0][0, 0])
    decayed = _extrapolate(Ez_c, m_i, u, decay=np.array([F]))[:, 0]
    assert np.allclose(decayed[:4], F ** u[:4])
    assert np.all(np.diff(decayed[:4]) < 0)           # toward the prior mean


def test_position_decay_declines_a_prior_the_closed_form_cannot_express():
    one = [dict(kind='ou', tau=20.0)]
    assert gpfa.position_decay(one, 16.0, 3).shape == (3,)
    assert gpfa.position_decay([dict(kind='matern32', tau=320.0)], 16.0, 2) is None
    assert gpfa.position_decay([dict(kind='const'), dict(kind='ou', tau=20.0)],
                               16.0, 2) is None


def test_the_cache_epoch_separates_frames_the_old_code_wrote():
    from latent_gp.latents import LatentConfig
    import latent_gp.latents as mod
    base = dict(cells_path='x.parquet', n_dims=6, n_fast=1, fast_tau=20.0,
                fast_kind='ou', slow_kind='const', bin_factor=8)
    before = LatentConfig(**base).tag
    old = mod.CACHE_EPOCH
    try:
        mod.CACHE_EPOCH = old - 1
        assert LatentConfig(**base).tag != before
    finally:
        mod.CACHE_EPOCH = old
