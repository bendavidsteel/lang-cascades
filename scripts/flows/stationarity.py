"""Stationarity tests for the latent-GP factor trajectories.

The tests answer one question for the paper: are the peaks of the density
landscape attractors of a stationary system, or is the system drifting?

What the latent-GP model changes, relative to testing PPCA coordinates:

  The prior contains the answer.  `fast_kind='wiener'` is a unit root by
  construction and `slow_kind='const'` freezes a dimension outright, so a unit
  root test on the posterior mean partly re-measures the prior. Two things
  follow. The fast and slow blocks are reported separately -- the slow block is
  a positive control, not evidence -- and the primary evidence is the
  model-free variogram over raw cell statistics, which assumes no prior at all.
  Ranking the priors themselves against held-out predictive likelihood is
  `latent_gp.sweep` (--taus / --ou-tau), not this script.

  The state is smoothed.  `coord_*` at t depends on observations after t, which
  is two-sided dependence a unit root test reads as persistence, so the tests
  read the filtered `causal_*` state. The control is the exception and has to
  be: a `slow_kind='const'` dimension is constant only in the smoothed state,
  because the filtered state is a running estimate of that constant and
  converges over the whole record rather than a burn-in. The fast block is
  unaffected -- its filtered-to-smoothed gap is flat in time rather than
  decaying -- and is reported on both states, which on this data does not
  change the verdict.

  The grid is interpolated.  `interp_days` resamples the fit's bin grid by
  linear interpolation, so most rows carry no new information and the
  autocorrelation between them is an artefact of the resampler. The latents are
  refitted here on the native bin grid.

  The panel is unbalanced.  Rows span each seed's observed bins only, so seeds
  enter and leave. Over 2022--2026 that alone moves the ensemble mean. Window
  and spread statistics run on a balanced sub-panel; the unbalanced figure is
  reported alongside to show how much of the drift was composition.

  Trajectories are not independent.  A factor model exists precisely because
  they share common factors, so Fisher's method over per-series p-values (the
  Maddala--Wu panel test) is anti-conservative and rejects on nothing at this
  N. The panel test here is cross-sectionally augmented (Pesaran's CADF/CIPS),
  and its null is calibrated by a block bootstrap that resamples the same time
  blocks for every seed, preserving the common factor.

Tests, and what each one can and cannot say:

1. Variogram over raw cell statistics -- model-free drift test with a
   permutation null. Saturation with lag means bounded, mean-reverting drift;
   sustained growth is a random walk. See `latent_gp.variogram`.
2. Mean squared displacement, raw and cross-sectionally demeaned. Characterises
   the dynamics; MSD shape alone does not decide stationarity, since a
   stationary Ornstein--Uhlenbeck process is also linear in tau at short lag.
3. CADF / CIPS panel unit root test, bootstrap-calibrated.
   H0: every series has a unit root.
4. KPSS, bootstrap-calibrated. H0: level (or trend) stationarity.
5. Window mean and variance drift, balanced against unbalanced.
6. Ensemble spread -- is the cloud dispersing, or translating rigidly?
7. Zivot--Andrews break test on the common factor. ADF and KPSS rejecting
   together indicates trend-stationarity *or* a structural break, and over this
   period a break is the more plausible reading; this separates them.

References:
    Dickey & Fuller (1979), JASA 74, 427-431.
    Kwiatkowski, Phillips, Schmidt & Shin (1992), J. Econometrics 54, 159-178.
    Pesaran (2007), "A simple panel unit root test in the presence of cross-
        section dependence", J. Applied Econometrics 22, 265-312.
    Zivot & Andrews (1992), J. Business & Economic Statistics 10, 251-270.
    Einstein (1905), Annalen der Physik 17, 549-560.
    Hyndman & Athanasopoulos (2021), Forecasting: Principles and Practice.
"""

import dataclasses
import os

import hydra
import numpy as np
import polars as pl
from scipy import optimize, stats

import splits
from latent_gp import drop_prior_dominated

# Latent units: `latent_gp.latents._standardise` scales each dimension by its
# training-region sd, so every displacement below is in sd, not in PCs.
UNIT = 'sd'

# KPSS asymptotic critical values, Kwiatkowski et al. (1992) table 1. Only for
# reference next to the bootstrap null, which is what the verdict uses.
KPSS_CRIT = {'c': {0.10: 0.347, 0.05: 0.463, 0.01: 0.739},
             'ct': {0.10: 0.119, 0.05: 0.146, 0.01: 0.216}}


# --------------------------------------------------------------------------
# ordinary least squares pieces, vectorised over panel units where it matters
# --------------------------------------------------------------------------

def _ols_t(X, y, col):
    """t statistic on coefficient `col` of an OLS fit, or nan if degenerate."""
    n, k = X.shape
    if n <= k + 1:
        return np.nan
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - k
    xtx_inv = np.linalg.pinv(X.T @ X)
    var = float(resid @ resid) / dof * xtx_inv[col, col]
    if not np.isfinite(var) or var <= 0:
        return np.nan
    return float(beta[col] / np.sqrt(var))


def _lagmat(v, p):
    """Columns v[t-1] .. v[t-p], aligned so row 0 is t = p."""
    return np.column_stack([v[p - k:len(v) - k] for k in range(1, p + 1)]) \
        if p else np.empty((len(v) - p, 0))


def adf_t(y, p=1, regression='c', augment=None):
    """t statistic on the level coefficient of a (augmented) Dickey-Fuller fit.

    `augment` is the cross-section mean series; supplying it makes this
    Pesaran's CADF, whose t statistic is free of the common factor. Without it
    this is the ordinary per-series ADF.

    `p` is fixed rather than chosen per series: an order selected on the real
    data and re-selected on each bootstrap surrogate would make the statistic
    and its null distribution incomparable.
    """
    y = np.asarray(y, dtype=float)
    T = len(y)
    if T < 4 * (p + 2):
        return np.nan
    dy = np.diff(y)
    rows = T - 1 - p
    if rows <= 0:
        return np.nan
    cols = [np.ones(rows), y[p:T - 1]]                  # const, y_{t-1}
    if regression == 'ct':
        cols.append(np.arange(rows, dtype=float))
    if p:
        cols.append(_lagmat(dy, p))
    if augment is not None:
        a = np.asarray(augment, dtype=float)
        da = np.diff(a)
        cols += [a[p:T - 1], da[p:]]                    # abar_{t-1}, d abar_t
        if p:
            cols.append(_lagmat(da, p))
    X = np.column_stack([c if c.ndim > 1 else c[:, None] for c in cols])
    return _ols_t(X, dy[p:], col=1)


def cips(Z, p=1, regression='c'):
    """Pesaran's CIPS: the mean CADF t over units, and the t's themselves.

    Z is (T, M): one column per trajectory, one dimension at a time.
    """
    abar = Z.mean(axis=1)
    t = np.array([adf_t(Z[:, m], p, regression, augment=abar)
                  for m in range(Z.shape[1])])
    ok = np.isfinite(t)
    return (float(np.mean(t[ok])) if ok.any() else np.nan), t


def kpss_stat(Z, regression='c', nlags=None):
    """KPSS statistic per column of Z (T, M), vectorised.

    Implemented here rather than called per series from statsmodels so that a
    bootstrap null costs one pass over an array instead of M x B calls.
    """
    Z = np.asarray(Z, dtype=float)
    T = Z.shape[0]
    X = np.ones((T, 1)) if regression == 'c' else \
        np.column_stack([np.ones(T), np.arange(T, dtype=float)])
    beta, *_ = np.linalg.lstsq(X, Z, rcond=None)
    e = Z - X @ beta
    S = np.cumsum(e, axis=0)
    if nlags is None:
        nlags = int(np.ceil(12 * (T / 100.0) ** 0.25))
    nlags = int(min(nlags, T - 1))
    # Newey-West long-run variance with Bartlett weights
    s2 = np.sum(e ** 2, axis=0) / T
    for k in range(1, nlags + 1):
        w = 1.0 - k / (nlags + 1.0)
        s2 = s2 + 2.0 * w * np.sum(e[k:] * e[:-k], axis=0) / T
    s2 = np.maximum(s2, 1e-12)
    return np.sum(S ** 2, axis=0) / (T ** 2 * s2)


# --------------------------------------------------------------------------
# bootstrap nulls
# --------------------------------------------------------------------------

def block_resample(X, block, rng):
    """Resample X along axis 0 in contiguous blocks of length `block`.

    The same block sequence is applied to every unit, so whatever the units
    share at a given time -- the common factor the whole model is built on --
    survives resampling. Drawing an independent sequence per unit would destroy
    exactly the dependence that makes Fisher's method wrong here.
    """
    T = X.shape[0]
    block = max(1, min(int(block), T))
    starts = rng.integers(0, T - block + 1, size=int(np.ceil(T / block)))
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:T]
    return X[idx]


def unit_root_surrogate(Z, block, rng):
    """A driftless random walk with Z's increment distribution and dependence.

    Increments are centred per unit, so the null is a random walk *without*
    drift -- the alternative an ADF fit with a constant is testing against.
    """
    d = np.diff(Z, axis=0)
    d = d - d.mean(axis=0, keepdims=True)
    return np.cumsum(np.vstack([np.zeros_like(Z[:1]), block_resample(d, block, rng)]),
                     axis=0)


def stationary_surrogate(Z, block, rng):
    """Level-stationary series keeping Z's short-run autocorrelation.

    Resampling blocks of the levels preserves dependence inside a block and
    destroys any trend spanning them, which is what H0 of KPSS asserts.
    """
    return block_resample(Z - Z.mean(axis=0, keepdims=True), block, rng)


def bootstrap_p(observed, surrogates, tail):
    """Bootstrap p value, (1 + #as extreme) / (1 + B) so it is never zero."""
    s = np.asarray([v for v in surrogates if np.isfinite(v)], dtype=float)
    if not len(s) or not np.isfinite(observed):
        return np.nan
    hits = np.sum(s <= observed) if tail == 'lower' else np.sum(s >= observed)
    return float((1.0 + hits) / (1.0 + len(s)))


def default_block(T):
    """Block length ~ T^(1/3), the usual rate for a moving-block bootstrap."""
    return max(2, int(round(T ** (1 / 3))))


# A frozen dimension's smoothed state is constant only up to the smoother's
# round-off, which grows with the problem: ~1e-7 of the cross-sectional spread
# on the full fit against ~1e-15 on a tenth of it. So the floor is relative.
# Real motion sits at ~0.5 of the cross-sectional spread, seven orders above.
FROZEN_REL_TOL = 1e-4


def _degenerate(Z, log, what):
    """A dimension the prior froze has no variation to test.

    The slow block is `slow_kind='const'` and cannot move, so running the
    machinery over it yields nan and a warning rather than a result. Saying so
    is the point: it is the control that shows a live test can tell the
    difference.

    Measured against the spread between trajectories, not against an absolute
    floor: an absolute one is a threshold on round-off, and which side of it
    the fit lands on is a fact about the fit's size, not about the data.
    """
    scale = float(Z.mean(axis=0).std())          # spread of the per-seed levels
    floor = max(1e-12, FROZEN_REL_TOL * scale)
    live = float(np.mean(Z.std(axis=0) > floor))
    if live >= 0.5:
        return None
    log(f"  {what}: {1 - live:.0%} of series are constant "
        f"-> frozen, no variation to test")
    return {'degenerate': True, 'live_share': live, 'p_value': np.nan}


def combine_dims(by_dim, reject_key, stat_key, stat_agg):
    """One verdict per block from its live dimensions, without picking one.

    Reporting the dimension with the smallest p would be a selection: the fast
    block is two dimensions wide, and the minimum of two p values rejects at
    roughly twice the nominal rate. So the block rejects only when every live
    dimension does, and the p reported is the weakest of them. Dimensions the
    prior froze are not evidence either way and are left out of the count.
    """
    live = [r for r in by_dim if not r.get('degenerate')]
    if not live:
        return {'degenerate': True, 'n_live': 0, 'n_reject': 0,
                'p_value': np.nan, stat_key: np.nan, reject_key: False,
                'by_dim': by_dim}
    n_reject = sum(bool(r[reject_key]) for r in live)
    return {'degenerate': False, 'n_live': len(live), 'n_reject': n_reject,
            'p_value': max(r['p_value'] for r in live),
            stat_key: stat_agg([r[stat_key] for r in live]),
            reject_key: n_reject == len(live), 'by_dim': by_dim}


def panel_unit_root(Z, p=1, regression='c', n_boot=199, seed=0, log=print):
    """CIPS against a bootstrap null of a driftless random-walk panel.

    Z is (T, M). A CIPS below the null's lower tail rejects the unit root,
    i.e. is evidence *for* stationarity.
    """
    dead = _degenerate(Z, log, f'CIPS({regression})')
    if dead is not None:
        return {**dead, 'cips': np.nan, 'per_unit_t': np.full(Z.shape[1], np.nan),
                'null': np.array([]), 'regression': regression, 'p_lags': p,
                'block': 0, 'rejects_unit_root': False}
    T = Z.shape[0]
    block = default_block(T)
    obs, per_unit = cips(Z, p, regression)
    rng = np.random.default_rng(seed)
    null = [cips(unit_root_surrogate(Z, block, rng), p, regression)[0]
            for _ in range(n_boot)]
    pval = bootstrap_p(obs, null, tail='lower')
    log(f"  CIPS({regression}, p={p}) = {obs:+.3f}   "
        f"null {np.nanmean(null):+.3f} +/- {np.nanstd(null):.3f} "
        f"(block {block}, B={n_boot})   p = {pval:.3f}")
    return {'cips': obs, 'per_unit_t': per_unit, 'null': np.array(null),
            'p_value': pval, 'block': block, 'regression': regression, 'p_lags': p,
            'degenerate': False,
            'rejects_unit_root': bool(np.isfinite(pval) and pval < 0.05)}


def panel_kpss(Z, regression='c', n_boot=199, seed=0, log=print):
    """Share of series rejecting stationarity, against a stationary null.

    The share is compared to a bootstrap null rather than to the asymptotic
    critical value: at this T the asymptotics are optimistic, and the series
    are cross-sectionally dependent, so the share itself has a much wider null
    distribution than a binomial would suggest.
    """
    dead = _degenerate(Z, log, f'KPSS({regression})')
    if dead is not None:
        return {**dead, 'reject_share': np.nan, 'null': np.array([]),
                'crit_5pct': KPSS_CRIT[regression][0.05], 'block': 0,
                'regression': regression, 'rejects_stationarity': False}
    T = Z.shape[0]
    block = default_block(T)
    crit = KPSS_CRIT[regression][0.05]
    obs = float(np.mean(kpss_stat(Z, regression) > crit))
    rng = np.random.default_rng(seed)
    null = [float(np.mean(kpss_stat(stationary_surrogate(Z, block, rng),
                                    regression) > crit))
            for _ in range(n_boot)]
    pval = bootstrap_p(obs, null, tail='upper')
    log(f"  KPSS({regression}) reject share = {obs:.3f}   "
        f"null {np.nanmean(null):.3f} +/- {np.nanstd(null):.3f} "
        f"(block {block}, B={n_boot})   p = {pval:.3f}")
    return {'reject_share': obs, 'null': np.array(null), 'p_value': pval,
            'block': block, 'regression': regression, 'crit_5pct': crit,
            'degenerate': False,
            'rejects_stationarity': bool(np.isfinite(pval) and pval < 0.05)}


# --------------------------------------------------------------------------
# building a balanced panel out of the latent frame
# --------------------------------------------------------------------------

@dataclasses.dataclass
class Panel:
    """A balanced (T, M, K) block of latent state, with its posterior sd."""
    Z: np.ndarray
    SD: np.ndarray
    times: np.ndarray          # datetime64, length T
    seeds: list
    dt_days: float

    @property
    def weights(self):
        """Precision weights. A state the fit could not pin down is nearly all
        prior, and averaging it in unweighted drags the ensemble mean toward
        the prior mean rather than toward anything measured."""
        return 1.0 / np.maximum(self.SD, 1e-6) ** 2

    def block(self, dims):
        return dataclasses.replace(self, Z=self.Z[:, :, dims],
                                   SD=self.SD[:, :, dims])

    def wmean(self):
        """Precision-weighted cross-sectional mean, (T, K)."""
        w = self.weights
        return (self.Z * w).sum(axis=1) / w.sum(axis=1)

    def demeaned(self):
        """State with the weighted cross-sectional mean removed at each t."""
        return self.Z - self.wmean()[:, None, :]


def balanced_panel(df, state_col, sd_col, n_dims, dt_days,
                   min_bins=30, min_seeds=10, log=print):
    """The largest (window x seed) rectangle of fully observed latent state.

    Candidate windows come from the quantiles of the observed seed spans; the
    rectangle maximising seeds x bins wins, subject to floors on both. The
    floors matter: maximising the product alone will trade most of the span for
    a few more seeds, and the panel tests get their power from the span.
    Reported so the paper can say which window the statistics describe.
    """
    span = df.group_by('filter_value').agg(
        pl.col('createtime').min().alias('lo'), pl.col('createtime').max().alias('hi'))
    lo = span['lo'].to_numpy().astype('datetime64[s]').astype(np.int64)
    hi = span['hi'].to_numpy().astype('datetime64[s]').astype(np.int64)
    qs = np.linspace(0, 1, 21)
    best = None
    for a in np.unique(np.quantile(lo, qs)):
        for b in np.unique(np.quantile(hi, qs)):
            if b <= a:
                continue
            n_seed = int(np.sum((lo <= a) & (hi >= b)))
            n_bin = (b - a) / 86400.0 / dt_days
            if n_seed < min_seeds or n_bin < min_bins:
                continue
            score = n_seed * n_bin
            if best is None or score > best[0]:
                best = (score, a, b, n_seed)
    if best is None:
        raise ValueError(f'no seed set covers >= {min_bins} bins with '
                         f'>= {min_seeds} seeds')
    _, a, b, n_seed = best
    t_lo = np.datetime64(int(a), 's')
    t_hi = np.datetime64(int(b), 's')

    keep = span.filter((pl.col('lo') <= pl.lit(t_lo.astype('datetime64[us]')))
                       & (pl.col('hi') >= pl.lit(t_hi.astype('datetime64[us]'))))
    sub = df.filter(pl.col('filter_value').is_in(keep['filter_value'].to_list())
                    & (pl.col('createtime') >= pl.lit(t_lo.astype('datetime64[us]')))
                    & (pl.col('createtime') <= pl.lit(t_hi.astype('datetime64[us]')))) \
            .sort(['filter_value', 'createtime'])

    times = np.sort(sub['createtime'].unique().to_numpy())
    seeds = sub['filter_value'].unique().sort().to_list()
    # a seed missing a bin inside the window would break the rectangle
    counts = sub.group_by('filter_value').agg(pl.len().alias('n'))
    full = counts.filter(pl.col('n') == len(times))['filter_value'].to_list()
    if len(full) < len(seeds):
        log(f"  dropping {len(seeds) - len(full)} seeds with gaps inside the window")
        sub = sub.filter(pl.col('filter_value').is_in(list(full))).sort(
            ['filter_value', 'createtime'])
        seeds = sorted(full)
    if len(seeds) < min_seeds:
        raise ValueError(f'balanced window leaves {len(seeds)} seeds, '
                         f'below the {min_seeds} floor')

    T, M = len(times), len(seeds)
    Z = sub[state_col].to_numpy().reshape(M, T, n_dims).transpose(1, 0, 2)
    SD = sub[sd_col].to_numpy().reshape(M, T, n_dims).transpose(1, 0, 2)
    log(f"  balanced panel: {M} seeds x {T} bins "
        f"({str(times[0])[:10]} to {str(times[-1])[:10]}, {dt_days:g}-day grid)")
    return Panel(Z=Z, SD=SD, times=times, seeds=seeds, dt_days=dt_days)


# --------------------------------------------------------------------------
# displacement, window statistics, spread, breaks
# --------------------------------------------------------------------------

def msd(Z, dt_days, max_lag=None, n_points=20, label='', log=print):
    """Mean squared displacement against lag in days.

    MSD(tau) = <|z(t+tau) - z(t)|^2> over t and over trajectories. Fits

        power law            MSD ~ tau^alpha
        drift + diffusion    MSD = 2 d D tau + (v tau)^2

    with D and |v|^2 constrained non-negative, and checks whether the long-lag
    tail plateaus. Lags are in days: the fit grid is regular, so a lag in rows
    is a lag in time, which was not true of the per-post frame this replaced.

    Lags are log spaced. On a linear grid a process that saturates early puts
    almost every point on the plateau, and the power law then fits the one
    step up onto it rather than the scaling.

    MSD shape does not decide stationarity on its own -- a stationary
    Ornstein--Uhlenbeck process is linear in tau before it saturates. Read it
    with the panel tests.
    """
    T, M, K = Z.shape
    if max_lag is None:
        max_lag = T // 2
    lags = np.unique(np.geomspace(1, max_lag, n_points).round().astype(int))
    vals, errs = [], []
    for lag in lags:
        d = Z[lag:] - Z[:T - lag]
        sq = np.sum(d ** 2, axis=2).ravel()
        vals.append(sq.mean())
        errs.append(sq.std() / np.sqrt(len(sq)))
    lags_days = lags * dt_days
    vals = np.array(vals)
    errs = np.array(errs)

    alpha, log_const, r_alpha, _, _ = stats.linregress(np.log(lags_days),
                                                       np.log(vals + 1e-12))
    # nnls, not lstsq then clip: a clipped coefficient leaves the reported D
    # and |v| describing a different curve from the one that was fitted.
    # Neither term can bend downwards, so a saturating MSD drives this R2
    # negative -- that is the saturation showing, not a failure.
    design = np.column_stack([lags_days, lags_days ** 2])
    coeffs, _ = optimize.nnls(design, vals)
    D_fit = coeffs[0] / (2 * K)
    v_fit = np.sqrt(coeffs[1])
    pred = design @ coeffs
    ss_tot = np.sum((vals - vals.mean()) ** 2)
    r2 = 1 - np.sum((vals - pred) ** 2) / ss_tot if ss_tot > 0 else np.nan

    # Fractional rise across the tail, and no significance test on it: the MSD
    # estimates at neighbouring lags share the same trajectories, so the
    # residuals of a fit through them are strongly correlated and its p value
    # does not mean what it appears to. A plateau that keeps creeping up by a
    # few percent is finite-sample bias, not growth.
    n_tail = max(3, len(lags) // 3)
    tail_growth = float((vals[-1] - vals[-n_tail]) / (np.mean(vals[-n_tail:]) + 1e-12))
    # a stationary process reaches MSD -> 2 Var(z); how close the tail gets
    plateau = float(vals[-1] / (2 * np.sum(Z.reshape(-1, K).var(axis=0)) + 1e-12))
    saturates = bool(abs(tail_growth) < 0.1)

    log(f"\n  MSD{f' [{label}]' if label else ''}  ({K} dims, {M} trajectories)")
    log(f"    MSD({lags_days[0]:.0f}d) = {vals[0]:.4f} {UNIT}^2, "
        f"MSD({lags_days[-1]:.0f}d) = {vals[-1]:.4f} {UNIT}^2")
    log(f"    power law   MSD ~ tau^{alpha:.3f}   R2 = {r_alpha ** 2:.4f}")
    log(f"    drift+diff  D = {D_fit:.3e} {UNIT}^2/day, "
        f"|v| = {v_fit:.3e} {UNIT}/day   R2 = {r2:.4f}")
    log(f"    tail over last {n_tail} lags "
        f"({lags_days[-n_tail]:.0f}-{lags_days[-1]:.0f}d): {tail_growth:+.3f}, "
        f"MSD/2Var = {plateau:.2f} "
        f"-> {'saturates' if saturates else 'still growing'}")
    return {'lags_days': lags_days, 'msd': vals, 'stderr': errs, 'alpha': alpha,
            'alpha_r2': r_alpha ** 2, 'D': D_fit, 'v': v_fit, 'fit_r2': r2,
            'tail_growth': tail_growth, 'plateau_ratio': plateau,
            'saturates': saturates, 'label': label}


def window_drift(panel, n_windows=6, log=print):
    """Mean and variance change between the first and last time window.

    Cohen's d is in pooled-sd units. Because the latent is already standardised
    to training-region sd, d and the raw displacement differ only by the
    window spread.
    """
    T = panel.Z.shape[0]
    edges = np.linspace(0, T, n_windows + 1).round().astype(int)
    w = panel.weights
    means, varis = [], []
    for i in range(n_windows):
        s = slice(edges[i], edges[i + 1])
        zz, ww = panel.Z[s].reshape(-1, panel.Z.shape[2]), w[s].reshape(-1, panel.Z.shape[2])
        mu = (zz * ww).sum(0) / ww.sum(0)
        means.append(mu)
        varis.append((ww * (zz - mu) ** 2).sum(0) / ww.sum(0))
    means, varis = np.array(means), np.array(varis)
    drift = means[-1] - means[0]
    var_rel = (varis[-1] - varis[0]) / (varis[0] + 1e-12)
    d = drift / np.sqrt((varis[0] + varis[-1]) / 2 + 1e-12)

    log(f"\n  Window drift ({n_windows} windows over "
        f"{(panel.times[-1] - panel.times[0]).astype('timedelta64[D]').astype(int)} days)")
    cohen = "Cohen's d"
    log(f"    {'dim':>3} | {'mu_0':>8} | {'d_mu':>8} | {'var_0':>8} | "
        f"{'d_var/var_0':>11} | {cohen:>10}")
    for k in range(len(drift)):
        log(f"    {k:>3} | {means[0, k]:>+8.4f} | {drift[k]:>+8.4f} | "
            f"{varis[0, k]:>8.4f} | {var_rel[k]:>+10.1%} | {d[k]:>+10.3f}")
    log(f"    max |d| = {np.max(np.abs(d)):.3f}  "
        f"(0.2 small, 0.5 medium, 0.8 large), "
        f"max |d_var/var_0| = {np.max(np.abs(var_rel)):.1%}")
    return {'window_means': means, 'window_vars': varis, 'mean_drift': drift,
            'var_change_rel': var_rel, 'cohens_d': d,
            'max_abs_d': float(np.max(np.abs(d))),
            'max_abs_var_rel': float(np.max(np.abs(var_rel))),
            # signed, for the prose: the drift's direction is the whole point,
            # and an absolute value silently turns a contraction into a growth
            'max_d': float(d[np.argmax(np.abs(d))]),
            'max_var_rel': float(var_rel[np.argmax(np.abs(var_rel))]),
            'n_windows': n_windows}


def unbalanced_window_drift(df, state_col, dims, n_windows=6, log=print):
    """`window_drift` over every row, balanced or not -- the composition check.

    Any gap between this and the balanced figure is seeds entering and leaving
    the panel, not the ensemble moving.
    """
    t = df['createtime'].to_numpy().astype('datetime64[s]').astype(np.int64)
    edges = np.linspace(t.min(), t.max() + 1, n_windows + 1)
    win = np.clip(np.searchsorted(edges, t, side='right') - 1, 0, n_windows - 1)
    Z = df[state_col].to_numpy()[:, dims]
    first, last = Z[win == 0], Z[win == n_windows - 1]
    drift = last.mean(0) - first.mean(0)
    d = drift / np.sqrt((first.var(0) + last.var(0)) / 2 + 1e-12)
    n_seed_first = df.filter(pl.Series(win == 0))['filter_value'].n_unique()
    n_seed_last = df.filter(pl.Series(win == n_windows - 1))['filter_value'].n_unique()
    log(f"\n  Unbalanced window drift: max |d| = {np.max(np.abs(d)):.3f}   "
        f"seeds present {n_seed_first} -> {n_seed_last}")
    return {'cohens_d': d, 'max_abs_d': float(np.max(np.abs(d))),
            'n_seeds_first': n_seed_first, 'n_seeds_last': n_seed_last}


def ensemble_spread(panel, log=print):
    """Is the cloud dispersing, or translating with a fixed shape?

    Distance of each trajectory from the weighted cross-sectional centroid,
    averaged over trajectories, regressed on time. This is dispersion of the
    ensemble, not divergence of nearby pairs -- it is not a Lyapunov exponent.
    """
    r = np.sqrt(np.sum(panel.demeaned() ** 2, axis=2)).mean(axis=1)
    days = ((panel.times - panel.times[0]).astype('timedelta64[D]')).astype(float)
    slope, intercept, rv, pv, _ = stats.linregress(days, r)
    ratio = r[-1] / (r[0] + 1e-12)
    log(f"\n  Ensemble spread: {r[0]:.4f} -> {r[-1]:.4f} {UNIT} "
        f"({ratio:.2f}x), slope {slope:+.3e} {UNIT}/day, "
        f"R2 = {rv ** 2:.3f}, p = {pv:.2e}")
    dispersing = bool(slope > 0 and pv < 0.05 and ratio > 1.5)
    log(f"    -> {'dispersing' if dispersing else 'spread held roughly constant'}")
    return {'mean_dist': r, 'days': days, 'slope': slope, 'r_squared': rv ** 2,
            'p_value': pv, 'spread_ratio': float(ratio), 'dispersing': dispersing}


def common_break(panel, dims, log=print):
    """Zivot--Andrews on the common factor of each dimension.

    ADF and KPSS rejecting together means trend-stationary *or* a structural
    break, which the paper cannot leave ambiguous over a period containing
    several. Run on the cross-sectional mean because a break in the landscape
    is a break shared across trajectories, not one seed changing its mind.
    """
    from statsmodels.tsa.stattools import zivot_andrews
    mu = panel.wmean()
    out = []
    for i, k in enumerate(dims):
        try:
            stat, pval, crit, lag, bp = zivot_andrews(mu[:, i], regression='c')
        except (ValueError, np.linalg.LinAlgError) as exc:
            log(f"    dim {k}: Zivot-Andrews failed ({exc})")
            out.append({'dim': k, 'stat': np.nan, 'p_value': np.nan,
                        'break_date': None, 'rejects': False})
            continue
        date = str(panel.times[min(bp, len(panel.times) - 1)])[:10]
        log(f"    dim {k}: ZA = {stat:+.3f}, p = {pval:.4f}, "
            f"break at {date} -> {'break' if pval < 0.05 else 'no break'}")
        out.append({'dim': k, 'stat': float(stat), 'p_value': float(pval),
                    'break_date': date, 'rejects': bool(pval < 0.05)})
    return out


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_latents(cfg, spec, log=print):
    """Filtered latent state on the fit's native bin grid.

    Both states are returned. The tests read the filtered `causal_*`, because
    the smoothed state at t is a function of observations after t and a unit
    root test reads that as persistence -- the same reason
    `latent_gp.compare_obs` reports both. The smoothed state comes along
    because it is the only one on which the frozen slow block is a valid
    control: see `analyse`.

    `interp_days` is forced to zero: the exported grid is linear interpolation
    between bin centres, so seven of every eight rows carry no new information
    and their autocorrelation belongs to the resampler.

    Refitting under a changed `interp_days` keys a separate cache entry, so
    this does not disturb the landscape model's latents.
    """
    import latent_space
    from latent_gp import LatentConfig, build_latents, coord_cols

    lcfg, _, kw = latent_space.fit_args(
        cfg, dataclasses.replace(LatentConfig.from_cfg(cfg), interp_days=0.0), spec)
    df = build_latents(lcfg, spec, **dict(kw, log=log))
    smoothed, causal, sd = coord_cols(cfg.n_dims)
    df = latent_space.keep_platform(cfg, df).filter(pl.col('filter_value') != '') \
           .select(['createtime', 'filter_value', causal, smoothed, sd]) \
           .sort(['filter_value', 'createtime'])
    return df, causal, smoothed, sd, 2.0 * lcfg.bin_factor


def run_variogram(cfg, log=print):
    """The model-free test, over the same cells the latent fit reads."""
    from latent_gp import cells as gp_cells
    from latent_gp import variogram

    df, meta = gp_cells.load(cfg.latents.cells_path, cfg.latents.bin_factor,
                             min_target_volume=cfg.min_target_volume)
    by_stat = variogram.curves(df, meta, log=log)
    got = by_stat[variogram.MEAN_STANCE]
    for name, c in by_stat.items():
        variogram.show(name, c['real'], c['null'])
    v = variogram.verdict(got['real'], got['null'])
    log(f"  drift sd ~ {v['drift_sd']:.4f}, tail "
        f"{v['tail_lags_days'][0]:.0f}-{v['tail_lags_days'][1]:.0f}d carries "
        f"{v['tail_share']:.1%} of the rise -> "
        f"{'bounded (mean-reverting)' if v['saturates'] else 'still growing'}")
    return v


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def cfg_get(cfg, path, default):
    """Dotted lookup that tolerates cfg=None, as the tests pass."""
    for key in path.split('.'):
        if cfg is None:
            return default
        cfg = cfg.get(key, None) if hasattr(cfg, 'get') else getattr(cfg, key, None)
    return default if cfg is None else cfg


def _fmt(x, digits=3):
    return '--' if x is None or not np.isfinite(x) else f'{x:.{digits}f}'


def write_tex(results, cfg, out_dir=None):
    """A table of the tests and a macro file, so the prose cites no typed-in
    numbers.

    Written to the top of `out_dir`, beside the per-configuration directories
    `latent_space.latent_root` addresses rather than inside one: the paper
    pastes in one stationarity table, not one per fit.
    """
    out_dir = out_dir or cfg_get(cfg, 'out_dir', './out')
    os.makedirs(out_dir, exist_ok=True)
    fast, slow = results['blocks']['fast'], results['blocks']['slow']
    p = results['panel']
    rows = [
        ('Variogram (raw cells)', 'model-free',
         f"tail share {_fmt(results['variogram']['tail_share'])}",
         'bounded drift' if results['variogram']['saturates'] else 'unbounded'),
        ('MSD', 'fast',
         f"$\\alpha = {_fmt(results['msd_raw']['alpha'], 2)}$",
         'saturates' if results['msd_raw']['saturates'] else 'grows'),
        ('MSD, demeaned', 'fast',
         f"$\\alpha = {_fmt(results['msd_demeaned']['alpha'], 2)}$",
         'saturates' if results['msd_demeaned']['saturates'] else 'grows'),
        ('CIPS panel unit root', 'fast',
         f"${_fmt(p['fast']['cips'])}$ ($p \\leq {_fmt(p['fast']['p_value'])}$) "
         f"on {p['fast']['n_reject']}/{p['fast']['n_live']} dims",
         'reject unit root' if p['fast']['rejects_unit_root'] else 'unit root stands'),
        ('KPSS reject share', 'fast',
         f"{_fmt(p['fast_kpss']['reject_share'])} "
         f"($p \\leq {_fmt(p['fast_kpss']['p_value'])}$)",
         'reject stationarity' if p['fast_kpss']['rejects_stationarity'] else 'consistent'),
        ('Window mean drift', 'fast',
         f"$d = {_fmt(results['window']['max_d'])}$",
         f"vs {_fmt(results['window_unbalanced']['max_abs_d'])} unbalanced"),
        ('Window variance drift', 'fast',
         f"${100 * results['window']['max_var_rel']:+.1f}\\%$", ''),
        ('Ensemble spread', 'fast',
         f"{_fmt(results['spread']['spread_ratio'], 2)}$\\times$",
         'dispersing' if results['spread']['dispersing'] else 'constant'),
        ('Zivot--Andrews (common factor)', 'fast',
         '; '.join(f"dim {b['dim']}: {b['break_date']}"
                   for b in results['breaks'] if b['rejects']) or 'no break',
         ''),
        ('CIPS panel unit root', 'fast, smoothed state',
         f"${_fmt(p['fast_smoothed']['cips'])}$ "
         f"($p \\leq {_fmt(p['fast_smoothed']['p_value'])}$)"
         if p.get('fast_smoothed') else '--',
         'sensitivity to the state'),
    ]
    if 'slow' in p:
        rows.append((
            'CIPS panel unit root', 'slow (control)',
            'no variation' if p['slow'].get('degenerate')
            else f"${_fmt(p['slow']['cips'])}$ ($p = {_fmt(p['slow']['p_value'])}$)",
            f"frozen ({cfg_get(cfg, 'latents.slow_kind', 'const')}), smoothed state"))
    path = os.path.join(out_dir, 'stationarity.tex')
    with open(path, 'w') as f:
        f.write('\\begin{tabular}{llll}\n\\toprule\n')
        f.write('Test & Block & Statistic & Reading \\\\\n\\midrule\n')
        for name, block, stat, verdict in rows:
            f.write(f'{name} & {block} & {stat} & {verdict} \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')
        control = (f"against the frozen slow block (dims {slow[0]}--{slow[-1]}) "
                   'as a control' if slow else 'with no slow block configured')
        f.write('\\caption{Stationarity tests on the latent-GP trajectories, '
                f"fast block (dims 0--{len(fast) - 1}) {control}. "
                'Panel tests are cross-sectionally augmented and '
                'bootstrap-calibrated.}\n')
        f.write('\\label{tab:stationarity}\n\\end{table}\n')

    macros = os.path.join(out_dir, 'stationarity_macros.tex')
    with open(macros, 'w') as f:
        for name, val in [
            ('statWindows', results['window']['n_windows']),
            ('statMaxCohenD', _fmt(results['window']['max_d'])),
            ('statMaxVarPct', f"{100 * results['window']['max_var_rel']:+.0f}"),
            ('statUnbalCohenD', _fmt(results['window_unbalanced']['max_abs_d'])),
            ('statCIPS', _fmt(p['fast']['cips'])),
            ('statCIPSp', _fmt(p['fast']['p_value'])),
            ('statKPSSshare', _fmt(p['fast_kpss']['reject_share'], 2)),
            ('statSpreadRatio', _fmt(results['spread']['spread_ratio'], 2)),
            ('statMSDalpha', _fmt(results['msd_raw']['alpha'], 2)),
            ('statDriftSd', _fmt(results['variogram']['drift_sd'], 4)),
            ('statPanelSeeds', len(results['panel_shape'][1])),
            ('statPanelBins', results['panel_shape'][0]),
            ('statGridDays', f"{results['dt_days']:g}"),
        ]:
            f.write(f'\\newcommand{{\\{name}}}{{{val}}}\n')
    print(f'\nWrote {path} and {macros}')
    return path, macros


def summarise(results, log=print):
    """A structured verdict rather than a vote.

    The old script averaged `is_stationary` over five tests that answer
    different questions and are not independent -- and counted MSD twice, raw
    and demeaned. Each line below is a separate claim, and the reader can see
    which one carries the conclusion.
    """
    p = results['panel']
    v = results['variogram']
    log('\n' + '=' * 62)
    log('SUMMARY')
    log('=' * 62)
    log(f"  drift present (model-free variogram):  "
        f"{'yes' if v['span'] > 0 else 'no'} (drift sd ~ {v['drift_sd']:.4f})")
    log(f"  drift bounded / mean-reverting:        "
        f"{'yes' if v['saturates'] else 'no -- consistent with a random walk'}")
    log(f"  unit root rejected on the fast block:  "
        f"{'yes' if p['fast']['rejects_unit_root'] else 'no'} "
        f"({p['fast']['n_reject']}/{p['fast']['n_live']} dims, "
        f"weakest CIPS p = {_fmt(p['fast']['p_value'])})")
    log(f"  stationarity rejected (KPSS):          "
        f"{'yes' if p['fast_kpss']['rejects_stationarity'] else 'no'} "
        f"({p['fast_kpss']['n_reject']}/{p['fast_kpss']['n_live']} dims, "
        f"weakest p = {_fmt(p['fast_kpss']['p_value'])})")
    sm = p.get('fast_smoothed')
    if sm is not None:
        log(f"    same on the smoothed state:         "
            f"{'yes' if sm['rejects_unit_root'] else 'no'} "
            f"({sm['n_reject']}/{sm['n_live']} dims) -- the state choice "
            f"{'does not change' if sm['rejects_unit_root'] == p['fast']['rejects_unit_root'] else 'CHANGES'} the verdict")
    ctl = p.get('slow')
    log(f"  control, frozen slow block:            "
        + ('none -- every dimension is fast' if ctl is None
           else 'reports frozen, as it must' if ctl.get('degenerate')
           else 'REPORTS MOTION -- the test is measuring itself'))
    breaks = [b for b in results['breaks'] if b['rejects']]
    log(f"  structural break in the common factor: "
        f"{'yes -- ' + ', '.join(b['break_date'] for b in breaks) if breaks else 'no'}")
    log(f"  ensemble dispersing:                   "
        f"{'yes' if results['spread']['dispersing'] else 'no (translating)'}")
    log(f"  window mean drift, max |d|:            "
        f"{results['window']['max_abs_d']:.3f} balanced, "
        f"{results['window_unbalanced']['max_abs_d']:.3f} unbalanced "
        f"(the gap is composition, not motion)")

    if p['fast']['rejects_unit_root'] and p['fast_kpss']['rejects_stationarity']:
        log('\n  ADF and KPSS both reject. That is trend-stationarity or a '
            'structural break,')
        log('  not a random walk -- the Zivot-Andrews line above separates them.')
    elif not p['fast']['rejects_unit_root'] and p['fast_kpss']['rejects_stationarity']:
        log('\n  Unit root stands and stationarity is rejected: the fast block '
            'is a random walk.')
        log('  Note the prior on it is Wiener, which asserts exactly that -- '
            'read the variogram,')
        log('  which assumes no prior, before concluding.')
    elif p['fast']['rejects_unit_root'] and not p['fast_kpss']['rejects_stationarity']:
        log('\n  Unit root rejected, stationarity not: the fast block is '
            'stationary.')
    else:
        log('\n  Neither test rejects: inconclusive at this T.')

    log('\n  The prior is not neutral. Rank wiener against ou on held-out '
        'predictive')
    log('  likelihood with `python -m latent_gp.sweep --taus ... --ou-tau ...` '
        'before')
    log('  reporting any of this as a property of the data rather than of the '
        'fit.')


def analyse(df, state_col, sd_col, dt_days, fast, slow, variogram_result,
            smoothed_col=None, n_boot=199, n_windows=6, min_bins=30,
            min_seeds=10, log=print):
    """Every test, over an already-loaded latent frame.

    Split out from `main` so the whole chain -- including the summary and the
    table, whose key names are otherwise only exercised at the end of a fit
    that takes an hour -- runs on a synthetic frame in the test suite.

    `smoothed_col` is what makes the slow block a control. Under
    `slow_kind='const'` the *smoothed* slow state is exactly constant in time,
    so a test that finds it moving is measuring itself. The *filtered* slow
    state is not constant: it is a running estimate of a constant, and it
    converges over the whole record rather than a burn-in, so on the filtered
    state the control reports motion that is entirely the filter. The fast
    block is unaffected -- its filtered-to-smoothed gap is flat in time, not a
    decaying transient -- but it is reported on both states anyway, since the
    choice is exactly the kind a reviewer asks about.
    """
    log('\n--- Balanced panel ---')
    panel = balanced_panel(df, state_col, sd_col, len(fast) + len(slow), dt_days,
                           min_bins=min_bins, min_seeds=min_seeds, log=log)
    fast_panel = panel.block(fast)
    if smoothed_col is None:
        smooth_panel = panel
    else:
        smooth_panel = balanced_panel(df, smoothed_col, sd_col,
                                      len(fast) + len(slow), dt_days,
                                      min_bins=min_bins, min_seeds=min_seeds,
                                      log=lambda *a, **k: None)
    # A homogeneous prior leaves no slow block, and then there is no control to
    # run: the variogram carries the model-free case on its own.
    slow_panel = smooth_panel.block(slow) if slow else None
    if slow_panel is None:
        log('\n  No slow block -- every dimension is fast, so the frozen '
            'control is not available.')

    results = {'blocks': {'fast': fast, 'slow': slow},
               'dt_days': dt_days,
               'panel_shape': (panel.Z.shape[0], panel.seeds),
               'variogram': variogram_result}

    log('\n=== 2. Mean squared displacement ===')
    results['msd_raw'] = msd(fast_panel.Z, dt_days, label='fast, raw', log=log)
    results['msd_demeaned'] = msd(fast_panel.demeaned(), dt_days, log=log,
                                  label='fast, cross-sectionally demeaned')
    if slow_panel is not None:
        results['msd_slow'] = msd(slow_panel.Z, dt_days, label='slow, control',
                                  log=log)

    log('\n=== 3-4. Panel unit root and stationarity tests ===')
    panel_res = {}
    blocks = [('fast', fast_panel)]
    if slow_panel is not None:
        blocks.append(('slow', slow_panel))
    blocks.append(('fast_smoothed', smooth_panel.block(fast)))
    for name, blk in blocks:
        log(f"  [{name} block]")
        # one dimension at a time; CIPS is defined per series, not per vector
        roots = [panel_unit_root(blk.Z[:, :, k], n_boot=n_boot, log=log)
                 for k in range(blk.Z.shape[2])]
        kpsses = [panel_kpss(blk.Z[:, :, k], n_boot=n_boot, log=log)
                  for k in range(blk.Z.shape[2])]
        panel_res[name] = combine_dims(roots, 'rejects_unit_root', 'cips',
                                       lambda v: float(np.mean(v)))
        panel_res[f'{name}_kpss'] = combine_dims(kpsses, 'rejects_stationarity',
                                                 'reject_share', min)
        log(f"    block verdict: unit root rejected on "
            f"{panel_res[name]['n_reject']}/{panel_res[name]['n_live']} live dims, "
            f"stationarity rejected on "
            f"{panel_res[f'{name}_kpss']['n_reject']}/"
            f"{panel_res[f'{name}_kpss']['n_live']}")
    # trend-stationarity is the specific alternative the paper claims
    log('  [fast block, trend regression]')
    panel_res['fast_ct'] = [panel_unit_root(fast_panel.Z[:, :, k], regression='ct',
                                            n_boot=n_boot, log=log)
                            for k in range(fast_panel.Z.shape[2])]
    results['panel'] = panel_res

    log('\n=== 5. Window drift ===')
    results['window'] = window_drift(fast_panel, n_windows=n_windows, log=log)
    results['window_unbalanced'] = unbalanced_window_drift(
        df, state_col, fast, n_windows=n_windows, log=log)

    log('\n=== 6. Ensemble spread ===')
    results['spread'] = ensemble_spread(fast_panel, log=log)

    log('\n=== 7. Structural breaks in the common factor ===')
    results['breaks'] = common_break(fast_panel, fast, log=log)

    summarise(results, log=log)
    return results


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    if cfg.latents.method != 'gpfa':
        raise ValueError(
            'stationarity.py tests the latent-GP trajectories; '
            f'cfg.latents.method is {cfg.latents.method!r}')

    spec = splits.SplitSpec.from_cfg(cfg)
    n_fast = cfg.latents.n_fast
    fast, slow = (list(range(min(n_fast, cfg.n_dims))),
                  list(range(min(n_fast, cfg.n_dims), cfg.n_dims)))
    print(f"Fast block dims {fast} (kind={cfg.latents.get('fast_kind', 'wiener')}, "
          f"tau={cfg.latents.fast_tau}); slow block dims {slow} "
          f"(kind={cfg.latents.slow_kind}) -- control, not evidence.")

    print('\n=== 1. Variogram over raw cell statistics (model-free) ===')
    variogram_result = run_variogram(cfg, log=print)

    print('\n--- Loading latents (filtered state, native grid) ---')
    df, state_col, smoothed_col, sd_col, dt_days = load_latents(cfg, spec, log=print)
    df = drop_prior_dominated(df, sd_col, fast, cfg.get('max_posterior_sd', 0.8))

    results = analyse(df, state_col, sd_col, dt_days, fast, slow, variogram_result,
                      smoothed_col=smoothed_col,
                      n_boot=cfg.get('stationarity_boot', 199),
                      n_windows=cfg.get('stationarity_windows', 6))
    write_tex(results, cfg)
    return results



if __name__ == '__main__':
    main()
