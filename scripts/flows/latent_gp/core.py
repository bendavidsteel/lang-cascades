"""Latent-GP factor model: shared machinery.

    f_mjt = W_j . z_m(t) + b_j          W, b global; z per-seed
    z_mk(.) ~ GP, represented as a linear-Gaussian state-space model

Posts enter only through per-cell sufficient statistics, so a cell's variable
number of posts collapses to a fixed-size emission (see `factorize`).

The prior is built from additive components (`component`); non-stationary ones
have no stationary covariance, so their P0 is set from the scale the latent can
plausibly reach rather than from the kernel.

Accuracy caveat: sequential smoothing degrades as the per-step process noise Q
approaches zero. Matern and IWP-2 components with a timescale far beyond the
data span fall in that regime and must not be swept -- see test_smoother.py,
which checks every prior against a dense exact posterior.
"""

import numpy as np
import jax
import jax.numpy as jnp

from . import _jax  # noqa: F401  -- enables x64 before any array is created

JITTER = 1e-12
SMOOTH_JITTER = 1e-10


# --------------------------------------------------------------- SSM priors

def component(kind, dt, tau=None, var=1.0, p0=None):
    """One additive prior component for a single latent dimension.

    Returns (F, Q, P0) blocks; slot 0 of each block is the position that the
    observation reads. `tau` is the timescale over which the level moves by
    about sqrt(var), which makes the kinds comparable on one sweep.
    """
    if kind == 'const':
        return np.ones((1, 1)), np.zeros((1, 1)), np.array([[var]])

    if kind == 'wiener':                         # random walk on the level
        q = var / tau
        return (np.ones((1, 1)), np.array([[q * dt]]),
                np.array([[var if p0 is None else p0]]))

    if kind == 'iwp2':                           # integrated Wiener: (level, slope)
        q = 3.0 * var / tau ** 3
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        P0 = np.diag([var if p0 is None else p0, var / tau ** 2])
        return F, Q, P0

    if kind == 'ou':                             # mean-reverting, but rough
        # tau is scaled so the short-lag increment variance matches wiener's at
        # the same tau, which makes the pair a test of confinement alone
        F = np.array([[np.exp(-dt / (2.0 * tau))]])
        P0 = np.array([[var]])
        return F, Q_stationary(F, P0), P0

    if kind == 'matern32':                       # stationary, mean-reverting
        lam = np.sqrt(3.0) / tau
        e = np.exp(-lam * dt)
        F = e * np.array([[1 + lam * dt, dt], [-lam ** 2 * dt, 1 - lam * dt]])
        P0 = np.array([[var, 0.0], [0.0, lam ** 2 * var]])
        return F, Q_stationary(F, P0), P0

    raise ValueError(kind)


def Q_stationary(F, P0):
    return P0 - F @ P0 @ F.T


def position_decay(comps, dt, K):
    """Per-dimension correlation of the position slot across one bin.

    Interpolating between bin centres is a prior bridge, and the bridge is
    linear only where the position does not revert. Returns None when a
    dimension carries anything but a single one-state component, which is the
    case the closed form in latents._interpolate covers.
    """
    per_dim = comps if (comps and isinstance(comps[0], (list, tuple))) else [comps] * K
    if len(per_dim) != K:
        raise ValueError(f'got {len(per_dim)} component lists for K={K}')
    out = []
    for cs in per_dim:
        if len(cs) != 1:
            return None
        F, _, _ = component(dt=dt, **cs[0])
        if F.shape[0] != 1:
            return None
        out.append(float(F[0, 0]))
    return np.asarray(out)


def build_ssm(dt, comps, K):
    """Returns F, Q, P0, S.

    comps is either a list of component dicts shared by every latent dimension,
    or a list of K such lists to give dimensions different dynamics. Sharing one
    prior across dimensions leaves the factor model free to rotate; giving them
    different dynamics breaks that symmetry, which identifies the latent basis
    but means the axes must not be rotated afterwards.
    """
    per_dim = comps if (comps and isinstance(comps[0], (list, tuple))) else [comps] * K
    if len(per_dim) != K:
        raise ValueError(f'got {len(per_dim)} component lists for K={K}')

    blocks = [[component(dt=dt, **c) for c in cs] for cs in per_dim]
    sizes = [sum(bb[0].shape[0] for bb in bl) for bl in blocks]
    offs = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    sdim = int(offs[-1])

    F = np.zeros((sdim, sdim)); Q = np.zeros((sdim, sdim)); P0 = np.zeros((sdim, sdim))
    S = np.zeros((K, sdim))
    for k in range(K):
        off = int(offs[k])
        for Fb, Qb, Pb in blocks[k]:
            sz = Fb.shape[0]
            sl = slice(off, off + sz)
            F[sl, sl] = Fb; Q[sl, sl] = Qb; P0[sl, sl] = Pb
            S[k, off] = 1.0                       # observed latent sums components
            off += sz
    Q += JITTER * np.eye(sdim)
    return F, Q, P0, S


def is_heterogeneous(comps):
    """True when dimensions were given different dynamics, so no rotation."""
    return bool(comps) and isinstance(comps[0], (list, tuple))


# ------------------------------------------------------------------ E-step

def make_smoother(F, Q, P0, S, filtered=False):
    """Information-form Kalman filter + RTS smoother, vmapped over seeds.

    Observations enter as per-bin information (B_t, c_t) = (S' G_t S, S' g_t),
    so a bin's variable number of posts needs no fixed-size emission and empty
    bins are automatically a no-op. The update is written as (I + P B)^-1 P,
    which never inverts a precision that may be singular or zero.

    Returns the mean and covariance of the observed latent S x. With
    filtered=True the backward pass is skipped, so the state at t depends only
    on observations up to t -- what an out-of-time evaluation needs on the
    input side, at the cost of a noisier estimate.
    """
    F = jnp.asarray(F); Q = jnp.asarray(Q); P0 = jnp.asarray(P0); S = jnp.asarray(S)
    sdim = F.shape[0]
    I = jnp.eye(sdim)

    def forward(carry, obs):
        m, P = carry                      # prediction entering step t
        B, c = obs
        Mx = I + P @ B
        Pf = jnp.linalg.solve(Mx, P)
        mf = jnp.linalg.solve(Mx, m + P @ c)
        Pf = 0.5 * (Pf + Pf.T)
        return (F @ mf, F @ Pf @ F.T + Q), (mf, Pf, m, P)

    def backward(carry, x):
        ms, Ps = carry
        mf, Pf, mp, Pp = x                # mp, Pp are the prediction into t+1
        Ppj = Pp + SMOOTH_JITTER * jnp.eye(sdim)
        C = jnp.linalg.solve(Ppj, F @ Pf).T
        ms_t = mf + C @ (ms - mp)
        Ps_t = Pf + C @ (Ps - Pp) @ C.T
        return (ms_t, 0.5 * (Ps_t + Ps_t.T)), (ms_t, Ps_t)

    def project(m, P):
        return (jnp.einsum('kj,tj->tk', S, m),
                jnp.einsum('kj,tjl,il->tki', S, P, S))

    def one(B, c):
        (_, _), (mf, Pf, mp, Pp) = jax.lax.scan(forward, (jnp.zeros(sdim), P0), (B, c))
        if filtered:
            return project(mf, Pf)
        init = (mf[-1], Pf[-1])
        xs = (mf[:-1], Pf[:-1], mp[1:], Pp[1:])
        _, (ms, Ps) = jax.lax.scan(backward, init, xs, reverse=True)
        ms = jnp.concatenate([ms, mf[-1][None]], 0)
        Ps = jnp.concatenate([Ps, Pf[-1][None]], 0)
        return project(ms, Ps)

    return jax.jit(jax.vmap(one))


def to_information(G, g, S):
    """Latent-space (G_t, g_t) -> state-space information (B_t, c_t)."""
    B = jnp.einsum('jk,mtkl,il->mtji', S.T, G, S.T)
    c = jnp.einsum('jk,mtk->mtj', S.T, g)
    return B, c


def assemble(d, W, prec, resid, K):
    """Per-cell precision and pseudo-target -> per-bin (G_t, g_t).

    prec is the cell's precision on f, resid its precision-weighted target
    offset; the Gaussian and ordinal likelihoods differ only in these two.

    Scattered one entry of the K x K block at a time. Scattering the whole
    block at once needs a (cells, K, K) intermediate, which on the finest grid
    is several times the size of the result and does not fit on the card.
    """
    Wj = W[d['j']]
    MT = d['M'] * d['T']
    g = jnp.zeros((MT, K)).at[d['flat']].add(resid[:, None] * Wj)
    ent = {}
    for a in range(K):
        for b in range(a, K):
            ent[a, b] = jnp.zeros(MT).at[d['flat']].add(prec * Wj[:, a] * Wj[:, b])
    G = jnp.stack([jnp.stack([ent[min(a, b), max(a, b)] for b in range(K)], -1)
                   for a in range(K)], -2)
    return G.reshape(d['M'], d['T'], K, K), g.reshape(d['M'], d['T'], K)


# ------------------------------------------------------------------ M-step

def m_step(d, Ez, Ezz, prec, target, K, ridge=1e-4, w_ridge=None):
    """Weighted least squares for (W_j, b_j) given the posterior over z.

    `prec` weights each cell and `target` is its pseudo-observation on f; under
    a Gaussian likelihood these are n/sigma^2 and the cell mean.

    `w_ridge` is a Gaussian prior precision on the loadings, shrinking each
    target by A_j / (A_j + w_ridge): A_j accumulates that target's site
    precision, so a target the data barely constrains shrinks and one that
    carries a dimension does not. It has to be an absolute scale -- a ridge
    proportional to A_j divides every loading by the same constant and changes
    no comparison. `ridge` stays a singularity guard, and applies alone to the
    intercept: shrinking b would pull a target's mean stance toward neutral,
    which is a claim about the data rather than regularisation.
    """
    M, T = d['M'], d['T']
    Eu = jnp.concatenate([Ez, jnp.ones((M, T, 1))], -1).reshape(M * T, K + 1)
    second = (Ezz + Ez[..., None] * Ez[..., None, :]).reshape(M * T, K, K)
    Euu = jnp.zeros((M * T, K + 1, K + 1))
    Euu = Euu.at[:, :K, :K].set(second)
    Euu = Euu.at[:, :K, K].set(Eu[:, :K])
    Euu = Euu.at[:, K, :K].set(Eu[:, :K])
    Euu = Euu.at[:, K, K].set(1.0)

    # entry at a time, for the reason given in assemble
    ent = {}
    for a in range(K + 1):
        for b in range(a, K + 1):
            ent[a, b] = jnp.zeros(d['J']).at[d['j']].add(prec * Euu[d['flat'], a, b])
    A = jnp.stack([jnp.stack([ent[min(a, b), max(a, b)] for b in range(K + 1)], -1)
                   for a in range(K + 1)], -2)
    c = jnp.zeros((d['J'], K + 1)).at[d['j']].add((prec * target)[:, None] * Eu[d['flat']])
    pen = jnp.concatenate([jnp.full(K, ridge if w_ridge is None else w_ridge),
                           jnp.array([ridge])])
    Wb = jnp.linalg.solve(A + jnp.diag(pen), c[..., None])[..., 0]
    return Wb[:, :K], Wb[:, K]


def identify(W, Ez, K):
    """Whiten the latent, then order axes by loading energy (as PCA does).

    W R with R^-1 z is the same model, so the basis is fixed post-hoc; this
    changes nothing about fit or predictions.
    """
    Z = Ez.reshape(-1, K)
    Cz = (Z.T @ Z) / Z.shape[0] + 1e-8 * jnp.eye(K)
    L = jnp.linalg.cholesky(Cz)
    U, Sv, Vt = jnp.linalg.svd(W @ L, full_matrices=False)
    R = Vt @ jnp.linalg.inv(L)
    return U * Sv, jnp.einsum('ij,mtj->mti', R, Ez)


# ------------------------------------------------------------------ metrics

def roughness(x):
    if len(x) < 20:
        return np.nan
    d2 = np.diff(x, n=2, axis=0)
    return np.mean((d2 ** 2).sum(1)) / np.mean(((x - x.mean(0)) ** 2).sum(1))


def diff_acf1(x):
    if len(x) < 20:
        return np.nan
    dd = np.diff(x, axis=0); dd = dd - dd.mean(0)
    return (dd[:-1] * dd[1:]).sum() / (dd ** 2).sum()
