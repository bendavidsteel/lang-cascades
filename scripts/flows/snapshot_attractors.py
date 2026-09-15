"""Fixed points of the flow field each snapshot panel draws.

`plot_nn_potential.py` draws the streamplots; this reports what they converge
on, so the prose cites located attractors rather than positions read off the
page. Same models, same panel times, same Boltzmann marginalisation over the
non-displayed dimensions, so a number here describes the figure as drawn.

Positions are given as percentiles of the figure's own coordinate cloud as well
as in latent units: the axis ticks are already that cloud's p10, mean and p90
(see `describe_dimensions.py`), and the units themselves are only whatever
`latent_gp.core.identify` whitened the fit to.

    python scripts/flows/snapshot_attractors.py +plots=[time_snapshots]
"""

import datetime
import json
import os

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import omegaconf
import polars as pl
import scipy.optimize
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import minimum_filter
from tqdm import tqdm

from plnn.models import DeepTimePhiPLNN

import latent_space
import splits
import sweep_runs
from nn_potential import INITIAL_DATE, UNIT_DAYS, run_dir
from plot_nn_potential import compute_marginalized_grad_phi, fit_period_spans

# The panels the platform row shows, in its order. plot_nn_potential keeps its
# own copy; these have to agree for the two to describe the same figure.
PLATFORMS = ('twitter', 'tiktok', 'instagram', 'bluesky')

# Finer than the streamplot's grid: this one seeds a root find rather than
# drawing a line, and a coarse minimum can sit a cell away from the zero.
GRID_RES = 80

# A candidate is a grid cell whose flow is a local minimum and small relative to
# the panel; anything above this is a slack region, not a fixed point.
CANDIDATE_FRAC = 0.05

# scipy.root converges to ~1e-12 on a real zero. A candidate that stalls at a
# soft spot in the field lands orders of magnitude above that.
RESIDUAL_TOL = 1e-6

DAY = 1.0 / UNIT_DAYS
SETTLE_TOL = 0.25       # LD units within an attractor counts as arrived
SETTLE_CAP = 20.0       # years to integrate before giving up
N_TRAJ = 4000


def flow(model, t, z, marginal, mc_dropout, key, n_marginal=256):
    """-grad phi marginalised over the non-displayed dimensions, confined as the
    figure confines it."""
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    grad_phi, _ = compute_marginalized_grad_phi(
        model, t, z, 0, 1, marginal, mc_dropout=mc_dropout, key=key,
        n_marginal=n_marginal)
    f = -np.array(grad_phi)
    norms = np.linalg.norm(z, ord=2, axis=1, keepdims=True)
    return np.where(norms > model.confinement_threshold, 0.0, f)


def grid_flow(model, t, xrange, yrange, marginal, mc_dropout, key, res=GRID_RES):
    x = np.linspace(*xrange, res, dtype=np.float64)
    y = np.linspace(*yrange, res, dtype=np.float64)
    xs, ys = np.meshgrid(x, y)
    f = flow(model, t, np.stack([xs.ravel(), ys.ravel()], 1),
             marginal, mc_dropout, key)
    return x, y, f[:, 0].reshape(xs.shape), f[:, 1].reshape(xs.shape)


def jacobian(model, t, p, marginal, mc_dropout, key, h=1e-2):
    """df/dz by central differences; its eigenvalues classify the fixed point."""
    p = np.asarray(p, dtype=np.float64)
    pts = np.stack([p + [h, 0], p - [h, 0], p + [0, h], p - [0, h]])
    f = flow(model, t, pts, marginal, mc_dropout, key)
    return np.stack([(f[0] - f[1]) / (2 * h), (f[2] - f[3]) / (2 * h)], axis=1)


def fixed_points(model, t, x, y, fu, fv, marginal, mc_dropout, key):
    """Every zero of the flow inside the panel, classified."""
    mag = np.hypot(fu, fv)
    xs, ys = np.meshgrid(x, y)
    local = mag == minimum_filter(mag, size=5, mode='nearest')
    seeds = sorted(((xs[i, j], ys[i, j], mag[i, j]) for i, j in zip(*np.where(local))),
                   key=lambda c: c[2])

    found = []
    for gx, gy, gmag in seeds:
        if gmag > CANDIDATE_FRAC * mag.max():
            break
        sol = scipy.optimize.root(
            lambda p: flow(model, t, p, marginal, mc_dropout, key)[0],
            np.array([gx, gy]), method='hybr', options={'xtol': 1e-8, 'eps': 1e-3})
        p, resid = sol.x, float(np.linalg.norm(sol.fun))
        if not (sol.success and resid < RESIDUAL_TOL):
            continue
        if not (x[0] <= p[0] <= x[-1] and y[0] <= p[1] <= y[-1]):
            continue
        if any(np.hypot(*(p - q['pos'])) < 1e-3 for q in found):
            continue
        ev = np.real(np.linalg.eigvals(
            jacobian(model, t, p, marginal, mc_dropout, key)))
        found.append({
            'pos': p,
            'kind': ('attractor' if (ev < 0).all()
                     else 'repeller' if (ev > 0).all() else 'saddle'),
            'eigenvalues': np.sort(ev),
            'residual': resid,
        })
    return found


def settle(interp, starts, attractors):
    """Integrate the frozen field from each start: years to reach an attractor,
    and which one. Deterministic -- the learned noise is orders below the basin
    scale, so a trajectory's endpoint is set by where it starts."""
    p = starts.copy()
    t_hit = np.full(len(p), np.nan)
    basin = np.full(len(p), -1)
    live = np.ones(len(p), bool)
    for step in range(int(SETTLE_CAP / DAY)):
        if not live.any():
            break
        q = p[live]
        k1 = interp(q)
        k2 = interp(q + 0.5 * DAY * k1)
        k3 = interp(q + 0.5 * DAY * k2)
        k4 = interp(q + DAY * k3)
        q = q + (DAY / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        p[live] = q
        d = np.linalg.norm(q[:, None, :] - attractors[None, :, :], axis=2)
        nearest = d.argmin(1)
        hit = d[np.arange(len(q)), nearest] < SETTLE_TOL
        if hit.any():
            idx = np.where(live)[0][hit]
            t_hit[idx] = (step + 1) * DAY
            basin[idx] = nearest[hit]
            live[idx] = False
    if live.any():
        d = np.linalg.norm(p[live][:, None, :] - attractors[None, :, :], axis=2)
        basin[live] = d.argmin(1)
    return t_hit, basin


def describe(x, y, fu, fv, points, coords, ecdf, sigma):
    """What the panel does to the accounts in it, rather than to its grid."""
    lin = [RegularGridInterpolator((y, x), c, bounds_error=False, fill_value=None)
           for c in (fu, fv)]
    interp = lambda p: np.stack([c(p[:, ::-1]) for c in lin], axis=1)

    inside = ((coords[:, 0] >= x[0]) & (coords[:, 0] <= x[-1])
              & (coords[:, 1] >= y[0]) & (coords[:, 1] <= y[-1]))
    coords = coords[inside]
    rng = np.random.default_rng(0)
    pts = coords[rng.choice(len(coords), min(N_TRAJ, len(coords)), replace=False)]

    # a month of drift, read off the percentile-ticked axes
    month = pts.copy()
    for _ in range(30):
        k1 = interp(month)
        k2 = interp(month + 0.5 * DAY * k1)
        k3 = interp(month + 0.5 * DAY * k2)
        k4 = interp(month + DAY * k3)
        month = month + (DAY / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    attractors = np.array([q['pos'] for q in points if q['kind'] == 'attractor'])
    mag = np.hypot(fu, fv)
    row = {
        'n_attractors': len(attractors),
        'n_saddles': sum(q['kind'] == 'saddle' for q in points),
        'grid_flow_median': float(np.median(mag)),
        'grid_flow_max': float(mag.max()),
        'flow_at_data_median': float(np.median(np.linalg.norm(interp(pts), axis=1))),
        'pct_per_month_ld1': float(np.median(np.abs(ecdf[0](month[:, 0])
                                                    - ecdf[0](pts[:, 0])))),
        'pct_per_month_ld2': float(np.median(np.abs(ecdf[1](month[:, 1])
                                                    - ecdf[1](pts[:, 1])))),
        'centroid_ld1': float(coords[:, 0].mean()),
        'centroid_ld2': float(coords[:, 1].mean()),
        'sigma': float(sigma),
    }
    if len(attractors):
        t_hit, basin = settle(interp, pts, attractors)
        row['settle_days_median'] = float(np.nanmedian(t_hit) * UNIT_DAYS)
        row['settle_days_p90'] = float(np.nanpercentile(t_hit, 90) * UNIT_DAYS)
        row['settled_frac'] = float(np.isfinite(t_hit).mean())
        row['basin_shares'] = [float((basin == i).mean()) for i in range(len(attractors))]
    return row


def analyse(panels, xrange, yrange, ecdf, mc_dropout, key):
    """Locate and describe the fixed points of every panel in a row."""
    fixed, described = [], []
    for name, model, t, marginal, coords in tqdm(panels, desc='Panels'):
        x, y, fu, fv = grid_flow(model, t, xrange, yrange, marginal, mc_dropout, key)
        points = fixed_points(model, t, x, y, fu, fv, marginal, mc_dropout, key)
        for q in points:
            # a half-life needs a decaying mode, which a saddle's unstable
            # direction is not
            half = (np.sort(np.log(2) / np.abs(q['eigenvalues']) * UNIT_DAYS)
                    if q['kind'] == 'attractor' else [None, None])
            fixed.append({
                'panel': name, 'kind': q['kind'],
                'ld1': float(q['pos'][0]), 'ld2': float(q['pos'][1]),
                'ld1_pct': float(ecdf[0](q['pos'][0])),
                'ld2_pct': float(ecdf[1](q['pos'][1])),
                'eig_min': float(q['eigenvalues'][0]),
                'eig_max': float(q['eigenvalues'][1]),
                'half_life_days_fast': half[0],
                'half_life_days_slow': half[1],
                'residual': q['residual'],
            })
        described.append({'panel': name, 't': float(t),
                          **describe(x, y, fu, fv, points, coords, ecdf,
                                     model.get_sigma())})
    return fixed, described


def time_panels(cfg, model, target_df, coord_col, marginal):
    """One panel per span of the fit period, as the time row draws them."""
    spans = fit_period_spans(target_df['createtime'], splits.SplitSpec.from_cfg(cfg))
    out = []
    for lo, hi in spans:
        t = ((lo - INITIAL_DATE).days + (hi - INITIAL_DATE).days) / (2 * UNIT_DAYS)
        coords = target_df.filter(pl.col('createtime').is_between(lo, hi))
        out.append((f'{lo:%b %Y}--{hi:%b %Y}', model, t, marginal,
                    coords[coord_col].to_numpy()[:, :2]))
    return out


def platform_panels(cfg, target_df, coord_col, dtype):
    """One panel per platform, each its own landscape over the whole period."""
    end = target_df['createtime'].max()
    end = end.date() if isinstance(end, datetime.datetime) else end
    t = (((datetime.date(2022, 1, 1) - INITIAL_DATE.date()).days
          + (end - INITIAL_DATE.date()).days) / (2 * UNIT_DAYS))

    out = []
    for platform in PLATFORMS:
        pcfg = omegaconf.OmegaConf.merge(cfg, {'platform': platform})
        state = sweep_runs.state_path(run_dir(pcfg))
        if state is None:
            raise SystemExit(
                f'no {platform} landscape under {run_dir(pcfg)}; it needs one '
                'model per platform, as the platform row does.')
        model, _ = DeepTimePhiPLNN.load(state, dtype=dtype)
        coords = latent_space.keep_platform(pcfg, target_df)[coord_col].to_numpy()
        marginal = coords[:, :cfg.n_dims] if cfg.n_dims > 2 else None
        out.append((platform, model, t, marginal, coords[:, :2]))
    return out


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    coord_col = f'coord_{cfg.n_dims}d'
    target_df, _, _ = latent_space.load(cfg)
    target_df = target_df.rename({latent_space.COORD: coord_col})
    coords = target_df[coord_col].to_numpy()

    # the window plot_nn_potential draws, so a position maps onto the panel
    xrange = (np.percentile(coords[:, 0], 0.5), np.percentile(coords[:, 0], 99.5))
    yrange = (np.percentile(coords[:, 1], 0.5), np.percentile(coords[:, 1], 99.5))
    ecdf = []
    for d in (0, 1):
        s = np.sort(coords[:, d])
        ecdf.append(lambda v, s=s: 100.0 * np.searchsorted(s, v) / len(s))

    rng = np.random.default_rng(seed=42)
    key = jax.random.PRNGKey(int(rng.integers(2**32)))
    key, modelkey, _, _ = jax.random.split(key, 4)
    dtype = jnp.float32
    marginal = coords[:, :cfg.n_dims] if cfg.n_dims > 2 else None

    plots = set(cfg.get('plots', ['time_snapshots']))
    panels = []
    if 'time_snapshots' in plots:
        state = sweep_runs.state_path(run_dir(cfg))
        if state is None:
            raise SystemExit(f'no landscape under {run_dir(cfg)}')
        model, _ = DeepTimePhiPLNN.load(state, dtype=dtype)
        panels += time_panels(cfg, model, target_df, coord_col, marginal)
    if 'platform_snapshots' in plots:
        panels += platform_panels(cfg, target_df, coord_col, dtype)
    if not panels:
        raise SystemExit(f'nothing to do: plots={sorted(plots)} names no '
                         'snapshot row')

    fixed, described = analyse(panels, xrange, yrange, ecdf, cfg.mc_dropout, modelkey)

    out_dir = cfg.get('out_dir', './out')
    os.makedirs(out_dir, exist_ok=True)
    tag = 'platform' if 'platform_snapshots' in plots else 'time'
    points_df = pl.DataFrame(fixed)
    points_df.write_parquet(
        os.path.join(out_dir, f'snapshot_attractors_{tag}.parquet.zstd'),
        compression='zstd')
    with open(os.path.join(out_dir, f'snapshot_panels_{tag}.json'), 'w') as fh:
        json.dump(described, fh, indent=1)

    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(points_df.select('panel', 'kind', 'ld1', 'ld1_pct', 'ld2', 'ld2_pct',
                               'half_life_days_fast', 'half_life_days_slow'))
    for row in described:
        print(f"{row['panel']}: grid |f| {row['grid_flow_median']:.1f} median / "
              f"{row['grid_flow_max']:.1f} peak, at data "
              f"{row['flow_at_data_median']:.1f}, "
              f"{row.get('pct_per_month_ld2', float('nan')):.1f} pctile-pts/month "
              f"on LD2, settle {row.get('settle_days_median', float('nan')):.0f} d, "
              f"basins {row.get('basin_shares')}")
    return points_df


if __name__ == '__main__':
    main()
