"""How reproducible is the subspace the factor model finds?

The latent basis is identified only up to an invertible transform, so no single
loading vector is comparable across two fits. The subspace its columns span is:
orthonormalise both loading matrices and take the singular values of Qa' Qb, and
those are the cosines of the principal angles between the two spans, one per
dimension. Their mean square is the share of the space the fits agree on, and a
random pair of K-dimensional subspaces of R^J scores K/J.

W W' is invariant the same way and says the same thing in the targets' own
units: which targets the latent claims move together.

    --split seeds   disjoint halves of the population -- estimation noise
    --split time    the first half of the study period against the second

Takes the latents.* block of the trial being reported, and runs from the repo
root:

    python scripts/flows/gpfa_subspace.py --data <cells> --dims 5 --n-fast 6 \
        --fast-kind wiener --fast-tau 10 --slow-kind wiener --slow-tau 2560 \
        --w-ridge 1 --obs-model hard --split seeds --reps 5
"""
import argparse

import numpy as np
import polars as pl
from latent_gp import cells, latents
from latent_gp.fit import fit, prior_components

TOL = 1e-8


def span(W, weight=None):
    """Orthonormal basis for the column space of W, in the weighted metric.

    Weighting by volume counts a target the fit barely constrains for less; its
    loading is mostly the ridge, and unweighted it still spends a full axis of
    R^J on the angle.
    """
    if weight is not None:
        W = W * np.sqrt(weight)[:, None]
    u, s, _ = np.linalg.svd(W, full_matrices=False)
    return u[:, s > s[0] * TOL]


def angles(Wa, Wb, weight=None):
    """Cosines of the principal angles between two loading spans, descending."""
    s = np.linalg.svd(span(Wa, weight).T @ span(Wb, weight), compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def coupling_r(Wa, Wb, weight=None):
    """Correlation of the target-by-target couplings the two fits imply."""
    if weight is not None:
        w = np.sqrt(weight)[:, None]
        Wa, Wb = Wa * w, Wb * w
    iu = np.triu_indices(Wa.shape[0], 1)
    return float(np.corrcoef((Wa @ Wa.T)[iu], (Wb @ Wb.T)[iu])[0, 1])


def energy(W):
    """Share of the loading energy each dimension carries."""
    s = np.linalg.svd(W, compute_uv=False) ** 2
    return s / s.sum()


def permuted_overlap(Wa, Wb, weight, reps, rng):
    """Overlap when one fit's loadings are reassigned to other targets.

    A sharper null than K/J: it keeps both fits' loading magnitudes and only
    breaks the correspondence between them.
    """
    out = np.empty(reps)
    for i in range(reps):
        out[i] = (angles(Wa, Wb[rng.permutation(Wb.shape[0])], weight) ** 2).mean()
    return out


def subset(df, meta, keep_m=None, t_lo=0, t_hi=None):
    """Re-index a slice of the cell frame so it packs as a fit of its own."""
    t_hi = meta['T'] if t_hi is None else t_hi
    out = df.filter((pl.col('t') >= t_lo) & (pl.col('t') < t_hi))
    seeds = meta['seeds']
    if keep_m is not None:
        order = sorted(int(m) for m in keep_m)
        out = out.filter(pl.col('m').is_in(order))
        out = out.with_columns(pl.col('m').replace_strict(
            {m: i for i, m in enumerate(order)}))
        seeds = [meta['seeds'][m] for m in order]
    if t_lo:
        out = out.with_columns(pl.col('t') - t_lo)
    return out, dict(meta, M=len(seeds), T=int(t_hi - t_lo), seeds=seeds)


def fit_part(df, meta, n_arch, logL, a, comps):
    """The loadings one half of the data gives, on the pinned target index."""
    d = cells.pack(cells.deflate(df, a.rho), meta,
                   None if n_arch is None else n_arch[df['row'].to_numpy()])
    r = fit(d, comps, meta['dt'], a.dims, a.iters, seed=a.seed, logL=logL,
            w_ridge=a.w_ridge)
    return r['W']


ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--dims', type=int, required=True)
ap.add_argument('--n-fast', type=int, required=True)
ap.add_argument('--fast-kind', required=True)
ap.add_argument('--fast-tau', type=float, required=True)
ap.add_argument('--slow-kind', required=True)
ap.add_argument('--slow-tau', type=float, required=True)
ap.add_argument('--split', default='seeds', choices=['seeds', 'time'])
ap.add_argument('--reps', type=int, default=5, help='seed split only')
ap.add_argument('--perm', type=int, default=200)
ap.add_argument('--seeds', default='', help='pinned seed list from pin_seeds.py')
ap.add_argument('--bin-factor', type=int, default=8)
ap.add_argument('--rho', type=float, default=0.0)
ap.add_argument('--iters', type=int, default=25)
ap.add_argument('--w-ridge', type=float, default=1e-4)
ap.add_argument('--obs-model', default='hard')
ap.add_argument('--temperature', type=float, default=1.0)
ap.add_argument('--resolution', type=int, default=6)
ap.add_argument('--calibration', default='')
ap.add_argument('--min-target-volume', type=int, default=400)
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()

keep = None
if a.seeds:
    keep = pl.read_parquet(a.seeds)['filter_value'].to_list()
df, meta = cells.load(a.data, a.bin_factor, seeds=keep,
                      min_target_volume=a.min_target_volume)
obs_df, n_arch, logL = latents.observation(
    df, a.obs_model, a.temperature, a.resolution, calibration_path=a.calibration)
obs_df = obs_df.with_row_index('row')
comps = prior_components(a.dims, a.n_fast, a.fast_tau, a.slow_kind, a.slow_tau,
                         fast_kind=a.fast_kind)

vol = np.zeros(meta['J'])
per_j = df.group_by('j').agg(pl.col('n').sum())
vol[per_j['j'].to_numpy()] = per_j['n'].to_numpy()

reps = a.reps if a.split == 'seeds' else 1
print(f"M={meta['M']} J={meta['J']} T={meta['T']} K={a.dims} "
      f"dt={meta['dt']:.0f}d  split {a.split}, {reps} rep(s), obs {a.obs_model}")

rng = np.random.default_rng(a.seed)
rows = []
for rep in range(reps):
    if a.split == 'seeds':
        order = rng.permutation(meta['M'])
        parts = [subset(obs_df, meta, keep_m=order[:meta['M'] // 2]),
                 subset(obs_df, meta, keep_m=order[meta['M'] // 2:])]
    else:
        half = meta['T'] // 2
        parts = [subset(obs_df, meta, t_hi=half),
                 subset(obs_df, meta, t_lo=half)]
    Wa, Wb = [fit_part(d, m, n_arch, logL, a, comps) for d, m in parts]
    for label, wt in (('unweighted', None), ('volume', vol)):
        cos = angles(Wa, Wb, wt)
        rows.append(dict(rep=rep, weight=label, cos=cos,
                         overlap=float((cos ** 2).mean()),
                         coupling=coupling_r(Wa, Wb, wt)))
    print(f"  rep {rep}: cells {len(parts[0][0]):,} / {len(parts[1][0]):,}  "
          f"energy a {np.round(energy(Wa), 3)} b {np.round(energy(Wb), 3)}")

null = permuted_overlap(Wa, Wb, vol, a.perm, rng)

hdr = (f"\n{'weight':12}" + ''.join(f"{'cos%d' % (k + 1):>8}" for k in range(a.dims))
       + f"{'overlap':>10}{'coupling':>10}")
print(hdr); print('-' * (len(hdr) - 1))
for label in ('unweighted', 'volume'):
    sel = [r for r in rows if r['weight'] == label]
    cos = np.mean([r['cos'] for r in sel], 0)
    print(f"{label:12}" + ''.join(f"{c:8.3f}" for c in cos)
          + f"{np.mean([r['overlap'] for r in sel]):10.3f}"
          + f"{np.mean([r['coupling'] for r in sel]):10.3f}")

vol_rows = [r for r in rows if r['weight'] == 'volume']
cos = np.mean([r['cos'] for r in vol_rows], 0)
print(f"\ncos: cosine of the k-th principal angle, 1 = the two fits agree on "
      f"that direction.\noverlap: mean cos^2, the share of the space they "
      f"share.  coupling: correlation\nof the off-diagonal W W', how much the "
      f"two fits agree on which targets move together.")
print(f"\nchance overlap: {a.dims / meta['J']:.3f} for random subspaces; "
      f"{null.mean():.3f} (95th {np.quantile(null, 0.95):.3f}) "
      f'with the loadings reassigned across targets')
print(f"dimensions reproducing at cos > 0.9: {int((cos > 0.9).sum())} of {a.dims}"
      f"   > 0.7: {int((cos > 0.7).sum())}   (volume-weighted)")
