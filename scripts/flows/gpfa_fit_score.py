"""Held-out fit of one latent-GP configuration, scored as latent_gp.sweep does.

The sweep's objective says how well the landscape predicts motion; this says how
well the factor model predicts the labels it was fitted to. Two contiguous
holdouts per seed: a forecast tail and an interior block.

Scoring is against hard labels whichever observation model is fitted, so RPS and
LL do not compare across them -- RES is the column that does.

Takes the latents.* block of the trial being reported, not the landscape's
hyperparameters, and runs from the repo root:

    python scripts/flows/gpfa_fit_score.py --data <cells> --dims 5 \
        --n-fast 6 --fast-kind wiener --fast-tau 10 --slow-kind wiener \
        --slow-tau 2560 --w-ridge 1 --obs-model hard
"""
import argparse
from math import erf, sqrt

import numpy as np
from latent_gp import cells, latents, metrics
from latent_gp.fit import fit, prior_components, score

def climatologies(df, train_mask, meta):
    """Static label rates a dynamic latent has to beat, coarse to fine.

    The finest is the series' own training mean: it knows this seed's usual
    stance on this target and nothing about when, so beating it is what makes
    the latent's time variation worth fitting.
    """
    tr = df.filter(train_mask)
    cnt = np.stack([tr['n_neg'].to_numpy(), tr['n_neu'].to_numpy(),
                    tr['n_pos'].to_numpy()], 1).astype(np.float64)
    m, j, J = tr['m'].to_numpy(), tr['j'].to_numpy(), meta['J']

    glob = cnt.sum(0)
    glob = glob / glob.sum()
    by_j = np.stack([np.bincount(j, w, J) for w in cnt.T], 1)
    by_mj = {}
    key = m.astype(np.int64) * J + j
    order = np.argsort(key)
    k_s, c_s = key[order], cnt[order]
    edges = np.flatnonzero(np.diff(k_s)) + 1
    for blk in np.split(np.arange(len(k_s)), edges):
        by_mj[int(k_s[blk[0]])] = c_s[blk].sum(0)
    return glob, by_j, by_mj, J


def baseline_probs(ev, glob, by_j, by_mj, J, level):
    """Per-held-out-cell label rates under one climatology, with back-off."""
    if level == 'global':
        p = np.repeat(glob[None], len(ev['n']), 0)
    elif level == 'target':
        tot = by_j.sum(1, keepdims=True)
        rate = np.where(tot > 0, by_j / np.maximum(tot, 1e-12), glob[None])
        p = rate[ev['j']]
    else:
        p = np.empty((len(ev['n']), 3))
        tot_j = by_j.sum(1, keepdims=True)
        rate_j = np.where(tot_j > 0, by_j / np.maximum(tot_j, 1e-12), glob[None])
        for i, (mi, ji) in enumerate(zip(ev['m'], ev['j'])):
            c = by_mj.get(int(mi) * J + int(ji))
            p[i] = c / c.sum() if c is not None and c.sum() > 0 else rate_j[ji]
    return p[:, 0], p[:, 1], p[:, 2]


ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--dims', type=int, required=True)
ap.add_argument('--n-fast', type=int, required=True)
ap.add_argument('--fast-kind', required=True)
ap.add_argument('--fast-tau', type=float, required=True)
ap.add_argument('--slow-kind', required=True)
ap.add_argument('--slow-tau', type=float, required=True)
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

df, meta = cells.load(a.data, a.bin_factor, min_target_volume=a.min_target_volume)
fc, inte = cells.holdout_masks(df, meta)
train_mask = ~fc & ~inte
# built before the observation model touches the counts, so the target does not
# move with the model under test
ev = {'forecast (tail)': cells.eval_set(df.filter(fc)),
      'interior block': cells.eval_set(df.filter(inte))}

obs_df, n_arch, logL = latents.observation(
    df, a.obs_model, a.temperature, a.resolution, calibration_path=a.calibration)
tr = cells.deflate(obs_df.filter(train_mask), a.rho)
d = cells.pack(tr, meta, None if n_arch is None else n_arch[train_mask])
print(f"M={meta['M']} J={meta['J']} T={meta['T']} K={a.dims} dt={meta['dt']:.0f}d "
      f"train cells {len(tr):,}  obs {a.obs_model}")

comps = prior_components(a.dims, a.n_fast, a.fast_tau, a.slow_kind, a.slow_tau,
                         fast_kind=a.fast_kind)
r = fit(d, comps, meta['dt'], a.dims, a.iters, seed=a.seed, logL=logL,
        w_ridge=a.w_ridge)
print(f"fitted threshold c = {r['c']:.4f}   "
      f"implied neutral share at f=0: "
      f"{erf(r['c'] / sqrt(2)):.4f}\n")

hdr = (f"{'holdout':18} {'RPS':>9} {'LL/post':>10} {'MSE':>9}   "
       f"{'neuREL':>8} {'neuRES':>8} {'polREL':>8} {'polRES':>8}")
print(hdr); print('-' * len(hdr))
score_of = {}
for name, e in ev.items():
    s = score(r, e)
    score_of[name] = s
    print(f"{name:18} {s['rps']:9.5f} {s['ll']:10.5f} {s['mse']:9.5f}   "
          f"{s['neutrality']['rel']:8.5f} {s['neutrality']['res']:8.5f} "
          f"{s['polarity']['rel']:8.5f} {s['polarity']['res']:8.5f}")
print('\nRPS, MSE, REL: lower better.  LL, RES: higher better.')
print('REL is calibration, RES is discrimination.')

glob, by_j, by_mj, J = climatologies(df, train_mask, meta)
print(f"\n{'holdout':18} {'baseline':14} {'RPSS':>8} {'R2(mean)':>9}")
print('-' * 53)
for name, e in ev.items():
    n_tot = e['n'].sum()
    rps_m = score_of[name]['rps']
    mse_m = score_of[name]['mse']
    for level, label in (('global', 'corpus rate'), ('target', 'per target'),
                         ('series', 'per series')):
        b_neg, b_neu, b_pos = baseline_probs(e, glob, by_j, by_mj, J, level)
        rps_b = float(metrics.rps(b_neg, b_neu, b_pos, e).sum() / n_tot)
        mse_b = float(np.average(((b_pos - b_neg) - e['mean']) ** 2, weights=e['n']))
        print(f"{name if level == 'global' else '':18} {label:14} "
              f"{1 - rps_m / rps_b:8.4f} {1 - mse_m / mse_b:9.4f}")
print('\nSkill scores: share of the baseline\'s error the latent removes.')
for name, e in ev.items():
    print(f"{name}: {int(e['n'].sum()):,} posts in {len(e['n']):,} cells, "
          f"uncertainty neu {score_of[name]['neutrality']['unc']:.5f} "
          f"pol {score_of[name]['polarity']['unc']:.5f}")

# RES is discrimination and UNC is the most any forecast of these labels could
# resolve, so the ratio is the share of achievable discrimination reached. It
# carries no calibration term, which is why it is readable where the skill
# scores above are not.
print(f"\n{'holdout':18} {'polarity':>10} {'neutrality':>12}   "
      "(RES / UNC: share of achievable discrimination)")
print('-' * 72)
for name in ev:
    s = score_of[name]
    print(f"{name:18} {s['polarity']['res'] / s['polarity']['unc']:10.1%} "
          f"{s['neutrality']['res'] / s['neutrality']['unc']:12.1%}")
