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
from latent_gp import cells, latents
from latent_gp.fit import fit, prior_components, score

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
out = {}
for name, e in ev.items():
    s = score(r, e)
    out[name] = s
    print(f"{name:18} {s['rps']:9.5f} {s['ll']:10.5f} {s['mse']:9.5f}   "
          f"{s['neutrality']['rel']:8.5f} {s['neutrality']['res']:8.5f} "
          f"{s['polarity']['rel']:8.5f} {s['polarity']['res']:8.5f}")
print('\nRPS, MSE, REL: lower better.  LL, RES: higher better.')
print('REL is calibration, RES is discrimination.')
for name, e in ev.items():
    print(f"{name}: {int(e['n'].sum()):,} posts in {len(e['n']):,} cells, "
          f"uncertainty neu {out[name]['neutrality']['unc']:.5f} "
          f"pol {out[name]['polarity']['unc']:.5f}")
