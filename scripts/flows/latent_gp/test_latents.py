"""End-to-end checks for the split-aware latent pipeline, on synthetic cells.

The leakage check is the point of the file: perturbing a held-out trajectory's
posts must leave every training trajectory's exported coordinate bit-identical.
That fails if the global parameters see held-out data, and it fails if per-seed
inference lets seeds mix.

Run as: python -m latent_gp.test_latents
"""

import datetime
import os
import sys
import tempfile

import numpy as np
import polars as pl
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import splits                                  # noqa: E402
from latent_gp import (LatentConfig, build_latents, build_loadings,   # noqa: E402
                       build_params, coord_cols, loading_matrix)
from latent_gp import cells as gp_cells                          # noqa: E402

M, J, T, K_TRUE, C_TRUE = 40, 25, 60, 3, 1.2
BIN_DAYS = 2


def synth(path, flip_seeds=(), flip_after=None):
    """Cells from a known factor model: 2 frozen dimensions plus one drifting.

    Each seed draws from its own stream, so flipping one seed cannot perturb
    another through the shared generator -- otherwise the leakage check below
    would be measuring the sampler, not the model.

    `flip_after` reverses every seed from that bin on, which is how the rolling
    origin is checked: a fold must not notice it. It is applied to the drawn
    counts rather than to f, because the seed's stream is shared across targets
    and resampling would perturb bins before the flip as well.
    """
    rng = np.random.default_rng(0)
    W = rng.normal(size=(J, K_TRUE)) * 0.8
    b = rng.normal(size=J) * 0.3
    z = np.empty((M, T, K_TRUE))
    z[:, :, :2] = rng.normal(size=(M, 1, 2))                       # frozen
    z[:, :, 2] = np.cumsum(rng.normal(scale=0.15, size=(M, T)), 1)  # drifting

    rows = []
    t0 = datetime.datetime(2020, 1, 1)
    for m in range(M):
        rng = np.random.default_rng(1000 + m)
        for j in range(J):
            n = rng.poisson(6, size=T)
            f = z[m] @ W[j] + b[j]
            if m in flip_seeds:
                f = -f
            p_neg = norm.cdf(-C_TRUE - f)
            p_pos = norm.cdf(f - C_TRUE)
            for t in np.flatnonzero(n > 0):
                counts = rng.multinomial(
                    n[t], [p_neg[t], 1 - p_neg[t] - p_pos[t], p_pos[t]])
                rows.append((f'seed{m:03d}', f'target{j:02d}',
                             t0 + datetime.timedelta(days=int(t) * BIN_DAYS),
                             float(counts[2] - counts[0]),
                             float(counts[2] + counts[0]), int(n[t])))
    df = pl.DataFrame(rows, orient='row',
                      schema=['SeedName', 'target', 'bin', 's_sum', 's2_sum', 'n'])
    if flip_after is not None:
        # f -> -f swaps the outer categories under a symmetric threshold, so
        # negating s_sum is itself a draw from the flipped model
        cut = t0 + datetime.timedelta(days=flip_after * BIN_DAYS)
        df = df.with_columns(pl.when(pl.col('bin') >= cut)
                               .then(-pl.col('s_sum'))
                               .otherwise(pl.col('s_sum')).alias('s_sum'))
    df.write_parquet(path, compression='zstd')
    return W, b, z


def run(path, spec, seed_split, lcfg):
    return build_latents(lcfg, spec, seed_split, log=lambda *a: None)


def main():
    rng = np.random.default_rng(0)
    spec = splits.SplitSpec(holdout_days=30)
    lcfg_kw = dict(n_dims=3, n_fast=1, fast_tau=20.0, bin_factor=1,
                   iters=8, infer_iters=5, min_target_volume=0)

    with tempfile.TemporaryDirectory() as td:
        clean = os.path.join(td, 'clean.parquet.zstd')
        W, b, z = synth(clean)

        seeds = [f'seed{m:03d}' for m in range(M)]
        traj = splits.assign_trajectory_split(seeds, spec)
        seed_split = dict(zip(traj['filter_value'], traj['traj_split']))
        n_by = {s: sum(v == s for v in seed_split.values()) for s in splits.TRAJ_SPLITS}
        print(f'trajectory split: {n_by}')
        assert all(n_by[s] > 0 for s in splits.TRAJ_SPLITS), n_by

        a = run(clean, spec, seed_split, LatentConfig(cells_path=clean, **lcfg_kw))

        # same data, but every held-out trajectory's stances reversed
        held = {s for s, v in seed_split.items() if v != 'train'}
        flipped = os.path.join(td, 'flipped.parquet.zstd')
        synth(flipped, flip_seeds={int(s[4:]) for s in held})
        bcfg = LatentConfig(cells_path=flipped, **lcfg_kw)
        bb = run(flipped, spec, seed_split, bcfg)

        c_coord, c_causal, c_sd = coord_cols(3)
        key = ['filter_value', 'createtime']
        j = a.select(key + [c_coord]).join(
            bb.select(key + [c_coord]).rename({c_coord: 'other'}), on=key, how='inner')
        j = j.join(traj, on='filter_value', how='left')

        def gap(rows):
            x = np.stack(rows[c_coord].to_numpy())
            y = np.stack(rows['other'].to_numpy())
            return float(np.abs(x - y).max())

        tr = gap(j.filter(pl.col('traj_split') == 'train'))
        ho = gap(j.filter(pl.col('traj_split') != 'train'))
        print(f'max |delta| on train trajectories: {tr:.3e}')
        print(f'max |delta| on held-out trajectories: {ho:.3e}')
        assert tr < 1e-9, f'held-out data reached the training representation ({tr:.2e})'
        assert ho > 1e-3, f'perturbation had no effect at all ({ho:.2e}) -- test is vacuous'

        # exported frame is well formed and spans only observed bins
        assert set(a.columns) >= {'createtime', 'filter_value', c_coord, c_causal,
                                  c_sd, 'n_posts', 'traj_split'}
        assert a['filter_value'].n_unique() == M
        assert a[c_coord].dtype == pl.Array(pl.Float64, 3), a[c_coord].dtype
        assert np.stack(a[c_sd].to_numpy()).min() >= 0

        # the drifting dimension must actually move and the frozen ones must not
        coords = np.stack(a.sort(['filter_value', 'createtime'])[c_coord].to_numpy())
        per_seed = coords.reshape(M, -1, 3)
        within = per_seed.std(1).mean(0)
        print('within-trajectory sd per latent dim:', np.round(within, 3))
        assert within[0] > 2 * within[1:].max(), \
            f'the fast dimension is not the one that moves: {within}'

        cau = np.stack(a.sort(['filter_value', 'createtime'])[c_causal].to_numpy())
        assert np.isfinite(cau).all()
        print('causal coords finite, shape', cau.shape)

        # interpolating onto a finer grid must reproduce the bin-grid values
        # exactly at the original bin centres, and add rows in between
        fine = run(clean, spec, seed_split,
                   LatentConfig(cells_path=clean, interp_days=0.5, **lcfg_kw))
        assert len(fine) > 1.9 * len(a), (len(fine), len(a))
        shared = a.select(key + [c_coord]).join(
            fine.select(key + [c_coord]).rename({c_coord: 'fine'}), on=key, how='inner')
        assert len(shared) == len(a), (len(shared), len(a))
        d = np.abs(np.stack(shared[c_coord].to_numpy())
                   - np.stack(shared['fine'].to_numpy())).max()
        print(f'{len(fine)} interpolated rows from {len(a)}; '
              f'max |delta| at original bin centres {d:.2e}')
        assert d < 1e-12, d

        check_loadings(td, clean, spec, seed_split, lcfg_kw, W, b, z)
        check_rolling_origin(td, seed_split, lcfg_kw)
        check_projection(td, clean, spec, seed_split, lcfg_kw, a)

    print('\nall latent-pipeline checks passed')


def check_loadings(td, clean, spec, seed_split, lcfg_kw, W_true, b_true, z_true):
    """The exported loadings must reconstruct the cell scores the fit models.

    A cell's score f is invariant to the latent basis, unlike W and z
    separately, so it is the one thing that can be compared against the truth
    whatever the axes came out as -- and reconstructing it from the *exported*
    coordinates is what catches loadings left in the fit's own units.
    """
    cache = os.path.join(td, 'loading_cache')
    lcfg = LatentConfig(cells_path=clean, **lcfg_kw)
    seen = []
    kw = dict(cache_root=cache, log=lambda *a: seen.append(' '.join(map(str, a))))

    loadings = build_loadings(lcfg, spec, seed_split, **kw)
    # one directory per fit, so the landscape models trained on it can sit
    # inside rather than beside every other configuration's
    fit, = os.listdir(cache)
    assert sorted(os.listdir(os.path.join(cache, fit))) == \
        ['latents.parquet.zstd', 'loadings.parquet.zstd', 'params.json'], fit

    # one fit serves both, so asking for the sibling must not refit
    seen.clear()
    a = build_latents(lcfg, spec, seed_split, **kw)
    assert any('reusing cached' in m for m in seen), seen

    # a fit predating an output regains it without the frames beside it moving:
    # those are what the landscape models on top were trained on
    latents_path = os.path.join(cache, fit, 'latents.parquet.zstd')
    before = open(latents_path, 'rb').read()
    os.remove(os.path.join(cache, fit, 'params.json'))
    params = build_params(lcfg, spec, seed_split, **kw)
    assert set(params) == {'c', 'mu', 'sd', 'dt', 'n_dims'}, params
    assert open(latents_path, 'rb').read() == before, 'the refit rewrote the latents'

    assert loadings['target'].to_list() == [f'target{j:02d}' for j in range(J)]
    assert loadings['loading'].dtype == pl.Array(pl.Float64, 3), loadings['loading'].dtype
    dim_first, targets = loading_matrix(loadings)
    assert dim_first.shape == (3, J), dim_first.shape
    assert targets == loadings['target'].to_list()

    coords = np.stack(a.sort(['filter_value', 'createtime'])[coord_cols(3)[0]]
                      .to_numpy()).reshape(M, -1, 3)
    assert coords.shape == z_true.shape, (coords.shape, z_true.shape)
    f_hat = np.einsum('mtk,jk->mtj', coords, loadings['loading'].to_numpy()) \
        + loadings['intercept'].to_numpy()
    f_true = np.einsum('mtk,jk->mtj', z_true, W_true) + b_true
    r = float(np.corrcoef(f_hat.ravel(), f_true.ravel())[0, 1])
    print(f'cell-score correlation with truth: {r:.3f}')
    assert r > 0.9, f'loadings do not reconstruct the cell scores ({r:.3f})'


def check_rolling_origin(td, seed_split, lcfg_kw):
    """A rolled-back origin must not see the data after its window.

    The perturbation lands entirely past the window's end, so the fold's
    coordinates have to come out bit-identical -- and the control confirms the
    same perturbation does move an unrolled run, which is what makes that
    meaningful.
    """
    # bin centres are t0 + 2t + 1 days, so a 40-day offset over 60 bins ends
    # the window at bin 40 and the perturbation starts well past it
    spec = splits.SplitSpec(holdout_days=30, origin_offset_days=40)
    flip_after = 45

    clean = os.path.join(td, 'clean.parquet.zstd')
    late = os.path.join(td, 'late.parquet.zstd')
    synth(late, flip_after=flip_after)

    c_coord = coord_cols(3)[0]
    key = ['filter_value', 'createtime']

    def coords(path, sp):
        got = run(path, sp, seed_split, LatentConfig(cells_path=path, **lcfg_kw))
        return got.select(key + [c_coord])

    a, bb = coords(clean, spec), coords(late, spec)
    assert len(a) == len(bb), (len(a), len(bb))
    j = a.join(bb.rename({c_coord: 'other'}), on=key, how='inner')
    assert len(j) == len(a), (len(j), len(a))
    rolled = float(np.abs(np.stack(j[c_coord].to_numpy())
                          - np.stack(j['other'].to_numpy())).max())

    full = splits.SplitSpec(holdout_days=30)
    u, v = coords(clean, full), coords(late, full)
    ju = u.join(v.rename({c_coord: 'other'}), on=key, how='inner')
    unrolled = float(np.abs(np.stack(ju[c_coord].to_numpy())
                            - np.stack(ju['other'].to_numpy())).max())

    print(f'rows kept: {len(a)} rolled vs {len(u)} unrolled')
    print(f'max |delta| from a post-window flip: rolled {rolled:.3e}, '
          f'unrolled {unrolled:.3e}')
    assert len(a) < len(u), 'the rolled origin kept just as many bins'
    assert rolled < 1e-9, f'data past the window end reached the fold ({rolled:.2e})'
    assert unrolled > 1e-3, f'the flip moves nothing even unrolled ({unrolled:.2e})'


PLATFORMS = ('twitter', 'tiktok')


def handles(clean, path):
    """The same cells keyed by platform handle instead of by seed.

    Each seed's cells are partitioned between two handles rather than copied, so
    a handle is a real subset of that seed's posts and the two together are
    exactly the seed -- which is what makes the projected coordinates comparable
    against the pooled ones.
    """
    df = pl.read_parquet(clean).with_columns(
        pl.col('target').str.slice(-1).cast(pl.Int64).alias('_j'))
    plat = pl.when(pl.col('_j') % 2 == 0).then(pl.lit(PLATFORMS[0])) \
             .otherwise(pl.lit(PLATFORMS[1]))
    df.with_columns(
        (pl.col('SeedName') + '-' + plat + '-handle').alias('PlatformHandleID')
    ).drop('_j').write_parquet(path, compression='zstd')
    return path


def check_projection(td, clean, spec, seed_split, lcfg_kw, pooled):
    """A projected fit must report the pooled fit's axes, not its own.

    The loadings and the standardisation are the axes, so they have to come back
    identical; what is actually inferred is z, and a handle's z has to track the
    seed it is part of. Rotate the basis and the second check fails while the
    first still passes, which is why both are here.
    """
    cache = os.path.join(td, 'projection_cache')
    path = handles(clean, os.path.join(td, 'handles.parquet.zstd'))
    kw = dict(cache_root=cache, log=lambda *a: None)

    ref = LatentConfig(cells_path=clean, **lcfg_kw)
    lcfg = LatentConfig(cells_path=path, ref_cells_path=clean,
                        traj_col=gp_cells.HANDLE_COL, **lcfg_kw)

    # a handle inherits its seed's cell: the reference fit already saw that
    # seed's posts, so hashing the handle would hold out a trajectory whose own
    # data shaped the axes it is scored in
    key = gp_cells.traj_seeds(path, gp_cells.HANDLE_COL)
    handle_split = splits.seed_split(gp_cells.seed_names(path, gp_cells.HANDLE_COL),
                                     spec, key=key)
    assert len(handle_split) == len(PLATFORMS) * M, len(handle_split)
    assert all(handle_split[h] == seed_split[s] for h, s in key.items())

    got = build_latents(lcfg, spec, handle_split, ref_seed_split=seed_split, **kw)
    assert build_loadings(lcfg, spec, handle_split, ref_seed_split=seed_split,
                          **kw).equals(build_loadings(ref, spec, seed_split, **kw))
    assert build_params(lcfg, spec, handle_split, ref_seed_split=seed_split,
                        **kw) == build_params(ref, spec, seed_split, **kw)

    # the reference fit sits in its own directory, not the projection's
    assert len(os.listdir(cache)) == 2, os.listdir(cache)

    c_coord = coord_cols(3)[0]
    key_cols = ['filter_value', 'createtime']
    per_seed = got.with_columns(
        pl.col('filter_value').str.split('-').list.get(0).alias('seed')) \
        .group_by(['seed', 'createtime']) \
        .agg([pl.col(c_coord).arr.get(k).mean().alias(f'd{k}') for k in range(3)]) \
        .join(pooled.select(key_cols + [c_coord]), left_on=['seed', 'createtime'],
              right_on=key_cols, how='inner')
    assert len(per_seed) == len(pooled), (len(per_seed), len(pooled))

    ref_coords = np.stack(per_seed[c_coord].to_numpy())
    for k in range(3):
        r = float(np.corrcoef(per_seed[f'd{k}'].to_numpy(), ref_coords[:, k])[0, 1])
        print(f'projected dim {k} vs pooled: r = {r:.3f}')
        assert r > 0.9, f'dimension {k} is not the pooled one ({r:.3f})'


if __name__ == '__main__':
    main()
