"""Smoke test of the training script's data path on synthetic cells.

Exercises load_latent_df -> build_training_pairs -> label_pairs and asserts the
scenario cells are populated and leak-free, without needing plnn or a GPU.

`check_platform_run` does the same for a platform-specific landscape, whose
trajectories are accounts projected onto the pooled fit's axes.

Run as: python test_nested_split_pipeline.py
"""

import os
import sys
import tempfile
import types

import jax
import numpy as np
import polars as pl
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# plnn and wandb are only needed to train; stub them so the data path can run
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

import splits                                    # noqa: E402
import latent_space                              # noqa: E402
import nn_potential as nnp                       # noqa: E402
from latent_gp import cells as gp_cells          # noqa: E402
from latent_gp.test_latents import PLATFORMS, handles, synth, M   # noqa: E402


def make_cfg(cells_path, out_dir):
    return OmegaConf.create({
        'n_dims': 3, 'platform': 'all', 'min_target_volume': 0,
        'out_dir': out_dir, 'sigma': 0.3, 'learning_rate': 1e-3,
        'rolling_mean_window': 292, 'trend_path': './data/trends',
        'split': {'holdout_days': 30, 'origin_offset_days': 0, 'train_frac': 0.70,
                  'val_frac': 0.10, 'seed': 42},
        'latents': {
            'method': 'gpfa', 'cells_path': cells_path,
            'bin_factor': 1, 'interp_days': 1.0, 'n_fast': 1, 'fast_tau': 20.0,
            'fast_kind': 'wiener', 'slow_kind': 'const', 'slow_tau': 2560.0,
            'rho': 0.0, 'iters': 8, 'infer_iters': 5, 'causal_state': False,
            'obs_model': 'hard', 'obs_temperature': 1.0, 'prob_resolution': 6,
            'prob_floor': 0.01, 'calibration_path': '', 'seed': 0,
        },
    })


def check_displacement_identity():
    """skill = 2*rho*R - R^2 must hold exactly, and rho^2 must bound skill."""
    rng = np.random.default_rng(0)
    n, K = 20_000, 6
    d = rng.normal(size=(n, K))
    for alpha, noise_sd in ((0.0, 0.0), (0.0, 0.3), (0.05, 0.3), (0.5, 0.3), (2.0, 0.1)):
        p = alpha * d + noise_sd * rng.normal(size=(n, K))
        m = nnp.compute_metrics(((d - p) ** 2).sum(1), (d ** 2).sum(1),
                                (p ** 2).sum(1), (d * p).sum(1))
        rho, R = m['direction_rho'], m['displacement_ratio']
        assert abs(m['skill_score'] - (2 * rho * R - R ** 2)) < 1e-9, (alpha, m)
        assert m['skill_ceiling'] + 1e-12 >= m['skill_score'], (alpha, m)
        assert abs(m['skill_ceiling'] - rho ** 2) < 1e-12
        print(f'  alpha={alpha:<4} rho={rho:+.4f} R={R:.4f} '
              f'skill={m["skill_score"]:+.5f} ceiling={m["skill_ceiling"]:.5f}')

    # a model that only rediscovers the momentum rule must show no gain
    m = rng.normal(size=(n, K))
    d2 = 0.4 * m + rng.normal(size=(n, K))
    for label, p, want_gain in (
            ('model == momentum', m.copy(), False),
            ('model is momentum rescaled', 3.0 * m, False),
            ('model adds an orthogonal signal', 0.4 * d2 + 0.1 * m, True)):
        met = nnp.compute_metrics(
            ((d2 - p) ** 2).sum(1), (d2 ** 2).sum(1), (p ** 2).sum(1), (d2 * p).sum(1),
            (m ** 2).sum(1), (d2 * m).sum(1), (p * m).sum(1))
        gain = met['ceiling_gain']
        print(f"  {label:32} mom_rho={met['momentum_rho']:+.3f} "
              f"ceiling={met['skill_ceiling']:.4f} gain={gain:.4f}")
        if want_gain:
            assert gain > 0.05, (label, met)
        else:
            assert gain < 1e-6, (label, met)

    # a fully collapsed model must read as collapsed, not as merely unhelpful
    m = nnp.compute_metrics((d ** 2).sum(1), (d ** 2).sum(1),
                            np.zeros(n), np.zeros(n))
    assert m['displacement_ratio'] == 0.0 and m['direction_rho'] == 0.0
    assert m['skill_ceiling'] == 0.0 and m['amplitude_vs_optimal'] == float('inf')
    print('  collapsed model: rho=0, ceiling=0, skill=0 -- distinguishable')


def check_per_dimension_gain():
    """The per-dimension gain must not depend on how many dimensions move.

    Pooled rho already does not, so long as nothing moves in the frozen
    directions. What breaks it is drift predicted where there is none: that
    lands in E|p|^2 alone, so a configuration with one moving dimension is
    charged for five directions of it and one with six for none. The
    per-dimension average scores only the dimensions that move and reports the
    rest as spurious_drift_ratio.
    """
    rng = np.random.default_rng(0)
    n, K, r = 20_000, 6, 0.5

    def build(n_moving, spurious):
        d = np.zeros((n, K))
        p = np.zeros((n, K))
        m = np.zeros((n, K))
        for k in range(n_moving):
            d[:, k] = rng.normal(size=n)
            p[:, k] = r * d[:, k] + np.sqrt(1 - r ** 2) * rng.normal(size=n)
            m[:, k] = 0.3 * d[:, k] + np.sqrt(1 - 0.09) * rng.normal(size=n)
        for k in range(n_moving, K):
            p[:, k] = spurious * rng.normal(size=n)
        return nnp.compute_metrics(
            ((d - p) ** 2).sum(1), (d ** 2).sum(1), (p ** 2).sum(1), (d * p).sum(1),
            (m ** 2).sum(1), (d * m).sum(1), (p * m).sum(1),
            dim_moments=(d ** 2, p ** 2, m ** 2, d * p, d * m, p * m))

    for spurious in (0.0, 0.5):
        got = {nm: build(nm, spurious) for nm in (1, 2, 3, 6)}
        for nm, met in got.items():
            assert met['n_moving_dims'] == nm, (nm, met['n_moving_dims'])
            assert abs(met['direction_rho_per_dim'] - r) < 0.02, (nm, met)
        spread = max(m['ceiling_gain_per_dim'] for m in got.values()) \
            - min(m['ceiling_gain_per_dim'] for m in got.values())
        assert spread < 0.01, ('per-dim gain tracks the dimension count', spread)
        pooled = max(m['ceiling_gain'] for m in got.values()) \
            - min(m['ceiling_gain'] for m in got.values())
        print(f'  spurious={spurious}: per-dim gain spread {spread:.5f}, '
              f'pooled {pooled:.5f}, '
              f'ratio reported {got[1]["spurious_drift_ratio"]:.3f}')
        if spurious == 0.0:
            assert pooled < 0.01, pooled          # nothing to leak, so pooled holds too
        else:
            # the leak is what the pooled number cannot separate out
            assert pooled > 0.05, pooled
            assert got[6]['spurious_drift_ratio'] == 0.0, got[6]

    # a dimension carrying almost none of the motion still counts once
    d = np.zeros((n, K)); p = np.zeros((n, K)); m = np.zeros((n, K))
    for k in range(K):
        rk = 0.9 if k == 0 else 0.05
        sd = 10.0 if k == 0 else 1.0
        d[:, k] = rng.normal(0, sd, n)
        p[:, k] = rk * d[:, k] + sd * np.sqrt(1 - rk ** 2) * rng.normal(size=n)
        m[:, k] = 0.3 * d[:, k] + sd * np.sqrt(1 - 0.09) * rng.normal(size=n)
    met = nnp.compute_metrics(
        ((d - p) ** 2).sum(1), (d ** 2).sum(1), (p ** 2).sum(1), (d * p).sum(1),
        (m ** 2).sum(1), (d * m).sum(1), (p * m).sum(1),
        dim_moments=(d ** 2, p ** 2, m ** 2, d * p, d * m, p * m))
    assert met['direction_rho'] > 0.8, met            # pooled follows the big dimension
    assert abs(met['direction_rho_per_dim'] - (0.9 + 5 * 0.05) / 6) < 0.02, met
    print(f"  variance-weighted rho {met['direction_rho']:.4f} vs "
          f"unweighted {met['direction_rho_per_dim']:.4f}")


def check_platform_run(td, cells, out_dir):
    """One landscape per platform, all of them in the pooled fit's space.

    The representation is shared -- every platform reads the same projected fit
    -- so what separates the runs is `platform` alone, and what must not
    separate them is the axes.
    """
    handle_cells = handles(cells, os.path.join(td, 'handles.parquet.zstd'))
    cfg = make_cfg(handle_cells, out_dir)
    cfg.latents.ref_cells_path = cells
    cfg.latents.traj_col = gp_cells.HANDLE_COL
    spec = splits.SplitSpec.from_cfg(cfg)

    _, pooled_components, pooled_targets = latent_space.load(make_cfg(cells, out_dir))
    _, components, targets = latent_space.load(cfg)
    assert targets == pooled_targets
    assert np.array_equal(components, pooled_components), \
        'the projection reported axes of its own'

    key = latent_space.split_key(cfg)
    seed_splits = splits.seed_split(gp_cells.seed_names(cells), spec)
    assert len(key) == len(PLATFORMS) * M

    dirs = set()
    for platform in PLATFORMS:
        cfg.platform = platform
        df = nnp.load_latent_df(cfg, spec)
        assert df.columns == ['createtime', 'filter_value', 'x0']
        assert df['filter_value'].n_unique() == M, platform
        assert df['filter_value'].str.contains(f'-{platform}-').all()

        pairs = nnp.build_training_pairs(cfg, df, smooth=False)
        labelled = splits.label_pairs(pairs, spec, time_col='next_createtime', key=key)
        train = splits.training_rows(labelled)
        for traj in splits.TRAJ_SPLITS:
            for time in splits.TIME_SPLITS:
                cell = splits.select(labelled, traj, time)
                splits.check_leakage(train, cell, splits.scenario_name(traj, time))
                assert len(cell) > 0, f'{platform} {traj}_{time} is empty'

        # a handle is on the side of the boundary its seed is on, or the axes
        # were fitted on posts belonging to a trajectory being held out here
        by_seed = dict(zip(labelled['filter_value'].to_list(),
                           labelled['traj_split'].to_list()))
        assert all(by_seed[h] == seed_splits[key[h]] for h in by_seed)
        dirs.add(nnp.run_dir(cfg))

    assert len(dirs) == len(PLATFORMS), dirs
    assert len({os.path.dirname(d) for d in dirs}) == 1, \
        'the platforms did not share one representation'
    print(f'{len(PLATFORMS)} platform landscapes over one projected fit, '
          'each in the pooled axes')

    cfg.platform = 'all'
    seed_cfg = make_cfg(cells, out_dir)
    seed_cfg.platform = PLATFORMS[0]
    try:
        nnp.load_latent_df(seed_cfg, spec)
    except ValueError as e:
        assert 'handle-keyed' in str(e), e
    else:
        raise AssertionError('a platform of a seed-keyed run should not resolve')
    print('a platform run on seed-keyed latents is refused, not silently empty')


def main():
    with tempfile.TemporaryDirectory() as td:
        cells = os.path.join(td, 'cells.parquet.zstd')
        synth(cells)
        cfg = make_cfg(cells, os.path.join(td, 'out'))
        spec = splits.SplitSpec.from_cfg(cfg)

        target_df = nnp.load_latent_df(cfg, spec)
        assert target_df['filter_value'].n_unique() == M
        assert target_df.columns == ['createtime', 'filter_value', 'x0']

        pairs = nnp.build_training_pairs(cfg, target_df, smooth=False)
        labelled = splits.label_pairs(pairs, spec, time_col='next_createtime')
        print(splits.summarise(labelled))

        train = splits.training_rows(labelled)
        seen = 0
        for traj in splits.TRAJ_SPLITS:
            for time in splits.TIME_SPLITS:
                name = splits.scenario_name(traj, time)
                cell = splits.select(labelled, traj, time)
                splits.check_leakage(train, cell, name)
                assert len(cell) > 0, f'{name} is empty'
                seen += len(cell)
        assert seen == len(labelled), (seen, len(labelled))
        print(f'all 6 scenario cells populated and leak-free ({seen} pairs)')

        # the cache must round-trip to the identical latent
        again = nnp.load_latent_df(cfg, spec)
        a = np.stack(target_df['x0'].to_numpy())
        b = np.stack(again['x0'].to_numpy())
        assert np.array_equal(a, b), 'cached latents differ from the fitted ones'
        print('latent cache round-trips exactly')

        # the momentum baseline must actually carry data. concat_arr hides a
        # null component as NaN inside a non-null array, so this checks the
        # plumbing rather than the formula -- the formula passed while every
        # scored batch was silently feeding NaN.
        assert 'has_prev' in pairs.columns
        kept = pairs.filter(pl.col('has_prev'))
        assert 0 < len(kept) < len(pairs), (len(kept), len(pairs))
        a = np.stack(kept['x0'].to_numpy())
        b = np.stack(kept['xm1'].to_numpy())
        assert np.isfinite(b).all(), 'NaN predecessor survived the has_prev filter'
        assert np.abs(a - b).mean() > 1e-9, 'predecessor is identical to the state'
        print(f'momentum inputs: {len(kept)}/{len(pairs)} pairs have a finite '
              f'predecessor, mean |x0-xm1|={np.abs(a - b).mean():.4f}')

        # df_to_data bounds its own memory; the loader promotes back to float64,
        # which is the dtype evaluate_pairs must match
        assert jax.config.jax_enable_x64, 'latent_gp should have enabled x64'
        sample = nnp.df_to_data(pairs.head(4))[0][0]
        for k in ('t0', 'x0', 't1', 'x1'):
            assert np.asarray(sample[k]).dtype == np.float32, (k, sample[k])
        print('df_to_data emits float32; loader promotes to', nnp.SOLVE_DTYPE.__name__)

        # apply_split's nested branch must agree with selecting the cell directly
        for scenario in ('val_out', 'test_out', 'val_in'):
            tr, cell = nnp.apply_split(pairs, 'nested', 0.8, spec=spec, scenario=scenario)
            want = splits.select(labelled, *scenario.rsplit('_', 1))
            assert len(cell) == len(want) and len(tr) == len(train), scenario
        print('apply_split nested branch matches direct scenario selection')

        # the landscape model must be written under the fit it was trained on
        assert nnp.run_dir(cfg).startswith(latent_space.latent_dir(cfg) + os.sep)
        assert os.path.exists(os.path.join(latent_space.latent_dir(cfg),
                                           'latents.parquet.zstd'))

        # run_dir must separate configurations that change what the model sees,
        # landscape hyperparameters included -- trials sharing a fit used to
        # share a directory and overwrite each other's checkpoints
        d1 = nnp.run_dir(cfg)
        cfg.sigma = 0.4
        d2 = nnp.run_dir(cfg)
        assert os.path.dirname(d1) == os.path.dirname(d2), (d1, d2)
        cfg.latents.n_fast = 2
        d3 = nnp.run_dir(cfg)
        cfg.split.holdout_days = 91
        d4 = nnp.run_dir(cfg)
        assert len({d1, d2, d3, d4}) == 4, (d1, d2, d3, d4)
        print('run_dir separates latent, split and landscape configurations')

        check_platform_run(td, cells, os.path.join(td, 'out'))

    print('displacement decomposition:')
    check_displacement_identity()
    print('per-dimension gain:')
    check_per_dimension_gain()
    print('\nnested-split pipeline smoke test passed')


if __name__ == '__main__':
    main()
