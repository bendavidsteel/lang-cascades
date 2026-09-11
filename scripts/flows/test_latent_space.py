import numpy as np
from omegaconf import OmegaConf

import latent_space
from latent_space import ranking_quality


def _q(components, volumes, n_top=2):
    return ranking_quality(np.asarray(components, dtype=float),
                           np.asarray(volumes, dtype=float), n_top=n_top)


def test_dimensions_topped_by_the_same_targets_are_not_distinct():
    same = [[9.0, 9.0, 0.1, 0.1], [9.0, 9.0, 0.1, 0.1]]
    assert _q(same, [1, 2, 3, 4])['unweighted/distinctness'] == 0.0


def test_dimensions_topped_by_disjoint_targets_are_fully_distinct():
    apart = [[9.0, 9.0, 0.1, 0.1], [0.1, 0.1, 9.0, 9.0]]
    assert _q(apart, [1, 2, 3, 4])['unweighted/distinctness'] == 1.0


def test_prevalence_is_the_volume_percentile_of_the_targets_that_top_a_dim():
    # loadings pick the two rarest targets, so they sit at percentile 0 and 1/3
    rare = [[9.0, 9.0, 0.1, 0.1], [9.0, 9.0, 0.1, 0.1]]
    got = _q(rare, [1, 2, 3, 4])['unweighted/prevalence']
    assert got == np.mean([0.0, 1 / 3])


def test_volume_weighting_lifts_prevalence_when_the_loadings_favour_rare_targets():
    # dimension 1's largest loading is on the rarest target; a common one is
    # close enough behind that sqrt(volume) reorders the pair
    components = [[9.0, 8.0, 0.1, 0.1], [0.1, 0.1, 9.0, 8.0]]
    volumes = [1, 1000, 1, 1000]
    q = _q(components, volumes, n_top=1)
    assert q['by_volume/prevalence'] > q['unweighted/prevalence']


def test_score_is_zero_when_either_axis_collapses():
    same = [[9.0, 9.0, 0.1, 0.1], [9.0, 9.0, 0.1, 0.1]]
    assert _q(same, [1, 2, 3, 4])['unweighted/score'] == 0.0


def test_a_single_dimension_has_nothing_to_be_distinct_from():
    assert _q([[9.0, 9.0, 0.1, 0.1]], [1, 2, 3, 4])['unweighted/distinctness'] == 1.0


def test_no_volumes_means_no_metrics():
    assert ranking_quality(np.ones((2, 4)), None) == {}


def _cfg(**latents):
    """Enough of a config for the path helpers."""
    base = {'method': 'gpfa', 'cells_path': 'cells.parquet.zstd', 'n_fast': 2,
            'fast_tau': 10.0, 'fast_kind': 'ou', 'slow_kind': 'const',
            'slow_tau': 2560.0, 'bin_factor': 8, 'interp_days': 2.0, 'rho': 0.0,
            'iters': 25, 'infer_iters': 15, 'obs_model': 'hard',
            'obs_temperature': 1.0, 'prob_resolution': 6, 'prob_floor': 0.01,
            'calibration_path': '', 'seed': 0}
    base.update(latents)
    return OmegaConf.create({
        'n_dims': 6, 'min_target_volume': 400, 'trend_path': './trend',
        'dim_reduction_method': 'ppca', 'rolling_mean_window': 292,
        'out_dir': './out', 'latents': base,
        'split': {'holdout_days': 365, 'origin_offset_days': 0,
                  'train_frac': 0.7, 'val_frac': 0.1, 'seed': 42}})


def test_two_gpfa_fits_do_not_share_one_dimension_labels_file(tmp_path):
    # 'gpfa' alone does not tell two fits apart, so a shared file would caption
    # one fit's axes with the other's names
    cells = tmp_path / 'cells.parquet.zstd'
    cells.write_bytes(b'cells')
    a = latent_space.dimension_labels_path(_cfg(n_fast=1, cells_path=str(cells)))
    b = latent_space.dimension_labels_path(_cfg(n_fast=2, cells_path=str(cells)))
    assert a != b
    assert a.endswith('dimension_labels.json')


def test_a_precomputed_method_keeps_its_labels_under_the_trend():
    got = latent_space.dimension_labels_path(_cfg(method='ppca'))
    assert got == './trend/ppca_dimension_labels.json'
