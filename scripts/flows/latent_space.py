"""The representation the analysis scripts read: trajectories plus loadings.

`cfg.latents.method` picks between them. 'gpfa' fits the latent-GP factor
model here, inside the split boundary, exactly as the landscape model does --
the loadings come out of that same fit, so they describe the axes the
trajectories actually move along. The precomputed methods read coords and a
component matrix written over the whole dataset.

With `cfg.latents.ref_cells_path` set the fit is a projection: the trajectories
come from a second aggregate -- one per platform handle rather than one per
seed -- and the axes from the pooled fit over the first. That is what lets a
platform-specific landscape be read against the pooled one; see latent_gp.

Loadings are returned (K, J) with the target names indexing the columns, which
is how PCA components arrive and so what code written against them expects.
"""

import datetime
import logging
import os

import numpy as np
import polars as pl

import splits
from latent_gp import (LatentConfig, build_latents, build_loadings, coord_cols,
                       drop_prior_dominated,
                       fit_dir, loading_matrix, reference)
from latent_gp import cells as gp_cells

logger = logging.getLogger(__name__)

OUT_ROOT = './out'


def latent_root(cfg):
    """Everything a configuration produces lives under here, fit first."""
    return os.path.join(cfg.get('out_dir', OUT_ROOT),
                        os.path.basename(cfg.trend_path.rstrip('/')))


def latent_dir(cfg):
    """Where this configuration's representation is cached."""
    root = latent_root(cfg)
    if cfg.latents.method == 'gpfa':
        return fit_dir(root, LatentConfig.from_cfg(cfg),
                       splits.SplitSpec.from_cfg(cfg))
    dims = '_'.join(str(d) for d in range(cfg.n_dims))
    return os.path.join(root, f'dims_{dims}_rm{cfg.rolling_mean_window}_'
                              f'{splits.SplitSpec.from_cfg(cfg).tag}')


# One name whatever the dimensionality, so nothing downstream hard-codes it
COORD = 'coord'

# The precomputed coords span the whole history; the analysis starts here. A
# gpfa aggregate is built over its own period, so applying this to it again
# would be a second, hidden opinion about the study window.
PRECOMPUTED_START = datetime.datetime(2022, 1, 1)

# Targets a dimension is named from, which is what a description reads.
TOP_TARGETS = 10


def load(cfg, smooth=True, spec=None):
    """Returns (trajectories, (K, J) loadings, target names).

    The gpfa latent is smooth in time by construction, so `smooth` only ever
    applies to the precomputed coords -- averaging the latent again would just
    widen the window each point already covers.

    `spec` overrides the split cfg implies, for a fold at a rolled-back origin.
    """
    if cfg.latents.method == 'gpfa':
        return _gpfa(cfg, spec)
    return _precomputed(cfg, smooth)


def name(cfg):
    """Filename prefix for artifacts describing the chosen representation.

    Keeps a gpfa fit's dimension labels off the precomputed method's, whose
    axes are a different basis over a different target set.
    """
    return 'gpfa' if cfg.latents.method == 'gpfa' else cfg.dim_reduction_method


def dimension_labels_path(cfg):
    """Where the labels naming this representation's dimensions live.

    Beside the fit for a gpfa latent: `name` alone does not tell two fits
    apart, and a shared file captions one fit's axes with another's names. The
    precomputed methods keep theirs under the trend, where the basis is fixed
    per method.
    """
    if cfg.latents.method == 'gpfa':
        return os.path.join(latent_dir(cfg), 'dimension_labels.json')
    return os.path.join(cfg.trend_path, f'{name(cfg)}_dimension_labels.json')


def target_volumes(cfg, targets):
    """Posts behind each target, for weighting a loading ranking.

    The fit solves a weighted least squares per target against a fixed ridge,
    so a low-volume target's loading is large when it is poorly constrained as
    readily as when it carries the axis, and ranking on magnitude alone puts
    the two side by side. None where the representation reports no volume.
    """
    if cfg.latents.method != 'gpfa':
        return None
    path = LatentConfig.from_cfg(cfg).cells_path
    vol = pl.scan_parquet(path).group_by('target') \
        .agg(pl.col('n').sum().alias('v')).collect(engine='streaming')
    lookup = dict(zip(vol['target'].to_list(), vol['v'].to_list()))
    return np.array([lookup.get(t, 0.0) for t in targets], dtype=float)


def n_moving_dims(cfg):
    """Dimensions that actually drift, which is all a mover ranking can use.

    Under the gpfa prior only the first n_fast dimensions have drifting
    dynamics; the rest are frozen per trajectory, so their movement is zero by
    construction rather than by measurement.
    """
    if cfg.latents.method != 'gpfa':
        return cfg.n_dims
    return min(cfg.latents.n_fast, cfg.n_dims)


def ranking_quality(components, volumes, n_top=TOP_TARGETS):
    """How readable the per-dimension target table is, under both rankings.

    A dimension is named from the targets that top it, so the table is only
    interpretable if a reader recognises those targets and if the dimensions do
    not all name themselves after the same few. `prevalence` is their mean
    volume percentile; `distinctness` is the share of the pooled top-n that is
    not shared between dimensions, 1 when every dimension names a disjoint set
    and 0 when they all name the same one. `score` requires both.

    Both rankings are scored from the one fit because the choice between them
    changes nothing the model is scored on -- only the order the table is read
    in -- so there is no reason to spend a trial on each.
    """
    if volumes is None:
        return {}
    order = np.argsort(np.argsort(volumes))
    percentile = order / max(len(volumes) - 1, 1)
    out = {}
    for name, w in (('unweighted', None), ('by_volume', volumes)):
        out.update({f'{name}/{k}': v for k, v in
                    _quality(components, percentile, n_top, w).items()})
    return out


def _quality(components, percentile, n_top, weights):
    score = np.abs(components)
    if weights is not None:
        score = score * np.sqrt(weights)
    n_top = min(n_top, score.shape[1])
    top = np.argsort(score, axis=1)[:, -n_top:]
    k = top.shape[0]
    spread = (len(np.unique(top)) - n_top) / (n_top * k - n_top) if k > 1 else 1.0
    prevalence = float(percentile[top].mean())
    return {'prevalence': prevalence, 'distinctness': float(spread),
            'score': float(np.sqrt(max(prevalence, 0.0) * max(spread, 0.0)))}


def axis_prefix(cfg):
    """What to call one axis of the representation.

    A gpfa axis is a latent dimension, which the write-up abbreviates LD; only
    the precomputed methods produce principal components.
    """
    return 'LD' if cfg.latents.method == 'gpfa' else 'PC'


def dimension_priors(cfg):
    """The GP prior each dimension was fitted under, as (kind, tau).

    Read back through the same call the fit used, so the two cannot disagree
    about which dimensions were the drifting ones. `tau` is in days -- the fit's
    grid spacing is -- and is None for a frozen dimension.
    """
    from latent_gp import fit as fit_mod

    lcfg = LatentConfig.from_cfg(cfg)
    comps = fit_mod.prior_components(lcfg.n_dims, lcfg.n_fast, lcfg.fast_tau,
                                     lcfg.slow_kind, lcfg.slow_tau,
                                     fast_kind=lcfg.fast_kind)
    # a homogeneous mix comes back as one shared list, a genuine mix per dimension
    per_dim = comps if (comps and isinstance(comps[0], (list, tuple))) \
        else [comps] * lcfg.n_dims
    return [(c[0]['kind'], c[0].get('tau')) for c in per_dim]


def dimension_variance_share(cfg):
    """Each dimension's share of the variance it drives in the linear predictor.

    PCA orders its axes by explained variance, so the index carries that
    meaning; a GPFA fit does not order its axes at all. This is the closest
    analogue: dimension k shifts a target's score by W[k] z_k, so it contributes
    var(z_k) ||W[k]||^2. Cross-dimension covariance is left out, so the shares
    partition the total only approximately.
    """
    if cfg.latents.method != 'gpfa':
        return None
    target_df, components, _ = load(cfg)
    z = np.stack(target_df[COORD].to_numpy())
    contrib = z.var(axis=0) * (np.asarray(components) ** 2).sum(axis=1)
    total = contrib.sum()
    return contrib / total if total > 0 else contrib


def dimension_quality(cfg, n_top=TOP_TARGETS):
    """ranking_quality for the fit cfg names, off the loadings it cached."""
    if cfg.latents.method != 'gpfa':
        return {}
    lcfg, spec, kw = fit_args(cfg)
    components, targets = loading_matrix(build_loadings(lcfg, spec, **kw))
    return ranking_quality(components, target_volumes(cfg, targets), n_top)


def traj_col(cfg):
    """The seed field one trajectory is keyed by.

    A gpfa fit aggregates by cfg.latents.traj_col, which need not be the
    cfg.filter_column the precomputed coords were built over; joining a
    trajectory back to its posts on the wrong one matches nothing at all.
    """
    if cfg.latents.method != 'gpfa':
        return cfg.filter_column
    return LatentConfig.from_cfg(cfg).traj_col


def keep_platform(cfg, df):
    """Trajectories belonging to cfg.platform, or all of them when 'all'.

    A trajectory names its platform only when it is one account rather than one
    person, so asking for a platform of a seed-keyed run selects nothing at all;
    that is a misconfiguration, not an empty result.
    """
    if cfg.platform == 'all':
        return df
    kept = df.filter(pl.col('filter_value').cast(pl.String)
                     .str.to_lowercase().str.contains(f'-{cfg.platform}-'))
    if len(kept) == 0:
        raise ValueError(
            f'no trajectory names platform {cfg.platform!r}: a platform run '
            'needs a handle-keyed representation, not a seed-keyed one')
    return kept


def split_key(cfg):
    """Trajectory id -> the id whose hash decides its train/val/test cell.

    None unless the run is keyed by platform handle, where it maps each handle
    back to its seed: the axes were fitted over whole seeds, so a handle has to
    land on the same side of the boundary as the rest of that person.
    """
    if cfg.latents.method != 'gpfa':
        return None
    lcfg = LatentConfig.from_cfg(cfg)
    return gp_cells.traj_seeds(lcfg.cells_path, lcfg.traj_col)


def fit_args(cfg, lcfg=None, spec=None):
    """(lcfg, spec, keyword arguments) every build_* call for this cfg takes.

    `lcfg` and `spec` override the ones cfg implies, for a caller that wants the
    same fit under one changed field or at a rolled-back origin.
    """
    spec = splits.SplitSpec.from_cfg(cfg) if spec is None else spec
    lcfg = LatentConfig.from_cfg(cfg) if lcfg is None else lcfg
    kw = dict(cache_root=latent_root(cfg), log=logger.info,
              seed_split=splits.seed_split(
                  gp_cells.seed_names(lcfg.cells_path, lcfg.traj_col), spec,
                  key=split_key(cfg)))
    ref = reference(lcfg)
    if ref is not None:
        kw['ref_seed_split'] = splits.seed_split(
            gp_cells.seed_names(ref.cells_path), spec)
    return lcfg, spec, kw


def _gpfa(cfg, spec=None):
    lcfg, spec, kw = fit_args(cfg, spec=spec)

    coord, causal, sd = coord_cols(lcfg.n_dims)
    state = causal if cfg.latents.causal_state else coord
    # the fast block is where the motion is, so a seed whose fast dims are
    # prior rather than measurement contributes the prior's dynamics and not
    # the data's -- the same filter the stationarity tests apply
    fast = list(range(min(lcfg.n_fast, lcfg.n_dims)))
    target_df = drop_prior_dominated(
        build_latents(lcfg, spec, **kw), sd, fast,
        cfg.get('max_posterior_sd', 0.8), log=logger.info) \
        .select(['createtime', 'filter_value', pl.col(state).alias(COORD)]) \
        .sort(['filter_value', 'createtime'])

    components, targets = loading_matrix(build_loadings(lcfg, spec, **kw))
    return target_df, components, targets


def _precomputed(cfg, smooth):
    trend_path = cfg.trend_path
    target_df = pl.read_parquet(
        os.path.join(trend_path, f'{cfg.dim_reduction_method}_coords.parquet.zstd'))
    coord_col = [c for c in target_df.columns if c.startswith('coord_')][0]
    n_dims = target_df.schema[coord_col].shape[0]

    target_df = target_df \
        .filter(pl.col('createtime') >= PRECOMPUTED_START) \
        .select(['createtime', 'filter_value', pl.col(coord_col).alias(COORD)]) \
        .sort(['filter_value', 'createtime'])
    if smooth:
        target_df = target_df \
            .with_columns([pl.col(COORD).arr.get(i).alias(f'dim_{i}')
                           for i in range(n_dims)]) \
            .rolling('createtime', period=f'{cfg.rolling_mean_window}d',
                     group_by='filter_value') \
            .agg([pl.col(f'dim_{i}').mean() for i in range(n_dims)]) \
            .with_columns(pl.concat_arr([f'dim_{i}' for i in range(n_dims)]).alias(COORD)) \
            .drop_nulls(COORD) \
            .select(['createtime', 'filter_value', COORD])

    component_df = pl.read_parquet(
        os.path.join(trend_path, f'{cfg.dim_reduction_method}_metadata.parquet.zstd'))
    if cfg.dim_reduction_method == 'sfa':
        components = component_df.filter(pl.col('n_components') == n_dims)['W'][0].to_numpy()
    elif cfg.dim_reduction_method in ['pca', 'ppca', 'pica']:
        components = np.stack(
            component_df.filter(pl.col('n_dims') == n_dims)['components'][0].to_numpy())
    else:
        raise ValueError(f'Unknown dim_reduction_method: {cfg.dim_reduction_method}')

    # these components are over the columns of the frame they were fitted on
    head = pl.read_parquet(
        os.path.join(trend_path, 'pivoted_and_imputed.parquet.zstd'), n_rows=1)
    targets = [c for c in head.columns if c not in ['createtime', 'filter_value']]
    assert len(targets) == components.shape[1]
    return target_df, components, targets
