"""Cell-level stance aggregate: loading, holdout masks and packing.

A cell is one (trajectory, target, time bin). The three ordinal category counts
are determined exactly by (n, s_sum, s2_sum) -- under hard labels as integers,
and under the classifier's probabilities as expected counts, since the same
three moments carry either. So switching between them never rebuilds the
aggregate.

A trajectory is a seed by default. An aggregate keyed by `PlatformHandleID`
splits each seed into one trajectory per platform instead, which is what the
platform-specific landscapes are fitted over; `traj_col` picks between them and
the rest of the pipeline is unchanged.

Expected counts at a temperature need the probabilities themselves, and take
them from the `q<i>` columns: the count of posts falling on lattice point i of
the simplex (see probs.py). They are optional, and absent for an aggregate
built from labels alone.
"""

import datetime
import re

import numpy as np
import polars as pl
import jax.numpy as jnp

from . import _jax  # noqa: F401  -- x64 before the first array

from . import probs as probs_mod

ARCH_RE = re.compile(r'^q(\d+)$')

# Trajectory key of an aggregate: one row per seed, or one per platform handle.
SEED_COL = 'SeedName'
HANDLE_COL = 'PlatformHandleID'


def arch_cols(columns):
    """Lattice count columns, in lattice-index order."""
    hits = [(int(m.group(1)), c) for c in columns for m in [ARCH_RE.match(c)] if m]
    return [c for _, c in sorted(hits)]


def stored_resolution(columns):
    """Resolution the lattice counts were built at, or None if there are none."""
    n = len(arch_cols(columns))
    if not n:
        return None
    for r in range(1, 33):
        if probs_mod.n_archetypes(r) == n:
            return r
    raise ValueError(f'{n} lattice columns match no simplex resolution')


def lattice(df, resolution):
    """Per-cell lattice counts, coarsening the stored grid if a coarser one is asked.

    Coarsening rounds an already-rounded point, so it does not match binning the
    raw probabilities at the coarse grid. Good enough to test how much the
    resolution matters, not for the fit the result is reported from.
    """
    cols = arch_cols(df.columns)
    if not cols:
        raise ValueError('aggregate carries no lattice counts: rebuild it from '
                         'the classifier probabilities')
    stored = stored_resolution(df.columns)
    counts = df.select(cols).to_numpy().astype(np.float64)
    if resolution == stored:
        return counts
    if resolution > stored:
        raise ValueError(f'resolution {resolution} is finer than the stored {stored}')
    tgt = probs_mod.assign(probs_mod.simplex_grid(stored), resolution)
    acc = np.zeros((probs_mod.n_archetypes(resolution), len(counts)))
    np.add.at(acc, tgt, counts.T)
    return acc.T


def load(path, bin_factor, seeds=None, min_target_volume=None, targets=None,
         traj_col=SEED_COL):
    """Re-bin the 2-day aggregate onto the latent grid and index seeds/targets.

    `seeds` restricts the seed set before indices are assigned, so a subset run
    still produces contiguous indices.

    `targets` pins the target index instead of reading it off the data, which is
    what lets a second aggregate be scored against loadings fitted on the first:
    column j has to mean the same target in both. Targets it names that this
    aggregate never mentions keep their column and simply carry no cells.

    Streamed: read eagerly, the full aggregate costs five times the memory of
    the frame it returns, and two fits sharing a machine will not fit.
    """
    lf = pl.scan_parquet(path)
    if seeds is not None:
        lf = lf.filter(pl.col(traj_col).is_in(list(seeds)))
    if targets is not None:
        lf = lf.filter(pl.col('target').is_in(list(targets)))
    elif min_target_volume:
        keep = (lf.group_by('target').agg(pl.col('n').sum().alias('v'))
                  .filter(pl.col('v') >= min_target_volume).select('target'))
        lf = lf.join(keep, on='target', how='inner')

    def one(expr):
        return lf.select(expr).collect(engine='streaming')

    seed_names = one(pl.col(traj_col).unique().sort())[traj_col].to_list()
    targets = (list(targets) if targets is not None
               else one(pl.col('target').unique().sort())['target'].to_list())
    t0 = one(pl.col('bin').min())['bin'][0]

    arch = arch_cols(lf.collect_schema().names())
    lf = lf.with_columns([
        pl.col(traj_col).replace_strict({s: i for i, s in enumerate(seed_names)}).alias('m'),
        pl.col('target').replace_strict({s: i for i, s in enumerate(targets)}).alias('j'),
        ((pl.col('bin') - pl.lit(t0)).dt.total_days() // (2 * bin_factor))
            .cast(pl.Int64).alias('t'),
    ])
    # group_by does not preserve order, and row order sets the reduction order
    # of the scatters downstream -- sort so repeated runs agree bit for bit
    df = lf.group_by(['m', 'j', 't']).agg(
        [pl.col('n').sum(), pl.col('s_sum').sum(), pl.col('s2_sum').sum()]
        + [pl.col(c).sum() for c in arch]
    ).sort(['m', 'j', 't']).with_columns([
        pl.col('n').cast(pl.Float64),
        ((pl.col('s2_sum') + pl.col('s_sum')) / 2).alias('n_pos'),
        ((pl.col('s2_sum') - pl.col('s_sum')) / 2).alias('n_neg'),
        (pl.col('n').cast(pl.Float64) - pl.col('s2_sum')).alias('n_neu'),
    ]).collect(engine='streaming')
    meta = dict(seeds=seed_names, targets=targets, t0=t0,
                M=len(seed_names), J=len(targets), T=int(df['t'].max()) + 1,
                dt=2.0 * bin_factor, arch=arch)
    return df, meta


def bin_times(meta):
    """Datetime at the centre of each latent bin.

    The centre, not the left edge, so a bin's time label is unbiased with
    respect to the posts it aggregates.
    """
    half = datetime.timedelta(days=meta['dt'] / 2)
    return [meta['t0'] + datetime.timedelta(days=meta['dt'] * t) + half
            for t in range(meta['T'])]


def cutoff_bin(meta, holdout_days, origin_offset_days=0):
    """First bin index inside the holdout window, and the cutoff date.

    Bins are assigned by their centre, so a bin straddling the cutoff falls on
    whichever side holds most of its posts.
    """
    times = bin_times(meta)
    cutoff = times[-1] - datetime.timedelta(days=origin_offset_days + holdout_days)
    for t, ts in enumerate(times):
        if ts >= cutoff:
            return t, cutoff
    return meta['T'], cutoff


def window_end_bin(meta, origin_offset_days=0):
    """One past the last bin inside the holdout window."""
    times = bin_times(meta)
    end = times[-1] - datetime.timedelta(days=origin_offset_days)
    for t, ts in enumerate(times):
        if ts > end:
            return t
    return meta['T']


def truncate(df, meta, t_end):
    """Drop bins from t_end on and shrink the grid to match.

    A rolled-back origin has to look like the data stopping there. Leaving the
    later bins in place would let the smoother carry them back into states
    inside the window, which is the leak the origin offset exists to prevent.
    """
    if t_end >= meta['T']:
        return df, meta
    return df.filter(pl.col('t') < t_end), dict(meta, T=int(t_end))


def holdout_masks(df, meta, frac=0.15, seed=0):
    """Two contiguous holdouts per seed: a forecast tail and an interior block.

    Contiguous blocks are what identify a timescale -- random per-cell holdout
    cannot, since neighbouring bins predict a held-out cell from the level
    alone. The tail tests extrapolation, the interior block interpolation.
    """
    rng = np.random.default_rng(seed)
    span = df.group_by('m').agg(pl.col('t').min().alias('lo'), pl.col('t').max().alias('hi'))
    lo = dict(zip(span['m'].to_list(), span['lo'].to_list()))
    hi = dict(zip(span['m'].to_list(), span['hi'].to_list()))

    fc_lo, in_lo, in_hi = {}, {}, {}
    for m in range(meta['M']):
        w = max(int(frac * (hi[m] - lo[m] + 1)), 2)
        fc_lo[m] = hi[m] - w + 1
        room = fc_lo[m] - lo[m] - 2 * w                 # keep a gap either side
        if room > 0:
            in_lo[m] = lo[m] + w + int(rng.integers(0, room))
            in_hi[m] = in_lo[m] + w
        else:
            in_lo[m] = in_hi[m] = -1

    m_arr = df['m'].to_numpy(); t_arr = df['t'].to_numpy()
    fc = np.array([t >= fc_lo[m] for m, t in zip(m_arr, t_arr)])
    inte = np.array([in_lo[m] <= t < in_hi[m] for m, t in zip(m_arr, t_arr)])
    return fc, inte


def deflate(df, rho):
    """Effective sample size per cell: posts in a burst are not independent.

    n_eff = n / (1 + (n-1) rho) is the classic design-effect correction; rho=0
    recovers the iid-per-post model.
    """
    if rho <= 0:
        return df.with_columns(pl.col('n').alias('n_eff'))
    return df.with_columns(
        (pl.col('n') / (1.0 + (pl.col('n') - 1.0) * rho)).alias('n_eff'))


def pack(df, meta, n_arch=None):
    """Arrays for the E/M steps, with per-cell counts scaled to n_eff.

    `n_arch` is the (cells, lattice) count matrix for the soft-evidence
    likelihood, already aligned to `df`'s rows.
    """
    n = df['n'].to_numpy().astype(np.float64)
    ne = df['n_eff'].to_numpy().astype(np.float64)
    scale = ne / n
    out = {
        'j': jnp.asarray(df['j'].to_numpy()),
        'flat': jnp.asarray(df['m'].to_numpy() * meta['T'] + df['t'].to_numpy()),
        'n': jnp.asarray(ne),
        's': jnp.asarray(df['s_sum'].to_numpy() * scale),
        's2': jnp.asarray(df['s2_sum'].to_numpy() * scale),
        'n_pos': jnp.asarray(df['n_pos'].to_numpy() * scale),
        'n_neg': jnp.asarray(df['n_neg'].to_numpy() * scale),
        'n_neu': jnp.asarray(df['n_neu'].to_numpy() * scale),
        'M': meta['M'], 'J': meta['J'], 'T': meta['T'], 'N': float(ne.sum()),
    }
    if n_arch is not None:
        # left on the host: the site computation only ever reads a chunk at a
        # time
        out['n_arch'] = np.asarray(n_arch) * scale[:, None]
    return out


def eval_set(df):
    """Held-out cells scored on their observed mean, weighted by real counts."""
    return dict(
        m=df['m'].to_numpy(), j=df['j'].to_numpy(), t=df['t'].to_numpy(),
        mean=(df['s_sum'] / df['n']).to_numpy(),
        n=df['n'].to_numpy().astype(np.float64),
        n_pos=df['n_pos'].to_numpy(), n_neg=df['n_neg'].to_numpy(),
        n_neu=df['n_neu'].to_numpy(),
    )


def seed_names(path, traj_col=SEED_COL):
    """Trajectory ids present in a cell file, without loading the counts."""
    return (pl.scan_parquet(path).select(traj_col).unique()
              .collect()[traj_col].sort().to_list())


def traj_seeds(path, traj_col):
    """Trajectory id -> the seed it belongs to.

    A handle-keyed aggregate has to inherit its seed's train/val/test cell: the
    representation was fitted over whole seeds, so splitting one seed's handles
    across the boundary would hold a trajectory out of the landscape whose own
    posts had already shaped the axes it is scored in.
    """
    if traj_col == SEED_COL:
        return None
    got = (pl.scan_parquet(path).select([traj_col, SEED_COL]).unique()
             .collect(engine='streaming'))
    return dict(zip(got[traj_col].to_list(), got[SEED_COL].to_list()))
