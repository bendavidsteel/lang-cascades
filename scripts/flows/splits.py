"""Nested trajectory x time split for landscape-model evaluation.

Trajectories are split train/val/test by a hash of their id rather than by
position in a shuffled list, so the assignment survives any upstream filtering
(platform, minimum volume) instead of shifting when the seed set changes. A run
whose trajectories are platform handles passes a `key` mapping each handle to
its seed, so both sides of the boundary agree on whole people.

Time is split by a holdout window. A pair counts as out-of-time when its
*target* t1 falls in the holdout, which is what keeps every training target
inside the training period even for long horizons.

The window sits at the end of the data by default. `origin_offset_days` moves
its right edge back and discards everything after it, so the fit cannot see
past the window it is scored on -- which is what makes a rolling-origin
evaluation a forecast at every origin rather than only at the last one.

Crossing the two gives the scenarios the evaluation reports separately:

    train x in    trajectory seen, time seen     -- training data
    val/test x in trajectory unseen, time seen   -- imputation
    train x out   trajectory seen, time unseen   -- forecast
    val/test x out both unseen                   -- the real generalisation test

Hyperparameters are selected on val x out and reported once on test x out;
selecting on test x out would spend the only clean estimate we have.
"""

import dataclasses
import datetime
import hashlib

import numpy as np
import polars as pl

TRAJ_SPLITS = ('train', 'val', 'test')
# Canonical name for a pair's target timestamp; the 1-step and horizon pair
# builders each call it something else, and time_split keys off it.
TARGET_TIME = 'target_time'
TIME_SPLITS = ('in', 'out')

# Holdout window lengths, in days. Reported together they describe how
# predictive performance decays as the unseen period lengthens.
HOLDOUT_DAYS = (91, 182, 365, 730)

# Origins for a rolling evaluation, as days trimmed from the end of the data.
# Half-year strides against a ~4.5 year span, so consecutive windows overlap:
# the folds are not independent, but the alternative is two of them.
ORIGIN_OFFSET_DAYS = (0, 182, 365, 547, 730)


@dataclasses.dataclass(frozen=True)
class SplitSpec:
    """Everything that determines which rows a model is allowed to see."""

    holdout_days: int
    origin_offset_days: int = 0
    train_frac: float = 0.70
    val_frac: float = 0.10
    seed: int = 42

    def __post_init__(self):
        if not 0 < self.train_frac < 1 or not 0 <= self.val_frac < 1:
            raise ValueError('fractions must lie in (0, 1)')
        if self.train_frac + self.val_frac >= 1:
            raise ValueError('train_frac + val_frac leaves no test trajectories')
        if self.holdout_days <= 0:
            raise ValueError('holdout_days must be positive')
        if self.origin_offset_days < 0:
            raise ValueError('origin_offset_days must be non-negative')

    @classmethod
    def from_cfg(cls, cfg):
        return cls(holdout_days=cfg.split.holdout_days,
                   origin_offset_days=cfg.split.origin_offset_days,
                   train_frac=cfg.split.train_frac,
                   val_frac=cfg.split.val_frac,
                   seed=cfg.split.seed)

    @property
    def test_frac(self):
        return 1.0 - self.train_frac - self.val_frac

    @property
    def tag(self):
        """Short identifier for run directories and cache keys."""
        # only extended when rolling, so existing runs and the latent cache
        # they share keep their key
        origin = f'_o{self.origin_offset_days}' if self.origin_offset_days else ''
        return (f'h{self.holdout_days}{origin}_tr{self.train_frac:g}'
                f'_va{self.val_frac:g}_s{self.seed}')


def _unit_hash(values, seed):
    """Deterministic uniform [0, 1) draw per value, independent of order."""
    out = np.empty(len(values), dtype=np.float64)
    for i, v in enumerate(values):
        digest = hashlib.blake2b(f'{seed}:{v}'.encode(), digest_size=8).digest()
        out[i] = int.from_bytes(digest, 'big') / 2.0 ** 64
    return out


def assign_trajectory_split(filter_values, spec, key=None):
    """Map each trajectory id to 'train' / 'val' / 'test'.

    `key` maps an id to the id whose hash decides it, which is how a run over
    platform handles puts every handle of a seed in the same cell as the seed.
    """
    values = sorted(set(str(v) for v in filter_values))
    hashed = [key.get(v, v) for v in values] if key else values
    u = _unit_hash(hashed, spec.seed)
    label = np.where(
        u < spec.train_frac, 'train',
        np.where(u < spec.train_frac + spec.val_frac, 'val', 'test'))
    return pl.DataFrame({'filter_value': values, 'traj_split': label})


def seed_split(filter_values, spec, key=None):
    """`assign_trajectory_split` as the id -> split dict the latent fit takes."""
    traj = assign_trajectory_split(filter_values, spec, key)
    return dict(zip(traj['filter_value'].to_list(), traj['traj_split'].to_list()))


def time_window(times, spec):
    """The holdout window, as (cutoff, end).

    `end` is where the data is treated as stopping. At the default offset that
    is the last observation, so the window is the tail of the data.
    """
    last = max(times) if not isinstance(times, pl.Series) else times.max()
    end = last - datetime.timedelta(days=spec.origin_offset_days)
    return end - datetime.timedelta(days=spec.holdout_days), end


def time_cutoff(times, spec):
    """Start of the holdout window."""
    return time_window(times, spec)[0]


def label_pairs(pairs, spec, cutoff=None, time_col='future_createtime', end=None,
                key=None):
    """Stamp traj_split and time_split onto a (t0, x0, t1, x1) frame.

    time_split keys off the pair's target time, so a pair that starts before
    the cutoff and lands after it is out-of-time. Pairs landing past the
    window's end are dropped rather than labelled: at a rolled-back origin they
    are the future this fold is not allowed to be scored on.
    """
    if time_col not in pairs.columns:
        raise ValueError(f'{time_col!r} missing; pair builders must keep the '
                         'target timestamp to assign time_split on t1')
    auto_cutoff, auto_end = time_window(pairs[time_col], spec)
    cutoff = auto_cutoff if cutoff is None else cutoff
    end = auto_end if end is None else end

    traj = assign_trajectory_split(pairs['filter_value'].unique().to_list(), spec, key)
    return pairs.filter(pl.col(time_col) <= end) \
        .with_columns(pl.col('filter_value').cast(pl.String)) \
        .join(traj, on='filter_value', how='left') \
        .with_columns(
            pl.col(time_col).alias(TARGET_TIME),
            pl.when(pl.col(time_col) < cutoff).then(pl.lit('in'))
              .otherwise(pl.lit('out')).alias('time_split'),
        )


def scenario_name(traj_split, time_split):
    return f'{traj_split}_{time_split}'


def select(df, traj_split, time_split):
    """Rows for one scenario cell; traj_split/time_split may be tuples."""
    traj = (traj_split,) if isinstance(traj_split, str) else tuple(traj_split)
    time = (time_split,) if isinstance(time_split, str) else tuple(time_split)
    return df.filter(pl.col('traj_split').is_in(traj)
                     & pl.col('time_split').is_in(time))


def training_rows(df):
    """The only rows a landscape model may fit on."""
    return select(df, 'train', 'in')


def check_leakage(train_df, eval_df, scenario, time_col=TARGET_TIME):
    """Assert an evaluation set is disjoint from training in the intended way.

    Raises rather than warns: a silent leak here invalidates every number the
    evaluation produces.
    """
    traj_split, time_split = scenario.rsplit('_', 1)
    if len(eval_df) == 0:
        return

    got_traj = set(eval_df['traj_split'].unique().to_list())
    got_time = set(eval_df['time_split'].unique().to_list())
    if got_traj != {traj_split} or got_time != {time_split}:
        raise AssertionError(
            f'{scenario}: mixed labels {sorted(got_traj)} x {sorted(got_time)}')

    if traj_split != 'train':
        shared = set(eval_df['filter_value'].unique().to_list()) \
            & set(train_df['filter_value'].unique().to_list())
        if shared:
            raise AssertionError(
                f'{scenario}: {len(shared)} trajectories also in train, '
                f'e.g. {sorted(shared)[:3]}')

    if time_split == 'out':
        latest_train = train_df[time_col].max()
        earliest_eval = eval_df[time_col].min()
        # no training pair at this horizon leaves nothing for an evaluation
        # pair to precede; a rolled-back origin runs out of in-time room at the
        # long horizons before it runs out of out-of-time ones
        if latest_train is None:
            return
        if earliest_eval <= latest_train:
            raise AssertionError(
                f'{scenario}: target at {earliest_eval} precedes the last '
                f'training target {latest_train}')


def summarise(df):
    """Pair counts and trajectory counts per scenario cell."""
    return df.group_by(['traj_split', 'time_split']).agg(
        pl.len().alias('n_pairs'),
        pl.col('filter_value').n_unique().alias('n_trajectories'),
    ).sort(['traj_split', 'time_split'])
