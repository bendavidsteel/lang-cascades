"""Rolling-holdout evaluation for the landscape model.

Two scans, both of which refit the latent representation as well as the
landscape model for every run, because the split boundary moves:

  horizon   one origin at the end of the data, several window lengths. Says how
            performance decays as the unseen period lengthens.
  rolling   one window length at several origins. Says whether the number the
            fixed holdout reports is typical or a fluke of the last year of
            data, which is what a final evaluation has to answer.

The rolling scan is the expensive one -- a fold costs a full run -- so it is
for reporting a settled configuration, not for selecting one.

    python rolling_holdout.py --rolling
    python rolling_holdout.py --rolling --holdout-days 182 --offsets 0 182 365
    python rolling_holdout.py --windows 91 365 -- sigma=0.3 n_dims=6
    python rolling_holdout.py --rolling --plot-only
"""

import argparse
import os
import subprocess
import sys

import polars as pl

import splits

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
# train_in is the fit, not a test; the other five are what the design buys.
SCENARIOS = ['val_in', 'test_in', 'train_out', 'val_out', 'test_out']
LABELS = {
    'val_in': 'unseen person, seen time',
    'test_in': 'unseen person, seen time (test)',
    'train_out': 'seen person, unseen time',
    'val_out': 'both unseen (val)',
    'test_out': 'both unseen (test)',
}
# Headline metrics, in reporting order; whichever of them the runs recorded.
METRICS = ['ceiling_gain', 'skill_score', 'median_skill', 'frac_better',
           'direction_rho', 'n']


def run_window(holdout_days, offset, overrides):
    cmd = [sys.executable, os.path.join(HERE, 'nn_potential.py'),
           f'split.holdout_days={holdout_days}',
           f'split.origin_offset_days={offset}'] + list(overrides)
    print(f'\n=== holdout {holdout_days}d  origin -{offset}d ===\n{" ".join(cmd)}',
          flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def collect(out_root, prefix='step'):
    """Gather every scenario_metrics file written under out/."""
    name = f'scenario_metrics_{prefix}.parquet.zstd'
    paths = [os.path.join(r, name) for r, _, fs in os.walk(out_root) if name in fs]
    if not paths:
        raise SystemExit(f'no {name} under {out_root} -- run the windows first')
    df = pl.concat([pl.read_parquet(p) for p in paths], how='diagonal_relaxed')
    # runs predating the rolling origin wrote neither column, and every one of
    # them was an unrolled window
    for col, fill in (('origin_offset_days', 0), ('latent_tag', '')):
        if col not in df.columns:
            df = df.with_columns(pl.lit(fill).alias(col))
    return df.with_columns(pl.col('origin_offset_days').fill_null(0),
                           pl.col('latent_tag').fill_null(''))


def one_configuration(df, latent_tag=None):
    """Narrow to a single latent configuration.

    out/ accumulates every run ever made, and averaging skill over folds that
    used different hyperparameters would quietly invent a result.
    """
    if latent_tag:
        df = df.filter(pl.col('latent_tag') == latent_tag)
        if len(df) == 0:
            raise SystemExit(f'no runs with latent_tag {latent_tag}')
        return df
    tags = df['latent_tag'].unique().to_list()
    if len(tags) > 1:
        cols = [c for c in ('n_dims', 'n_fast', 'fast_tau', 'fast_kind', 'slow_kind')
                if c in df.columns]
        summary = df.group_by('latent_tag').agg(
            *[pl.col(c).unique().sort().alias(c) for c in cols],
            pl.col('origin_offset_days').unique().sort().alias('offsets'))
        with pl.Config(tbl_cols=-1, fmt_str_lengths=60):
            raise SystemExit(f'{len(tags)} latent configurations match:\n'
                             f'{summary.sort("latent_tag")}\n'
                             'pass --latent-tag to pick one '
                             "(runs predating the column show as '')")
    return df


def order_scenarios(df):
    rank = {s: i for i, s in enumerate(SCENARIOS)}
    return df.filter(pl.col('scenario').is_in(SCENARIOS)) \
        .with_columns(pl.col('scenario').replace_strict(rank).alias('_r')) \
        .sort('_r').drop('_r')


def fold_summary(df):
    """Mean and spread of each headline metric across the folds."""
    cols = [m for m in METRICS if m in df.columns]
    aggs = [pl.len().alias('folds')]
    for m in cols:
        aggs += [pl.col(m).mean().alias(f'{m}_mean'),
                 pl.col(m).std().alias(f'{m}_sd')]
    return order_scenarios(df.group_by('scenario').agg(*aggs))


def plot_horizon(df, out_path):
    import matplotlib
    ax = _axes()
    for scenario in SCENARIOS:
        rows = df.filter(pl.col('scenario') == scenario).sort('holdout_days')
        if len(rows) == 0:
            continue
        ax.plot(rows['holdout_days'], rows['skill_score'], marker='o',
                label=f'{scenario} — {LABELS[scenario]}')
    ax.set_xscale('log')
    ax.set_xticks(sorted(df['holdout_days'].unique().to_list()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('holdout window (days)')
    ax.set_title('Landscape-model skill by scenario and holdout length')
    _finish(ax, out_path)


def plot_rolling(df, out_path):
    """Skill per origin, with each scenario's across-fold mean behind it."""
    ax = _axes()
    for scenario in SCENARIOS:
        rows = df.filter(pl.col('scenario') == scenario).sort('origin_offset_days')
        if len(rows) == 0:
            continue
        line, = ax.plot(rows['origin_offset_days'], rows['skill_score'],
                        marker='o', label=f'{scenario} — {LABELS[scenario]}')
        ax.axhline(rows['skill_score'].mean(), color=line.get_color(),
                   lw=0.8, alpha=0.4)
    ax.invert_xaxis()                       # later origins to the right
    ax.set_xlabel('origin (days trimmed from the end of the data)')
    ax.set_title('Landscape-model skill by scenario and forecast origin')
    _finish(ax, out_path)


def _axes():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _, ax = plt.subplots(figsize=(7, 4.5))
    return ax


def _finish(ax, out_path):
    ax.axhline(0, color='0.6', lw=0.8, ls='--')
    ax.set_ylabel('skill vs no-movement baseline')
    ax.legend(fontsize=8, frameon=False)
    ax.figure.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ax.figure.savefig(out_path, dpi=150)
    print(f'wrote {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rolling', action='store_true',
                    help='scan forecast origins at one window length')
    ap.add_argument('--windows', type=int, nargs='+', default=list(splits.HOLDOUT_DAYS),
                    help='holdout lengths for the horizon scan')
    ap.add_argument('--holdout-days', type=int, default=365,
                    help='window length held fixed by the rolling scan')
    ap.add_argument('--offsets', type=int, nargs='+',
                    default=list(splits.ORIGIN_OFFSET_DAYS),
                    help='origins for the rolling scan, as days off the end')
    ap.add_argument('--prefix', default='step')
    ap.add_argument('--latent-tag', default=None,
                    help='disambiguate when out/ holds several configurations')
    ap.add_argument('--out-root', default=os.path.join(REPO, 'out'))
    ap.add_argument('--fig', default=None)
    ap.add_argument('--plot-only', action='store_true')
    args, overrides = ap.parse_known_args()
    overrides = [o for o in overrides if o != '--']

    windows = [args.holdout_days] if args.rolling else args.windows
    offsets = args.offsets if args.rolling else [0]
    if not args.plot_only:
        for days in windows:
            for offset in offsets:
                run_window(days, offset, overrides)

    df = one_configuration(collect(args.out_root, args.prefix), args.latent_tag)
    df = df.filter(pl.col('holdout_days').is_in(windows)
                   & pl.col('origin_offset_days').is_in(offsets))
    if len(df) == 0:
        raise SystemExit('no runs match the requested windows and origins')

    keys = ['holdout_days', 'origin_offset_days', 'scenario']
    shown = [c for c in keys + METRICS if c in df.columns]
    print(order_scenarios(df).select(shown).sort(keys))

    fig = args.fig or os.path.join(
        REPO, 'figs',
        f'rolling_holdout_{"origin" if args.rolling else "horizon"}_skill.png')
    if args.rolling:
        print('\nacross folds:')
        print(fold_summary(df))
        plot_rolling(df, fig)
    else:
        plot_horizon(df, fig)


if __name__ == '__main__':
    main()
