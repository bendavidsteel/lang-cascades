import datetime
import logging
import os

import hydra
import numpy as np
import polars as pl
from scipy import stats

import latent_space
from latent_space import COORD
from multiple_testing import benjamini_hochberg

logger = logging.getLogger(__name__)

def load_text_df(cfg, columns=['id', 'createtime', 'seed', 'Document', 'Targets', 'Stances']):
    dir_path = cfg.base_stance_path
    df = pl.read_parquet([os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path) if file_name.endswith('.parquet.zstd')], columns=columns)
    # The corpus stamps createtime UTC-aware, a gpfa latent's is naive off the
    # aggregate's bin grid; polars picks no supertype for the two, so every
    # comparison against a trajectory's window fails unless this side is naive.
    if df.schema['createtime'].time_zone is not None:
        df = df.with_columns(pl.col('createtime').dt.convert_time_zone('UTC')
                             .dt.replace_time_zone(None))
    return df


def covered_years(target_df):
    """Years whose mid-year the trajectories reach, for the sliding windows.

    The corpus grows, so a hard-coded list stops analysing the newest year
    without saying so.
    """
    lo = target_df['createtime'].min()
    hi = target_df['createtime'].max()
    return [y for y in range(lo.year, hi.year + 1)
            if lo <= datetime.datetime(y, 7, 1) <= hi]


def compute_movement_per_user(target_df: pl.DataFrame, dim_idx: int, per_year: bool = True) -> pl.DataFrame:
    """Compute movement in a specific dimension for each user, optionally grouped by year."""
    group_cols = ['filter_value', 'year'] if per_year else ['filter_value']

    df = target_df \
        .with_columns(pl.col(COORD).arr.get(dim_idx).alias('dim_value'))

    if per_year:
        df = df.with_columns(pl.col('createtime').dt.year().alias('year'))

    return df \
        .filter(pl.col('dim_value').is_not_null()) \
        .sort(['filter_value', 'createtime']) \
        .group_by(group_cols) \
        .agg([
            pl.col('dim_value').first().alias('start_value'),
            pl.col('dim_value').last().alias('end_value'),
            pl.col('createtime').min().alias('start_date'),
            pl.col('createtime').max().alias('end_date'),
            pl.len().alias('n_observations')
        ]) \
        .filter(
            (pl.col('n_observations') >= 2) &
            pl.col('start_value').is_not_null() &
            pl.col('end_value').is_not_null()
        ) \
        .with_columns([
            (pl.col('end_value') - pl.col('start_value')).alias('movement')
        ])


def get_top_movers(movement_df: pl.DataFrame, direction: str, n_top: int = 3) -> pl.DataFrame:
    """Get top N movers in a given direction ('positive' or 'negative')."""
    if direction == 'positive':
        return movement_df.sort('movement', descending=True).head(n_top)
    else:
        return movement_df.sort('movement', descending=False).head(n_top)


def get_percentile_movers(movement_df: pl.DataFrame, direction: str, percentile: float) -> pl.DataFrame:
    """Get top percentile of movers in a given direction ('positive' or 'negative')."""
    n_top = max(1, int(movement_df.height * percentile))
    return get_top_movers(movement_df, direction, n_top)


def get_heavy_loading_targets(components: np.ndarray, targets: list, dim_idx: int, n_targets: int = 5,
                              weights: np.ndarray = None) -> list:
    """Get targets that load heavily on a specific dimension.

    `weights` is a per-target post count. The fit constrains a low-volume
    target's loading barely at all, so ranking on magnitude alone puts those at
    the top; sqrt(weights) ranks by the target's share of the variance the
    dimension drives instead. Reported loadings are unweighted either way.
    """
    component = components[dim_idx]
    abs_loadings = np.abs(component)
    score = abs_loadings if weights is None else abs_loadings * np.sqrt(weights)
    top_indices = np.argsort(score)[-n_targets:][::-1]

    return [
        {
            'target': targets[idx],
            'loading': component[idx],
            'rank_score': float(score[idx]),
            'direction': 'positive' if component[idx] > 0 else 'negative'
        }
        for idx in top_indices
    ]


STANCE_SCORES = {'FAVOR': 1.0, 'NEUTRAL': 0.0, 'AGAINST': -1.0}


def directional_stance_shift(table, stances, expected_sign):
    """One-sided test that the favour/against balance moved the way expected.

    Rows are early/late counts, columns stances scored +1 favour, 0 neutral,
    -1 against; expected_sign is +1 when the mover should be shifting towards
    favour and -1 away from it. An omnibus chi-squared fires on any reshuffle,
    including favour and against both draining into neutral, so the statistic
    here is the shift in mean stance score, tested against the null that the
    two periods are a random split of the pooled counts (linear-by-linear
    association). Returns (p_value, shift in score points), the p-value nan
    where there is no contrast to test, so that a correction counts it as no
    test rather than as one that failed.
    """
    scores = np.array([STANCE_SCORES.get(s, 0.0) for s in stances], dtype=float)
    early = table[0].astype(float)
    late = table[1].astype(float)
    n_early, n_late = early.sum(), late.sum()
    n = n_early + n_late
    if n_early == 0 or n_late == 0 or n < 3:
        return np.nan, 0.0

    shift = float(late @ scores / n_late - early @ scores / n_early)

    pooled = (early + late) / n
    mean_score = pooled @ scores
    var_score = pooled @ (scores - mean_score) ** 2
    if var_score <= 0:
        return np.nan, 0.0

    z = (late @ scores - n_late * mean_score) / np.sqrt(n_early * n_late / (n - 1) * var_score)
    return float(stats.norm.sf(z * expected_sign)), shift


def stance_table(counts: pl.DataFrame):
    """Early/late contingency table over the stances present, and its columns."""
    early = counts.filter(pl.col('is_early'))
    late = counts.filter(~pl.col('is_early'))
    all_stances = sorted(counts['Stance'].unique().to_list())
    early_dict = dict(zip(early['Stance'].to_list(), early['len'].to_list()))
    late_dict = dict(zip(late['Stance'].to_list(), late['len'].to_list()))
    table = np.array([
        [early_dict.get(s, 0) for s in all_stances],
        [late_dict.get(s, 0) for s in all_stances]
    ])
    return table, all_stances, early_dict, late_dict


def compute_target_significance(
    stance_counts: pl.DataFrame,
    expected_signs: dict,
    alpha: float = 0.05,
    min_observations: int = 10
) -> pl.DataFrame:
    """Test each target's stance shift in the direction its loading predicts.

    One family per block of movers: every heavy target the block is read on is
    one test, and the handful the table keeps are the survivors of a hundred,
    so they are corrected together. Targets with fewer than min_observations in
    either period are not tested and hold no place in the family.
    """
    results = []

    for target in stance_counts['Target'].unique().to_list():
        target_data = stance_counts.filter(pl.col('Target') == target)
        early = target_data.filter(pl.col('is_early'))
        late = target_data.filter(~pl.col('is_early'))

        total_early = early['len'].sum() if early.height > 0 else 0
        total_late = late['len'].sum() if late.height > 0 else 0

        if total_early < min_observations or total_late < min_observations:
            results.append({'Target': target, 'p_value': np.nan, 'stance_shift': 0.0})
            continue

        table, all_stances, _, _ = stance_table(target_data)
        p_value, shift = directional_stance_shift(table, all_stances, expected_signs[target])
        results.append({'Target': target, 'p_value': p_value, 'stance_shift': shift})

    df = pl.DataFrame(results, schema={'Target': pl.Utf8, 'p_value': pl.Float64,
                                       'stance_shift': pl.Float64})
    q_values, significant = benjamini_hochberg(df['p_value'].to_numpy(), alpha=alpha)
    return df.with_columns([
        pl.Series('q_value', q_values),
        pl.Series('significant', significant),
    ])


def compute_stance_changes_for_user(
    text_df: pl.DataFrame,
    filter_value: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    targets: list,
    filter_col: str,
    movement_direction: str = 'positive',
    min_observations: int = 5
) -> dict:
    """Compute stance changes for a user, pooled by target loading direction.

    Every pooled test is returned, significant or not: the candidates are one
    family, and which of them survives is decided over the whole pool by the
    caller rather than user by user.
    """
    midpoint = start_date + (end_date - start_date) / 2
    target_names = [t['target'] for t in targets]
    target_info_df = pl.DataFrame(targets).rename({'target': 'Target'})

    # Filter to user's documents within the time range, explode, and join loading direction
    user_stances = text_df \
        .filter(
            (pl.col(filter_col) == filter_value) &
            (pl.col('createtime') >= start_date) &
            (pl.col('createtime') <= end_date)
        ) \
        .explode(['Targets', 'Stances']) \
        .rename({'Targets': 'Target', 'Stances': 'Stance'}) \
        .filter(pl.col('Target').is_in(target_names)) \
        .join(target_info_df.select(['Target', 'direction']), on='Target') \
        .with_columns((pl.col('createtime') < midpoint).alias('is_early'))

    if user_stances.height == 0:
        return {}

    movement_sign = 1 if movement_direction == 'positive' else -1

    results = {}
    for loading_dir in ['positive', 'negative']:
        dir_stances = user_stances.filter(pl.col('direction') == loading_dir)
        if dir_stances.height == 0:
            continue

        # Count stances pooled across all targets in this loading direction
        stance_counts = dir_stances \
            .group_by(['Stance', 'is_early']) \
            .len()

        early_counts = stance_counts.filter(pl.col('is_early'))
        late_counts = stance_counts.filter(~pl.col('is_early'))

        total_early = early_counts['len'].sum() if early_counts.height > 0 else 0
        total_late = late_counts['len'].sum() if late_counts.height > 0 else 0

        if total_early < min_observations or total_late < min_observations:
            continue

        table, all_stances, early_dict, late_dict = stance_table(stance_counts)

        # a target loading negatively on the dimension is expected to lose
        # favour as its mover travels the positive way along it, and vice versa
        expected_sign = movement_sign * (1 if loading_dir == 'positive' else -1)
        p_value, shift = directional_stance_shift(table, all_stances, expected_sign)

        # Compute percentages
        stance_changes = {}
        for s in all_stances:
            early_n = early_dict.get(s, 0)
            late_n = late_dict.get(s, 0)
            early_pct = early_n / total_early * 100 if total_early > 0 else 0
            late_pct = late_n / total_late * 100 if total_late > 0 else 0
            stance_changes[s] = {
                'early_pct': early_pct,
                'late_pct': late_pct,
                'change': late_pct - early_pct
            }

        n_targets = dir_stances['Target'].n_unique()
        label = f'{loading_dir}_loading_targets'
        results[label] = {
            'loading_direction': loading_dir,
            'stance_changes': stance_changes,
            'early_n': total_early,
            'late_n': total_late,
            'p_value': p_value,
            'stance_shift': shift,
            'n_targets': n_targets
        }

    return results


def compute_aggregate_stance_changes(
    text_df: pl.DataFrame,
    movers_df: pl.DataFrame,
    targets: list,
    filter_col: str,
    movement_direction: str = 'positive',
    n_top_targets: int = 3,
    significance_threshold: float = 0.05,
    min_observations: int = 10
) -> dict:
    """Compute pooled stance changes across a group of movers on specific targets, filtered by significance."""
    target_names = [t['target'] for t in targets]
    target_info_df = pl.DataFrame(targets).rename({'target': 'Target'})

    # Compute midpoints for each mover's time window
    movers_with_mid = movers_df \
        .select(['filter_value', 'start_date', 'end_date']) \
        .with_columns([
            pl.col('filter_value').cast(pl.Utf8),
            (pl.col('start_date') + (pl.col('end_date') - pl.col('start_date')) / 2).alias('midpoint'),
        ])

    # Join movers with text data, filter to relevant time windows, and explode targets
    combined = text_df \
        .join(movers_with_mid, left_on=filter_col, right_on='filter_value') \
        .filter(
            (pl.col('createtime') >= pl.col('start_date')) &
            (pl.col('createtime') <= pl.col('end_date'))
        ) \
        .explode(['Targets', 'Stances']) \
        .rename({'Targets': 'Target', 'Stances': 'Stance'}) \
        .filter(pl.col('Target').is_in(target_names)) \
        .with_columns((pl.col('createtime') < pl.col('midpoint')).alias('is_early'))

    if combined.height == 0:
        return {}

    # Compute stance distributions in early vs late periods
    stance_counts = combined \
        .group_by(['Target', 'Stance', 'is_early']) \
        .len()

    totals = stance_counts \
        .group_by(['Target', 'is_early']) \
        .agg(pl.col('len').sum().alias('total'))

    stance_pcts = stance_counts \
        .join(totals, on=['Target', 'is_early']) \
        .with_columns((pl.col('len') / pl.col('total') * 100).alias('pct'))

    early_pcts = stance_pcts.filter(pl.col('is_early')) \
        .select(['Target', 'Stance', pl.col('pct').alias('early_pct'), pl.col('len').alias('early_n')])
    late_pcts = stance_pcts.filter(~pl.col('is_early')) \
        .select(['Target', 'Stance', pl.col('pct').alias('late_pct'), pl.col('len').alias('late_n')])

    changes_df = early_pcts \
        .join(late_pcts, on=['Target', 'Stance'], how='full', coalesce=True) \
        .with_columns([
            pl.col('early_pct').fill_null(0.0),
            pl.col('late_pct').fill_null(0.0),
            pl.col('early_n').fill_null(0),
            pl.col('late_n').fill_null(0),
        ]) \
        .with_columns((pl.col('late_pct') - pl.col('early_pct')).alias('change'))

    # A target loading negatively on the dimension is expected to lose favour
    # as the group travels the positive way along it, and vice versa; the test
    # is one-sided in that direction, so surviving it is already the filter on
    # which way the stances moved.
    movement_sign = 1 if movement_direction == 'positive' else -1
    expected_signs = {t['target']: movement_sign * (1 if t['loading'] > 0 else -1)
                      for t in targets}
    significance_df = compute_target_significance(stance_counts, expected_signs,
                                                  significance_threshold, min_observations)

    # Rank on how far the stances actually moved, weighted by the same score
    # that picked the heavy-loading targets in the first place: ranking on
    # p-value alone puts the highest-volume targets on top whatever the size of
    # their shift.
    valid_targets = changes_df \
        .group_by('Target') \
        .agg([
            pl.col('early_n').sum().alias('total_early'),
            pl.col('late_n').sum().alias('total_late'),
        ]) \
        .filter((pl.col('total_early') > 0) & (pl.col('total_late') > 0)) \
        .join(significance_df, on='Target') \
        .filter(pl.col('significant')) \
        .join(target_info_df.select(['Target', 'loading', 'rank_score']), on='Target') \
        .sort(pl.col('rank_score') * pl.col('stance_shift').abs(), descending=True) \
        .head(n_top_targets)

    if valid_targets.height == 0:
        return {}

    top_changes = changes_df \
        .join(valid_targets.select(['Target', 'total_early', 'total_late', 'p_value',
                                    'q_value', 'stance_shift']), on='Target') \
        .join(target_info_df, on='Target')

    results = {}
    for target in valid_targets['Target'].to_list():
        target_rows = top_changes.filter(pl.col('Target') == target)
        target_info = target_rows.select(['loading', 'direction', 'total_early', 'total_late',
                                          'p_value', 'q_value', 'stance_shift']).row(0, named=True)

        stance_changes = {}
        for row in target_rows.iter_rows(named=True):
            stance_changes[row['Stance']] = {
                'early_pct': row['early_pct'],
                'late_pct': row['late_pct'],
                'change': row['change']
            }

        results[target] = {
            'loading': target_info['loading'],
            'loading_direction': target_info['direction'],
            'stance_changes': stance_changes,
            'early_n': target_info['total_early'],
            'late_n': target_info['total_late'],
            'p_value': target_info['p_value'],
            'q_value': target_info['q_value'],
            'stance_shift': target_info['stance_shift']
        }

    return results


def analyze_dimension_movements(
    target_df: pl.DataFrame,
    text_df: pl.DataFrame,
    components: np.ndarray,
    targets: list,
    filter_col: str,
    n_dims: int = 3,
    years: list = None,
    n_top_movers: int = 3,
    n_candidate_movers: int = 50,
    n_heavy_targets: int = 5,
    percentiles: list = [0.01, 0.10],
    per_year: bool = True,
    axis_prefix: str = 'LD',
    target_weights: np.ndarray = None,
    significance_threshold: float = 0.05
):
    """Analyze movement patterns across dimensions, time periods, and directions."""
    results = {}
    if years is None:
        years = covered_years(target_df)

    for dim_idx in range(n_dims):
        dim_name = f'{axis_prefix}{dim_idx + 1}'
        results[dim_name] = {}

        # Get heavy loading targets for this dimension
        heavy_targets = get_heavy_loading_targets(components, targets, dim_idx, n_heavy_targets,
                                                  weights=target_weights)

        # Build dict of time periods to analyze
        if per_year:
            # Use sliding windows centered on mid-year, each half max 1 year wide
            time_periods = {}
            for year in years:
                mid = datetime.datetime(year, 7, 1)
                window_start = mid - datetime.timedelta(days=365)
                window_end = mid + datetime.timedelta(days=365)
                window_df = target_df.filter(
                    (pl.col('createtime') >= window_start) \
                    & (pl.col('createtime') <= window_end)
                )
                movement_df = compute_movement_per_user(window_df, dim_idx, per_year=False)
                if movement_df is not None and movement_df.height > 0:
                    time_periods[year] = movement_df
        else:
            movement_df = compute_movement_per_user(target_df, dim_idx, per_year=False)
            time_periods = {'all_time': movement_df}

        for period_key, period_movement in time_periods.items():
            if period_movement.height == 0:
                continue

            results[dim_name][period_key] = {'heavy_targets': heavy_targets}

            for direction in ['positive', 'negative']:
                # Get a larger candidate pool, filter to those with significant
                # pooled stance changes, then keep top N by movement
                candidates = get_top_movers(period_movement, direction, n_candidate_movers)

                # The whole pool is screened before any of it is kept: the
                # movers on the page are the survivors of fifty candidates
                # tested on both loading pools, so that is the family.
                scanned = []
                for row in candidates.iter_rows(named=True):
                    stance_changes = compute_stance_changes_for_user(
                        text_df,
                        row['filter_value'],
                        row['start_date'],
                        row['end_date'],
                        heavy_targets,
                        filter_col,
                        movement_direction=direction
                    )
                    scanned.extend((row['filter_value'], label, group)
                                   for label, group in stance_changes.items())

                q_values, keep = benjamini_hochberg(
                    np.array([group['p_value'] for _, _, group in scanned]),
                    alpha=significance_threshold)

                survivors = {}
                for (filter_value, label, group), q, ok in zip(scanned, q_values, keep):
                    if not ok:
                        continue
                    group['q_value'] = float(q)
                    survivors.setdefault(filter_value, {})[label] = group

                movers_info = []
                for row in candidates.iter_rows(named=True):
                    stance_changes = survivors.get(row['filter_value'])
                    if not stance_changes:
                        continue

                    movers_info.append({
                        'filter_value': row['filter_value'],
                        'movement': row['movement'],
                        'start_value': row['start_value'],
                        'end_value': row['end_value'],
                        'n_observations': row['n_observations'],
                        'stance_changes': stance_changes,
                    })
                    if len(movers_info) >= n_top_movers:
                        break

                results[dim_name][period_key][direction] = movers_info

                # Compute percentile group statistics
                for pct in percentiles:
                    pct_movers = get_percentile_movers(period_movement, direction, pct)
                    pct_label = f'top_{int(pct * 100)}pct'
                    agg_stance_changes = compute_aggregate_stance_changes(
                        text_df, pct_movers, heavy_targets, filter_col,
                        movement_direction=direction,
                        significance_threshold=significance_threshold
                    )
                    results[dim_name][period_key][f'{direction}_{pct_label}'] = {
                        'n_users': pct_movers.height,
                        'mean_movement': pct_movers['movement'].mean(),
                        'median_movement': pct_movers['movement'].median(),
                        'stance_changes': agg_stance_changes,
                    }

    return results


STANCE_ORDER = ['FAVOR', 'NEUTRAL', 'AGAINST']

def print_analysis_results(results: dict, title: str = None):
    """Print analysis results in a formatted way."""
    if title:
        print(f"\n{'#'*80}")
        print(f"  {title}")
        print(f"{'#'*80}")

    for dim_name, dim_data in results.items():
        print(f"\n{'='*80}")
        print(f"DIMENSION: {dim_name}")
        print(f"{'='*80}")

        for period_key, year_data in sorted(dim_data.items()):
            if period_key == 'heavy_targets':
                continue

            if period_key == 'all_time':
                period_label = "ALL TIME"
            else:
                mid = datetime.datetime(period_key, 7, 1)
                w_start = mid - datetime.timedelta(days=365)
                w_end = mid + datetime.timedelta(days=365)
                period_label = f"WINDOW: mid-{period_key} ({w_start.strftime('%b %Y')} - {w_end.strftime('%b %Y')})"
            print(f"\n{'-'*60}")
            print(period_label)
            print(f"{'-'*60}")

            # heavy_targets = year_data.get('heavy_targets', [])
            # print(f"\nHeavy loading targets for {dim_name}:")
            # for t in heavy_targets:
            #     print(f"  • {t['target']}: loading={t['loading']:.4f} ({t['direction']})")

            # for direction in ['positive', 'negative']:
            #     movers = year_data.get(direction, [])
            #     dir_label = "POSITIVE" if direction == 'positive' else "NEGATIVE"
            #     print(f"\nTop {dir_label} movers with significant pooled stance changes:")

            #     if not movers:
            #         print("  No movers with significant stance changes found")
            #         continue

            #     for i, mover in enumerate(movers, 1):
            #         print(f"\n  {i}. {mover['filter_value']}")
            #         print(f"     Movement: {mover['movement']:.4f} ({mover['start_value']:.4f} → {mover['end_value']:.4f})")
            #         print(f"     Observations: {mover['n_observations']}")

            #         stance_changes = mover.get('stance_changes', {})
            #         if stance_changes:
            #             print(f"     Pooled stance changes by loading direction:")
            #             for group_key, group_data in stance_changes.items():
            #                 dir_label = group_data['loading_direction'].upper()
            #                 print(f"       {dir_label}-loading targets ({group_data['n_targets']} targets, n_early={group_data['early_n']}, n_late={group_data['late_n']}, p={group_data['p_value']:.4f}):")
            #                 for stance in sorted(group_data['stance_changes'], key=lambda s: (STANCE_ORDER.index(s) if s in STANCE_ORDER else len(STANCE_ORDER))):
            #                     change_data = group_data['stance_changes'][stance]
            #                     change_str = f"+{change_data['change']:.1f}" if change_data['change'] >= 0 else f"{change_data['change']:.1f}"
            #                     print(f"         {stance}: {change_data['early_pct']:.1f}% → {change_data['late_pct']:.1f}% ({change_str}%)")
            #         else:
            #             print(f"     No significant stance changes on heavy-loading targets")

            # Print percentile group summaries
            for direction in ['positive', 'negative']:
                dir_label = "POSITIVE" if direction == 'positive' else "NEGATIVE"
                for key, pct_data in sorted(year_data.items()):
                    if not key.startswith(f'{direction}_top_'):
                        continue
                    pct_label = key.replace(f'{direction}_', '')
                    print(f"\n  {dir_label} movers - {pct_label} ({pct_data['n_users']} users):")
                    print(f"    Mean movement: {pct_data['mean_movement']:.4f}")
                    print(f"    Median movement: {pct_data['median_movement']:.4f}")

                    stance_changes = pct_data.get('stance_changes', {})
                    if stance_changes:
                        print(f"    Top significant stance changes (pooled across group):")
                        for target, target_data in stance_changes.items():
                            print(f"      {target} (loading={target_data['loading']:.4f}, n_early={target_data['early_n']}, n_late={target_data['late_n']}, shift={target_data['stance_shift']:+.3f}, q={target_data['q_value']:.4f}):")
                            for stance in sorted(target_data['stance_changes'], key=lambda s: (STANCE_ORDER.index(s) if s in STANCE_ORDER else len(STANCE_ORDER))):
                                change_data = target_data['stance_changes'][stance]
                                change_str = f"+{change_data['change']:.1f}" if change_data['change'] >= 0 else f"{change_data['change']:.1f}"
                                print(f"        {stance}: {change_data['early_pct']:.1f}% → {change_data['late_pct']:.1f}% ({change_str}%)")
                    else:
                        print(f"    No significant stance changes")

def write_latex_table(results: dict, output_path: str):
    """Write stance change results as a LaTeX table.

    Columns: Dim., Dir., Percentile, Target, FAVOR, NEUTRAL, AGAINST. The first
    three name one group of movers and are printed on its first row only, so a
    group reads as a block rather than repeating its own label down the page.
    """
    rows = []
    for dim_name, dim_data in results.items():
        for period_key, year_data in sorted(dim_data.items()):
            if period_key == 'heavy_targets':
                continue

            for direction in ['positive', 'negative']:
                dir_label = "Pos." if direction == 'positive' else "Neg."
                for key, pct_data in sorted(year_data.items()):
                    if not key.startswith(f'{direction}_top_'):
                        continue

                    pct_num = int(key.replace(f'{direction}_top_', '').replace('pct', ''))
                    # the end of the axis is its own column, so the percentile
                    # is the size of the group rather than a threshold whose
                    # direction the reader has to infer from < or >
                    pct_label = f"Top {pct_num}\\%"

                    stance_changes = pct_data.get('stance_changes', {})
                    for target, target_data in stance_changes.items():
                        change_vals = {}
                        for stance, change_data in target_data['stance_changes'].items():
                            change_vals[stance] = (change_data['early_pct'], change_data['late_pct'])
                        rows.append({
                            'group': (dim_name, period_key, direction, pct_num),
                            'dim': dim_name,
                            'dir': dir_label,
                            'percentile': pct_label,
                            'target': target,
                            'changes': change_vals,
                        })

    if not rows:
        return

    lines = []
    lines.append("\\begin{tabular}{llllrrr}")
    lines.append("\\toprule")
    lines.append("Dim. & Dir. & Percentile & Target & Favor & Neutral & Against \\\\")
    lines.append("\\midrule")
    prev_group = None
    for row in rows:
        opens_group = row['group'] != prev_group
        if prev_group is not None and opens_group:
            lines.append("\\midrule")
        head = ((row['dim'], row['dir'], row['percentile']) if opens_group
                else ('', '', ''))
        prev_group = row['group']
        vals = []
        for stance in STANCE_ORDER:
            early_pct, late_pct = row['changes'].get(stance, (0, 0))
            val = f"{early_pct:.1f}→{late_pct:.1f}"
            vals.append(val)
        target_escaped = row['target'].replace('_', '\\_').replace('&', '\\&')
        lines.append(f"{head[0]} & {head[1]} & {head[2]} & {target_escaped} & "
                     f"{vals[0]} & {vals[1]} & {vals[2]} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"LaTeX table written to {output_path}")


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    logging.basicConfig(level=logging.INFO)
    logger.info("Loading data...")

    target_df, components, targets = latent_space.load(cfg)
    # a frozen dimension has no movement to rank movers by
    n_moving = latent_space.n_moving_dims(cfg)
    n_short, n_long = min(2, n_moving), min(3, n_moving)
    prefix = latent_space.axis_prefix(cfg)
    # ranked the way the dimension table names its dimensions, so the targets a
    # mover's stance changes are read on are the ones the axis is named after
    weights = (latent_space.target_volumes(cfg, targets)
               if cfg.latents.get('rank_by_volume', True) else None)

    logger.info("Loading text data for stance analysis...")
    # a gpfa run keys trajectories by latents.traj_col, not by filter_column
    filter_col = latent_space.traj_col(cfg)
    text_df = load_text_df(cfg)
    text_df = text_df.with_columns(pl.col('seed').struct.field(filter_col)).drop('seed')

    if filter_col == 'SeedName':
        years = covered_years(target_df)
        logger.info(f"Analyzing per-year movements ({n_short} dimensions)...")
        per_year_results = analyze_dimension_movements(
            target_df=target_df,
            text_df=text_df,
            components=components,
            targets=targets,
            filter_col=filter_col,
            n_dims=n_short,
            years=years,
            n_top_movers=3,
            n_heavy_targets=100,
            per_year=True,
            axis_prefix=prefix,
            target_weights=weights
        )
        print_analysis_results(per_year_results, title=f"PER-YEAR ANALYSIS ({prefix}1-{prefix}{n_short})")
        for year in years:
            year_results = {dim: {k: v for k, v in data.items() if k == year} for dim, data in per_year_results.items()}
            write_latex_table(year_results, f'./out/per_year_stance_changes_{year}.tex')

        logger.info(f"Analyzing all-time movements ({n_long} dimensions)...")
        all_time_results = analyze_dimension_movements(
            target_df=target_df,
            text_df=text_df,
            components=components,
            targets=targets,
            filter_col=filter_col,
            n_dims=n_long,
            n_top_movers=3,
            n_heavy_targets=100,
            per_year=False,
            axis_prefix=prefix,
            target_weights=weights
        )
        print_analysis_results(all_time_results, title=f"ALL-TIME ANALYSIS ({prefix}1-{prefix}{n_long})")
        write_latex_table(all_time_results, './out/all_time_stance_changes.tex')
    elif filter_col == 'PlatformHandleID':
        for platform in ['twitter', 'instagram', 'bluesky', 'tiktok']:
            platform_target_df = target_df.filter(
                pl.col('filter_value').cast(pl.String)\
                    .str.to_lowercase()\
                    .str.contains(f'-{platform}-')
            )
            platform_text_df = text_df.filter(
                pl.col(filter_col).cast(pl.String)\
                    .str.to_lowercase()\
                    .str.contains(f'-{platform}-')
            )

            if platform_target_df.height == 0:
                logger.info(f"No data for platform {platform}, skipping")
                continue

            logger.info(f"Analyzing {platform} movements ({n_short} dimensions)...")
            platform_results = analyze_dimension_movements(
                target_df=platform_target_df,
                text_df=platform_text_df,
                components=components,
                targets=targets,
                filter_col=filter_col,
                n_dims=n_short,
                n_top_movers=3,
                n_heavy_targets=100,
                per_year=False,
                axis_prefix=prefix,
                target_weights=weights
            )
            print_analysis_results(platform_results, title=f"{platform.upper()} PLATFORM ANALYSIS ({prefix}1-{prefix}{n_short})")
            write_latex_table(platform_results, f'./out/{platform}_platform_stance_changes.tex')

if __name__ == '__main__':
    main()
