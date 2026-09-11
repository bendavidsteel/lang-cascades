"""Re-evaluate a trained landscape model on per-category val subsets to
recover per-sample losses (so we can plot error bars), test significance,
and plot bar charts by category. Loads the most recent local checkpoint
matching the hydra config."""

import os

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from scipy import stats

from plnn.dataset import LandscapeSimulationDataset, NumpyLoader
from plnn.models import DeepTimePhiPLNN

from nn_potential import \
    apply_split, \
    run_dir, \
    build_horizon_pairs, \
    rolling_frame, \
    compute_training_split, \
    df_to_data, \
    evaluate_dataloader, \
    load_seed_metadata, \
    load_target_df
from sweep_runs import state_path as select_state

import latent_space
import splits


# Horizons to evaluate and plot
HORIZONS = [30]
# Categories to break down by
CATEGORY_COLS = ['MainType', 'Party']
# Exclude degenerate subsets with near-zero baseline
MIN_BASELINE_MSE = 1e-5

# Plot aggregator (mirrors eval_horizons.py terminology):
#   'median_per_pair' = median(model_i / baseline_i) with IQR shown as a
#                       horizontal box plot of the per-pair ratios.
#   'median_ratio'    = median(model) / median(baseline) with Q1/Q3 of the
#                       model losses rescaled by median(baseline) as error
#                       bars. Robust to near-zero baseline pairs (which
#                       dominate per-pair-ratio aggregators).
PLOT_AGGREGATOR = 'median_ratio'

# Semantic colors for Canadian Party values (case-insensitive match)
PARTY_COLORS = {
    'liberal': '#D71920',
    'lpc': '#D71920',
    'conservative': '#1A4782',
    'cpc': '#1A4782',
    'ndp': '#F58220',
    'new democratic': '#F58220',
    'bloc': '#33B2CC',
    'bloc québécois': '#33B2CC',
    'bloc quebecois': '#33B2CC',
    'bq': '#33B2CC',
    'green': '#3D9B35',
    'gpc': '#3D9B35',
    'ppc': '#83478D',
    "people's party": '#83478D',
}


def category_colors(category, values):
    """Return bar colors per value for MainType/Party; default blue otherwise."""
    if category == 'Party':
        return [PARTY_COLORS.get(str(v).lower(), '#1565C0') for v in values]
    if category == 'MainType':
        cmap = plt.cm.tab10
        return [cmap(i % 10) for i in range(len(values))]
    return ['#1565C0'] * len(values)


def _parse_subset(subset):
    """Split 'MainType_politician' into ('MainType', 'politician')."""
    for prefix in CATEGORY_COLS:
        if subset.startswith(prefix + '_'):
            return (prefix, subset[len(prefix) + 1:])
    return ('other', subset)


def model_dir_for_cfg(cfg):
    """Reconstruct the training output directory from the run config."""
    n_dims = cfg.n_dims
    dims = list(range(n_dims))
    trend_name = os.path.basename(cfg.trend_path.rstrip('/'))
    # same layout the trainer writes, including the latent and split tags
    dir_path = run_dir(cfg)
    if cfg.latents.method != 'gpfa' and cfg.rolling_mean_window != 100:
        dir_path = f"{dir_path}_rm{cfg.rolling_mean_window}"
    return dir_path


def build_val_subsets(val_paired):
    """Build per-(MainType/Party) value subsets of val_paired."""
    subsets = []
    for col in CATEGORY_COLS:
        if col not in val_paired.columns:
            continue
        for v in val_paired.drop_nulls(col).filter(pl.col(col) != '')[col].unique().sort().to_list():
            subsets.append((f'{col}_{v}', val_paired.filter(pl.col(col) == v)))
    return subsets


def evaluate_subset(model, subset_df, eval_batch_size, key):
    """Run evaluate_dataloader on a subset_df, returning per-sample losses."""
    subset_data = df_to_data(subset_df)
    subset_dataset = LandscapeSimulationDataset(data=subset_data)
    subset_loader = NumpyLoader(
        subset_dataset,
        batch_size=min(eval_batch_size, len(subset_dataset)),
        shuffle=False,
    )
    return evaluate_dataloader(model, subset_loader, key)


def compute_breakdown_metrics(cfg):
    """Load the most recent local model checkpoint and compute per-subset
    metrics at each horizon in HORIZONS. Returns a polars DataFrame with one
    row per (horizon, subset) including per-sample loss summaries (mean, std, n)
    for both model and no-movement baseline.
    """
    n_dims = cfg.n_dims
    dims = list(range(n_dims))

    print("Loading data...", flush=True)
    if cfg.latents.method == 'gpfa':
        target_df, _, _ = latent_space.load(cfg, smooth=False)
        target_df = target_df.rename({latent_space.COORD: 'x0'}) \
            .select(['createtime', 'filter_value', 'x0']) \
            .sort(['filter_value', 'createtime'])
    else:
        target_df = load_target_df(cfg)
    rolling_df = rolling_frame(cfg, target_df, dims)
    print(f"  Timestep rows: {len(rolling_df)}", flush=True)

    val_filter_values, cutoff_time = compute_training_split(cfg)
    spec = splits.SplitSpec.from_cfg(cfg)
    split_key = latent_space.split_key(cfg)

    state_path = select_state(model_dir_for_cfg(cfg))
    print(f"Loading model state from {state_path}...", flush=True)
    model, _ = DeepTimePhiPLNN.load(state_path, dtype=jnp.float32)

    seed_df = load_seed_metadata(cfg)

    rows = []
    losses_by_key = {}
    rng = np.random.default_rng(42)
    key = jax.random.PRNGKey(int(rng.integers(2**32)))

    for horizon in HORIZONS:
        print(f"\n--- Horizon: {horizon}d ---", flush=True)
        paired_df = build_horizon_pairs(rolling_df, horizon, dims)
        if len(paired_df) == 0:
            print(f"  No pairs at {horizon}d; skipping.", flush=True)
            continue

        _, val_paired = apply_split(
            paired_df, cfg.split_type, cfg.train_fraction,
            val_filter_values=val_filter_values, cutoff_time=cutoff_time,
            spec=spec, scenario=cfg.objective_scenario, key=split_key,
        )
        val_paired = val_paired.with_columns(pl.col('filter_value').cast(pl.String)) \
            .join(seed_df, left_on='filter_value', right_on='SeedName', how='left')
        print(f"  Val pairs: {len(val_paired)}", flush=True)
        if len(val_paired) == 0:
            continue

        subsets = build_val_subsets(val_paired)
        for subset_name, subset_df in subsets:
            if len(subset_df) == 0:
                continue
            key, subkey = jax.random.split(key)
            model_losses, baseline_losses = evaluate_subset(
                model, subset_df, cfg.eval_batch_size, subkey)
            n_pairs = len(model_losses)
            n_traj = subset_df['filter_value'].n_unique()
            model_mse = float(np.mean(model_losses))
            baseline_mse = float(np.mean(baseline_losses))
            ratio = model_mse / baseline_mse if baseline_mse > 0 else float('nan')
            frac_better = float(np.mean(model_losses < baseline_losses))
            model_arr = np.asarray(model_losses, dtype=np.float64)
            baseline_arr = np.asarray(baseline_losses, dtype=np.float64)
            losses_by_key[(horizon, subset_name)] = {
                'model_loss': model_arr,
                'baseline_loss': baseline_arr,
            }
            pair_ratios = model_arr / np.maximum(baseline_arr, 1e-12)
            ratio_q1 = float(np.percentile(pair_ratios, 25))
            ratio_median = float(np.median(pair_ratios))
            ratio_q3 = float(np.percentile(pair_ratios, 75))
            print(
                f"  {subset_name:40s} n_pairs={n_pairs:5d} n_traj={n_traj:4d} "
                f"model_mse={model_mse:.5f} baseline_mse={baseline_mse:.5f} "
                f"ratio={ratio:.4f} median={ratio_median:.4f} "
                f"frac_better={frac_better:.3f}",
                flush=True,
            )
            category, value = _parse_subset(subset_name)
            rows.append({
                'horizon': horizon,
                'subset': subset_name,
                'category': category,
                'value': value,
                'model_mse': model_mse,
                'baseline_mse': baseline_mse,
                'ratio': ratio,
                'ratio_q1': ratio_q1,
                'ratio_median': ratio_median,
                'ratio_q3': ratio_q3,
                'frac_better': frac_better,
                'n_pairs': n_pairs,
                'n_trajectories': int(n_traj),
            })

    return pl.DataFrame(rows), losses_by_key


def test_significance(metrics_df, losses_by_key):
    """One-sided paired Wilcoxon signed-rank test on per-pair losses.

    H1: model_loss < baseline_loss per pair (median of differences negative).

    Paired because each pair contributes both a model_loss and a baseline_loss
    on the same (filter_value, t0, t1). Per-pair difficulty varies wildly and
    both losses inherit it, so pairing carries real signal — Wilcoxon
    exploits that; Mann-Whitney would discard it.
    """
    p_values = []
    for row in metrics_df.iter_rows(named=True):
        losses = losses_by_key.get((row['horizon'], row['subset']))
        if losses is None or len(losses['model_loss']) < 2:
            p_values.append(np.nan)
            continue
        try:
            res = stats.wilcoxon(
                losses['model_loss'], losses['baseline_loss'],
                alternative='less',
            )
            p_values.append(float(res.pvalue))
        except ValueError:
            p_values.append(np.nan)
    return metrics_df.with_columns(pl.Series('p_value', p_values))


def benjamini_hochberg(p_values, alpha=0.05):
    """Apply Benjamini-Hochberg correction. Returns array of booleans."""
    valid = ~np.isnan(p_values)
    n = np.sum(valid)
    if n == 0:
        return np.full(len(p_values), False)

    order = np.argsort(p_values[valid])
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)

    adjusted = p_values.copy()
    adjusted[valid] = p_values[valid] * n / ranks

    significant = np.full(len(p_values), False)
    significant[valid] = adjusted[valid] < alpha
    return significant


def significance_stars(p):
    if np.isnan(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'


def plot_category_bars(df, category, horizon, fig_dir, losses_by_key):
    """Plot the per-category breakdown for one (category, horizon). The
    PLOT_AGGREGATOR module constant selects the visualization:
    - 'median_per_pair': horizontal box plot of per-pair model/baseline ratios.
    - 'median_ratio': horizontal box plot whose median, Q1, Q3, and 1.5×IQR
      whiskers are computed on the model losses and rescaled by
      median(baseline). Matches the `median_ratio` aggregator in
      eval_horizons.py for the central tendency.
    """
    sub = df.filter(
        (pl.col('category') == category)
        & (pl.col('horizon') == horizon)
        & (pl.col('baseline_mse') > MIN_BASELINE_MSE)
    )
    if len(sub) == 0:
        return

    summaries = []
    for r in sub.iter_rows(named=True):
        losses = losses_by_key.get((horizon, r['subset']))
        if losses is None:
            continue
        model = losses['model_loss']
        baseline = losses['baseline_loss']
        if PLOT_AGGREGATOR == 'median_per_pair':
            ratios = model / np.maximum(baseline, 1e-12)
            summaries.append({
                'value': r['value'],
                'p_value': r['p_value'],
                'n_trajectories': r['n_trajectories'],
                'center': float(np.median(ratios)),
                'q3': float(np.quantile(ratios, 0.75)),
                'ratios': ratios,
            })
        elif PLOT_AGGREGATOR == 'median_ratio':
            base_med = float(np.median(baseline))
            if not (base_med > 0):
                continue
            m_q1 = float(np.quantile(model, 0.25))
            m_med = float(np.median(model))
            m_q3 = float(np.quantile(model, 0.75))
            m_iqr = m_q3 - m_q1
            low_fence = m_q1 - 1.5 * m_iqr
            high_fence = m_q3 + 1.5 * m_iqr
            in_low = model[model >= low_fence]
            in_high = model[model <= high_fence]
            whislo = float(in_low.min()) if in_low.size else m_q1
            whishi = float(in_high.max()) if in_high.size else m_q3
            summaries.append({
                'value': r['value'],
                'p_value': r['p_value'],
                'n_trajectories': r['n_trajectories'],
                'center': m_med / base_med,
                'q1': m_q1 / base_med,
                'q3': m_q3 / base_med,
                'whislo': whislo / base_med,
                'whishi': whishi / base_med,
            })
        else:
            raise ValueError(f"Unknown PLOT_AGGREGATOR: {PLOT_AGGREGATOR}")

    if not summaries:
        return

    summaries.sort(key=lambda x: x['center'])
    values = [s['value'] for s in summaries]
    p_vals = np.array([s['p_value'] for s in summaries])
    n_trajs = np.array([s['n_trajectories'] for s in summaries])

    n = len(values)
    # Keep per-row axes height constant by budgeting a fixed inch allowance for
    # the x-axis label/ticks; otherwise tight_layout's constant bottom overhead
    # eats a different fraction of the figure for different n, making boxes
    # come out different sizes between categories.
    row_height = 0.3
    bottom_overhead = 0.6
    fig_height = bottom_overhead + row_height * n
    fig, ax = plt.subplots(1, 1, figsize=(4, fig_height))
    ax.set_position([
        0.25,
        bottom_overhead / fig_height,
        0.72,
        (row_height * n) / fig_height,
    ])
    colors = category_colors(category, values)
    y_pos = np.arange(n)

    if PLOT_AGGREGATOR == 'median_per_pair':
        datasets = [s['ratios'] for s in summaries]
        parts = ax.boxplot(
            datasets, positions=y_pos, vert=False, widths=0.6,
            patch_artist=True, showfliers=False,
            medianprops={'color': 'black', 'linewidth': 1.0},
            whiskerprops={'color': 'black', 'linewidth': 0.8},
            capprops={'color': 'black', 'linewidth': 0.8},
            boxprops={'edgecolor': 'black', 'linewidth': 0.5},
        )
        for box, c in zip(parts['boxes'], colors):
            box.set_facecolor(c)
            box.set_alpha(0.75)
        whisker_xs = np.concatenate([w.get_xdata() for w in parts['whiskers']])
        bulk_lo = min(float(whisker_xs.min()), 1.0)
        bulk_hi = max(float(whisker_xs.max()), 1.0)
        xlabel = 'Method loss / No-movement loss (per pair)'
    else:  # 'median_ratio'
        stats_list = [{
            'med': s['center'],
            'q1': s['q1'],
            'q3': s['q3'],
            'whislo': s['whislo'],
            'whishi': s['whishi'],
            'fliers': [],
        } for s in summaries]
        parts = ax.bxp(
            stats_list, positions=y_pos, vert=False, widths=0.6,
            patch_artist=True, showfliers=False,
            medianprops={'color': 'black', 'linewidth': 1.0},
            whiskerprops={'color': 'black', 'linewidth': 0.8},
            capprops={'color': 'black', 'linewidth': 0.8},
            boxprops={'edgecolor': 'black', 'linewidth': 0.5},
        )
        for box, c in zip(parts['boxes'], colors):
            box.set_facecolor(c)
            box.set_alpha(0.75)
        whisker_xs = np.concatenate([w.get_xdata() for w in parts['whiskers']])
        bulk_lo = min(float(whisker_xs.min()), 1.0)
        bulk_hi = max(float(whisker_xs.max()), 1.0)
        xlabel = 'MSE / Median no-movement MSE'

    ax.axvline(1.0, color='black', linewidth=0.5, linestyle='--')
    ax.set_yticks(y_pos)
    display_values = [str(v).capitalize() for v in values] if category == 'MainType' else values
    ax.set_yticklabels(display_values, fontsize=8)
    # Tighten y-limits so boxes sit flush against the axes edges (boxes have
    # widths=0.6, so 0.5 above/below the outermost positions leaves a 0.2 gap,
    # matching the inter-row gap — no extra slack on top of that).
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlabel(xlabel, fontsize=8)

    span = max(bulk_hi - bulk_lo, 1e-6)
    # Hard-cap the upper limit at 1.3 to focus on the no-skill region; bars or
    # whiskers extending past that are visually clipped.
    # ax.set_xlim(max(bulk_lo - span * 0.05, 0.0), min(bulk_hi + span * 0.35, 1.3))
    ax.set_xlim(right=5)
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    right_edge = ax.get_xlim()[1]

    for i, (s, p, nt) in enumerate(zip(summaries, p_vals, n_trajs)):
        stars = significance_stars(p)
        label = f'{stars}  n={nt}' if stars else f'n={nt}'
        # Anchor just inside the right edge of the box (q3), clamped to the
        # visible plot edge for boxes that extend past the upper xlim.
        x = min(s['q3'], right_edge) - x_range * 0.01
        # ax.text(x, i, label, va='center', ha='right', fontsize=7)

    os.makedirs(fig_dir, exist_ok=True)
    fname = f'eval_breakdown_{category}_{horizon}d.png'
    fig.savefig(os.path.join(fig_dir, fname), dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved {fname}")


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    print("Re-evaluating model on val subsets...")
    metrics_df, losses_by_key = compute_breakdown_metrics(cfg)
    if len(metrics_df) == 0:
        print("No metrics produced; nothing to plot.")
        return

    print("\nRunning significance tests (Wilcoxon signed-rank, BH-corrected)...")
    metrics_df = test_significance(metrics_df, losses_by_key)

    sig_col = np.full(len(metrics_df), False)
    for horizon in metrics_df['horizon'].unique().to_list():
        mask = metrics_df['horizon'].to_numpy() == horizon
        p_vals = metrics_df['p_value'].to_numpy().copy()
        p_vals[~mask] = np.nan
        sig_col |= benjamini_hochberg(p_vals)
    metrics_df = metrics_df.with_columns(pl.Series('significant', sig_col))

    for horizon in sorted(metrics_df['horizon'].unique().to_list()):
        print(f"\n=== {horizon}-day horizon ===")
        h_df = metrics_df.filter(pl.col('horizon') == horizon) \
            .filter(pl.col('baseline_mse') > MIN_BASELINE_MSE) \
            .sort(['category', 'ratio'], descending=[False, False])
        for row in h_df.iter_rows(named=True):
            stars = significance_stars(row['p_value'])
            print(
                f"  {row['category']:12s} | {row['value']:40s} | "
                f"ratio={row['ratio']:.4f} | frac_better={row['frac_better']:.3f} | "
                f"n_traj={row['n_trajectories']:5d} | {stars}"
            )

    trend_name = os.path.basename(cfg.trend_path.rstrip('/'))
    fig_dir = f'./figs/{trend_name}'
    print(f"\nPlotting to {fig_dir}...")
    for horizon in HORIZONS:
        for category in CATEGORY_COLS:
            plot_category_bars(metrics_df, category, horizon, fig_dir, losses_by_key)

    print("Done.")


if __name__ == '__main__':
    main()
