import os

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from plnn.dataset import LandscapeSimulationDataset, NumpyLoader
from plnn.models import DeepTimePhiPLNN

from nn_potential import df_to_data, compute_rolling_means, build_horizon_pairs, \
    evaluate_dataloader, compute_training_split, apply_split
from sweep_runs import state_path as select_state

HORIZON_DAYS = [7, 14, 30, 60, 90]


def print_comparison(name, model_losses, baseline_losses):
    """Print comparison statistics for a given split."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<25} {'Model':<15} {'Baseline':<15}")
    print(f"  {'-' * 55}")

    model_rmse = np.sqrt(model_losses)
    baseline_rmse = np.sqrt(baseline_losses)

    rows = [
        ("Mean MSE", np.mean(model_losses), np.mean(baseline_losses)),
        ("Median MSE", np.median(model_losses), np.median(baseline_losses)),
        ("Mean RMSE", np.mean(model_rmse), np.mean(baseline_rmse)),
        ("Median RMSE", np.median(model_rmse), np.median(baseline_rmse)),
        ("Std RMSE", np.std(model_rmse), np.std(baseline_rmse)),
        ("90th pct RMSE", np.percentile(model_rmse, 90), np.percentile(baseline_rmse, 90)),
    ]
    for label, m, b in rows:
        print(f"  {label:<25} {m:<15.6f} {b:<15.6f}")

    frac_better = np.mean(model_losses < baseline_losses)
    print(f"\n  Model beats baseline: {frac_better * 100:.1f}% of samples")

    skill = 1.0 - np.mean(model_losses) / np.mean(baseline_losses)
    print(f"  Skill score (1 - MSE_model/MSE_baseline): {skill:.4f}", flush=True)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    dims = cfg.dims
    trend_name = os.path.basename(cfg.trend_path.rstrip('/'))

    if cfg.platform != 'all':
        dir_path = f'./out/{trend_name}/dims_{"_".join([str(d) for d in dims])}_{cfg.platform}'
    else:
        dir_path = f'./out/{trend_name}/dims_{"_".join([str(d) for d in dims])}'

    if cfg.rolling_mean_window != 100:
        dir_path = f"{dir_path}_rm{cfg.rolling_mean_window}"

    # Load and prepare rolling mean trajectories
    print("Loading data...", flush=True)
    target_path = os.path.join(cfg.trend_path, 'pca_coords.parquet.zstd')
    target_df = pl.read_parquet(target_path, columns=['createtime', 'filter_value', 'coord_21d'])

    if cfg.platform != 'all':
        target_df = target_df.filter(
            pl.col('filter_value').cast(pl.String)\
                .str.to_lowercase()\
                .str.contains(f'-{cfg.platform}-')
        )

    target_df = target_df.filter(pl.col('filter_value') != '')\
        .select(['createtime', 'filter_value', 'coord_21d'])\
        .sort(['filter_value', 'createtime'])\
        .rename({'coord_21d': 'x0'})

    print("Computing rolling means...", flush=True)
    rolling_df = compute_rolling_means(cfg, target_df, dims)
    print(f"Rolling mean rows: {len(rolling_df)}", flush=True)

    # Load trained model
    dtype = jnp.float32
    state_path = select_state(dir_path)
    print(f"Loading model from: {state_path}", flush=True)
    model, _ = DeepTimePhiPLNN.load(state_path, dtype=dtype)

    seed = 42
    rng = np.random.default_rng(seed=seed)
    key = jax.random.PRNGKey(int(rng.integers(2**32)))

    # Recover the same train/val split metadata used during training
    val_filter_values, cutoff_time = compute_training_split(cfg)

    # Evaluate at each horizon
    for horizon in HORIZON_DAYS:
        print(f"\n{'#' * 60}", flush=True)
        print(f"  HORIZON: {horizon} days", flush=True)
        print(f"{'#' * 60}", flush=True)

        print("  Building pairs...", flush=True)
        paired_df = build_horizon_pairs(rolling_df, horizon, dims)
        print(f"  Paired samples: {len(paired_df)}", flush=True)

        if len(paired_df) == 0:
            print("  No pairs found at this horizon, skipping.", flush=True)
            continue

        _, val_df = apply_split(
            paired_df, cfg.split_type, cfg.train_fraction,
            val_filter_values=val_filter_values, cutoff_time=cutoff_time,
        )
        print(f"  Val samples: {len(val_df)}", flush=True)

        val_data = df_to_data(val_df)
        val_dataset = LandscapeSimulationDataset(data=val_data)
        val_dataloader = NumpyLoader(
            val_dataset,
            batch_size=min(cfg.eval_batch_size, len(val_dataset)),
            shuffle=False,
        )

        key, subkey = jax.random.split(key)
        print("  Running model inference...", flush=True)
        val_model_losses, val_baseline_losses = evaluate_dataloader(model, val_dataloader, subkey)
        print_comparison(f"Validation ({horizon}d horizon)", val_model_losses, val_baseline_losses)


if __name__ == '__main__':
    main()
