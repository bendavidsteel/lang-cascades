"""Write the seed list every trial is scored on.

The posterior-sd filter drops a seed whose fast dimensions are prior rather
than measurement, but which seeds those are depends on n_fast and n_dims, so
each trial was graded on a population its own configuration chose. Across one
sweep that ranged from 1465 to 4384 of 4384 seeds, and the trials keeping
fewest scored highest.

Post count is the same quantity the sd filter was standing in for -- sd is high
exactly where posts are few -- and it does not move with the latent
configuration, so it can be fixed once and reused.

    python scripts/flows/pin_seeds.py min_seed_posts=1200
"""

import logging

import hydra
import polars as pl

logger = logging.getLogger(__name__)


def seed_posts(cells_path, seed_col='SeedName'):
    """Total labelled posts per seed, over every bin and target."""
    return pl.scan_parquet(cells_path) \
        .group_by(seed_col).agg(pl.col('n').sum().alias('posts')) \
        .collect() \
        .rename({seed_col: 'filter_value'})


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    floor = cfg.get('min_seed_posts', 1200)
    per = seed_posts(cfg.latents.cells_path, cfg.filter_column)
    keep = per.filter(pl.col('posts') >= floor).sort('posts', descending=True)
    if keep.height < 2:
        raise ValueError(f'min_seed_posts={floor} leaves {keep.height} seeds')

    out = cfg.latents.seed_path
    keep.write_parquet(out, compression='zstd')
    logger.info(f'{keep.height} of {per.height} seeds at >= {floor} posts -> {out}')
    logger.info(f'  posts: min {keep["posts"].min()} '
                f'median {keep["posts"].median()} max {keep["posts"].max()}')


if __name__ == '__main__':
    main()
