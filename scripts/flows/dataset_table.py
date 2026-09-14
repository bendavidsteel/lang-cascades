import os
import re

import polars as pl

# the actor types the dataset is built from, in the order they are reported
MAIN_TYPES = ['politician', 'influencer', 'foreign']


def _labelled(df, col):
    """Rows carrying a non-blank value of col."""
    return df.drop_nulls(col).filter(pl.col(col) != '')


def post_counts(df, col):
    return _labelled(df, col).group_by(col).agg(pl.len().alias('num_posts'))


def people_counts(df, col):
    """People per value of col, each person assigned the value they post under most.

    MainType and Party both vary across a person's posts, so counting a person
    under every value they ever carry would make the blocks sum past the total.
    """
    return _labelled(df, col) \
        .group_by(['SeedName', col]).agg(pl.len().alias('n')) \
        .group_by('SeedName').agg(
            pl.col(col).sort_by(['n', col], descending=[True, False]).first()) \
        .group_by(col).agg(pl.len().alias('num_people'))


def counts_by(df, col, order=None):
    """(label, num_posts, num_people) per value of col, dropping blanks."""
    counts = post_counts(df, col).join(people_counts(df, col), on=col, how='left') \
        .with_columns(pl.col('num_people').fill_null(0))
    if order is None:
        counts = counts.sort('num_posts', descending=True)
    else:
        counts = counts.with_columns(
            pl.col(col).replace_strict(order, range(len(order)), default=len(order))
            .alias('_rank')).sort(['_rank', col]).drop('_rank')
    return list(counts.iter_rows())


def platform_counts(df):
    """(platform, num_posts, num_handles) -- a platform row counts accounts, not people."""
    counts = df.group_by('platform').agg(
        pl.len().alias('num_posts'),
        pl.col('PlatformHandleID').n_unique().alias('num_handles'),
    ).sort('platform')
    return [(platform.capitalize(), num_posts, num_handles)
            for platform, num_posts, num_handles in counts.iter_rows()]


def main():
    stance_data_path = './data/stance_targets/2022-01-01-onwards_noun_phrase_stance'
    output_path = './out/dataset_table.tex'

    file_paths = [
        os.path.join(stance_data_path, file)
        for file in os.listdir(stance_data_path)
        if re.search(r'\d{4}_\d{1,2}_doc_targets_with_stance.parquet.zstd', file)
    ]

    if not file_paths:
        raise ValueError("No stance data files found in the data directory")

    df = pl.read_parquet(file_paths, columns=['id', 'platform', 'seed'])
    df = df.unique(['id', 'platform'])
    df = df.with_columns([
        pl.col('seed').struct.field(f)
        for f in ('SeedName', 'PlatformHandleID', 'MainType', 'Party')
    ])

    total_num_posts = len(df)
    total_num_users = df['SeedName'].n_unique()

    blocks = [
        ('Platform', 'Num Handles', platform_counts(df)),
        ('Figure type', 'Num People', [
            (main_type.capitalize(), num_posts, num_people)
            for main_type, num_posts, num_people in counts_by(df, 'MainType', MAIN_TYPES)
        ]),
        ('Party', 'Num People', counts_by(df, 'Party')),
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("\\begin{tabular}{lrr}\n")
        f.write("\\toprule\n")

        for i, (heading, unit, rows) in enumerate(blocks):
            if not rows:
                continue
            if i:
                f.write("\\midrule\n")
            f.write(f"{heading} & Num Posts & {unit} \\\\\n")
            f.write("\\midrule\n")
            for label, num_posts, num_people in rows:
                f.write(f"{label} & {num_posts:,} & {num_people:,} \\\\\n")

        f.write("\\midrule\n")
        f.write(f"Total & {total_num_posts:,} & {total_num_users:,} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Dataset statistics by platform, figure type, and party. "
                "Platform rows count accounts, so a person posting from accounts on "
                "several platforms is counted in each of them; the other blocks count "
                "people, each assigned the type or party they post under most often.}\n")
        f.write("\\label{tab:dataset}\n")
        f.write("\\end{table}\n")

    print(f"LaTeX table written to {output_path}")

if __name__ == '__main__':
    main()
