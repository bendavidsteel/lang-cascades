"""The table saying which latent dimension carries which piece of metadata.

An axis is usually read off the targets it loads on, which cannot separate
three things that move together in Canadian federal politics: whether a user
holds office at all, which side they are on, and whether their side is the one
in government. Each row here is one of those asked of every axis, so the axis
that answers it is the one whose column moves.

Office is asked of members of parliament alone. Over everyone it would also be
answering whether a party's supporters include more commentators than another's.
"""
import logging
import os

import hydra
import numpy as np

import latent_space
from compare_dimensions import _cohens_d, contrast_rows, load_user_means

logger = logging.getLogger(__name__)

# Rows to show, as (key into contrast_rows, how to name it in the table).
ROWS = [
    ('role', 'Role: influencer vs.\\ politician'),
    ('flank', 'Flank: Con.\\,$+$\\,PPC vs.\\ NDP\\,$+$\\,Green'),
    ('office_federal_mps', 'Office: Liberal vs.\\ opposition, MPs only'),
]

TABLE_END = """    \\bottomrule
\\end{tabular}"""

# The dimensions the paper reads. The fit has more, and they carry none of the
# three contrasts, so a column each would be three columns of noise.
MAX_DIMS = 3


def table_start(n_dims, prefix):
    heads = ' & '.join(f"\\textbf{{{prefix}{k + 1}}}" for k in range(n_dims))
    return f"""\\begin{{tabular}}{{l|{'c' * n_dims}}}
\\toprule
\\textbf{{Contrast}} & {heads} \\\\
\\midrule"""


def cell(d, strongest):
    """One effect size, the row's largest one in bold."""
    return f"$\\mathbf{{{d:+.2f}}}$" if strongest else f"${d:+.2f}$"


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    dim_cols = [f'x0_{i}' for i in range(cfg.n_dims)]
    n_dims = min(cfg.n_dims, MAX_DIMS)
    prefix = latent_space.axis_prefix(cfg)

    logger.info("Loading per-user mean positions...")
    rows = contrast_rows(load_user_means(cfg, dim_cols), dim_cols)

    lines = [table_start(n_dims, prefix)]
    for key, label in ROWS:
        _, a, b = rows[key]
        d, lo, hi = _cohens_d(a, b)
        # ranked over every dimension, so the bold is not the largest of three
        # when a dimension the table leaves out carries the contrast
        strongest = int(np.nanargmax(np.abs(d)))
        cells = ' & '.join(cell(d[k], k == strongest) for k in range(n_dims))
        lines.append(f"    {label} & {cells} \\\\")
        logger.info(
            f"{label}: n={len(a)}/{len(b)}, "
            f"d={np.array2string(d, precision=2, sign='+')}, "
            f"CI lo={np.array2string(lo, precision=2, sign='+')}, "
            f"hi={np.array2string(hi, precision=2, sign='+')}")
    lines.append(TABLE_END)

    os.makedirs('out', exist_ok=True)
    out_path = os.path.join('out', 'construct_table.tex')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
