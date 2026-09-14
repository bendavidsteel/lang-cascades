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
\\end{tabular}}"""


def table_start(n_dims, prefix):
    """The header, and the column spacing that keeps it inside a text width.

    Grouped, so the narrower spacing ends with the table rather than carrying
    on into whatever the paper puts next.
    """
    heads = ' & '.join(f"\\textbf{{{prefix}{k + 1}}}" for k in range(n_dims))
    return f"""{{\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{l|r|{'c' * n_dims}}}
\\toprule
\\textbf{{Contrast}} & \\textbf{{$n$}} & {heads} \\\\
\\midrule"""


def cell(d, lo, hi, strongest):
    """One effect size over its interval, the row's largest one in bold."""
    point = f"\\mathbf{{{d:+.2f}}}" if strongest else f"{d:+.2f}"
    return (f"\\shortstack{{${point}$ \\\\ "
            f"{{\\scriptsize $[{lo:+.1f}, {hi:+.1f}]$}}}}")


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    n_dims = cfg.n_dims
    dim_cols = [f'x0_{i}' for i in range(n_dims)]
    prefix = latent_space.axis_prefix(cfg)

    logger.info("Loading per-user mean positions...")
    rows = contrast_rows(load_user_means(cfg, dim_cols), dim_cols)

    lines = [table_start(n_dims, prefix)]
    for key, label in ROWS:
        _, a, b = rows[key]
        d, lo, hi = _cohens_d(a, b)
        strongest = int(np.nanargmax(np.abs(d)))
        cells = ' & '.join(cell(d[k], lo[k], hi[k], k == strongest)
                           for k in range(n_dims))
        lines.append(f"    {label} & {len(a)}/{len(b)} & {cells} \\\\")
        logger.info(f"{label}: n={len(a)}/{len(b)}, "
                    f"d={np.array2string(d, precision=2, sign='+')}")
    lines.append(TABLE_END)

    os.makedirs('out', exist_ok=True)
    out_path = os.path.join('out', 'construct_table.tex')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
