
import json
import logging
import os

import hydra

import latent_space

NUM_CATS = 3

# the groups describe_dimensions cuts the dimension into, and what to head them
# with: the quantiles it split on, not a polarity, since the cut is by rank
GROUPS = {
    3: ((('negative', '0--5\\%'),
         ('neutral', '5\\% -- 95\\%'),
         ('positive', '95\\% -- 100\\%'))),
    5: ((('very_negative', '0--1\\%'),
         ('negative', '1\\% -- 10\\%'),
         ('neutral', '10\\% -- 90\\%'),
         ('positive', '90\\% -- 99\\%'),
         ('very_positive', '99\\% -- 100\\%'))),
}

TABLE_END = """    \\bottomrule
\\end{tabularx}"""


def format_prior(kind, tau):
    """One dimension's GP prior, short enough for a table cell."""
    if kind == 'const':
        return 'frozen'
    names = {'ou': 'OU', 'wiener': 'Wiener', 'matern32': "Mat\\'ern 3/2",
             'iwp2': 'IWP(2)'}
    return f"{names.get(kind, kind)}, $\\tau$={tau:.0f}\\,d"


def table_start(groups):
    """The header, widened to however many groups a dimension is cut into."""
    n = len(groups)
    heads = ' & '.join(f"\\textbf{{{h}}}" for _, h in groups)
    return f"""\\begin{{tabularx}}{{\\textwidth}}{{c|c|c|X|{'X' * n}}}
\\toprule
& & & & \\multicolumn{{{n}}}{{c}}{{\\textbf{{Description}}}} \\\\
\\cmidrule(lr){{5-{4 + n}}}
\\textbf{{Dim.}} & \\textbf{{Prior}} & \\textbf{{Var.}} & \\textbf{{Targets}} & {heads} \\\\
\\midrule"""


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    logging.info("Loading data...")

    # Save dimension labels to file
    dim_label_path = latent_space.dimension_labels_path(cfg)
    with open(dim_label_path, 'r') as f:
        dimension_labels = json.load(f)

    groups = GROUPS[NUM_CATS]

    # the index is a name, not a rank: a gpfa fit does not order its axes
    priors = latent_space.dimension_priors(cfg)
    shares = latent_space.dimension_variance_share(cfg)

    table_lines = [table_start(groups)]

    max_dim = 5
    for dim_idx in sorted([int(i) for i in dimension_labels.keys()]):
        if dim_idx >= max_dim:
            break
        labels = dimension_labels[str(dim_idx)][f'{NUM_CATS}_cat']

        text_labels = [labels[key].replace('&', '\\&') for key, _ in groups]

        kind, tau = priors[dim_idx]
        share = '--' if shares is None else f"{100 * shares[dim_idx]:.0f}\\%"
        line = f"       {dim_idx+1} & {format_prior(kind, tau)} & {share} & "
        # drop the axis name the label opens with, however long it is
        top_targets = labels['top_features'].partition('(')[2].split(', ')[:4]
        top_targets = [t.replace('&', '\\&') for t in top_targets]
        line += ',\\newline '.join(top_targets) + " & "
        line += ' & '.join(text_labels) + " \\\\"
        table_lines.append(line)

    table_lines.append(TABLE_END)
    table_tex = '\n'.join(table_lines)
    os.makedirs('out', exist_ok=True)
    with open(os.path.join('out', 'dimension_table.tex'), 'w') as f:
        f.write(table_tex)


if __name__ == '__main__':
    main()
