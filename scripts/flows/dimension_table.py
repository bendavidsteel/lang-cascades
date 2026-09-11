
import json
import logging
import os

import hydra

import latent_space

TABLE_END = """    \\bottomrule
\\end{tabularx}"""

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    logging.info("Loading data...")

    trend_path = cfg.trend_path
    trend_name = os.path.basename(trend_path.rstrip('/'))
    keywords = None
    dir_name = f"{trend_name}/all"

    # Save dimension labels to file
    dim_label_path = latent_space.dimension_labels_path(cfg)
    with open(dim_label_path, 'r') as f:
        dimension_labels = json.load(f)

    num_cats = 5

    if num_cats == 5:
        TABLE_START = """\\begin{tabularx}{\\textwidth}{c|X|XXXXX}
\\toprule 
& & \\multicolumn{5}{c}{\\textbf{Description}} \\\\
\\cmidrule(lr){3-7}
\\textbf{Dim.} & \\textbf{Targets} & \\textbf{0-1\%} & \\textbf{1\% - 10\%} & \\textbf{10\% - 90\%} & \\textbf{90\% - 99\%} & \\textbf{99\% - 100\%} \\\\
\\midrule"""
    elif num_cats == 3:
        TABLE_START = """\\begin{tabularx}{\\textwidth}{c|X|XXX}
\\toprule
& & \\multicolumn{3}{c}{\\textbf{Description}} \\\\
\\cmidrule(lr){3-5}
\\textbf{Dim.} & \\textbf{Targets} & \\textbf{Negative} & \\textbf{Neutral} & \\textbf{Positive} \\\\
\\midrule"""

    table_lines = [TABLE_START]

    max_dim = 5
    for dim_idx in sorted([int(i) for i in dimension_labels.keys()]):
        if dim_idx >= max_dim:
            break
        labels = dimension_labels[str(dim_idx)][f'{num_cats}_cat']

        if num_cats == 5:
            text_labels = [
                labels['very_negative'],
                labels['negative'],
                labels['neutral'],
                labels['positive'],
                labels['very_positive']
            ]
        elif num_cats == 3:
            text_labels = [
                labels['negative'],
                labels['neutral'],
                labels['positive']
            ]
        text_labels = [lbl.replace('&', '\\&') for lbl in text_labels]

        line = f"       {dim_idx+1} & "
        top_targets = labels['top_features'][5:].split(', ')[:4]
        top_targets = [t.replace('&', '\\&') for t in top_targets]
        line += ',\\newline '.join(top_targets) + " & "
        if num_cats == 5:
            line += f"{text_labels[0]} & "
            line += f"{text_labels[1]} & "
            line += f"{text_labels[2]} & "
            line += f"{text_labels[3]} & "
            line += f"{text_labels[4]} \\\\"
        elif num_cats == 3:
            line += f"{text_labels[0]} & "
            line += f"{text_labels[1]} & "
            line += f"{text_labels[2]} \\\\"
        table_lines.append(line)

    table_lines.append(TABLE_END)
    table_tex = '\n'.join(table_lines)
    output_path = os.path.join('figs', dir_name)
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join('out', 'dimension_table.tex'), 'w') as f:
        f.write(table_tex)


if __name__ == '__main__':
    main()
