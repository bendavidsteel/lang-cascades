"""Score the model's stance targets and stance labels against manual coding.

This is the in-domain evaluation: the coded pairs are drawn from the study corpus
itself, not from the benchmarks the classifier was tuned on.

Reads one or more CSVs exported by the coding page (make_coding_page.py) and reports:

  * coverage -- how much of the sample was actually coded
  * sample composition -- posts and coded pairs per platform, year, political group
    and text provenance, with the strata a thin sample cannot support flagged
  * target-extraction validity -- the share of extracted targets a coder judged
    relevant to the post, which is the precision of target extraction, at pair and
    post level
  * stance -- accuracy, per-class precision/recall/F1 (the coder is the gold
    standard), macro/micro/weighted averages, the confusion matrix, and
    chance-corrected agreement
  * calibration -- how the model's label frequencies compare with the coder's, and,
    where the sample carries per-class probabilities, a reliability table with ECE,
    MCE, Brier score and NLL
  * subgroup error rates -- extraction and stance error per platform, year,
    political group, actor type and text provenance, with the worst-group gap
  * every pair where the model and the coder disagree, for error analysis

Agreement statistics: with one coder the model-vs-coder pair is scored with Cohen's
kappa (each rater gets its own marginals) and with Fleiss's kappa, which for two
raters is Scott's pi (raters share pooled marginals). Fleiss's kappa is the headline
number once several coders' files are passed, in which case coder-vs-coder agreement
is reported too and the model is scored against the coders' majority label.

Confidence intervals are percentile bootstrap over pairs for headline numbers and
Wilson score intervals for the subgroup proportions, where cells get small enough
that the bootstrap degenerates. Both are seeded, so reruns match.

Columns the coding page does not carry -- text provenance, per-class probabilities --
can be joined on from the sample they were drawn from with --sample.

The report is written to stdout. --output also saves it; give that a .tex path and
the results come out as booktabs tabulars instead, one tabular per file, named after
that path, with no table environment or caption, so each can be \\input where its
caption is written.

Examples:
    python scripts/target_mining/score_coded_posts.py -i out/stance_coding_coder_2026-07-29.csv
    python scripts/target_mining/score_coded_posts.py -i out/stance_coding_coder_2026-07-29.csv \
        --sample out/classified_post_sample_250.csv --output out/coding_tables.tex
    python scripts/target_mining/score_coded_posts.py -i out/coder_bs.csv -i out/coder_jd.csv \
        --output out/coding_report.md
    python scripts/target_mining/score_coded_posts.py --selftest
"""

import argparse
import csv
import math
import os
import random
from collections import Counter, OrderedDict, defaultdict

STANCES = ['FAVOR', 'AGAINST', 'NEUTRAL']

# the four platforms the corpus covers; the sample is expected to reach all of them
EXPECTED_PLATFORMS = ['bluesky', 'instagram', 'tiktok', 'twitter']

# (candidate columns, label) the report breaks results down by; the first column
# present in the data wins, so a sample that records party_family is grouped by it
# and an older one falls back to the raw party name
BREAKDOWNS = [
    (('platform',), 'platform'),
    (('year',), 'year'),
    (('party_family', 'party'), 'political group'),
    (('main_type',), 'actor type'),
    (('actor_group',), 'actor group'),
    (('text_kind',), 'text provenance'),
    (('text_source',), 'text source'),
    (('model_stance',), 'model label'),
]

# conditioning on the model's own label makes its predictions constant within a
# group, which forces kappa to 0 and makes macro-F1 meaningless -- the per-class
# precision table already carries that information. text_source splits into a cell
# per platform-and-field combination, too thin for a rate; text_kind collapses it.
ERROR_BREAKDOWNS = [b for b in BREAKDOWNS
                    if b[0] not in (('model_stance',), ('text_source',))]

# the dimensions the sample is meant to be stratified over, for the coverage check
STRATA = [b for b in BREAKDOWNS if b[0] != ('model_stance',)]

# actor_group is the party and actor-type blocks merged into one column
# ("party:NDP", "type:influencer"), so the paper tables leave it to the text report
PAPER_ERROR_BREAKDOWNS = [b for b in ERROR_BREAKDOWNS if b[0] != ('actor_group',)]

# per-class probability columns, and the single confidence column accepted instead
PROB_COLUMN = 'model_prob_{}'.format
CONFIDENCE_COLUMN = 'model_confidence'

# text_source records which fields of a post the classified document was built from;
# these substrings collapse it to how the text came to be, which is what the
# transcript-derived subgroup turns on
TEXT_KINDS = [('transcript', 'transcript (ASR)'), ('ocr', 'image text (OCR)')]

# subgroup rates below this many pairs are reported but marked as too thin to read
DEFAULT_MIN_CELL = 20

# ---------------------------------------------------------------- statistics


def weighted_stats(triples, labels):
    """Accuracy, per-class P/R/F1 and Cohen's kappa from (gold, pred, weight) triples.

    Weights undo the two-stage design: capping targets per post under-samples pairs
    from posts with many classified targets, so each pair stands for
    n_post_targets / n_sampled_targets of them.
    """
    total = sum(w for _, _, w in triples)
    if not total:
        return None
    accuracy = sum(w for g, p, w in triples if g == p) / total
    per = OrderedDict()
    for c in labels:
        tp = sum(w for g, p, w in triples if g == c and p == c)
        fp = sum(w for g, p, w in triples if g != c and p == c)
        fn = sum(w for g, p, w in triples if g == c and p != c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per[c] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': tp + fn}
    present = [c for c in labels if per[c]['support'] or
               any(p == c for _, p, _ in triples)]
    macro_f1 = sum(per[c]['f1'] for c in present) / len(present) if present else 0.0
    gold_share = {c: sum(w for g, _, w in triples if g == c) / total for c in labels}
    pred_share = {c: sum(w for _, p, w in triples if p == c) / total for c in labels}
    expected = sum(gold_share[c] * pred_share[c] for c in labels)
    kappa = (accuracy - expected) / (1 - expected) if expected < 1 else float('nan')
    # Kish effective sample size: how much precision the weights cost
    effective = total ** 2 / sum(w * w for _, _, w in triples)
    return {'accuracy': accuracy, 'macro_f1': macro_f1, 'kappa': kappa,
            'per_class': per, 'n': len(triples), 'effective_n': effective}


def weights_of(rows):
    """pair_weight column if the sample recorded it, else 1.0 (unweighted)."""
    weights = []
    for r in rows:
        raw = (r.get('pair_weight') or '').strip()
        try:
            w = float(raw)
        except ValueError:
            w = 1.0
        weights.append(w if w > 0 else 1.0)
    return weights


def confusion(gold, pred, labels):
    """counts[(gold, pred)] for aligned label sequences."""
    counts = Counter(zip(gold, pred))
    return {(g, p): counts.get((g, p), 0) for g in labels for p in labels}


def prf(gold, pred, labels):
    """Per-class precision/recall/F1/support plus the three averages."""
    per = OrderedDict()
    for c in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per[c] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': tp + fn}

    n = len(gold)
    present = [c for c in labels if per[c]['support'] or
               any(p == c for p in pred)]        # ignore classes nobody used
    macro = {k: (sum(per[c][k] for c in present) / len(present) if present else 0.0)
             for k in ('precision', 'recall', 'f1')}
    total_support = sum(per[c]['support'] for c in labels) or 1
    weighted = {k: sum(per[c][k] * per[c]['support'] for c in labels) / total_support
                for k in ('precision', 'recall', 'f1')}
    accuracy = sum(1 for g, p in zip(gold, pred) if g == p) / n if n else 0.0
    # single-label multiclass: micro-P = micro-R = micro-F1 = accuracy
    return {'per_class': per, 'macro': macro, 'weighted': weighted,
            'accuracy': accuracy, 'micro_f1': accuracy, 'n': n}


def cohen_kappa(a, b, labels):
    """Chance-corrected agreement between two raters with separate marginals."""
    n = len(a)
    if not n:
        return float('nan')
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[c] / n) * (cb[c] / n) for c in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else float('nan')


def fleiss_kappa(table):
    """Fleiss's kappa from a table of per-item category counts.

    `table` is a list of rows, one per item, each row counting how many raters
    assigned each category. Rows must sum to the same number of raters. With two
    raters this is Scott's pi.
    """
    rows = [r for r in table if sum(r) > 0]
    if not rows:
        return float('nan')
    m = sum(rows[0])
    if m < 2 or any(sum(r) != m for r in rows):
        return float('nan')     # every item needs the same rater count
    n_items = len(rows)
    k = len(rows[0])

    agreement = sum((sum(c * c for c in row) - m) / (m * (m - 1)) for row in rows)
    p_bar = agreement / n_items
    p_cat = [sum(row[j] for row in rows) / (n_items * m) for j in range(k)]
    p_e = sum(p * p for p in p_cat)
    return (p_bar - p_e) / (1 - p_e) if p_e < 1 else float('nan')


def fleiss_from_pairs(rater_labels, labels):
    """Fleiss's kappa for aligned label lists (one list per rater)."""
    index = {c: i for i, c in enumerate(labels)}
    table = []
    for votes in zip(*rater_labels):
        row = [0] * len(labels)
        for v in votes:
            row[index[v]] += 1
        table.append(row)
    return fleiss_kappa(table)


def bootstrap_ci(items, stat, seed=42, reps=2000, alpha=0.05):
    """Percentile CI for a statistic computed over a list of items."""
    n = len(items)
    if n < 2:
        return (float('nan'), float('nan'))
    rng = random.Random(seed)
    values = []
    for _ in range(reps):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        v = stat(sample)
        if not (isinstance(v, float) and math.isnan(v)):
            values.append(v)
    if not values:
        return (float('nan'), float('nan'))
    values.sort()
    lo = values[max(0, int(math.floor(alpha / 2 * len(values))))]
    hi = values[min(len(values) - 1, int(math.ceil((1 - alpha / 2) * len(values))) - 1)]
    return (lo, hi)


def wilson_ci(successes, n, z=1.96):
    """Score interval for a proportion, which holds up where the bootstrap would not.

    Subgroup cells get small, and a percentile bootstrap over ten pairs returns an
    interval made of the few values those pairs can take.
    """
    if not n:
        return (float('nan'), float('nan'))
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def reliability_bins(items, n_bins=10):
    """Bin (confidence, correct, weight) triples by confidence.

    Returns one dict per occupied bin with the weighted mean confidence and the
    weighted accuracy, which is what a reliability diagram plots against each other.
    """
    bins = [[] for _ in range(n_bins)]
    for confidence, correct, weight in items:
        edge = min(n_bins - 1, max(0, int(confidence * n_bins)))
        bins[edge].append((confidence, correct, weight))
    out = []
    for i, rows in enumerate(bins):
        if not rows:
            continue
        total = sum(w for _, _, w in rows)
        out.append({
            'lo': i / n_bins, 'hi': (i + 1) / n_bins, 'n': len(rows), 'weight': total,
            'confidence': sum(c * w for c, _, w in rows) / total,
            'accuracy': sum(w for _, k, w in rows if k) / total,
        })
    return out


def calibration_error(items, n_bins=10):
    """Expected and maximum calibration error over binned confidences."""
    bins = reliability_bins(items, n_bins)
    total = sum(b['weight'] for b in bins)
    if not total:
        return {'ece': float('nan'), 'mce': float('nan'), 'bins': []}
    gaps = [(b, abs(b['accuracy'] - b['confidence'])) for b in bins]
    return {'ece': sum(b['weight'] * g for b, g in gaps) / total,
            'mce': max(g for _, g in gaps),
            'bins': bins}


def brier_score(items, labels):
    """Mean squared error of the whole probability vector against the one-hot gold.

    Ranges from 0 to 2 for a proper multiclass score, and unlike accuracy it moves
    when the model is right for weak reasons.
    """
    total = sum(w for _, _, w in items)
    if not total:
        return float('nan')
    return sum(w * sum((probs.get(c, 0.0) - (1.0 if gold == c else 0.0)) ** 2
                       for c in labels)
               for gold, probs, w in items) / total


def log_loss(items, floor=1e-12):
    """Mean negative log probability of the coder's label."""
    total = sum(w for _, _, w in items)
    if not total:
        return float('nan')
    return -sum(w * math.log(max(floor, probs.get(gold, 0.0)))
                for gold, probs, w in items) / total


def label_frequency(gold, pred, labels, weights=None):
    """Share of each label given by the coder and by the model.

    Equal shares are the weakest sense in which a classifier can be calibrated: it
    can hold while every individual pair is wrong, but a model that is off here
    biases every aggregate built on its labels.
    """
    weights = [1.0] * len(gold) if weights is None else weights
    total = sum(weights) or 1.0
    rows = OrderedDict()
    for c in labels:
        g = sum(w for x, w in zip(gold, weights) if x == c) / total
        p = sum(w for x, w in zip(pred, weights) if x == c) / total
        rows[c] = {'gold': g, 'model': p, 'diff': p - g,
                   'ratio': p / g if g else float('nan')}
    return rows


# ---------------------------------------------------------------- loading

def coder_name(path, rows):
    """Prefer the initials typed into the page; fall back to the file name."""
    names = {r['coder'].strip() for r in rows if r.get('coder', '').strip()}
    if len(names) == 1:
        return names.pop()
    return os.path.splitext(os.path.basename(path))[0]


# what the coder said is never taken from a joined-on table, only from their own file
CODING_COLUMNS = {'coded_relevant', 'coded_stance', 'note', 'coder', 'coded_at'}


def load_table(path):
    """Rows of a CSV or parquet file as dicts of strings, nulls as blanks."""
    if path.endswith('.parquet') or path.endswith('.parquet.zstd'):
        import polars as pl       # only samples written as parquet need it
        rows = pl.read_parquet(path).to_dicts()
    else:
        with open(path, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    return [{k: ('' if v is None else str(v)) for k, v in r.items()} for r in rows]


def load_aux(paths):
    """Per-pair columns the coding page does not carry, from the sample CSVs.

    Text provenance and the model's class probabilities live in the sample the pairs
    were drawn from; joining them back on is what lets one coded file be scored for
    calibration and text-source subgroups without recoding it.
    """
    aux = {}
    for path in paths:
        rows = load_table(path)
        missing = {'platform', 'post_id', 'target'} - set(rows[0] if rows else {})
        if missing:
            raise SystemExit(f'{path} cannot be joined on, missing: {sorted(missing)}')
        for r in rows:
            key = (r['platform'], r['post_id'], r['target'])
            aux.setdefault(key, {}).update(
                {k: v for k, v in r.items() if v and k not in CODING_COLUMNS})
    return aux


def add_text_kind(row):
    """Collapse text_source to how the text came to be: spoken, on-screen, written."""
    source = (row.get('text_source') or '').strip().lower()
    if source:
        row['text_kind'] = next((label for key, label in TEXT_KINDS if key in source),
                                'written')


def load_coding(paths, aux_paths=()):
    """Read each coded CSV into {pair_key: row}, keyed identically across coders."""
    aux = load_aux(aux_paths)
    coders = OrderedDict()
    meta = {}
    for path in paths:
        rows = load_table(path)
        missing = {'platform', 'post_id', 'target', 'model_stance',
                   'coded_relevant', 'coded_stance'} - set(rows[0] if rows else {})
        if missing:
            raise SystemExit(f'{path} is missing columns: {sorted(missing)}')
        name = coder_name(path, rows)
        if name in coders:
            name = f'{name} ({os.path.basename(path)})'
        table = OrderedDict()
        for r in rows:
            key = (r['platform'], r['post_id'], r['target'])
            for column, value in aux.get(key, {}).items():
                if not (r.get(column) or '').strip():
                    r[column] = value
            add_text_kind(r)
            table[key] = r
            meta.setdefault(key, r)
        coders[name] = table
    if aux:
        matched = sum(1 for t in coders.values() for k in t if k in aux)
        print(f'joined extra columns onto {matched} pair(s) from {len(aux)} aux rows')
    return coders, meta


def probs_of(row):
    """The model's per-class probabilities, if the sample recorded them."""
    probs = {}
    for c in STANCES:
        raw = (row.get(PROB_COLUMN(c.lower())) or '').strip()
        if not raw:
            return None
        try:
            probs[c] = float(raw)
        except ValueError:
            return None
    total = sum(probs.values())
    if total <= 0 or any(v < 0 for v in probs.values()):
        return None
    if abs(total - 1.0) > 0.02:      # a rounded or renormalised vector still scores
        probs = {c: v / total for c, v in probs.items()}
    return probs


def confidence_of(row):
    """Probability the model gave its own label, from the full vector or on its own."""
    probs = probs_of(row)
    if probs:
        return max(probs.values())
    raw = (row.get(CONFIDENCE_COLUMN) or '').strip()
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if 0.0 <= value <= 1.0 else None


def relevance_of(row):
    v = (row.get('coded_relevant') or '').strip()
    return {'1': True, '0': False}.get(v)


def stance_of(row):
    v = (row.get('coded_stance') or '').strip().upper()
    return v if v in STANCES else None


# ---------------------------------------------------------------- reporting

def present_column(rows, columns):
    """First of a breakdown's candidate columns that any row actually fills."""
    for column in columns:
        if any((r.get(column) or '').strip() for r in rows):
            return column
    return None


def in_breakdown(row, column):
    """Whether a row belongs in a breakdown at all.

    The party of an influencer or a foreign account is incidental, so the political
    group breakdown covers politicians; the actor type breakdown carries the rest.
    A row whose actor type is unrecorded is kept, since it cannot be ruled out.
    """
    if column not in ('party', 'party_family'):
        return True
    return (row.get('main_type') or '').strip().lower() in ('', 'politician')


def blank_label(row, column):
    """What an empty cell means, where it means something specific.

    A politician with no party holds a non-partisan office -- a mayoralty, a seat in
    a consensus-government legislature, a senate appointment -- rather than having
    an unrecorded one.
    """
    if column in ('party', 'party_family'):
        return 'unaffiliated politician'
    return '(blank)'


def group_of(row, column):
    return (row.get(column) or '').strip() or blank_label(row, column)


def stance_pairs(table, relevant_only):
    """(key, row, coder label, model label) for every pair both sides labelled."""
    pairs = []
    for key, r in table.items():
        gold = stance_of(r)
        pred = (r['model_stance'] or '').strip().upper()
        if gold is None or pred not in STANCES:
            continue
        if relevant_only and relevance_of(r) is not True:
            continue
        pairs.append((key, r, gold, pred))
    return pairs


class Report:
    """Collects lines for stdout and, optionally, a markdown file."""

    def __init__(self):
        self.lines = []

    def __call__(self, text=''):
        self.lines.append(text)
        print(text)

    def head(self, text, char='='):
        self('')
        self(text)
        self(char * len(text))

    def table(self, headers, rows):
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))
        self('  '.join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip())
        self('  '.join('-' * widths[i] for i in range(len(headers))))
        for row in rows:
            self('  '.join(str(c).ljust(widths[i]) for i, c in enumerate(row)).rstrip())


def pct(x):
    return 'n/a' if isinstance(x, float) and math.isnan(x) else f'{100 * x:.1f}%'


def num(x):
    return 'n/a' if isinstance(x, float) and math.isnan(x) else f'{x:.3f}'


def ci(bounds):
    lo, hi = bounds
    if any(isinstance(v, float) and math.isnan(v) for v in (lo, hi)):
        return ''
    return f'[{lo:.3f}, {hi:.3f}]'


def pct_ci(bounds):
    lo, hi = bounds
    if any(isinstance(v, float) and math.isnan(v) for v in (lo, hi)):
        return ''
    return f'[{100 * lo:.1f}%, {100 * hi:.1f}%]'


def report_coverage(out, coders, meta):
    out.head('Coverage')
    rows = []
    for name, table in coders.items():
        both = sum(1 for r in table.values()
                   if relevance_of(r) is not None and stance_of(r) is not None)
        rel_only = sum(1 for r in table.values()
                       if relevance_of(r) is not None and stance_of(r) is None)
        st_only = sum(1 for r in table.values()
                      if relevance_of(r) is None and stance_of(r) is not None)
        blank = len(table) - both - rel_only - st_only
        rows.append([name, len(table), both, rel_only, st_only, blank])
    out.table(['coder', 'pairs', 'fully coded', 'relevance only', 'stance only', 'blank'], rows)
    posts = {(k[0], k[1]) for k in meta}
    out('')
    out(f'{len(meta)} distinct (post, target) pairs over {len(posts)} posts.')


def report_relevance(out, name, table, seed):
    """Share of extracted targets the coder judged relevant to the post."""
    coded = [(k, r) for k, r in table.items() if relevance_of(r) is not None]
    if not coded:
        out('No relevance codes.')
        return
    flags = [relevance_of(r) for _, r in coded]
    share = sum(flags) / len(flags)
    lo, hi = bootstrap_ci(flags, lambda s: sum(s) / len(s), seed=seed)
    out.head(f'Target-extraction validity -- {name}', '-')
    out(f'{sum(flags)} of {len(flags)} extracted targets judged relevant to their post '
        f'= {pct(share)} 95% CI [{pct(lo)}, {pct(hi)}]')
    out('(this is the precision of target extraction as the coder sees it)')

    ws = weights_of([r for _, r in coded])
    if any(abs(w - 1.0) > 1e-9 for w in ws):
        wshare = sum(w for f, w in zip(flags, ws) if f) / sum(ws)
        eff = sum(ws) ** 2 / sum(w * w for w in ws)
        items = list(zip(flags, ws))
        wlo, whi = bootstrap_ci(
            items, lambda s: sum(w for f, w in s if f) / sum(w for _, w in s), seed=seed)
        out(f'weighted to all extracted pairs: {pct(wshare)} '
            f'95% CI [{pct(wlo)}, {pct(whi)}] (effective n {eff:.0f} of {len(ws)})')
        out('  the unweighted figure treats every sampled pair equally, which favours '
            'posts with few targets; the weighted one estimates the share over every '
            'classified pair in the corpus')
    else:
        out('unweighted: this sample carries no pair_weight column, so it estimates the '
            'per-post average rather than the share over all extracted pairs')

    # per post, since a post whose every target is spurious is a different failure
    # from one spurious target among several
    per_post = defaultdict(list)
    for (platform, post_id, _), r in coded:
        per_post[(platform, post_id)].append(relevance_of(r))
    any_relevant = sum(1 for flags in per_post.values() if any(flags))
    none_relevant = sum(1 for flags in per_post.values() if not any(flags))
    out('')
    out(f'{any_relevant} of {len(per_post)} posts have at least one relevant target '
        f'({pct(any_relevant / len(per_post))}); {none_relevant} have none')
    out(f'mean per-post share of relevant targets '
        f'{pct(sum(sum(f) / len(f) for f in per_post.values()) / len(per_post))}')
    out('extraction recall is not identified by this design: coders judge the targets '
        'the model proposed, and never name one it missed')

    # how the coder's two answers interact: if irrelevant targets nearly all get one
    # stance, the "all pairs" stance scores are partly measuring relevance
    both = [r for _, r in coded if stance_of(r) is not None]
    if both:
        out('')
        out('coder relevance x coder stance:')
        out.table(['relevance'] + STANCES + ['total'],
                  [[label] + [sum(1 for r in both if relevance_of(r) is flag
                                  and stance_of(r) == s) for s in STANCES]
                   + [sum(1 for r in both if relevance_of(r) is flag)]
                   for flag, label in [(True, 'relevant'), (False, 'not relevant')]])


def report_stance(out, name, table, seed, relevant_only):
    pairs = stance_pairs(table, relevant_only)
    scope = 'targets coded relevant' if relevant_only else 'all coded pairs'
    out.head(f'Stance vs {name} -- {scope} (n={len(pairs)})', '-')
    if len(pairs) < 2:
        out('Not enough coded pairs to score.')
        return None

    gold = [g for _, _, g, _ in pairs]
    pred = [p for _, _, _, p in pairs]
    scores = prf(gold, pred, STANCES)

    out(f'accuracy / micro-F1  {num(scores["accuracy"])}  '
        f'95% CI {ci(bootstrap_ci(list(zip(gold, pred)), lambda s: prf([g for g, _ in s], [p for _, p in s], STANCES)["accuracy"], seed=seed))}')
    out(f'macro-F1             {num(scores["macro"]["f1"])}  '
        f'95% CI {ci(bootstrap_ci(list(zip(gold, pred)), lambda s: prf([g for g, _ in s], [p for _, p in s], STANCES)["macro"]["f1"], seed=seed))}')
    out(f'weighted F1          {num(scores["weighted"]["f1"])}')
    ck = cohen_kappa(gold, pred, STANCES)
    fk = fleiss_from_pairs([gold, pred], STANCES)
    out(f"Cohen's kappa        {num(ck)}  "
        f'95% CI {ci(bootstrap_ci(list(zip(gold, pred)), lambda s: cohen_kappa([g for g, _ in s], [p for _, p in s], STANCES), seed=seed))}')
    out(f"Fleiss's kappa       {num(fk)}  (two raters, so this is Scott's pi)")

    ws = weights_of([r for _, r, _, _ in pairs])
    if any(abs(w - 1.0) > 1e-9 for w in ws):
        triples = [(g, p, w) for (_, _, g, p), w in zip(pairs, ws)]
        w_stat = weighted_stats(triples, STANCES)
        w_ci = bootstrap_ci(triples,
                            lambda s: weighted_stats(s, STANCES)['kappa'], seed=seed)
        out('')
        out('weighted to the pair-level population:')
        out(f'  accuracy {num(w_stat["accuracy"])}   macro-F1 {num(w_stat["macro_f1"])}   '
            f"Cohen's kappa {num(w_stat['kappa'])} 95% CI {ci(w_ci)}")
        out(f'  effective n {w_stat["effective_n"]:.0f} of {w_stat["n"]} coded pairs')

    out('')
    out('per class (coder = gold standard):')
    out.table(['class', 'precision', 'recall', 'f1', 'support (coder)', 'predicted (model)'],
              [[c, num(v['precision']), num(v['recall']), num(v['f1']), v['support'],
                sum(1 for p in pred if p == c)]
               for c, v in scores['per_class'].items()]
              + [['macro', num(scores['macro']['precision']), num(scores['macro']['recall']),
                  num(scores['macro']['f1']), len(gold), len(pred)]])

    out('')
    out('confusion matrix (rows = coder, columns = model):')
    cm = confusion(gold, pred, STANCES)
    out.table(['coder \\ model'] + STANCES + ['total'],
              [[g] + [cm[(g, p)] for p in STANCES] + [sum(cm[(g, p)] for p in STANCES)]
               for g in STANCES]
              + [['total'] + [sum(cm[(g, p)] for g in STANCES) for p in STANCES] + [len(gold)]])

    return pairs


def report_composition(out, meta, coders, min_cell):
    """What the coded sample actually covers, stratum by stratum."""
    out.head('Sample composition')
    rows = list(meta.values())
    coded = {k for table in coders.values() for k, r in table.items()
             if relevance_of(r) is not None and stance_of(r) is not None}
    posts = {(k[0], k[1]) for k in meta}
    out(f'{len(meta)} sampled (post, target) pairs over {len(posts)} posts; '
        f'{len(coded)} pairs fully coded by at least one coder.')
    out(f'cells with fewer than {min_cell} coded pairs are marked thin: reported, but '
        f'too small to read a rate off.')

    for columns, label in STRATA:
        column = present_column(rows, columns)
        if column is None:
            out('')
            out(f'{label}: not recorded in these files (looked for '
                f'{", ".join(columns)}) -- join it on with --sample')
            continue
        groups = defaultdict(lambda: {'pairs': 0, 'coded': 0, 'posts': set()})
        for key, r in meta.items():
            if not in_breakdown(r, column):
                continue
            group = groups[group_of(r, column)]
            group['pairs'] += 1
            group['posts'].add((key[0], key[1]))
            group['coded'] += key in coded
        out('')
        out(f'by {label} ({column}):')
        out.table(['group', 'posts', 'pairs', 'coded', ''],
                  [[g, len(v['posts']), v['pairs'], v['coded'],
                    'thin' if v['coded'] < min_cell else '']
                   for g, v in sorted(groups.items(),
                                      key=lambda kv: -kv[1]['coded'])])

    platform_column = present_column(rows, ('platform',))
    year_column = present_column(rows, ('year',))
    if platform_column and year_column:
        cells = Counter((group_of(r, platform_column), group_of(r, year_column))
                        for k, r in meta.items() if k in coded)
        platforms = sorted({p for p, _ in cells})
        years = sorted({y for _, y in cells})
        out('')
        out('coded pairs by platform x year:')
        out.table(['platform'] + years + ['total'],
                  [[p] + [cells.get((p, y), 0) for y in years]
                   + [sum(cells.get((p, y), 0) for y in years)] for p in platforms]
                  + [['total'] + [sum(cells.get((p, y), 0) for p in platforms)
                                  for y in years] + [sum(cells.values())]])

    out('')
    out('coverage the reviewer asked for:')
    seen = {group_of(r, 'platform').lower() for k, r in meta.items() if k in coded}
    absent = [p for p in EXPECTED_PLATFORMS if p not in seen]
    out(f'  platforms: {len(seen & set(EXPECTED_PLATFORMS))} of '
        f'{len(EXPECTED_PLATFORMS)} covered' + (f', missing {", ".join(absent)}'
                                                if absent else ''))
    years = {group_of(r, 'year') for k, r in meta.items() if k in coded} - {'(blank)'}
    out(f'  years: {len(years)} distinct'
        + (f' ({min(years)}-{max(years)})' if years else ''))
    group_column = present_column(rows, ('party_family', 'party'))
    if group_column:
        families = {group_of(r, group_column) for k, r in meta.items()
                    if k in coded and in_breakdown(r, group_column)}
        out(f'  political groups: {len(families)} named '
            f'({", ".join(sorted(families))})')
    else:
        out('  political groups: no party column in these files')
    if present_column(rows, ('text_kind',)):
        kinds = Counter(group_of(r, 'text_kind') for k, r in meta.items() if k in coded)
        out('  text provenance: '
            + ', '.join(f'{k} {v}' for k, v in kinds.most_common()))
    else:
        out('  text provenance: not recorded, so transcript-derived posts cannot be '
            'told apart -- rerun sample_classified_posts.py for a text_source column '
            'and pass the sample with --sample')


def report_calibration(out, name, table, seed, n_bins):
    """How well the model's own confidence, and its label rates, match the coder."""
    pairs = stance_pairs(table, relevant_only=True)
    scope = 'coded relevant'
    if not pairs:
        pairs, scope = stance_pairs(table, False), 'coded, relevance unknown'
    out.head(f'Calibration -- {name}', '-')
    if len(pairs) < 2:
        out('Not enough coded pairs to score.')
        return
    gold = [g for _, _, g, _ in pairs]
    pred = [p for _, _, _, p in pairs]
    weights = weights_of([r for _, r, _, _ in pairs])
    weighted = any(abs(w - 1.0) > 1e-9 for w in weights)

    out(f'label frequencies (n={len(pairs)} pairs {scope}):')
    freq = label_frequency(gold, pred, STANCES)
    wfreq = label_frequency(gold, pred, STANCES, weights) if weighted else None
    headers = ['class', 'coder', 'model', 'model - coder', 'ratio']
    body = [[c, pct(v['gold']), pct(v['model']), f"{100 * v['diff']:+.1f}pp",
             num(v['ratio'])] for c, v in freq.items()]
    if wfreq:
        headers += ['coder (wtd)', 'model (wtd)']
        body = [row + [pct(wfreq[row[0]]['gold']), pct(wfreq[row[0]]['model'])]
                for row in body]
    out.table(headers, body)
    worst = max(freq.items(), key=lambda kv: abs(kv[1]['diff']))
    out(f'largest label-rate gap: {worst[0]} '
        f'{"over" if worst[1]["diff"] > 0 else "under"}-predicted by '
        f'{abs(100 * worst[1]["diff"]):.1f}pp')
    out('(matching label rates is the weakest sense of calibration, but it is the one '
        'that biases aggregate stance series)')

    scored = [(r, g, p, w) for (_, r, g, p), w in zip(pairs, weights)
              if confidence_of(r) is not None]
    if not scored:
        out('')
        out('confidence calibration: this sample carries no class probabilities.')
        out(f'  record {PROB_COLUMN("favor")}/{PROB_COLUMN("against")}/'
            f'{PROB_COLUMN("neutral")} (or {CONFIDENCE_COLUMN} alone) per pair and '
            f'pass that file with --sample')
        return

    out('')
    out(f'confidence calibration on {len(scored)} of {len(pairs)} pairs '
        f'({n_bins} equal-width bins):')
    items = [(confidence_of(r), g == p, 1.0) for r, g, p, _ in scored]
    calibration = calibration_error(items, n_bins)
    out.table(['confidence bin', 'n', 'mean confidence', 'accuracy', 'gap'],
              [[f'{b["lo"]:.1f}-{b["hi"]:.1f}', b['n'], num(b['confidence']),
                num(b['accuracy']), f'{b["accuracy"] - b["confidence"]:+.3f}']
               for b in calibration['bins']])
    ece_ci = bootstrap_ci(items, lambda s: calibration_error(s, n_bins)['ece'], seed=seed)
    out(f'ECE {num(calibration["ece"])}  95% CI {ci(ece_ci)}   '
        f'MCE {num(calibration["mce"])}')
    mean_confidence = sum(c for c, _, _ in items) / len(items)
    accuracy = sum(1 for _, correct, _ in items if correct) / len(items)
    out(f'mean confidence {num(mean_confidence)} vs accuracy {num(accuracy)}: '
        f'{"over" if mean_confidence > accuracy else "under"}confident by '
        f'{abs(mean_confidence - accuracy):.3f}')

    vectors = [(g, probs_of(r), 1.0) for r, g, _, _ in scored if probs_of(r)]
    if vectors:
        out(f'Brier score {num(brier_score(vectors, STANCES))} (0 perfect, 2 worst)   '
            f'NLL {num(log_loss(vectors))}')
        out('')
        out('class-wise calibration (mean predicted probability vs observed rate):')
        out.table(['class', 'mean predicted', 'observed', 'difference'],
                  [[c, num(predicted), num(observed), f'{predicted - observed:+.3f}']
                   for c, predicted, observed in
                   ((c, sum(probs[c] for _, probs, _ in vectors) / len(vectors),
                     sum(1 for g, _, _ in vectors if g == c) / len(vectors))
                    for c in STANCES)])
    else:
        out('only a single confidence per pair, so Brier score and NLL are not '
            'available; record the full class vector for those')

    if weighted:
        weighted_items = [(confidence_of(r), g == p, w) for r, g, p, w in scored]
        weighted_calibration = calibration_error(weighted_items, n_bins)
        line = (f'weighted to the pair-level population: '
                f'ECE {num(weighted_calibration["ece"])}   '
                f'MCE {num(weighted_calibration["mce"])}')
        if vectors:
            weighted_vectors = [(g, probs_of(r), w) for r, g, _, w in scored
                                if probs_of(r)]
            line += (f'   Brier {num(brier_score(weighted_vectors, STANCES))}'
                     f'   NLL {num(log_loss(weighted_vectors))}')
        out('')
        out(line)


def subgroup_rows(items, min_cell):
    """(group, n, errors, rate, Wilson CI, thin) for {group: [error flags]}."""
    rows = []
    for group, flags in sorted(items.items(), key=lambda kv: -len(kv[1])):
        errors = sum(flags)
        rows.append([group, len(flags), errors, pct(errors / len(flags)),
                     pct_ci(wilson_ci(errors, len(flags))),
                     'thin' if len(flags) < min_cell else ''])
    return rows


def report_gap(out, rows, min_cell, label):
    """Spread in an error rate across the groups big enough to read one off."""
    usable = [(r[0], r[2] / r[1]) for r in rows if r[1] >= min_cell]
    if len(usable) < 2:
        out(f'  fewer than two {label} groups reach {min_cell} pairs, so no gap is '
            f'reported')
        return
    best = min(usable, key=lambda kv: kv[1])
    worst = max(usable, key=lambda kv: kv[1])
    out(f'  worst-group gap: {worst[0]} {pct(worst[1])} vs {best[0]} {pct(best[1])}, '
        f'{100 * (worst[1] - best[1]):.1f}pp apart')


def report_subgroups(out, name, table, seed, min_cell):
    """Extraction and stance error rates per subgroup, with the worst-group gap."""
    out.head(f'Subgroup error rates -- {name}', '-')
    rows = list(table.values())

    coded = [(k, r) for k, r in table.items() if relevance_of(r) is not None]
    pairs = stance_pairs(table, relevant_only=True)
    out(f'extraction error = share of extracted targets the coder judged not '
        f'relevant (n={len(coded)})')
    out(f'stance error = share of relevant targets given the wrong stance '
        f'(n={len(pairs)})')

    for columns, label in ERROR_BREAKDOWNS:
        column = present_column(rows, columns)
        if column is None:
            continue
        extraction = defaultdict(list)
        for _, r in coded:
            if in_breakdown(r, column):
                extraction[group_of(r, column)].append(not relevance_of(r))
        stance = defaultdict(list)
        for _, r, gold, pred in pairs:
            if in_breakdown(r, column):
                stance[group_of(r, column)].append(gold != pred)
        if len(extraction) < 2 and len(stance) < 2:
            continue

        out('')
        out(f'by {label} ({column}) -- extraction:')
        extraction_rows = subgroup_rows(extraction, min_cell)
        out.table(['group', 'coded', 'not relevant', 'extraction error', '95% CI', ''],
                  extraction_rows)
        report_gap(out, extraction_rows, min_cell, label)

        out('')
        out(f'by {label} ({column}) -- stance:')
        stance_rows = []
        for group, flags in sorted(stance.items(), key=lambda kv: -len(kv[1])):
            group_pairs = [(g, p) for _, r, g, p in pairs
                           if in_breakdown(r, column)
                           and group_of(r, column) == group]
            golds = [g for g, _ in group_pairs]
            preds = [p for _, p in group_pairs]
            scores = prf(golds, preds, STANCES)
            errors = sum(flags)
            stance_rows.append([group, len(flags), errors, pct(errors / len(flags)),
                                pct_ci(wilson_ci(errors, len(flags))),
                                num(scores['macro']['f1']),
                                num(cohen_kappa(golds, preds, STANCES)),
                                'thin' if len(flags) < min_cell else ''])
        out.table(['group', 'n', 'wrong', 'stance error', '95% CI', 'macro-F1',
                   "Cohen's kappa", ''], stance_rows)
        report_gap(out, stance_rows, min_cell, label)


def report_disagreements(out, pairs, limit):
    rows = [(k, r, g, p) for k, r, g, p in pairs if g != p]
    out.head(f'Disagreements ({len(rows)} of {len(pairs)})', '-')
    if not rows:
        out('None.')
        return
    shown = rows if limit <= 0 else rows[:limit]
    for k, r, g, p in shown:
        rel = relevance_of(r)
        flag = '' if rel is None else ('' if rel else '  [coded not relevant]')
        out(f'- "{r["target"]}" -- coder {g}, model {p}{flag}')
        out(f'  {r["seed_name"]} ({r["platform"]}, {r["year"]}) {r["post_url"]}')
        if (r.get('note') or '').strip():
            out(f'  note: {r["note"].strip()}')
    if len(rows) > len(shown):
        out(f'... {len(rows) - len(shown)} more (use --show-disagreements 0 for all)')


def report_between_coders(out, coders, seed):
    """Coder-vs-coder agreement, then the model against the coders' majority."""
    names = list(coders)
    out.head('Between coders')

    for field, getter, labels in [('relevance', relevance_of, [True, False]),
                                  ('stance', stance_of, STANCES)]:
        shared = [k for k in coders[names[0]]
                  if all(getter(coders[n].get(k, {})) is not None for n in names)]
        if len(shared) < 2:
            out(f'{field}: no pairs coded by every coder.')
            continue
        per_rater = [[getter(coders[n][k]) for k in shared] for n in names]
        fk = fleiss_from_pairs(per_rater, labels)
        out('')
        out(f'{field}: {len(shared)} pairs coded by all {len(names)} coders, '
            f"Fleiss's kappa {num(fk)}")
        if len(names) == 2:
            out(f"  Cohen's kappa {num(cohen_kappa(per_rater[0], per_rater[1], labels))}")
        rows = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                agree = sum(1 for a, b in zip(per_rater[i], per_rater[j]) if a == b)
                rows.append([names[i], names[j], f'{agree}/{len(shared)}',
                             pct(agree / len(shared)),
                             num(cohen_kappa(per_rater[i], per_rater[j], labels))])
        out.table(['coder A', 'coder B', 'agree', 'raw', "Cohen's kappa"], rows)

    # majority label per pair, ties dropped, then score the model against it
    shared = [k for k in coders[names[0]]
              if all(stance_of(coders[n].get(k, {})) is not None for n in names)]
    gold, pred, ties = [], [], 0
    for k in shared:
        votes = Counter(stance_of(coders[n][k]) for n in names)
        top = votes.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            ties += 1
            continue
        model = (coders[names[0]][k]['model_stance'] or '').strip().upper()
        if model in STANCES:
            gold.append(top[0][0])
            pred.append(model)
    if len(gold) >= 2:
        s = prf(gold, pred, STANCES)
        out('')
        out(f'model vs coder majority (n={len(gold)}, {ties} ties dropped): '
            f'accuracy {num(s["accuracy"])}, macro-F1 {num(s["macro"]["f1"])}, '
            f"Cohen's kappa {num(cohen_kappa(gold, pred, STANCES))}")


# ---------------------------------------------------------------- latex

LATEX_SPECIALS = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
                  '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
                  '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}

THIN_MARK = r'\,$^{\dagger}$'


def tex(value):
    """Escape a data-derived cell; markup is added around the result, not inside it."""
    return ''.join(LATEX_SPECIALS.get(c, c) for c in str(value))


def tex_pct(x):
    """A rate as a bare number; the column header carries the unit."""
    return 'n/a' if isinstance(x, float) and math.isnan(x) else f'{100 * x:.1f}'


def tex_pct_ci(bounds):
    lo, hi = bounds
    if any(isinstance(v, float) and math.isnan(v) for v in (lo, hi)):
        return ''
    return f'[{100 * lo:.1f}, {100 * hi:.1f}]'


def tex_row(cells):
    return ' & '.join(str(c) for c in cells) + r' \\'


def tex_block(label, span):
    return rf'\multicolumn{{{span}}}{{l}}{{\textit{{{tex(label)}}}}} \\'


def tex_tabular(spec, header, body):
    """One booktabs tabular, with any trailing rule dropped."""
    while body and body[-1] == r'\midrule':
        body.pop()
    return ([r'\begin{tabular}{' + spec + '}', r'\toprule'] + header
            + [r'\midrule'] + body + [r'\bottomrule', r'\end{tabular}'])


def tex_slug(text):
    return ''.join(c if c.isalnum() else '_' for c in text).strip('_').lower()


def tex_extraction(table, seed):
    coded = [(k, r) for k, r in table.items() if relevance_of(r) is not None]
    if not coded:
        return []
    flags = [relevance_of(r) for _, r in coded]
    share = sum(flags) / len(flags)
    body = [tex_row(['Extracted targets judged relevant', len(flags),
                     tex_pct(share),
                     tex_pct_ci(bootstrap_ci(flags, lambda s: sum(s) / len(s),
                                             seed=seed))])]
    weights = weights_of([r for _, r in coded])
    if any(abs(w - 1.0) > 1e-9 for w in weights):
        items = list(zip(flags, weights))
        body.append(tex_row([
            r'\quad weighted to all extracted pairs', len(flags),
            tex_pct(sum(w for f, w in items if f) / sum(weights)),
            tex_pct_ci(bootstrap_ci(
                items, lambda s: sum(w for f, w in s if f) / sum(w for _, w in s),
                seed=seed))]))
    per_post = defaultdict(list)
    for (platform, post_id, _), r in coded:
        per_post[(platform, post_id)].append(relevance_of(r))
    any_relevant = sum(1 for f in per_post.values() if any(f))
    body.append(tex_row(['Posts with at least one relevant target', len(per_post),
                         tex_pct(any_relevant / len(per_post)),
                         tex_pct_ci(wilson_ci(any_relevant, len(per_post)))]))
    body.append(tex_row(['Mean per-post share of relevant targets', len(per_post),
                         tex_pct(sum(sum(f) / len(f) for f in per_post.values())
                                 / len(per_post)), '']))
    return [('extraction',
             'target-extraction validity; recall is not identified by this design',
             tex_tabular('lrrr', [tex_row(['', '$n$', r'\%', r'95\% CI'])], body))]


def tex_scopes(table):
    """The two scopes every stance table is cut by, dropping any that is empty.

    Every extracted target is scored, and then only the ones the coder judged
    relevant: irrelevant targets are nearly all coded NEUTRAL, so the wider scope
    folds "this target does not belong to the post" into that class.
    """
    scopes = [('All coded pairs', stance_pairs(table, False)),
              ('Coded relevant', stance_pairs(table, True))]
    return [(label, pairs) for label, pairs in scopes if len(pairs) >= 2]


def tex_agreement(table, seed):
    scopes = tex_scopes(table)
    if not scopes:
        return []

    def cell(pairs, stat):
        gold = [g for _, _, g, _ in pairs]
        pred = [p for _, _, _, p in pairs]
        bounds = bootstrap_ci(list(zip(gold, pred)),
                              lambda s: stat([g for g, _ in s], [p for _, p in s]),
                              seed=seed)
        return f'{num(stat(gold, pred))} {ci(bounds)}'.strip()

    metrics = [
        ('Accuracy / micro-F1', lambda g, p: prf(g, p, STANCES)['accuracy']),
        ('Macro-F1', lambda g, p: prf(g, p, STANCES)['macro']['f1']),
        ('Weighted F1', lambda g, p: prf(g, p, STANCES)['weighted']['f1']),
        (r"Cohen's $\kappa$", lambda g, p: cohen_kappa(g, p, STANCES)),
    ]
    body = [tex_row(['$n$ (pairs)'] + [len(pairs) for _, pairs in scopes])]
    body += [tex_row([label] + [cell(pairs, stat) for _, pairs in scopes])
             for label, stat in metrics]
    return [('agreement',
             r'stance agreement with the coder, 95\% bootstrap CI in brackets',
             tex_tabular('l' + 'r' * len(scopes),
                         [tex_row([''] + [tex(label) for label, _ in scopes])], body))]


def tex_per_class(table):
    scopes = tex_scopes(table)
    if not scopes:
        return []
    scores = [(label, prf([g for _, _, g, _ in pairs], [p for _, _, _, p in pairs],
                          STANCES)) for label, pairs in scopes]
    header = [
        tex_row([''] + [rf'\multicolumn{{4}}{{c}}{{{tex(label)}}}'
                        for label, _ in scores]),
        ' '.join(rf'\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}'
                 for i in range(len(scores))),
        tex_row(['Class'] + ['P', 'R', 'F1', '$n$'] * len(scores)),
    ]
    body = []
    for c in STANCES:
        cells = []
        for _, score in scores:
            v = score['per_class'][c]
            cells += [num(v['precision']), num(v['recall']), num(v['f1']),
                      v['support']]
        body.append(tex_row([c.title()] + cells))
    body.append(r'\midrule')
    macro = []
    for _, score in scores:
        macro += [num(score['macro']['precision']), num(score['macro']['recall']),
                  num(score['macro']['f1']), score['n']]
    body.append(tex_row(['Macro'] + macro))
    return [('per_class', 'per-class precision and recall, coder as gold standard',
             tex_tabular('l' + 'rrrr' * len(scores), header, body))]


def tex_reliability_placeholder(n_bins):
    """The reliability table with a dash wherever a number will go.

    The classifier does not record its class probabilities yet, so the table is
    emitted empty to hold its final shape in a manuscript.
    """
    width = 1 / n_bins
    floor = 1 / len(STANCES)     # a softmax over k classes cannot be less confident
    body = []
    for i in range(n_bins):
        lo = i * width
        if lo + width > floor:
            body.append(tex_row([f'{lo:.1f}--{lo + width:.1f}'] + ['--'] * 4))
    body.append(r'\midrule')
    body.append(r'\multicolumn{5}{l}{ECE --, MCE --, Brier --, NLL --} \\')
    return [('reliability',
             'PLACEHOLDER confidence calibration: no class probabilities in this '
             'sample, so every number is a dash',
             tex_tabular('lrrrr',
                         [tex_row(['Confidence', '$n$', 'Mean conf.', 'Accuracy',
                                   'Gap'])], body))]


def tex_reliability(table, seed, n_bins):
    pairs = stance_pairs(table, relevant_only=True) or stance_pairs(table, False)
    weights = weights_of([r for _, r, _, _ in pairs])
    scored = [(r, g, p, w) for (_, r, g, p), w in zip(pairs, weights)
              if confidence_of(r) is not None]
    if len(scored) < 2:
        return tex_reliability_placeholder(n_bins)
    items = [(confidence_of(r), g == p, 1.0) for r, g, p, _ in scored]
    calibration = calibration_error(items, n_bins)
    body = [tex_row([f'{b["lo"]:.1f}--{b["hi"]:.1f}', b['n'], num(b['confidence']),
                     num(b['accuracy']), f'{b["accuracy"] - b["confidence"]:+.3f}'])
            for b in calibration['bins']]
    ece_ci = bootstrap_ci(items, lambda s: calibration_error(s, n_bins)['ece'],
                          seed=seed)
    summary = [f'ECE {num(calibration["ece"])} {ci(ece_ci)}',
               f'MCE {num(calibration["mce"])}']
    vectors = [(g, probs_of(r), 1.0) for r, g, _, _ in scored if probs_of(r)]
    if vectors:
        summary += [f'Brier {num(brier_score(vectors, STANCES))}',
                    f'NLL {num(log_loss(vectors))}']
    body.append(r'\midrule')
    body.append(rf'\multicolumn{{5}}{{l}}{{{", ".join(summary)}}} \\')
    return [('reliability', r'confidence calibration, 95\% bootstrap CI on ECE',
             tex_tabular('lrrrr',
                         [tex_row(['Confidence', '$n$', 'Mean conf.', 'Accuracy',
                                   'Gap'])], body))]


def tex_subgroups(table, min_cell):
    rows = list(table.values())
    coded = [(k, r) for k, r in table.items() if relevance_of(r) is not None]
    pairs = stance_pairs(table, relevant_only=True)
    body = []
    for columns, label in PAPER_ERROR_BREAKDOWNS:
        column = present_column(rows, columns)
        if column is None:
            continue
        groups = OrderedDict()

        def cell(name):
            return groups.setdefault(name, {'posts': set(), 'extraction': [],
                                            'stance': []})

        for (platform, post_id, _), r in coded:
            if not in_breakdown(r, column):
                continue
            group = cell(group_of(r, column))
            group['posts'].add((platform, post_id))
            group['extraction'].append(not relevance_of(r))
        for _, r, gold, pred in pairs:
            if in_breakdown(r, column):
                cell(group_of(r, column))['stance'].append((gold, pred))
        if len(groups) < 2:
            continue
        body.append(tex_block(label, 10))
        for group, v in sorted(groups.items(),
                               key=lambda kv: -len(kv[1]['extraction'])):
            extraction, stance = v['extraction'], v['stance']
            thin = len(stance) < min_cell or len(extraction) < min_cell
            cells = [tex(group) + (THIN_MARK if thin else ''), len(v['posts']),
                     len(extraction)]
            if extraction:
                errors = sum(extraction)
                cells += [tex_pct(errors / len(extraction)),
                          tex_pct_ci(wilson_ci(errors, len(extraction)))]
            else:
                cells += ['', '']
            cells.append(len(stance))
            if stance:
                golds = [g for g, _ in stance]
                preds = [p for _, p in stance]
                wrong = sum(1 for g, p in stance if g != p)
                cells += [tex_pct(wrong / len(stance)),
                          tex_pct_ci(wilson_ci(wrong, len(stance))),
                          num(prf(golds, preds, STANCES)['macro']['f1']),
                          num(cohen_kappa(golds, preds, STANCES))]
            else:
                cells += ['', '', '', '']
            body.append(tex_row(cells))
        body.append(r'\midrule')
    if not body:
        return []
    header = [
        tex_row(['', '', r'\multicolumn{3}{c}{Target extraction}',
                 r'\multicolumn{5}{c}{Stance}']),
        r'\cmidrule(lr){3-5} \cmidrule(lr){6-10}',
        tex_row(['Subgroup', 'Posts', '$n$', r'Error \%', r'95\% CI', '$n$',
                 r'Error \%', r'95\% CI', 'Macro-F1', r'$\kappa$']),
    ]
    return [('subgroups',
             f'sample composition and subgroup error rates; dagger marks fewer than '
             f'{min_cell} pairs in either task',
             tex_tabular('lrrrrrrrrr', header, body))]


def latex_tables(coders, meta, seed, min_cell, n_bins):
    """(file slug, lines) per result tabular, captions left to the author.

    One tabular per file so each can be \\input where its caption is written; the
    comment at the top of a file says what it holds and what the dagger means.
    """
    tables = []
    for name, table in coders.items():
        per_coder = (tex_extraction(table, seed) + tex_agreement(table, seed)
                     + tex_per_class(table) + tex_reliability(table, seed, n_bins)
                     + tex_subgroups(table, min_cell))
        suffix = '' if len(coders) == 1 else f'_{tex_slug(name)}'
        tables += [(slug + suffix, comment, lines)
                   for slug, comment, lines in per_coder]

    files = []
    for slug, comment, lines in tables:
        files.append((slug, [
            '% generated by scripts/target_mining/score_coded_posts.py',
            f'% {comment}',
            r'% needs \usepackage{booktabs}; wrap in your own table environment',
        ] + lines))
    return files


# ---------------------------------------------------------------- self-test

def selftest():
    """Check the statistics against hand-computable and published values."""
    gold = ['A', 'A', 'B', 'B', 'C']
    pred = ['A', 'B', 'B', 'B', 'C']
    s = prf(gold, pred, ['A', 'B', 'C'])
    assert abs(s['accuracy'] - 0.8) < 1e-12, s['accuracy']
    assert abs(s['per_class']['A']['precision'] - 1.0) < 1e-12
    assert abs(s['per_class']['A']['recall'] - 0.5) < 1e-12
    assert abs(s['per_class']['A']['f1'] - 2 / 3) < 1e-12
    assert abs(s['per_class']['B']['precision'] - 2 / 3) < 1e-12
    assert abs(s['per_class']['B']['recall'] - 1.0) < 1e-12
    assert abs(s['per_class']['B']['f1'] - 0.8) < 1e-12
    assert abs(s['macro']['f1'] - (2 / 3 + 0.8 + 1.0) / 3) < 1e-12
    assert abs(s['weighted']['f1'] - (2 * 2 / 3 + 2 * 0.8 + 1.0) / 5) < 1e-12
    assert abs(s['micro_f1'] - s['accuracy']) < 1e-12

    # 2x2 with po=0.70, pe=0.50 -> kappa=0.40
    a = ['y'] * 20 + ['y'] * 5 + ['n'] * 10 + ['n'] * 15
    b = ['y'] * 20 + ['n'] * 5 + ['y'] * 10 + ['n'] * 15
    assert abs(cohen_kappa(a, b, ['y', 'n']) - 0.4) < 1e-12, cohen_kappa(a, b, ['y', 'n'])
    assert abs(cohen_kappa(a, a, ['y', 'n']) - 1.0) < 1e-12

    # published Fleiss example: 10 items, 14 raters, 5 categories, kappa = 0.210
    table = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0],
             [2, 2, 8, 1, 1], [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2],
             [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    assert abs(fleiss_kappa(table) - 0.2099) < 5e-4, fleiss_kappa(table)
    assert abs(fleiss_kappa([[2, 0], [0, 2], [2, 0]]) - 1.0) < 1e-12

    # Fleiss with two raters == Scott's pi (pooled marginals), and differs from Cohen
    rng = random.Random(0)
    x = [rng.choice(STANCES) for _ in range(200)]
    y = [rng.choice(STANCES) for _ in range(200)]
    n = len(x)
    po = sum(1 for i, j in zip(x, y) if i == j) / n
    pooled = Counter(x) + Counter(y)
    pe = sum((pooled[c] / (2 * n)) ** 2 for c in STANCES)
    scott = (po - pe) / (1 - pe)
    assert abs(fleiss_from_pairs([x, y], STANCES) - scott) < 1e-12

    # weights of 1 must reproduce the unweighted numbers exactly
    triples = [(g, p, 1.0) for g, p in zip(gold, pred)]
    w = weighted_stats(triples, ['A', 'B', 'C'])
    assert abs(w['accuracy'] - s['accuracy']) < 1e-12
    assert abs(w['macro_f1'] - s['macro']['f1']) < 1e-12
    assert abs(w['kappa'] - cohen_kappa(gold, pred, ['A', 'B', 'C'])) < 1e-12
    assert abs(w['effective_n'] - len(gold)) < 1e-12

    # duplicating a row is the same as doubling its weight
    dup = [(g, p, 1.0) for g, p in zip(gold, pred)] + [(gold[0], pred[0], 1.0)]
    wt = [(gold[0], pred[0], 2.0)] + [(g, p, 1.0) for g, p in list(zip(gold, pred))[1:]]
    a, b = weighted_stats(dup, ['A', 'B', 'C']), weighted_stats(wt, ['A', 'B', 'C'])
    for key in ('accuracy', 'macro_f1', 'kappa'):
        assert abs(a[key] - b[key]) < 1e-12, (key, a[key], b[key])

    # uneven weights cost precision: effective n < number of rows
    uneven = weighted_stats([('A', 'A', 10.0), ('B', 'B', 1.0), ('C', 'A', 1.0)],
                            ['A', 'B', 'C'])
    assert uneven['effective_n'] < 3, uneven['effective_n']
    assert weights_of([{'pair_weight': '2.5'}, {'pair_weight': ''},
                       {'pair_weight': 'x'}, {}]) == [2.5, 1.0, 1.0, 1.0]

    lo, hi = bootstrap_ci([1] * 50 + [0] * 50, lambda s: sum(s) / len(s), reps=500)
    assert 0.3 < lo < 0.5 < hi < 0.7, (lo, hi)
    assert math.isnan(fleiss_kappa([[1, 0], [0, 2]]))     # unequal rater counts

    # Wilson interval: published bounds for 8/10, and it never leaves [0, 1]
    lo, hi = wilson_ci(8, 10)
    assert abs(lo - 0.4903) < 5e-4 and abs(hi - 0.9435) < 5e-4, (lo, hi)
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and abs(hi - 0.2775) < 5e-4, (lo, hi)
    a, b = wilson_ci(3, 17), wilson_ci(14, 17)
    assert abs(a[0] - (1 - b[1])) < 1e-12 and abs(a[1] - (1 - b[0])) < 1e-12
    assert wilson_ci(1, 1) == (max(0.0, wilson_ci(1, 1)[0]), 1.0)

    # calibration: the gap between mean confidence and accuracy, bin by bin
    exact = [(0.55, True, 1.0), (0.55, False, 1.0)]
    assert abs(calibration_error(exact, 10)['ece'] - 0.05) < 1e-12
    over = [(1.0, True, 1.0), (1.0, False, 1.0)]
    assert abs(calibration_error(over, 10)['ece'] - 0.5) < 1e-12
    assert abs(calibration_error(over, 10)['mce'] - 0.5) < 1e-12
    assert len(calibration_error(over, 10)['bins']) == 1     # empty bins are dropped
    # weighting a pair twice is the same as listing it twice
    dup = [(0.9, True, 1.0), (0.9, True, 1.0), (0.2, False, 1.0)]
    wtd = [(0.9, True, 2.0), (0.2, False, 1.0)]
    assert abs(calibration_error(dup, 10)['ece']
               - calibration_error(wtd, 10)['ece']) < 1e-12

    assert brier_score([('A', {'A': 1.0, 'B': 0.0, 'C': 0.0}, 1.0)],
                       ['A', 'B', 'C']) == 0.0
    third = {c: 1 / 3 for c in ('A', 'B', 'C')}
    assert abs(brier_score([('A', third, 1.0)], ['A', 'B', 'C']) - 2 / 3) < 1e-12
    assert abs(log_loss([('A', {'A': 0.5, 'B': 0.5, 'C': 0.0}, 1.0)])
               - math.log(2)) < 1e-12

    freq = label_frequency(['A', 'A', 'B'], ['A', 'B', 'B'], ['A', 'B'])
    assert abs(freq['A']['gold'] - 2 / 3) < 1e-12 and abs(freq['A']['model'] - 1 / 3) < 1e-12
    assert abs(freq['A']['diff'] + 1 / 3) < 1e-12 and abs(freq['A']['ratio'] - 0.5) < 1e-12

    # probability columns: parsed, renormalised, and rejected when incomplete
    row = {'model_prob_favor': '0.7', 'model_prob_against': '0.2',
           'model_prob_neutral': '0.1'}
    assert abs(confidence_of(row) - 0.7) < 1e-12
    assert abs(sum(probs_of({**row, 'model_prob_favor': '7', 'model_prob_against': '2',
                             'model_prob_neutral': '1'}).values()) - 1.0) < 1e-12
    assert probs_of({**row, 'model_prob_neutral': ''}) is None
    assert probs_of({**row, 'model_prob_neutral': 'x'}) is None
    assert confidence_of({'model_confidence': '0.42'}) == 0.42
    assert confidence_of({'model_confidence': '4.2'}) is None
    assert confidence_of({}) is None

    for source, kind in [('caption+transcript', 'transcript (ASR)'),
                         ('transcript', 'transcript (ASR)'),
                         ('caption+ocr', 'image text (OCR)'),
                         ('written', 'written'), ('repost text', 'written')]:
        row = {'text_source': source}
        add_text_kind(row)
        assert row['text_kind'] == kind, (source, row)
    blank = {'text_source': ''}
    add_text_kind(blank)
    assert 'text_kind' not in blank

    rows = [{'party': 'Liberal', 'party_family': ''}, {'party': '', 'party_family': ''}]
    assert present_column(rows, ('party_family', 'party')) == 'party'
    assert present_column(rows, ('text_kind',)) is None
    assert group_of(rows[1], 'party') == 'unaffiliated politician'
    assert group_of(rows[1], 'platform') == '(blank)'
    assert group_of({'party': 'Liberal', 'main_type': 'politician'}, 'party') == 'Liberal'
    # the political group breakdown is politicians, plus rows with no actor type
    for kind, keep in [('politician', True), ('influencer', False), ('foreign', False),
                       ('', True)]:
        assert in_breakdown({'main_type': kind}, 'party') is keep
        assert in_breakdown({'main_type': kind}, 'platform') is True
    print('selftest: all statistics checks pass')


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('-i', '--input', action='append', default=[],
                        help='coded CSV exported by the coding page (repeat per coder)')
    parser.add_argument('--output',
                        help='also write the report to this file; a .tex path instead '
                             'writes one booktabs tabular per file, named after that '
                             'path, e.g. --output out/coding.tex -> '
                             'out/coding_composition.tex and its siblings')
    parser.add_argument('-s', '--sample', action='append', default=[], metavar='CSV',
                        help='the sample the pairs were drawn from, to join on columns '
                             'the coding page does not carry (text_source, class '
                             'probabilities); repeatable')
    parser.add_argument('--show-disagreements', type=int, default=25,
                        help='how many disagreements to list (0 = all)')
    parser.add_argument('--min-cell', type=int, default=DEFAULT_MIN_CELL,
                        help='coded pairs a subgroup needs before its rate is read')
    parser.add_argument('--calibration-bins', type=int, default=10,
                        help='equal-width confidence bins in the reliability table')
    parser.add_argument('--seed', type=int, default=42, help='bootstrap seed')
    parser.add_argument('--selftest', action='store_true',
                        help='verify the statistics against known values and exit')
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.input:
        parser.error('give at least one -i/--input coded CSV (or --selftest)')

    coders, meta = load_coding(args.input, args.sample)
    out = Report()
    out(f'Coded stance evaluation -- {len(coders)} coder(s): {", ".join(coders)}')
    report_coverage(out, coders, meta)
    report_composition(out, meta, coders, args.min_cell)

    for name, table in coders.items():
        report_relevance(out, name, table, args.seed)

    for name, table in coders.items():
        for relevant_only in (False, True):
            pairs = report_stance(out, name, table, args.seed, relevant_only)
            if pairs and not relevant_only:
                report_disagreements(out, pairs, args.show_disagreements)

    for name, table in coders.items():
        report_calibration(out, name, table, args.seed, args.calibration_bins)
        report_subgroups(out, name, table, args.seed, args.min_cell)

    if len(coders) > 1:
        report_between_coders(out, coders, args.seed)

    if args.output:
        directory = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(directory, exist_ok=True)
        if args.output.endswith('.tex'):
            stem = args.output[:-len('.tex')]
            print()
            for slug, lines in latex_tables(coders, meta, args.seed, args.min_cell,
                                            args.calibration_bins):
                path = f'{stem}_{slug}.tex'
                with open(path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines) + '\n')
                print(f'wrote {path}')
        else:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write('```\n' + '\n'.join(out.lines) + '\n```\n')
            print(f'\nwrote {args.output}')


if __name__ == '__main__':
    main()
