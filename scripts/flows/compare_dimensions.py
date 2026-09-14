"""What tells one latent dimension from another.

describe_dimensions names each axis from the posts written at its extremes,
which says what an axis is about but not how two axes that name the same
targets differ. This reads the fit's geometry instead: the signed loadings the
axes share, how much of an axis is a standing position versus movement over the
study period, and which contrast among the party centroids each axis carries.
"""
import logging

import hydra
import numpy as np
import polars as pl
from scipy import stats

import latent_space
import splits
from nn_potential import load_latent_df, rolling_frame
from test_dimensionality import (MIN_GROUP_SIZE, MIN_POINTS_PER_USER,
                                 compute_user_means, load_seed_metadata_full)

logger = logging.getLogger(__name__)

N_TOP = 12


def _score(components, volumes):
    """Ranking weight per loading: its share of the variance the axis drives.

    A loading is a coefficient per unit of the axis, so magnitude alone ranks a
    target the fit barely constrains alongside one that defines the axis.
    """
    w = np.abs(components)
    return w if volumes is None else w * np.sqrt(volumes)


def report_loadings(components, targets, volumes, prefix, n_top=N_TOP):
    """Each axis's leading targets by sign, then the union across every axis.

    Split by sign because an axis is read from what it raises against what it
    lowers, and a ranking on magnitude alone can return one end of it.
    """
    score = _score(components, volumes)
    tops = []
    for k in range(components.shape[0]):
        order = np.argsort(np.where(components[k] > 0, score[k], -score[k]))
        print(f"\n  {prefix}{k + 1}")
        for label, side in (('raises', order[::-1][:n_top]), ('lowers', order[:n_top])):
            print(f"    {label}")
            for j in side:
                vol = '' if volumes is None else f"  n={volumes[j]:>8.0f}"
                print(f"      {targets[j]:<44} {components[k, j]:+.4f}{vol}")
        tops.append(np.concatenate([order[::-1][:n_top], order[:n_top]]))

    shared = sorted(set(int(j) for top in tops for j in top),
                    key=lambda j: -score[:, j].max())
    print(f"\n  Loadings of every leading target on every axis "
          f"(union of the top {n_top}):")
    print(f"    {'target':<44} " +
          ' '.join(f'{prefix}{k + 1:<6}' for k in range(components.shape[0])))
    for j in shared:
        row = ' '.join(f'{components[k, j]:+8.3f}' for k in range(components.shape[0]))
        print(f"    {targets[j]:<44} {row}")


def report_axis_overlap(components, volumes, prefix):
    """Cosine between the axes' loading vectors.

    Two axes can name the same targets and still be different axes; what makes
    them the same is agreeing on the signs of everything else too.
    """
    w = components if volumes is None else components * np.sqrt(volumes)
    unit = w / np.linalg.norm(w, axis=1, keepdims=True)
    cos = unit @ unit.T
    print("\n  Cosine between loading vectors (volume-weighted):")
    print(f"    {'':<6}" + ' '.join(f'{prefix}{k + 1:<6}' for k in range(len(cos))))
    for k, row in enumerate(cos):
        print(f"    {prefix}{k + 1:<4}" + ' '.join(f'{v:+7.3f}' for v in row))


def report_polarity(components, volumes, prefix):
    """Whether an axis is a contrast between two camps or a one-sided one.

    A bipolar axis raises one set of stances as it lowers another, so its
    loadings cancel; a unipolar axis moves a bloc of stances together against
    nothing in particular, and its signed mass survives the sum.
    """
    w = components if volumes is None else components * np.sqrt(volumes)
    balance = w.sum(axis=1) / np.abs(w).sum(axis=1)
    print("\n  Signed share of each axis's weighted loading mass "
          "(0 = two-sided contrast):")
    for k, b in enumerate(balance):
        print(f"    {prefix}{k + 1:<4} {b:+.3f}")


def axis_divergence(components, targets, volumes, a, b, prefix, n_top=10):
    """Where two axes that share their leading targets part company.

    A fitted axis has no inherent sign, so b is first turned to face a; what is
    left is the targets one axis takes a stand on and the other does not.
    """
    w = components if volumes is None else components * np.sqrt(volumes)
    flip = -1.0 if w[a] @ w[b] < 0 else 1.0
    gap, agree = w[a] - flip * w[b], np.abs(w[a] + flip * w[b])
    facing = f"{prefix}{b + 1}" if flip > 0 else f"-{prefix}{b + 1}"
    print(f"\n  {prefix}{a + 1} against {facing}, weighted loadings:")
    for label, order in (('furthest apart', np.argsort(-np.abs(gap))),
                         ('most alike', np.argsort(-agree))):
        print(f"    {label}")
        for j in order[:n_top]:
            print(f"      {targets[j]:<44} {prefix}{a + 1}={components[a, j]:+7.3f}  "
                  f"{facing}={flip * components[b, j]:+7.3f}")


def variance_split(rolling_df, dim_cols, prefix):
    """How much of each axis is who a user is, and how much is when.

    A standing disagreement and a swing in opinion look alike in a cross
    section. They separate here: an axis users hold a fixed place on has its
    variance between users, one they travel along has it within.
    """
    per_user = rolling_df.group_by('filter_value').agg(
        [pl.col(c).mean().alias(f'm_{c}') for c in dim_cols] +
        [pl.col(c).var().alias(f'v_{c}') for c in dim_cols] +
        [pl.len().alias('n_points')]
    ).filter(pl.col('n_points') >= MIN_POINTS_PER_USER)

    between = np.array([per_user[f'm_{c}'].var() for c in dim_cols])
    within = np.array([per_user[f'v_{c}'].mean() for c in dim_cols])

    # cross-sectional mean per time bin: the part of the movement every user
    # shares, as against drift that cancels out across the population
    common = rolling_df.group_by('createtime').agg(
        [pl.col(c).mean().alias(c) for c in dim_cols] + [pl.len().alias('n')]
    ).filter(pl.col('n') >= 30).sort('createtime')
    shared = np.array([common[c].var() for c in dim_cols])

    print(f"\n  {'axis':<6} {'between-user':>13} {'within-user':>12} "
          f"{'ICC':>7} {'common drift':>13} {'/within':>8}")
    for k, c in enumerate(dim_cols):
        icc = between[k] / (between[k] + within[k])
        print(f"  {prefix}{k + 1:<4} {between[k]:>13.4f} {within[k]:>12.4f} "
              f"{icc:>7.3f} {shared[k]:>13.4f} {shared[k] / within[k]:>8.3f}")
    return common


def report_drift(common, dim_cols, prefix, n_periods=10):
    """The population mean of each axis over the study period."""
    t = np.arange(common.height, dtype=float)
    print(f"\n  Systematic movement of the population mean over the period:")
    print(f"    {'axis':<6} {'range':>7} {'rho(t)':>8} {'slope/decade':>13}")
    for k, c in enumerate(dim_cols):
        y = common[c].to_numpy()
        rho = stats.spearmanr(t, y).statistic
        days = (common['createtime'][-1] - common['createtime'][0]).days
        slope = np.polyfit(t, y, 1)[0] * common.height * 3652.5 / max(days, 1)
        print(f"    {prefix}{k + 1:<4} {y.max() - y.min():>7.3f} {rho:>8.3f} "
              f"{slope:>13.3f}")

    stamps = common['createtime'].to_list()
    step = max(len(stamps) // n_periods, 1)
    rows = list(range(0, len(stamps), step))
    print(f"\n  {'date':<12} " + ' '.join(f'{prefix}{k + 1:<7}'
                                          for k in range(len(dim_cols))))
    for i in rows:
        vals = ' '.join(f'{common[c][i]:+8.3f}' for c in dim_cols)
        print(f"  {stamps[i].date()!s:<12} {vals}")


def level_positions(user_means_df, field, dim_cols, prefix, min_size=MIN_GROUP_SIZE,
                    sort_dim=2, n_show=10):
    """The extreme levels of a metadata field along one axis.

    A field with many small levels is read here rather than through eta^2,
    which says an axis separates the levels without saying which ones.
    """
    sub = user_means_df.drop_nulls(field).filter(pl.col(field) != '')
    agg = sub.group_by(field).agg(
        [pl.col(c).mean().alias(c) for c in dim_cols] + [pl.len().alias('n')]
    ).filter(pl.col('n') >= min_size).sort(dim_cols[sort_dim])
    if agg.height <= 2 * n_show:
        rows = list(range(agg.height))
    else:
        rows = list(range(n_show)) + [None] + list(range(agg.height - n_show, agg.height))
    print(f"\n  {field}, the levels at each end of {prefix}{sort_dim + 1}:")
    print(f"    {'level':<34} {'n':>4}  " +
          '  '.join(f'{prefix}{k + 1:<5}' for k in range(len(dim_cols))))
    for i in rows:
        if i is None:
            print('    ...')
            continue
        vals = '  '.join(f'{agg[c][i]:+6.2f}' for c in dim_cols)
        print(f"    {agg[field][i]:<34} {agg['n'][i]:>4}  {vals}")


# The party holding office in each province over most of the study period.
# Names are the metadata's own; a province whose governing party never reaches
# the size floor drops out of the test rather than being guessed at.
PROVINCIAL_GOVERNMENTS = {
    'Coalition avenir Québec',
    'Progressive Conservative Party of Ontario',
    'BC NDP',
    'United Conservative Party of Alberta',
    'Saskatchewan Party',
    'New Democratic Party of Manitoba',
    'Progressive Conservative Association of Nova Scotia',
    'Liberal Party of Newfoundland and Labrador',
    'Progressive Conservative Party of Prince Edward Island',
}


def incumbency_test(user_means_df, dim_cols, prefix, min_size=MIN_GROUP_SIZE):
    """Does an axis put a province's governing party below its opposition?

    Provincial parties cut across the left-right ordering -- a governing party
    is as often the left one as the right one -- so an axis that separates
    them by office rather than by flank is measuring something other than
    ideology. Each province is one paired comparison, and the province a party
    belongs to is read off its members rather than its name.
    """
    sub = user_means_df.drop_nulls(['ProvincialParty', 'Province']) \
        .filter((pl.col('ProvincialParty') != '') & (pl.col('Province') != ''))
    agg = sub.group_by('ProvincialParty').agg(
        [pl.col(c).mean().alias(c) for c in dim_cols] +
        [pl.col('Province').mode().sort().first().alias('Province'), pl.len().alias('n')]
    ).filter(pl.col('n') >= min_size)

    deltas = []
    print(f"\n  Governing party minus its opposition, by province:")
    print(f"    {'province':<22} {'governing party':<44} {'opp':>4}  " +
          '  '.join(f'{prefix}{k + 1:<5}' for k in range(len(dim_cols))))
    for (province,), rows in sorted(agg.group_by('Province'), key=lambda kv: kv[0]):
        gov = rows.filter(pl.col('ProvincialParty').is_in(PROVINCIAL_GOVERNMENTS))
        opp = rows.filter(~pl.col('ProvincialParty').is_in(PROVINCIAL_GOVERNMENTS))
        if gov.height != 1 or opp.height == 0:
            continue
        d = np.array([gov[c][0] - opp[c].mean() for c in dim_cols])
        deltas.append(d)
        row = '  '.join(f'{v:+6.2f}' for v in d)
        print(f"    {province:<22} {gov['ProvincialParty'][0]:<44} "
              f"{opp.height:>4}  {row}")

    if not deltas:
        return
    deltas = np.stack(deltas)
    below = (deltas < 0).sum(0)
    n = len(deltas)
    p = np.array([stats.binomtest(int(b), n, 0.5, alternative='less').pvalue
                  if b <= n / 2 else
                  stats.binomtest(int(b), n, 0.5, alternative='greater').pvalue
                  for b in below])
    print(f"\n    {'':<22} {'mean gap':>44} {'':>4}  " +
          '  '.join(f'{prefix}{k + 1:<5}' for k in range(len(dim_cols))))
    print(f"    {'':<22} {'governing below opposition in':>44} {'':>4}  " +
          '  '.join(f'{b:>2d}/{n:<3d}' for b in below))
    print(f"    {'':<22} {'mean gap, SDs of the level spread':>44} {'':>4}  " +
          '  '.join(f'{v:+6.2f}' for v in deltas.mean(0) /
                    np.array([agg[c].std() for c in dim_cols])))
    print(f"    {'':<22} {'sign test p':>44} {'':>4}  " +
          '  '.join(f'{v:>6.3f}' for v in p))


def _cohens_d(a, b, n_boot=1000, seed=42):
    """Separation of two sets of positions per dim, in pooled SDs, with a CI."""
    rng = np.random.default_rng(seed)

    def d(x, y):
        n, m = len(x), len(y)
        pooled = np.sqrt(((n - 1) * x.var(0, ddof=1) + (m - 1) * y.var(0, ddof=1))
                         / max(n + m - 2, 1))
        return np.where(pooled > 0, (x.mean(0) - y.mean(0)) / pooled, np.nan)

    draws = np.stack([d(a[rng.integers(len(a), size=len(a))],
                        b[rng.integers(len(b), size=len(b))])
                      for _ in range(n_boot)])
    return d(a, b), *np.percentile(draws, [2.5, 97.5], axis=0)


def report_contrasts(rows, dim_cols, prefix):
    """One line per construct: how far it moves each axis, in pooled SDs.

    Three things the metadata can say about a user -- whether they hold office
    at all, which side they are on, and whether their side is the one in
    government -- are separate questions that a single reading of the axes
    tends to run together. Each row is one of them, and an axis belongs to the
    row that moves it.
    """
    print(f"\n  {'contrast':<44} {'n':>11}  " +
          '  '.join(f'{prefix}{k + 1:<13}' for k in range(len(dim_cols))))
    for label, a, b in rows:
        if len(a) < MIN_GROUP_SIZE or len(b) < MIN_GROUP_SIZE:
            print(f"  {label:<44} {len(a):>5}/{len(b):<5}  too few users")
            continue
        d, lo, hi = _cohens_d(a, b)
        cells = '  '.join(f'{d[k]:+5.2f} [{lo[k]:+4.1f},{hi[k]:+4.1f}]'
                          for k in range(len(dim_cols)))
        print(f"  {label:<44} {len(a):>5}/{len(b):<5}  {cells}")


def _positions(df, dim_cols, *predicates):
    out = df
    for pred in predicates:
        out = out.filter(pred)
    return out.select(dim_cols).to_numpy()


def _logit_weights(a, b, n_boot=400, seed=42):
    """Standardized logistic weights separating two groups, with CIs.

    A contrast read one axis at a time credits every axis that correlates with
    the label; these credit only what an axis adds once the others are known,
    which is what says an axis carries a construct rather than borrowing it.

    Ridged, because a contrast the axes separate cleanly sends an unpenalised
    fit's weights off to wherever the optimiser stops.
    """
    from sklearn.linear_model import LogisticRegression

    x = np.vstack([a, b])
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    mu, sd = x.mean(0), x.std(0)
    scale = np.where(sd > 0, sd, 1.0)

    def fit(idx):
        model = LogisticRegression(C=1.0, max_iter=2000)
        model.fit((x[idx] - mu) / scale, y[idx])
        return model.coef_[0]

    rng = np.random.default_rng(seed)
    at, bt = np.arange(len(a)), len(a) + np.arange(len(b))
    draws = np.stack([fit(np.r_[rng.choice(at, at.size), rng.choice(bt, bt.size)])
                      for _ in range(n_boot)])
    return fit(np.arange(len(y))), *np.percentile(draws, [2.5, 97.5], axis=0)


def report_unique(rows, dim_cols, prefix):
    """The same contrasts, each axis credited only for what it alone explains."""
    print(f"\n  {'contrast':<44} {'n':>11}  " +
          '  '.join(f'{prefix}{k + 1:<13}' for k in range(len(dim_cols))))
    for label, a, b in rows:
        if min(len(a), len(b)) < 5 * len(dim_cols):
            continue
        w, lo, hi = _logit_weights(a, b)
        cells = '  '.join(f'{w[k]:+5.2f} [{lo[k]:+4.1f},{hi[k]:+4.1f}]'
                          for k in range(len(dim_cols)))
        print(f"  {label:<44} {len(a):>5}/{len(b):<5}  {cells}")


def axis_correlations(user_means_df, dim_cols, prefix, strata):
    """Correlation between axes across users, overall and inside one role.

    Two axes that both separate governments from oppositions are the same
    finding twice if users who sit high on one sit high on the other, and two
    findings if they do not.
    """
    for label, predicate in strata:
        sub = user_means_df if predicate is None else user_means_df.filter(predicate)
        if sub.height < 2 * len(dim_cols):
            continue
        corr = np.corrcoef(sub.select(dim_cols).to_numpy(), rowvar=False)
        print(f"\n  {label} (n={sub.height}):")
        print(f"    {'':<6}" + ' '.join(f'{prefix}{k + 1:<6}' for k in range(len(corr))))
        for k, row in enumerate(corr):
            print(f"    {prefix}{k + 1:<4}" + ' '.join(f'{v:+7.3f}' for v in row))


def construct_contrasts(user_means_df, dim_cols, prefix):
    """Role, flank and office, each as its own contrast over the same users.

    Office is read twice, once over every partisan and once inside a single
    elected role. The restricted reading is the one that settles whether an
    axis separates governments from oppositions or only separates sitting
    members from the candidates and staff a losing party has more of.
    """
    df = user_means_df
    opposition = ~pl.col('FederalParty').is_in(['Liberal'])
    mp = pl.col('SubType') == 'member of parliament'
    mla = pl.col('SubType') == 'member of the provincial legislature'
    in_gov = pl.col('ProvincialParty').is_in(list(PROVINCIAL_GOVERNMENTS))
    partisan = pl.col('FederalParty').is_in(
        ['Liberal', 'Conservative', 'NDP', 'Green', 'Bloc Québécois', 'PPC'])
    prov = pl.col('ProvincialParty').is_not_null() & (pl.col('ProvincialParty') != '')

    rows = [
        ('role: influencer vs politician',
         _positions(df, dim_cols, pl.col('MainType') == 'influencer'),
         _positions(df, dim_cols, pl.col('MainType') == 'politician')),
        ('flank: Con+PPC vs NDP+Green, federal',
         _positions(df, dim_cols, pl.col('FederalParty').is_in(['Conservative', 'PPC'])),
         _positions(df, dim_cols, pl.col('FederalParty').is_in(['NDP', 'Green']))),
        ('flank: Con+PPC vs NDP+Green, MPs only',
         _positions(df, dim_cols, mp, pl.col('FederalParty').is_in(['Conservative', 'PPC'])),
         _positions(df, dim_cols, mp, pl.col('FederalParty').is_in(['NDP', 'Green']))),
        ('office: federal Liberal vs rest',
         _positions(df, dim_cols, partisan, ~opposition),
         _positions(df, dim_cols, partisan, opposition)),
        ('office: federal Liberal vs rest, MPs only',
         _positions(df, dim_cols, mp, partisan, ~opposition),
         _positions(df, dim_cols, mp, partisan, opposition)),
        ('office: provincial government vs rest',
         _positions(df, dim_cols, prov, in_gov),
         _positions(df, dim_cols, prov, ~in_gov)),
        ('office: provincial government vs rest, MLAs',
         _positions(df, dim_cols, mla, prov, in_gov),
         _positions(df, dim_cols, mla, prov, ~in_gov)),
    ]
    report_contrasts(rows, dim_cols, prefix)
    print("\n  The same contrasts, as standardized logistic weights:")
    report_unique(rows, dim_cols, prefix)
    print("\n  Correlation between axes across users:")
    axis_correlations(user_means_df, dim_cols, prefix, [
        ('all users', None),
        ('politicians', pl.col('MainType') == 'politician'),
        ('members of a provincial legislature', mla & prov),
        ('members of parliament', mp & partisan),
    ])


def party_contrasts(user_means_df, dim_cols, prefix, min_size=MIN_GROUP_SIZE):
    """Each party against the mean of the others, per axis.

    The pairwise table says which two parties sit furthest apart; this says
    whether an axis is a line the parties order along or a wall between one of
    them and the rest.
    """
    sub = user_means_df.drop_nulls('FederalParty') \
        .filter(pl.col('FederalParty') != '')
    keep = sub.group_by('FederalParty').len() \
        .filter(pl.col('len') >= min_size)['FederalParty'].to_list()
    sub = sub.filter(pl.col('FederalParty').is_in(keep)).sort('filter_value')

    parties = sorted(keep)
    points = sub.select(dim_cols).to_numpy()
    codes = np.array([parties.index(p) for p in sub['FederalParty'].to_list()])
    counts = np.bincount(codes, minlength=len(parties)).astype(float)
    centroids = np.stack([points[codes == g].mean(0) for g in range(len(parties))])
    resid = points - centroids[codes]
    pooled = np.sqrt((resid ** 2).sum(0) / max(points.shape[0] - len(parties), 1))

    print(f"\n  Party against the mean of the other parties, in pooled "
          f"within-party SDs:")
    print(f"    {'party':<20} {'n':>4}  " +
          '  '.join(f'{prefix}{k + 1:<5}' for k in range(len(dim_cols))))
    for i, party in enumerate(parties):
        others = centroids[np.arange(len(parties)) != i].mean(0)
        row = '  '.join(f'{v:+6.2f}' for v in (centroids[i] - others) / pooled)
        print(f"    {party:<20} {counts[i]:>4.0f}  {row}")
    return parties, centroids, pooled


def named_contrasts(parties, centroids, pooled, dim_cols, prefix, contrasts):
    """Hand-picked contrasts between blocs of parties, per axis."""
    index = {p: i for i, p in enumerate(parties)}
    print(f"\n  {'contrast':<34} " +
          '  '.join(f'{prefix}{k + 1:<5}' for k in range(len(dim_cols))))
    for label, left, right in contrasts:
        if any(p not in index for p in left + right):
            continue
        a = centroids[[index[p] for p in left]].mean(0)
        b = centroids[[index[p] for p in right]].mean(0)
        row = '  '.join(f'{v:+6.2f}' for v in (a - b) / pooled)
        print(f"  {label:<34} {row}")


# axes whose leading targets overlap enough that a description cannot tell
# them apart, as (a, b) zero-based indices
AXIS_PAIRS = [(0, 1), (0, 2), (1, 2)]

CONTRASTS = [
    ('governing vs opposition', ['Liberal'],
     ['Conservative', 'NDP', 'Green', 'Bloc Québécois', 'PPC']),
    ('right vs left', ['Conservative', 'PPC'], ['NDP', 'Green']),
    ('anti-establishment vs rest', ['PPC'],
     ['Liberal', 'Conservative', 'NDP', 'Green', 'Bloc Québécois']),
    ('Quebec vs rest', ['Bloc Québécois'],
     ['Liberal', 'Conservative', 'NDP', 'Green', 'PPC']),
    ('Conservative vs NDP', ['Conservative'], ['NDP']),
    ('Conservative vs Liberal', ['Conservative'], ['Liberal']),
]


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    n_dims = cfg.n_dims
    dim_cols = [f'x0_{i}' for i in range(n_dims)]
    prefix = latent_space.axis_prefix(cfg)

    logger.info("Loading trajectories and loadings...")
    spec = splits.SplitSpec.from_cfg(cfg)
    _, components, targets = latent_space.load(cfg, spec=spec)
    volumes = latent_space.target_volumes(cfg, targets)

    print("\n=== Targets each axis loads on ===")
    report_loadings(np.asarray(components), targets, volumes, prefix)
    report_axis_overlap(np.asarray(components), volumes, prefix)
    report_polarity(np.asarray(components), volumes, prefix)
    for a, b in AXIS_PAIRS:
        axis_divergence(np.asarray(components), targets, volumes, a, b, prefix)

    target_df = load_latent_df(cfg, spec)
    rolling_df = rolling_frame(cfg, target_df, list(range(n_dims)))

    print("\n=== Standing position versus movement ===")
    common = variance_split(rolling_df, dim_cols, prefix)
    report_drift(common, dim_cols, prefix)

    seed_df = load_seed_metadata_full(cfg, latent_space.traj_col(cfg))
    rolling_df = rolling_df.with_columns(pl.col('filter_value').cast(pl.String)) \
        .join(seed_df, left_on='filter_value',
              right_on=latent_space.traj_col(cfg), how='left')
    user_means_df = compute_user_means(rolling_df, dim_cols)

    print("\n=== Which contrast among the parties each axis carries ===")
    parties, centroids, pooled = party_contrasts(user_means_df, dim_cols, prefix)
    named_contrasts(parties, centroids, pooled, dim_cols, prefix, CONTRASTS)

    for field in ('ProvincialParty', 'SubType'):
        for _, b in AXIS_PAIRS:
            level_positions(user_means_df, field, dim_cols, prefix, sort_dim=b,
                            n_show=40)

    incumbency_test(user_means_df, dim_cols, prefix)

    print("\n=== Role, flank and office as separate contrasts ===")
    construct_contrasts(user_means_df, dim_cols, prefix)


if __name__ == '__main__':
    main()
