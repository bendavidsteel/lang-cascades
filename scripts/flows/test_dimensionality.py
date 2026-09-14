"""Test whether 'political' MainType users have variance concentrated in
fewer of the existing latent dimensions than 'influencer' users, and identify
which latent dimensions separate seed types best.

Hypothesis (Converse 1964; Poole-Rosenthal NOMINATE; Baldassarri & Gelman 2008):
politicians' attitudes are more ideologically constrained, falling onto a single
common left-right axis more than the broader public (or influencers).

Two distinct quantities per user, treated separately:
- POSITION = user's mean coords across observations (one K-vector per user)
- TRAJECTORY = user's temporal variance Var(x_k(t)) (one K-vector per user)

Trajectories come from latent_space.load, so a gpfa run reads the same fit the
landscape was trained on and a precomputed run its own coords. Either way they
are already low-dimensional, so we work along the existing axes rather than
running a fresh per-user decomposition, which would be orientation-free and
answer a different question.

Entropy is in nats against a ceiling of log(n_dims), so a value here is only
comparable with one from a fit of the same dimensionality; H / log(n_dims) is
what carries across.

Three analyses:

1. Per-user trajectory-variance fractions: for each user, compute
       var_k       = Var(x_k(t)) along trajectory in dim k
       frac_k      = var_k / sum_j(var_j)
       max_frac    = max_k frac_k
       D_eff       = (sum var_k)^2 / sum(var_k^2)   (participation ratio)
   Compare distributions across MainType. Tests the temporal claim: do
   politicians drift along a single dim more than influencers?

2. Group-level position-variance fractions + Shannon entropy: for each
   MainType, take the array of per-user mean positions, compute Var across
   users in each latent dim, normalize. Tests the cross-sectional claim: are
   politicians' positions more ideologically constrained? Bootstrap CIs on
   the entropy come from resampling users with replacement.

3. Between-group variance per dim: for each metadata field (MainType, SubType,
   FederalParty, ProvincialParty, Province, NewsOutletCategory) and each latent
   dim, compute eta-squared from a one-way ANOVA on per-user mean positions.
   eta_k^2(G) = SS_between(G) / SS_total in dim k tells us what fraction of
   cross-user position variance in dim k is explained by group G.
"""
import logging
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

import latent_space
import splits
from nn_potential import load_latent_df, rolling_frame

logger = logging.getLogger(__name__)

MIN_POINTS_PER_USER = 30
GROUP_FIELDS = ['MainType', 'SubType', 'FederalParty', 'ProvincialParty', 'Province', 'NewsOutletCategory']
MIN_GROUP_SIZE = 5  # min users per level to include a group level in ANOVA


def load_seed_metadata_full(cfg, key):
    """Load extra seed fields beyond what nn_potential.load_seed_metadata gives.

    `key` is the seed field trajectories are named by, which a handle-keyed fit
    makes PlatformHandleID rather than SeedName.
    """
    dir_path = cfg.base_stance_path
    file_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.parquet.zstd')]
    df = pl.read_parquet(file_paths, columns=['seed'])
    fields = list(dict.fromkeys([key] + GROUP_FIELDS))
    return df.select(
        [pl.col('seed').struct.field(f) for f in fields]
    ).unique(key)


def per_user_dim_variance(rolling_df, dim_cols, n_dims):
    """Per-user variance along each existing latent dim.

    Returns a dataframe with one row per user holding:
      var_k, frac_k for k in 0..n_dims-1
      max_frac      = max_k frac_k             (concentration on one fixed axis)
      argmax_frac   = which dim dominates
      effective_dim = (sum var_k)^2 / sum(var_k^2)   (1 = pure 1D, n_dims = uniform)
    """
    var_cols = [pl.col(c).var().alias(f'var_{i}') for i, c in enumerate(dim_cols)]
    user_var_df = rolling_df.group_by(['filter_value', 'MainType']).agg(
        var_cols + [pl.len().alias('n_points')]
    ).filter(pl.col('n_points') >= MIN_POINTS_PER_USER)\
     .drop_nulls([f'var_{i}' for i in range(n_dims)])

    var_arr = user_var_df.select([f'var_{i}' for i in range(n_dims)]).to_numpy()
    total = var_arr.sum(axis=1, keepdims=True)
    keep = (total[:, 0] > 0)
    user_var_df = user_var_df.filter(pl.Series(keep))
    var_arr = var_arr[keep]
    total = total[keep]

    frac = var_arr / total
    max_frac = frac.max(axis=1)
    argmax_frac = frac.argmax(axis=1)
    eff_dim = (var_arr.sum(axis=1) ** 2) / (var_arr ** 2).sum(axis=1)

    return user_var_df.with_columns(
        [pl.Series(f'frac_{i}', frac[:, i]) for i in range(n_dims)] +
        [
            pl.Series('max_frac', max_frac),
            pl.Series('argmax_frac', argmax_frac),
            pl.Series('effective_dim', eff_dim),
        ]
    )


def compute_user_means(rolling_df, dim_cols):
    """One row per user: mean position per latent dim plus metadata fields."""
    return rolling_df.group_by('filter_value').agg(
        [pl.col(c).mean().alias(c) for c in dim_cols] +
        [pl.col(f).first().alias(f) for f in GROUP_FIELDS] +
        [pl.len().alias('n_points')]
    ).filter(pl.col('n_points') >= MIN_POINTS_PER_USER)


def _shannon_entropy(frac):
    """Shannon entropy in nats. Treats 0*log(0) as 0."""
    return float(-np.sum(np.where(frac > 0, frac * np.log(frac), 0.0)))


def _position_var_frac(positions):
    """Variance per dim across rows of `positions`, normalized to a fraction vector."""
    if positions.shape[0] < 2:
        return np.full(positions.shape[1], np.nan)
    var_per_dim = positions.var(axis=0, ddof=1)
    total = float(var_per_dim.sum())
    if total <= 0:
        return np.zeros_like(var_per_dim)
    return var_per_dim / total


def group_entropy_bootstrap(user_means_df, dim_cols, main_types, n_boot=500, seed=42):
    """Shannon entropy of position-variance fractions, with user-resample bootstrap CIs.

    For each MainType: take the (n_users, n_dims) array of per-user mean
    positions, compute variance per dim across users, normalize to a fraction
    vector, take Shannon entropy. Bootstrap by resampling users with
    replacement.
    """
    rng = np.random.default_rng(seed)
    n_dims = len(dim_cols)
    out = {}
    for main_type in main_types:
        sub = user_means_df.filter(pl.col('MainType') == main_type)
        n_users = sub.height
        if n_users < 2:
            continue
        positions = sub.select(dim_cols).to_numpy()

        obs_frac = _position_var_frac(positions)
        obs_entropy = _shannon_entropy(obs_frac)

        boot = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n_users, size=n_users)
            boot[b] = _shannon_entropy(_position_var_frac(positions[idx]))

        out[main_type] = {
            'frac': obs_frac,
            'entropy': obs_entropy,
            'effective_dim_exp_h': float(np.exp(obs_entropy)),
            'ci_low': float(np.quantile(boot, 0.025)),
            'ci_high': float(np.quantile(boot, 0.975)),
            'boot': boot,
            'n_users': n_users,
        }
        logger.info(
            f"  {main_type}: H={obs_entropy:.4f} nats "
            f"(95% CI [{out[main_type]['ci_low']:.4f}, {out[main_type]['ci_high']:.4f}]), "
            f"exp(H)={out[main_type]['effective_dim_exp_h']:.3f}, "
            f"H/log({n_dims})={obs_entropy/np.log(n_dims):.3f}"
        )
    return out


def compare_group_entropy(group_boot, pol_key, inf_key, n_dims):
    """Bootstrap CI on H_pol - H_inf using independent resamples."""
    if pol_key not in group_boot or inf_key not in group_boot:
        return
    pol = group_boot[pol_key]['boot']
    inf = group_boot[inf_key]['boot']
    n = min(len(pol), len(inf))
    diff = pol[:n] - inf[:n]
    obs_diff = group_boot[pol_key]['entropy'] - group_boot[inf_key]['entropy']
    p_two_sided = 2 * min(float(np.mean(diff <= 0)), float(np.mean(diff >= 0)))
    ceiling = float(np.log(n_dims))
    print(
        f"\n  Group-level entropy: {pol_key} H={group_boot[pol_key]['entropy']:.4f} "
        f"vs {inf_key} H={group_boot[inf_key]['entropy']:.4f}"
    )
    # nats against a log(n_dims) ceiling, so only the normalized value compares
    # with a fit of a different dimensionality
    print(
        f"  normalized H/log({n_dims})={group_boot[pol_key]['entropy']/ceiling:.4f} "
        f"vs {group_boot[inf_key]['entropy']/ceiling:.4f} (ceiling {ceiling:.4f} nats)"
    )
    print(
        f"  diff = H_{pol_key} - H_{inf_key} = {obs_diff:+.4f} nats "
        f"(95% bootstrap CI [{np.quantile(diff, 0.025):+.4f}, {np.quantile(diff, 0.975):+.4f}], "
        f"two-sided bootstrap p={p_two_sided:.4g})"
    )
    print(
        f"  Lower entropy = more concentrated. "
        f"Politicians {'less' if obs_diff > 0 else 'more'} concentrated than influencers."
    )


# ------------------------------------------------------------------ spectrum
#
# A per-axis variance fraction is a statement about the basis as much as about
# the group. Under a homogeneous prior the latent is free to rotate, so
# core.identify fixes the basis post-hoc -- and it whitens, which drives the
# pooled per-axis fractions to uniform whatever shape the cloud has. The
# eigenvalues of a group's own covariance survive both.


def _concentration(eigvals):
    """Rotation-invariant concentration summaries of a covariance spectrum.

    lambda1_share is the share of between-user variance a rank-1 model
    captures, which is what the ideological-constraint literature reports.
    The other two are effective dimension counts, in dimensions.
    """
    lam = np.sort(np.asarray(eigvals, dtype=float))[::-1]
    total = float(lam.sum())
    if not np.isfinite(total) or total <= 0:
        return {k: float('nan') for k in
                ('lambda1_share', 'participation_ratio', 'entropy', 'exp_h')}
    p = lam / total
    pos = p[p > 0]
    h = float(-np.sum(pos * np.log(pos)))
    return {
        'lambda1_share': float(p[0]),
        'participation_ratio': float(total ** 2 / float((lam ** 2).sum())),
        'entropy': h,
        'exp_h': float(np.exp(h)),
    }


# Whether a larger value of a statistic means more concentration. An effective
# dimension count runs the other way from a variance share.
CONCENTRATION_SIGN = {'lambda1_share': +1, 'participation_ratio': -1,
                      'entropy': -1, 'exp_h': -1}


def position_spectrum(positions):
    """Eigenvalues of the between-user covariance of mean positions."""
    if positions.shape[0] < positions.shape[1] + 2:
        return np.full(positions.shape[1], np.nan)
    return np.sort(np.linalg.eigvalsh(np.cov(positions, rowvar=False)))[::-1]


def _group_positions(user_means_df, dim_cols, main_type):
    return user_means_df.filter(pl.col('MainType') == main_type)\
        .select(dim_cols).to_numpy()


def _matched(positions, n, rng, replace=False):
    idx = rng.choice(positions.shape[0], size=n, replace=replace)
    return positions[idx]


def _matched_stat(positions, n, stat, rng, n_draw, spectrum=None, replace=False):
    """A concentration statistic measured at a fixed number of users.

    Sample eigenvalues spread apart as the user count falls, so a smaller group
    looks more concentrated for no reason but its size. Averaging over draws
    keeps the subsampling from adding noise of its own.
    """
    spectrum = position_spectrum if spectrum is None else spectrum
    vals = [_concentration(spectrum(_matched(positions, n, rng, replace)))[stat]
            for _ in range(n_draw)]
    return float(np.nanmean(vals))


def report_spectra(user_means_df, dim_cols, main_types):
    """Each group's position spectrum and its concentration, at matched n."""
    groups = {mt: _group_positions(user_means_df, dim_cols, mt)
              for mt in main_types}
    groups = {mt: p for mt, p in groups.items() if p.shape[0] >= len(dim_cols) + 2}
    if not groups:
        return {}
    n = min(p.shape[0] for p in groups.values())
    out = {}
    for mt, pos in groups.items():
        rng = np.random.default_rng(42)
        lam = position_spectrum(pos)
        c = _concentration(lam)
        out[mt] = dict(c, n_users=pos.shape[0], eigvals=lam, matched_n=n,
                       lambda1_share_matched=_matched_stat(
                           pos, n, 'lambda1_share', rng, 200))
        print(
            f"  {mt} (n={pos.shape[0]}): eigvals="
            f"[{', '.join(f'{v:.3f}' for v in lam)}]"
        )
        print(
            f"    lambda1 share={c['lambda1_share']:.4f} "
            f"(at n={n}: {out[mt]['lambda1_share_matched']:.4f}), "
            f"participation ratio={c['participation_ratio']:.3f}, "
            f"exp(H)={c['exp_h']:.3f} of {len(dim_cols)}"
        )
    return out


def compare_concentration(pol, inf, pol_key, inf_key, spectrum=None, n_dims=None,
                          stat='lambda1_share', n_draw=25, n_boot=1000,
                          n_perm=1000, seed=42):
    """Permutation test and bootstrap CI on a concentration difference.

    Both groups are measured at the same number of users, and the permutation
    null is built the same way, so the size difference between them cannot
    show up as a difference in concentration. `spectrum` maps a group's rows to
    its eigenvalues, so the same test runs on a raw or a noise-corrected one.
    """
    spectrum = position_spectrum if spectrum is None else spectrum
    n_dims = pol.shape[1] if n_dims is None else n_dims
    if min(pol.shape[0], inf.shape[0]) < n_dims + 2:
        return None
    n = min(pol.shape[0], inf.shape[0])
    rng = np.random.default_rng(seed)

    obs_pol = _matched_stat(pol, n, stat, rng, n_draw, spectrum)
    obs_inf = _matched_stat(inf, n, stat, rng, n_draw, spectrum)
    obs = obs_pol - obs_inf

    boot = np.array([
        _matched_stat(pol, n, stat, rng, 1, spectrum, replace=True) -
        _matched_stat(inf, n, stat, rng, 1, spectrum, replace=True)
        for _ in range(n_boot)])

    pool = np.vstack([pol, inf])
    null = np.empty(n_perm)
    for b in range(n_perm):
        idx = rng.permutation(pool.shape[0])
        null[b] = (_concentration(spectrum(pool[idx[:n]]))[stat] -
                   _concentration(spectrum(pool[idx[n:2 * n]]))[stat])
    # +1 so a p-value is never zero on a finite number of shuffles
    p_perm = (1 + int(np.sum(np.abs(null) >= abs(obs)))) / (n_perm + 1)

    print(f"\n  {stat} at n={n}: {pol_key}={obs_pol:.4f} vs {inf_key}={obs_inf:.4f}")
    print(
        f"  diff = {obs:+.4f} "
        f"(95% bootstrap CI [{np.quantile(boot, 0.025):+.4f}, "
        f"{np.quantile(boot, 0.975):+.4f}], "
        f"two-sided permutation p={p_perm:.4g})"
    )
    toward = obs * CONCENTRATION_SIGN[stat]
    print(
        f"  {pol_key.capitalize()}s "
        f"{'more' if toward > 0 else 'less'} concentrated than {inf_key}s"
        f"{'' if p_perm < 0.05 else ', but not distinguishably so'}."
    )
    return {'stat': stat, 'matched_n': n, pol_key: obs_pol, inf_key: obs_inf,
            'diff': obs, 'p_perm': p_perm,
            'ci_low': float(np.quantile(boot, 0.025)),
            'ci_high': float(np.quantile(boot, 0.975))}


def split_half_positions(rolling_df, dim_cols, min_per_half=10):
    """Two independent estimates of each user's mean position.

    Alternating observations rather than two blocks of time: a user who drifts
    would put the drift into the difference between the halves, which is what
    the correction reads as noise.
    """
    halved = rolling_df.sort(['filter_value', 'createtime']).with_columns(
        (pl.int_range(pl.len()).over('filter_value') % 2).alias('half'))
    agg = halved.group_by(['filter_value', 'MainType', 'half']).agg(
        [pl.col(c).mean().alias(c) for c in dim_cols] + [pl.len().alias('n')]
    ).filter(pl.col('n') >= min_per_half)
    a = agg.filter(pl.col('half') == 0).drop(['half', 'n'])
    b = agg.filter(pl.col('half') == 1).drop(['half', 'n', 'MainType'])
    return a.join(b, on='filter_value', suffix='_b')


def noise_corrected_spectrum(a, b):
    """Signal-covariance eigenvalues, with independent measurement noise removed.

    Each half's error is independent of the other's, so their cross-covariance
    carries the signal without the noise floor that inflates the low end of a
    within-half spectrum -- which flattens a spectrum and so hides exactly the
    concentration being tested for. An eigenvalue near zero can come out
    negative, which is the estimator being unbiased rather than a failure.
    """
    a = a - a.mean(0)
    b = b - b.mean(0)
    c = (a.T @ b) / (a.shape[0] - 1)
    return np.sort(np.linalg.eigvalsh((c + c.T) / 2))[::-1]


def halves_spectrum(rows):
    """noise_corrected_spectrum over rows holding both halves side by side."""
    k = rows.shape[1] // 2
    return noise_corrected_spectrum(rows[:, :k], rows[:, k:])


def group_halves(half_df, dim_cols, main_type):
    """One group's two half-estimates, stacked into a single array."""
    sub = half_df.filter(pl.col('MainType') == main_type)
    return sub.select(dim_cols + [f'{c}_b' for c in dim_cols]).to_numpy()


def report_noise_corrected(half_df, dim_cols, main_types):
    """Concentration after removing the per-user estimation noise floor."""
    out = {}
    for mt in main_types:
        rows = group_halves(half_df, dim_cols, mt)
        if rows.shape[0] < len(dim_cols) + 2:
            continue
        lam = halves_spectrum(rows)
        c = _concentration(lam)
        out[mt] = dict(c, n_users=rows.shape[0], eigvals=lam)
        print(
            f"  {mt} (n={rows.shape[0]}): eigvals="
            f"[{', '.join(f'{v:.3f}' for v in lam)}], "
            f"lambda1 share={c['lambda1_share']:.4f}, "
            f"exp(H)={c['exp_h']:.3f} of {len(dim_cols)}"
        )
    return out


def group_level_position_variance(user_means_df, dim_cols, main_types):
    """Per-MainType variance fractions across per-user mean positions.

    Each user contributes one position; we compute Var across users in each
    latent dim and normalize. This is the cross-sectional spread of the group,
    independent of how individuals move within their trajectories.
    """
    out = {}
    for main_type in main_types:
        sub = user_means_df.filter(pl.col('MainType') == main_type)
        if sub.height < 2:
            continue
        positions = sub.select(dim_cols).to_numpy()
        var_per_dim = positions.var(axis=0, ddof=1)
        total = float(var_per_dim.sum())
        if total <= 0:
            continue
        frac = var_per_dim / total
        out[main_type] = {'var': var_per_dim, 'frac': frac, 'n_users': sub.height}
        logger.info(
            f"  group {main_type}: n_users={sub.height}, "
            f"var={[f'{v:.4f}' for v in var_per_dim]}, frac={[f'{v:.3f}' for v in frac]}"
        )
    return out


def between_group_eta_squared(user_means_df, group_field, dim_cols):
    """Compute eta^2 (= SS_between / SS_total) per latent dim for one group field.

    Each row of user_means_df is one user's mean position in latent space plus
    metadata fields. Levels with fewer than MIN_GROUP_SIZE users are dropped,
    and rows with null/empty group label are dropped.
    """
    sub = user_means_df\
        .drop_nulls(group_field)\
        .filter(pl.col(group_field) != '')

    if sub.height < 2:
        return None

    level_counts = sub.group_by(group_field).len()
    keep = level_counts.filter(pl.col('len') >= MIN_GROUP_SIZE)[group_field].to_list()
    if len(keep) < 2:
        return None
    sub = sub.filter(pl.col(group_field).is_in(keep))

    n_total = sub.height
    grand_mean = sub.select(dim_cols).mean().to_numpy()[0]

    group_stats = sub.group_by(group_field).agg(
        [pl.col(c).mean().alias(f'{c}_gmean') for c in dim_cols] +
        [pl.len().alias('n')]
    )
    n_g = group_stats['n'].to_numpy()
    g_means = group_stats.select([f'{c}_gmean' for c in dim_cols]).to_numpy()

    ss_between = ((g_means - grand_mean) ** 2 * n_g[:, None]).sum(axis=0)
    points = sub.select(dim_cols).to_numpy()
    ss_total = ((points - grand_mean) ** 2).sum(axis=0)

    eta2 = np.where(ss_total > 0, ss_between / ss_total, 0.0)

    f_stats, p_values = [], []
    n_groups = len(keep)
    df_between = n_groups - 1
    df_within = n_total - n_groups
    for k in range(len(dim_cols)):
        ss_within = ss_total[k] - ss_between[k]
        if ss_within <= 0 or df_within <= 0 or df_between <= 0:
            f_stats.append(float('nan'))
            p_values.append(float('nan'))
            continue
        f = (ss_between[k] / df_between) / (ss_within / df_within)
        f_stats.append(float(f))
        p_values.append(float(1 - stats.f.cdf(f, df_between, df_within)))

    return {
        'group_field': group_field,
        'levels': keep,
        'n_users': n_total,
        'eta2': eta2,
        'f_stat': np.array(f_stats),
        'p_value': np.array(p_values),
    }


def run_between_group_analysis(user_means_df, dim_cols, n_dims):
    """eta^2 per (group_field, dim) on per-user mean positions."""
    results = []
    for field in GROUP_FIELDS:
        res = between_group_eta_squared(user_means_df, field, dim_cols)
        if res is None:
            logger.info(f"  {field}: skipped (too few labeled users / levels)")
            continue
        eta_str = ', '.join(f"d{i}={v:.3f}" for i, v in enumerate(res['eta2']))
        p_str = ', '.join(f"d{i}={v:.2g}" for i, v in enumerate(res['p_value']))
        logger.info(
            f"  {field}: n_users={res['n_users']}, levels={len(res['levels'])}, "
            f"eta^2 [{eta_str}], p [{p_str}]"
        )
        results.append(res)
    return results


def plot_between_group(results, n_dims, prefix, out_path):
    if not results:
        return
    fields = [r['group_field'] for r in results]
    eta2 = np.stack([r['eta2'] for r in results])  # (n_fields, n_dims)
    p_values = np.stack([r['p_value'] for r in results])

    fig, ax = plt.subplots(figsize=(1.6 * n_dims + 2, 0.5 * len(fields) + 2))
    im = ax.imshow(eta2, aspect='auto', cmap='viridis', vmin=0, vmax=max(eta2.max(), 1e-3))
    ax.set_xticks(range(n_dims))
    ax.set_xticklabels([f'{prefix}{i+1}' for i in range(n_dims)])
    ax.set_yticks(range(len(fields)))
    ax.set_yticklabels([f"{f} (n={r['n_users']}, k={len(r['levels'])})" for f, r in zip(fields, results)])
    for i in range(len(fields)):
        for j in range(n_dims):
            star = '***' if p_values[i, j] < 1e-3 else ('**' if p_values[i, j] < 1e-2 else ('*' if p_values[i, j] < 5e-2 else ''))
            ax.text(j, i, f"{eta2[i, j]:.2f}{star}", ha='center', va='center',
                    color='white' if eta2[i, j] < eta2.max() * 0.6 else 'black', fontsize=9)
    ax.set_title(r'Between-group variance per dim: $\eta^2$ = SS$_{between}$ / SS$_{total}$')
    fig.colorbar(im, ax=ax, label=r'$\eta^2$')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    logger.info(f"Saved {out_path}")
    plt.close(fig)


def federal_party_centroid_analysis(user_means_df, dim_cols, n_dims, min_party_size=MIN_GROUP_SIZE):
    """Per-dim FederalParty centroid spread and pairwise differences.

    For each latent dim k:
      - centroid_p,k    = mean of per-user mean positions among users in party p
      - centroid_std_k  = SD of centroid_p,k across parties
      - pooled_sd_k     = sqrt of pooled within-party variance on dim k
      - standardized_spread_k = centroid_std_k / pooled_sd_k
        (between-party SD measured in within-party SD units; comparable across dims)

    Pairwise: diff_{p,q,k} = centroid_p,k - centroid_q,k, with Cohen's d using
    pooled_sd_k. Same dim-level pooled SD for every pair so d-values rank by
    raw centroid distance per dim.
    """
    sub = user_means_df\
        .drop_nulls('FederalParty')\
        .filter(pl.col('FederalParty') != '')

    if sub.height < 2:
        logger.warning("Too few users with FederalParty for centroid analysis")
        return None, None

    counts = sub.group_by('FederalParty').len()
    keep = counts.filter(pl.col('len') >= min_party_size)['FederalParty'].to_list()
    if len(keep) < 2:
        logger.warning(f"Fewer than 2 FederalParty levels with >= {min_party_size} users; skipping")
        return None, None
    sub = sub.filter(pl.col('FederalParty').is_in(keep))

    party_stats = sub.group_by('FederalParty').agg(
        [pl.col(c).mean().alias(f'{c}_mean') for c in dim_cols] +
        [pl.col(c).std().alias(f'{c}_std') for c in dim_cols] +
        [pl.len().alias('n')]
    ).sort('FederalParty')

    parties = party_stats['FederalParty'].to_list()
    n_per_party = party_stats['n'].to_numpy()
    centroids = party_stats.select([f'{c}_mean' for c in dim_cols]).to_numpy()
    within_sd = party_stats.select([f'{c}_std' for c in dim_cols]).to_numpy()
    within_sd = np.nan_to_num(within_sd, nan=0.0)

    centroid_std = centroids.std(axis=0, ddof=1)
    centroid_range = centroids.max(axis=0) - centroids.min(axis=0)

    df_within = (n_per_party - 1).astype(float)[:, None]
    pooled_var = (within_sd ** 2 * df_within).sum(axis=0) / max(df_within.sum(), 1.0)
    pooled_sd = np.sqrt(pooled_var)
    standardized_spread = np.where(pooled_sd > 0, centroid_std / pooled_sd, np.nan)

    pairwise = []
    n_parties = len(parties)
    for i in range(n_parties):
        for j in range(i + 1, n_parties):
            diff = centroids[i] - centroids[j]
            d = np.where(pooled_sd > 0, diff / pooled_sd, np.nan)
            for k in range(n_dims):
                pairwise.append({
                    'party_a': parties[i],
                    'party_b': parties[j],
                    'dim': k,
                    'diff': float(diff[k]),
                    'abs_diff': float(abs(diff[k])),
                    'cohens_d': float(d[k]),
                    'abs_cohens_d': float(abs(d[k])),
                })
    pairwise_df = pl.DataFrame(pairwise)

    info = {
        'parties': parties,
        'n_per_party': n_per_party,
        'centroids': centroids,
        'within_sd': within_sd,
        'pooled_sd': pooled_sd,
        'centroid_std': centroid_std,
        'centroid_range': centroid_range,
        'standardized_spread': standardized_spread,
    }
    return info, pairwise_df


def plot_federal_party_centroids(party_info, n_dims, prefix, out_path):
    if party_info is None:
        return
    parties = party_info['parties']
    centroids = party_info['centroids']
    standardized_spread = party_info['standardized_spread']

    dim_order = np.argsort(-np.nan_to_num(standardized_spread, nan=-np.inf))

    fig, axes = plt.subplots(
        1, 2,
        figsize=(2.0 * n_dims + 6, max(0.4 * len(parties) + 2.5, 4)),
        gridspec_kw={'width_ratios': [1.4, 1]},
    )

    vmax = float(np.abs(centroids).max()) if centroids.size else 1.0
    ax = axes[0]
    im = ax.imshow(centroids[:, dim_order], aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n_dims))
    ax.set_xticklabels([f'{prefix}{d+1}' for d in dim_order])
    ax.set_yticks(range(len(parties)))
    ax.set_yticklabels([f"{p} (n={n})" for p, n in zip(parties, party_info['n_per_party'])])
    ax.set_title('FederalParty centroids (dims sorted by standardized spread)')
    for i in range(len(parties)):
        for j in range(n_dims):
            v = centroids[i, dim_order[j]]
            ax.text(
                j, i, f"{v:+.2f}", ha='center', va='center', fontsize=8,
                color='white' if abs(v) > vmax * 0.5 else 'black',
            )
    fig.colorbar(im, ax=ax, label='centroid (mean position)')

    ax = axes[1]
    ax.bar(range(n_dims), standardized_spread[dim_order], color='C0')
    ax.set_xticks(range(n_dims))
    ax.set_xticklabels([f'{prefix}{d+1}' for d in dim_order])
    ax.set_ylabel('centroid SD / pooled within-party SD')
    ax.set_title('Standardized between-party spread per dim')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    logger.info(f"Saved {out_path}")
    plt.close(fig)


def report_federal_party_centroids(party_info, pairwise_df, n_dims, prefix,
                                   top_pairs_per_dim=5):
    if party_info is None:
        return
    parties = party_info['parties']
    print(f"Parties (n_users): {dict(zip(parties, party_info['n_per_party'].tolist()))}")

    print("\nPer-dim spread of FederalParty centroids:")
    for k in range(n_dims):
        print(
            f"  {prefix}{k+1}: centroid SD={party_info['centroid_std'][k]:.4f}, "
            f"range={party_info['centroid_range'][k]:.4f}, "
            f"pooled within-SD={party_info['pooled_sd'][k]:.4f}, "
            f"standardized spread={party_info['standardized_spread'][k]:.3f}"
        )

    rank = np.argsort(-np.nan_to_num(party_info['standardized_spread'], nan=-np.inf))
    rank_str = ', '.join(
        f"{prefix}{int(d)+1} ({party_info['standardized_spread'][d]:.2f})" for d in rank
    )
    print(f"\nDims ranked by standardized between-party spread: {rank_str}")

    print(f"\nTop {top_pairs_per_dim} pairwise centroid differences per dim (by |Cohen's d|):")
    for k in range(n_dims):
        top = pairwise_df\
            .filter(pl.col('dim') == k)\
            .sort('abs_cohens_d', descending=True)\
            .head(top_pairs_per_dim)
        print(f"  {prefix}{k+1}:")
        for row in top.iter_rows(named=True):
            print(
                f"    {row['party_a']:<20} - {row['party_b']:<20}: "
                f"diff={row['diff']:+.3f}, d={row['cohens_d']:+.2f}"
            )

    print("\nLargest pairwise centroid differences overall (by |Cohen's d|):")
    overall_top = pairwise_df.sort('abs_cohens_d', descending=True).head(10)
    for row in overall_top.iter_rows(named=True):
        print(
            f"  {prefix}{row['dim']+1}: {row['party_a']:<20} - {row['party_b']:<20}: "
            f"diff={row['diff']:+.3f}, d={row['cohens_d']:+.2f}"
        )


def compare_pol_vs_inf(per_user_df, pol_key, inf_key, n_dims, prefix):
    """One-sided Mann-Whitney U: politicians concentrate on a fixed axis more than influencers."""
    pol = per_user_df.filter(pl.col('MainType') == pol_key)
    inf = per_user_df.filter(pl.col('MainType') == inf_key)
    print(f"\n=== {pol_key} (n={len(pol)}) vs {inf_key} (n={len(inf)}) ===")
    print("H1: politicians' trajectory variance is more concentrated on one fixed axis")

    tests = [('max_frac', 'greater'), ('effective_dim', 'less')]
    tests += [(f'frac_{i}', 'greater') for i in range(n_dims)]
    tests += [(f'frac_{i}', 'less') for i in range(n_dims)]

    for metric, alt in tests:
        a = pol[metric].to_numpy()
        b = inf[metric].to_numpy()
        if len(a) < 2 or len(b) < 2:
            continue
        u, p = stats.mannwhitneyu(a, b, alternative=alt)
        print(
            f"  {metric} ({alt}): {pol_key} med={np.median(a):.3f} (mean {np.mean(a):.3f}), "
            f"{inf_key} med={np.median(b):.3f} (mean {np.mean(b):.3f}); "
            f"U={u:.0f}, p={p:.4g}"
        )

    print(f"\n  Dominant-axis distribution (argmax of per-user frac):")
    for label, sub in [(pol_key, pol), (inf_key, inf)]:
        counts = sub.group_by('argmax_frac').len().sort('argmax_frac')
        total = sub.height
        parts = ', '.join(
            f"{prefix}{int(d)+1}={c/total:.1%}"
            for d, c in zip(counts['argmax_frac'].to_list(), counts['len'].to_list())
        )
        print(f"    {label}: {parts}")


def plot_results(per_user_df, group_var, n_dims, main_types, prefix, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cmap = plt.get_cmap('tab10')
    colors = {mt: cmap(i % 10) for i, mt in enumerate(main_types)}
    dim_labels = [f'{prefix}{i+1}' for i in range(n_dims)]

    ax = axes[0]
    width = 0.8 / max(len(main_types), 1)
    x = np.arange(n_dims)
    for i, main_type in enumerate(main_types):
        info = group_var.get(main_type)
        if info is None:
            continue
        ax.bar(
            x + (i - (len(main_types) - 1) / 2) * width, info['frac'], width,
            label=f"{main_type} (n_users={info['n_users']})", color=colors[main_type],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels)
    ax.set_ylabel('Fraction of position variance across users')
    ax.set_title('Group-level (per-user mean positions)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for i, main_type in enumerate(main_types):
        sub = per_user_df.filter(pl.col('MainType') == main_type)
        if sub.height == 0:
            continue
        fracs = np.stack([sub[f'frac_{d}'].to_numpy() for d in range(n_dims)], axis=1)
        med = np.median(fracs, axis=0)
        q25 = np.quantile(fracs, 0.25, axis=0)
        q75 = np.quantile(fracs, 0.75, axis=0)
        offset = (i - (len(main_types) - 1) / 2) * 0.18
        ax.errorbar(
            x + offset, med, yerr=[med - q25, q75 - med], fmt='o-',
            label=f'{main_type} (n={sub.height})', color=colors[main_type], capsize=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels)
    ax.set_ylabel('Fraction of trajectory variance (per user)')
    ax.set_title('Per-user (median ± IQR)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    for main_type in main_types:
        sub = per_user_df.filter(pl.col('MainType') == main_type)
        if sub.height == 0:
            continue
        ax.hist(
            sub['max_frac'].to_numpy(), bins=30, alpha=0.5, density=True,
            label=f'{main_type} (n={sub.height})', color=colors[main_type],
        )
    ax.set_xlabel('max-axis variance fraction per user')
    ax.set_ylabel('Density')
    ax.set_title('Per-user concentration on a fixed axis')
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    logger.info(f"Saved {out_path}")
    plt.close(fig)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    n_dims = cfg.n_dims
    dim_cols = [f'x0_{i}' for i in range(n_dims)]

    prefix = latent_space.axis_prefix(cfg)

    logger.info(f"Loading trajectories (n_dims={n_dims}, platform={cfg.platform})...")
    target_df = load_latent_df(cfg, splits.SplitSpec.from_cfg(cfg))
    rolling_df = rolling_frame(cfg, target_df, list(range(n_dims)))

    logger.info("Loading seed metadata...")
    seed_df = load_seed_metadata_full(cfg, latent_space.traj_col(cfg))

    # a gpfa fit may be keyed by handle rather than seed, and the metadata is
    # per seed either way
    rolling_df = rolling_df\
        .with_columns(pl.col('filter_value').cast(pl.String))\
        .join(seed_df, left_on='filter_value',
              right_on=latent_space.traj_col(cfg), how='left')

    main_type_df = rolling_df.drop_nulls('MainType').filter(pl.col('MainType') != '')
    main_types = sorted(main_type_df['MainType'].unique().to_list())
    logger.info(f"MainTypes found: {main_types}")

    logger.info("Computing per-user mean positions...")
    user_means_df = compute_user_means(rolling_df, dim_cols)
    main_type_means = user_means_df.drop_nulls('MainType').filter(pl.col('MainType') != '')
    logger.info(
        f"Users per MainType (>= {MIN_POINTS_PER_USER} points):\n"
        f"{main_type_means.group_by('MainType').len().sort('MainType')}"
    )

    logger.info("Group-level position-variance fractions per dim, by MainType...")
    group_var = group_level_position_variance(main_type_means, dim_cols, main_types)

    logger.info("Group-level position-variance Shannon entropy with user bootstrap...")
    group_boot = group_entropy_bootstrap(main_type_means, dim_cols, main_types)

    print("\n=== Concentration of mean positions (rotation-invariant) ===")
    if cfg.latents.method == 'gpfa':
        print("  The per-axis fractions above read the basis core.identify "
              "chose, which\n  is whitened, so they are near-uniform whatever "
              "shape a group has. These\n  eigenvalues are the same question "
              "asked of the group's own covariance.")
    spectra = report_spectra(main_type_means, dim_cols, main_types)

    logger.info(f"Per-user variance fractions per dim (>= {MIN_POINTS_PER_USER} points/user)...")
    per_user_df = per_user_dim_variance(main_type_df, dim_cols, n_dims)
    logger.info(f"Per-user metrics computed for {per_user_df.height} users")

    print("\n=== Per-user variance-fraction summary by MainType ===")
    summary = per_user_df.group_by('MainType').agg(
        [pl.len().alias('n_users'),
         pl.col('n_points').median().alias('median_n_points'),
         pl.col('max_frac').mean().alias('mean_max_frac'),
         pl.col('max_frac').median().alias('median_max_frac'),
         pl.col('effective_dim').mean().alias('mean_eff_dim'),
         pl.col('effective_dim').median().alias('median_eff_dim')] +
        [pl.col(f'frac_{i}').median().alias(f'median_frac_{i}') for i in range(n_dims)]
    ).sort('MainType')
    with pl.Config(tbl_width_chars=200, tbl_rows=50):
        print(summary)

    pol_key = next((mt for mt in main_types if 'olit' in mt.lower()), None)
    inf_key = next((mt for mt in main_types if 'nflu' in mt.lower()), None)
    if pol_key and inf_key:
        compare_pol_vs_inf(per_user_df, pol_key, inf_key, n_dims, prefix)
        compare_group_entropy(group_boot, pol_key, inf_key, n_dims)

        print("\n=== Is one group more uni-dimensional? (matched n) ===")
        pol_pos = _group_positions(main_type_means, dim_cols, pol_key)
        inf_pos = _group_positions(main_type_means, dim_cols, inf_key)
        for stat in ('lambda1_share', 'participation_ratio'):
            compare_concentration(pol_pos, inf_pos, pol_key, inf_key, stat=stat)

        print("\n=== The same, with the per-user noise floor removed ===")
        half_df = split_half_positions(main_type_df, dim_cols)
        report_noise_corrected(half_df, dim_cols, main_types)
        for stat in ('lambda1_share', 'participation_ratio'):
            compare_concentration(group_halves(half_df, dim_cols, pol_key),
                                  group_halves(half_df, dim_cols, inf_key),
                                  pol_key, inf_key, spectrum=halves_spectrum,
                                  n_dims=n_dims, stat=stat)
    else:
        logger.warning(
            f"Could not auto-detect political/influencer MainType keys "
            f"(found: {main_types}); skipping pairwise test."
        )

    trend_name = os.path.basename(cfg.trend_path.rstrip('/'))
    fig_dir = os.path.join('figs', trend_name)
    os.makedirs(fig_dir, exist_ok=True)
    suffix = f'_{latent_space.name(cfg)}'
    suffix += f'_{cfg.platform}' if cfg.platform != 'all' else ''
    out_fig = os.path.join(fig_dir, f'dimensionality_by_main_type{suffix}.png')
    plot_results(per_user_df, group_var, n_dims, main_types, prefix, out_fig)

    out_parquet = os.path.join(fig_dir, f'per_user_dimensionality{suffix}.parquet.zstd')
    per_user_df.write_parquet(out_parquet, compression='zstd')
    logger.info(f"Saved per-user metrics to {out_parquet}")

    print("\n=== Between-group variance per latent dimension (per-user means) ===")
    bg_results = run_between_group_analysis(user_means_df, dim_cols, n_dims)

    out_heatmap = os.path.join(fig_dir, f'between_group_eta_squared{suffix}.png')
    plot_between_group(bg_results, n_dims, prefix, out_heatmap)

    if bg_results:
        bg_records = []
        for r in bg_results:
            for k in range(n_dims):
                bg_records.append({
                    'group_field': r['group_field'],
                    'n_users': r['n_users'],
                    'n_levels': len(r['levels']),
                    'dim': k,
                    'eta_squared': float(r['eta2'][k]),
                    'f_stat': float(r['f_stat'][k]),
                    'p_value': float(r['p_value'][k]),
                })
        bg_df = pl.DataFrame(bg_records)
        with pl.Config(tbl_width_chars=200, tbl_rows=200):
            print(bg_df.sort(['group_field', 'dim']))
        out_bg_parquet = os.path.join(fig_dir, f'between_group_eta_squared{suffix}.parquet.zstd')
        bg_df.write_parquet(out_bg_parquet, compression='zstd')
        logger.info(f"Saved between-group eta^2 to {out_bg_parquet}")

    print("\n=== FederalParty centroid differences per latent dim ===")
    party_info, pairwise_df = federal_party_centroid_analysis(user_means_df, dim_cols, n_dims)
    if party_info is not None:
        report_federal_party_centroids(party_info, pairwise_df, n_dims, prefix)

        out_party_fig = os.path.join(fig_dir, f'federal_party_centroids{suffix}.png')
        plot_federal_party_centroids(party_info, n_dims, prefix, out_party_fig)

        out_party_parquet = os.path.join(fig_dir, f'federal_party_pairwise{suffix}.parquet.zstd')
        pairwise_df.write_parquet(out_party_parquet, compression='zstd')
        logger.info(f"Saved FederalParty pairwise centroid diffs to {out_party_parquet}")


if __name__ == '__main__':
    main()
