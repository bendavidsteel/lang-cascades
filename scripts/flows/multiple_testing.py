"""Multiple-comparison correction shared by the evaluations.

One family per figure panel: eval_breakdown corrects over a horizon's
subgroups, eval_horizons over a scenario's horizons and rivals.
"""

import numpy as np


def benjamini_hochberg(p_values, alpha=0.05):
    """Benjamini-Hochberg adjusted p-values, and which of them clear alpha."""
    valid = ~np.isnan(p_values)
    n = int(np.sum(valid))
    adjusted = np.full(len(p_values), np.nan)
    significant = np.full(len(p_values), False)
    if n == 0:
        return adjusted, significant

    order = np.argsort(p_values[valid])
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)

    scaled = p_values[valid] * n / ranks
    # BH steps up from the largest p, so an adjusted value can never sit above
    # one ranked below it
    stepped = np.minimum.accumulate(scaled[order][::-1])[::-1]
    out = np.empty_like(scaled)
    out[order] = stepped
    adjusted[valid] = np.minimum(out, 1.0)
    significant[valid] = adjusted[valid] < alpha
    return adjusted, significant


def significance_stars(q):
    """Stars for a BH-adjusted p-value, so the figure claims what BH allows."""
    if np.isnan(q):
        return ''
    if q < 0.001:
        return '***'
    if q < 0.01:
        return '**'
    if q < 0.05:
        return '*'
    return 'n.s.'
