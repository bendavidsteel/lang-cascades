import numpy as np
import pytest

# eval_breakdown loads the trained landscape on import, which needs the GPU
# machine's stack; the correction these tests cover is plain numpy
eval_breakdown = pytest.importorskip('eval_breakdown')


def test_an_adjusted_value_never_sits_above_one_ranked_below_it():
    p = np.array([0.01, 0.019, 0.03])

    adjusted, _ = eval_breakdown.benjamini_hochberg(p)

    # raw scaling gives 0.030, 0.0285, 0.030, which would rank the smallest
    # p-value as the least significant of the three
    assert adjusted[0] == pytest.approx(0.0285)
    assert np.all(np.diff(adjusted) >= 0)


def test_a_family_that_only_looks_significant_one_at_a_time_clears_nothing():
    # each p-value is under alpha on its own; none survives the correction
    p = np.array([0.02, 0.04, 0.06])

    adjusted, significant = eval_breakdown.benjamini_hochberg(p)

    assert adjusted == pytest.approx(np.full(3, 0.06))
    assert not significant.any()


def test_an_untested_subgroup_stays_out_of_the_family():
    p = np.array([0.001, np.nan, 0.5])

    adjusted, significant = eval_breakdown.benjamini_hochberg(p)

    # two tests, not three: a subgroup with no pairs must not shrink the others
    assert adjusted[0] == pytest.approx(0.002)
    assert np.isnan(adjusted[1])
    assert not significant[1]


def test_an_adjusted_value_is_still_a_probability():
    p = np.array([0.6, 0.9])

    adjusted, _ = eval_breakdown.benjamini_hochberg(p)

    assert adjusted.max() <= 1.0


def test_stars_follow_the_adjusted_value():
    assert eval_breakdown.significance_stars(0.0005) == '***'
    assert eval_breakdown.significance_stars(0.005) == '**'
    assert eval_breakdown.significance_stars(0.04) == '*'
    assert eval_breakdown.significance_stars(0.2) == 'n.s.'
    assert eval_breakdown.significance_stars(np.nan) == ''
