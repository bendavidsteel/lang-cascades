import datetime

import polars as pl
import pytest

# describe_dimensions pulls in toponymy and its vLLM wrappers, which only the
# GPU machine has; the banding these tests cover is plain polars
describe_dimensions = pytest.importorskip('describe_dimensions')


T0 = datetime.datetime(2024, 1, 1)


def day(n):
    return T0 + datetime.timedelta(days=n)


@pytest.fixture
def rows():
    """Two trajectories on a shared weekly grid; one drifts up, one stays put."""
    return pl.DataFrame({
        'filter_value': ['A'] * 4 + ['B'] * 4,
        'createtime': [day(0), day(7), day(14), day(21)] * 2,
        'dim_0': [0.0, 1.0, 2.0, 3.0, 0.1, 0.0, 0.1, 0.0],
    })


@pytest.fixture
def loadings():
    return pl.DataFrame({'Target': ['x', 'y'], 'Loading': [0.5, -0.2]})


def text(ids, times, users, targets):
    return pl.DataFrame({
        'id': ids,
        'createtime': times,
        'SeedName': users,
        'Document': [f'post {i}' for i in ids],
        'Targets': targets,
        'Stances': [['FAVOR'] * len(t) for t in targets],
    })


def band_of(out, doc_id):
    return out.filter(pl.col('id') == doc_id)['band_value'].item()


def test_a_document_takes_the_band_its_author_was_in_that_day(rows, loadings):
    docs = text(['early', 'late'], [day(0), day(21)], ['A', 'A'], [['x'], ['x']])

    out = describe_dimensions.band_documents(
        rows, docs, loadings, 'dim_0', 'SeedName', n_exemplars=0)

    # the author crosses the axis, so the two posts land at opposite ends
    # rather than both covering the whole span the author was seen over
    assert band_of(out, 'early') == 0.0
    assert band_of(out, 'late') == 3.0


def test_a_document_outside_the_trajectory_is_dropped(rows, loadings):
    docs = text(['stray'], [day(200)], ['B'], [['x']])

    out = describe_dimensions.band_documents(
        rows, docs, loadings, 'dim_0', 'SeedName', n_exemplars=0)

    assert out.height == 0


def test_a_document_counts_once_however_many_targets_it_names(rows, loadings):
    docs = text(['multi'], [day(7)], ['A'], [['x', 'y']])

    out = describe_dimensions.band_documents(
        rows, docs, loadings, 'dim_0', 'SeedName', n_exemplars=0)

    assert out.height == 1


def test_a_dimension_with_too_little_on_its_targets_falls_back(rows, loadings):
    docs = text(['off'], [day(7)], ['A'], [['unloaded']])

    out = describe_dimensions.band_documents(
        rows, docs, loadings, 'dim_0', 'SeedName', n_exemplars=1)

    # nothing names a loaded target, so the band is filled from everything
    # these authors wrote rather than left empty
    assert out.height == 1


def test_an_axis_is_labelled_at_the_two_deciles():
    plot_nn_potential = pytest.importorskip('plot_nn_potential')
    labels = {'0': {'2_cat': {'negative': 'low end', 'positive': 'high end',
                              'negative_threshold': -2.0,
                              'positive_threshold': 3.0}}}

    positions, names = plot_nn_potential.axis_tick_labels(labels, 0)

    # two ticks, ascending, and no middle: the 2_cat cut describes no middle
    assert positions == [-2.0, 3.0]
    assert names == ['low end', 'high end']


def test_an_axis_with_no_labels_keeps_its_own_ticks():
    plot_nn_potential = pytest.importorskip('plot_nn_potential')

    assert plot_nn_potential.axis_tick_labels({}, 0) == ([], [])


def test_the_snapshots_split_the_fit_period_evenly():
    import datetime

    plot_nn_potential = pytest.importorskip('plot_nn_potential')
    splits = pytest.importorskip('splits')

    spec = splits.SplitSpec(365, 0, 0.70, 0.10, 42)
    times = pl.Series([datetime.datetime(2022, 1, 8) + datetime.timedelta(days=16 * i)
                       for i in range(103)])

    spans = plot_nn_potential.fit_period_spans(times, spec)

    # the row covers the fit period end to end, in equal pieces, and stops
    # where the holdout begins rather than running into unseen data
    assert len(spans) == 3
    assert spans[0][0] == times.min()
    assert spans[-1][1] == splits.time_cutoff(times, spec)
    assert spans[0][1] == spans[1][0] and spans[1][1] == spans[2][0]
    widths = {(hi - lo).days for lo, hi in spans}
    assert max(widths) - min(widths) <= 1


def test_a_longer_dataset_widens_the_panels_rather_than_adding_them():
    import datetime

    plot_nn_potential = pytest.importorskip('plot_nn_potential')
    splits = pytest.importorskip('splits')

    spec = splits.SplitSpec(365, 0, 0.70, 0.10, 42)
    short = pl.Series([datetime.datetime(2022, 1, 1), datetime.datetime(2025, 1, 1)])
    long = pl.Series([datetime.datetime(2022, 1, 1), datetime.datetime(2027, 1, 1)])

    a = plot_nn_potential.fit_period_spans(short, spec)
    b = plot_nn_potential.fit_period_spans(long, spec)

    assert len(a) == len(b) == 3
    assert (b[0][1] - b[0][0]) > (a[0][1] - a[0][0])
