# Portable NumPy semantics. Expectations checked against NumPy 2.5.3 on CPython 3.14.4.
# Scope: sorting, searching, selection by condition, set operations, counting and binning.

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_sort_defaults_to_last_axis():
    a = np.array([[3, 1, 2], [9, 7, 8]])
    assert np.sort(a).tolist() == [[1, 2, 3], [7, 8, 9]]


def test_sort_along_first_axis():
    a = np.array([[3, 1], [1, 2], [2, 0]])
    assert np.sort(a, axis=0).tolist() == [[1, 0], [2, 1], [3, 2]]


def test_sort_with_axis_none_flattens():
    a = np.array([[3, 1], [1, 2], [2, 0]])
    assert np.sort(a, axis=None).tolist() == [0, 1, 1, 2, 2, 3]


def test_np_sort_returns_copy():
    a = np.array([3, 1, 2])
    r = np.sort(a)
    assert r.tolist() == [1, 2, 3]
    assert a.tolist() == [3, 1, 2]


def test_ndarray_sort_is_in_place_and_returns_none():
    a = np.array([3, 1, 2])
    assert a.sort() is None
    assert a.tolist() == [1, 2, 3]


def test_ndarray_sort_with_axis():
    a = np.array([[3, 1], [1, 2], [2, 0]])
    a.sort(axis=0)
    assert a.tolist() == [[1, 0], [2, 1], [3, 2]]


def test_sorting_strided_view_writes_through():
    a = np.array([5, 4, 3, 2, 1, 0])
    a[::2].sort()
    assert a.tolist() == [1, 4, 3, 2, 5, 0]


def test_sort_preserves_dtype():
    r = np.sort(np.array([3, -1, 2], dtype=np.int8))
    assert r.dtype == np.dtype("int8")
    assert r.tolist() == [-1, 2, 3]


def test_sort_floats_with_infinities_and_nan_last():
    r = np.sort(np.array([3.0, np.nan, -1.0, np.inf, -np.inf, np.nan]))
    assert r[:4].tolist() == [-np.inf, -1.0, 3.0, np.inf]
    assert np.isnan(r[4:]).tolist() == [True, True]


def test_argsort_places_nan_last():
    assert np.argsort(np.array([3.0, np.nan, -1.0])).tolist() == [2, 0, 1]


def test_sort_bool():
    assert np.sort(np.array([True, False, True, False])).tolist() == [False, False, True, True]


def test_sort_strings():
    assert np.sort(np.array(["pear", "apple", "fig"])).tolist() == ["apple", "fig", "pear"]


def test_argsort_returns_int64_indices():
    r = np.argsort(np.array([30, 10, 20]))
    assert r.dtype == np.dtype("int64")
    assert r.tolist() == [1, 2, 0]


def test_argsort_stable_keeps_tie_order():
    a = np.array([2, 1, 2, 1, 0, 1])
    assert np.argsort(a, kind="stable").tolist() == [4, 1, 3, 5, 0, 2]
    assert a.argsort(kind="stable").tolist() == [4, 1, 3, 5, 0, 2]


def test_argsort_along_axes():
    a = np.array([[3, 1, 2], [0, 5, 4]])
    assert np.argsort(a, axis=1).tolist() == [[1, 2, 0], [0, 2, 1]]
    assert np.argsort(a, axis=0).tolist() == [[1, 0, 0], [0, 1, 1]]


def test_argsort_indices_reproduce_sort():
    a = np.array([0.5, -2.0, 7.25, 3.0])
    assert a[np.argsort(a)].tolist() == np.sort(a).tolist()


def test_lexsort_uses_last_key_as_primary():
    secondary = np.array([3, 1, 2, 1])
    primary = np.array([1, 0, 1, 0])
    assert np.lexsort((secondary, primary)).tolist() == [1, 3, 2, 0]


def test_lexsort_is_stable_for_full_ties():
    keys = (np.array([0, 0, 0]), np.array([1, 1, 1]))
    assert np.lexsort(keys).tolist() == [0, 1, 2]


def test_lexsort_accepts_2d_key_array():
    keys = np.array([[9, 8, 7, 6], [1, 1, 0, 0]])
    assert np.lexsort(keys).tolist() == [3, 2, 1, 0]


def test_unique_returns_sorted_values():
    r = np.unique(np.array([3, 1, 2, 3, 1]))
    assert r.dtype == np.dtype("int64")
    assert r.tolist() == [1, 2, 3]


def test_unique_flattens_2d_input():
    assert np.unique(np.array([[2, 1], [1, 3]])).tolist() == [1, 2, 3]


def test_unique_of_empty_array():
    r = np.unique(np.array([], dtype=np.int64))
    assert r.shape == (0,)
    assert r.dtype == np.dtype("int64")


def test_unique_return_counts():
    values, counts = np.unique(np.array([4, 2, 4, 4, 9]), return_counts=True)
    assert values.tolist() == [2, 4, 9]
    assert counts.tolist() == [1, 3, 1]
    assert counts.dtype == np.dtype("int64")


def test_unique_return_index_gives_first_occurrence():
    values, index = np.unique(np.array([5, 3, 5, 1, 3]), return_index=True)
    assert values.tolist() == [1, 3, 5]
    assert index.tolist() == [3, 1, 0]


def test_unique_return_inverse_reconstructs_input():
    a = np.array([5, 3, 5, 1, 3])
    values, inverse = np.unique(a, return_inverse=True)
    assert inverse.tolist() == [2, 1, 2, 0, 1]
    assert values[inverse].tolist() == a.tolist()


def test_unique_inverse_keeps_input_shape_for_2d():
    a = np.array([[7, 5, 7], [5, 5, 9]])
    values, inverse = np.unique(a, return_inverse=True)
    assert values.tolist() == [5, 7, 9]
    assert inverse.shape == (2, 3)
    assert inverse.tolist() == [[1, 0, 1], [0, 0, 2]]


def test_unique_returns_all_outputs_in_documented_order():
    values, index, inverse, counts = np.unique(
        np.array([2, 0, 2]), return_index=True, return_inverse=True, return_counts=True
    )
    assert values.tolist() == [0, 2]
    assert index.tolist() == [1, 0]
    assert inverse.tolist() == [1, 0, 1]
    assert counts.tolist() == [1, 2]


def test_unique_collapses_nan_values():
    r = np.unique(np.array([1.0, np.nan, 0.5, np.nan]))
    assert r.shape == (3,)
    assert r[:2].tolist() == [0.5, 1.0]
    assert np.isnan(r[2])


def test_unique_of_strings():
    assert np.unique(np.array(["b", "a", "b"])).tolist() == ["a", "b"]


@pytest.mark.parametrize("side, expected", [("left", 1), ("right", 3)])
def test_searchsorted_side_for_repeated_value(side, expected):
    a = np.array([1, 2, 2, 3])
    result = np.searchsorted(a, 2, side=side)
    assert isinstance(result, np.int64)
    assert result == expected


def test_searchsorted_array_of_values():
    a = np.array([1, 2, 2, 3])
    r = np.searchsorted(a, [0, 2, 2.5, 4])
    assert r.dtype == np.dtype("int64")
    assert r.tolist() == [0, 1, 3, 4]
    assert a.searchsorted(np.array([2, 3]), side="right").tolist() == [3, 4]


def test_searchsorted_keeps_value_shape():
    a = np.array([10, 20, 30])
    assert np.searchsorted(a, [[5, 25], [30, 35]]).tolist() == [[0, 2], [2, 3]]


def test_where_selects_elementwise():
    r = np.where(np.array([True, False, True]), np.array([1, 2, 3]), np.array([10, 20, 30]))
    assert r.tolist() == [1, 20, 3]


def test_where_broadcasts_condition_and_scalars():
    a = np.arange(6).reshape(2, 3)
    assert np.where(a > 2, a, -1).tolist() == [[-1, -1, -1], [3, 4, 5]]
    column = np.array([[True], [False]])
    assert np.where(column, np.array([1, 2, 3]), 0).tolist() == [[1, 2, 3], [0, 0, 0]]


def test_where_promotes_branch_dtypes():
    r = np.where(np.array([True, False]), 1, 2.5)
    assert r.dtype == np.dtype("float64")
    assert r.tolist() == [1.0, 2.5]


def test_where_requires_both_or_neither_branch():
    with pytest.raises(ValueError):
        np.where(np.array([True]), 1)


def test_nonzero_1d_returns_one_element_tuple():
    result = np.nonzero(np.array([0, 3, 0, 4]))
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert result[0].dtype == np.dtype("int64")
    assert result[0].tolist() == [1, 3]


def test_nonzero_2d_returns_row_and_column_indices():
    a = np.array([[0, 2, 0], [3, 0, 4]])
    rows, cols = a.nonzero()
    assert rows.tolist() == [0, 1, 1]
    assert cols.tolist() == [1, 0, 2]


def test_nonzero_of_bool_mask():
    (index,) = np.nonzero(np.array([False, True, True]))
    assert index.tolist() == [1, 2]


def test_argwhere_returns_one_row_per_match():
    a = np.array([[0, 2, 0], [3, 0, 4]])
    r = np.argwhere(a)
    assert r.shape == (3, 2)
    assert r.tolist() == [[0, 1], [1, 0], [1, 2]]
    assert np.argwhere(np.array([0, 1, 0, 1])).tolist() == [[1], [3]]


def test_argwhere_with_no_matches_has_zero_rows():
    assert np.argwhere(np.zeros((2, 2))).shape == (0, 2)


def test_flatnonzero_uses_flat_positions():
    a = np.array([[0, 2, 0], [3, 0, 4]])
    assert np.flatnonzero(a).tolist() == [1, 3, 5]


def test_count_nonzero_total():
    a = np.array([[0, 2, 0], [3, 0, 4]])
    assert np.count_nonzero(a) == 3
    assert np.count_nonzero(np.array([True, False, True])) == 2


def test_count_nonzero_along_axis():
    a = np.array([[0, 2, 0], [3, 0, 4]])
    assert np.count_nonzero(a, axis=0).tolist() == [1, 1, 1]
    assert np.count_nonzero(a, axis=1).tolist() == [1, 2]
    assert np.count_nonzero(a, axis=1).dtype == np.dtype("int64")


def test_isin_tests_membership_elementwise():
    r = np.isin(np.array([1, 2, 3, 4]), [2, 4, 6])
    assert r.dtype == np.dtype("bool")
    assert r.tolist() == [False, True, False, True]


def test_isin_keeps_element_shape_and_flattens_test_elements():
    a = np.array([[1, 2], [3, 4]])
    assert np.isin(a, np.array([[4], [1]])).tolist() == [[True, False], [False, True]]


def test_isin_invert():
    r = np.isin(np.array([1, 2, 3]), [2], invert=True)
    assert r.tolist() == [True, False, True]


def test_intersect1d_returns_sorted_unique_common_values():
    assert np.intersect1d([3, 1, 2, 3], [3, 4, 1]).tolist() == [1, 3]


def test_union1d_returns_sorted_unique_values():
    assert np.union1d([3, 1], [2, 1]).tolist() == [1, 2, 3]


def test_setdiff1d_returns_sorted_unique_difference():
    assert np.setdiff1d([5, 1, 3, 1], [3]).tolist() == [1, 5]
    assert np.setdiff1d([1, 2], [1, 2]).shape == (0,)


def test_bincount_counts_each_value():
    r = np.bincount(np.array([0, 1, 1, 3]))
    assert r.dtype == np.dtype("int64")
    assert r.tolist() == [1, 2, 0, 1]


def test_bincount_with_weights_sums_weights():
    r = np.bincount(np.array([0, 1, 1, 3]), weights=np.array([0.5, 1.0, 2.0, 0.25]))
    assert r.dtype == np.dtype("float64")
    assert r.tolist() == [0.5, 3.0, 0.0, 0.25]


def test_bincount_minlength_pads_output():
    assert np.bincount(np.array([1, 1]), minlength=4).tolist() == [0, 2, 0, 0]
    assert np.bincount(np.array([3]), minlength=2).tolist() == [0, 0, 0, 1]
    assert np.bincount(np.array([], dtype=np.int64), minlength=3).tolist() == [0, 0, 0]


def test_bincount_rejects_negative_values():
    with pytest.raises(ValueError):
        np.bincount(np.array([0, -1]))


def test_histogram_with_bin_count_closes_last_bin():
    counts, edges = np.histogram(np.array([1, 2, 1, 4]), bins=3)
    assert counts.dtype == np.dtype("int64")
    assert counts.tolist() == [2, 1, 1]
    assert edges.dtype == np.dtype("float64")
    assert edges.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_histogram_with_explicit_edges():
    counts, edges = np.histogram(np.array([0.5, 1.0, 1.5, 2.0, 3.0]), bins=[0, 1, 2, 3])
    assert counts.tolist() == [1, 2, 2]
    assert edges.tolist() == [0, 1, 2, 3]


def test_histogram_range_excludes_outside_values():
    counts, edges = np.histogram(np.array([1, 2, 10, -1]), bins=2, range=(0, 4))
    assert counts.tolist() == [1, 1]
    assert edges.tolist() == [0.0, 2.0, 4.0]


def test_histogram_defaults_to_ten_bins():
    counts, edges = np.histogram(np.arange(10.0))
    assert counts.tolist() == [1] * 10
    assert edges.shape == (11,)
    assert_allclose(edges, np.linspace(0.0, 9.0, 11))


@pytest.mark.parametrize(
    "right, expected",
    [(False, [0, 1, 2, 2, 4, 4]), (True, [0, 0, 1, 2, 3, 4])],
)
def test_digitize_increasing_bins(right, expected):
    bins = np.array([0.0, 1.0, 2.5, 4.0])
    x = np.array([-1.0, 0.0, 1.0, 2.0, 4.0, 5.0])
    r = np.digitize(x, bins, right=right)
    assert r.dtype == np.dtype("int64")
    assert r.tolist() == expected


def test_digitize_decreasing_bins():
    bins = np.array([4.0, 2.5, 1.0, 0.0])
    x = np.array([-1.0, 0.0, 1.0, 2.0, 4.0, 5.0])
    assert np.digitize(x, bins).tolist() == [4, 3, 2, 2, 0, 0]


def test_argmax_and_argmin_pick_first_tie():
    a = np.array([1, 3, 3, 0, 0])
    assert np.argmax(a) == 1
    assert np.argmin(a) == 3
    assert isinstance(np.argmax(a), np.int64)


def test_argmax_without_axis_uses_flat_index():
    b = np.array([[1, 5, 5], [7, 7, 0]])
    assert np.argmax(b) == 3
    assert b.argmin() == 5


def test_argmax_and_argmin_along_axis():
    b = np.array([[1, 5, 5], [7, 7, 0]])
    assert np.argmax(b, axis=0).tolist() == [1, 1, 0]
    assert np.argmax(b, axis=1).tolist() == [1, 0]
    assert np.argmin(b, axis=1).tolist() == [0, 2]


def test_argmax_and_argmin_return_first_nan():
    a = np.array([1.0, np.nan, 3.0, np.nan])
    assert np.argmax(a) == 1
    assert np.argmin(a) == 1


def test_argmax_of_empty_array_raises_value_error():
    with pytest.raises(ValueError):
        np.argmax(np.array([]))
