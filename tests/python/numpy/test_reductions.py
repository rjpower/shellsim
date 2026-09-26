# Portable NumPy semantics. Expectations checked against NumPy 2.5.3 on CPython 3.14.4.
# Scope: reductions, statistics, cumulative operations, nan-aware variants, and their result dtypes.

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal


def test_sum_over_axis_none_int_negative_and_tuple():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    assert a.sum() == 21
    assert_array_equal(np.sum(a, axis=0), [5, 7, 9])
    assert_array_equal(np.sum(a, axis=1), [6, 15])
    assert_array_equal(np.sum(a, axis=-1), [6, 15])
    assert np.sum(a, axis=(0, 1)) == 21
    assert np.sum(a, axis=None) == 21


def test_keepdims_preserves_reduced_axes_as_length_one():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.sum(a, axis=0, keepdims=True).shape == (1, 3)
    assert np.max(a, axis=1, keepdims=True).shape == (2, 1)
    whole = np.sum(a, keepdims=True)
    assert whole.shape == (1, 1)
    assert_array_equal(whole, [[21]])
    assert np.mean(a, axis=(0,), keepdims=True).shape == (1, 3)
    assert_array_equal(a - a.mean(axis=1, keepdims=True), [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])


def test_prod_min_max_along_axes():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    assert_array_equal(np.prod(a, axis=0), [4, 10, 18])
    assert np.prod(a) == 720
    assert_array_equal(np.min(a, axis=1), [1, 4])
    assert_array_equal(np.max(a, axis=0), [4, 5, 6])
    assert_array_equal(np.max(a, axis=-1), [3, 6])
    assert np.min(a, axis=(0, 1)) == 1


def test_reductions_over_axis_tuples_in_three_dimensions():
    b = np.arange(24).reshape(2, 3, 4)
    assert_array_equal(b.sum(axis=(0, 2)), [60, 92, 124])
    assert_array_equal(b.max(axis=(1, 2)), [11, 23])
    assert b.mean(axis=(0, 2), keepdims=True).shape == (1, 3, 1)
    assert_array_equal(b.min(axis=-1), [[0, 4, 8], [12, 16, 20]])


def test_mean_along_axes_returns_float64_for_ints():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    assert a.mean() == 3.5
    assert_array_equal(np.mean(a, axis=0), [2.5, 3.5, 4.5])
    assert_array_equal(np.mean(a, axis=1), [2.0, 5.0])
    assert np.mean(a, axis=0).dtype == np.float64


def test_ptp_is_max_minus_min():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.ptp(a) == 5
    assert_array_equal(np.ptp(a, axis=0), [3, 3, 3])
    assert_array_equal(np.ptp(a, axis=1), [2, 2])


def test_var_and_std_with_ddof():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert x.var() == 1.25
    assert_allclose(x.var(ddof=1), 5.0 / 3.0, rtol=1e-15)
    assert_allclose(x.std(), np.sqrt(1.25), rtol=1e-15)
    assert_allclose(np.std(x, ddof=1), np.sqrt(5.0 / 3.0), rtol=1e-15)
    assert np.std([2, 4, 4, 4, 5, 5, 7, 9]) == 2.0
    assert np.var([2, 4, 4, 4, 5, 5, 7, 9]) == 4.0


def test_var_and_std_along_axes():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    assert_array_equal(np.var(a, axis=0), [2.25, 2.25, 2.25])
    assert_array_equal(np.std(a, axis=1, ddof=1), [1.0, 1.0])
    assert np.var(a, axis=1, keepdims=True).shape == (2, 1)


def test_median_of_odd_and_even_lengths_and_axes():
    assert np.median([3, 1, 2]) == 2.0
    assert np.median([4, 1, 3, 2]) == 2.5
    assert type(np.median([3, 1, 2])) is np.float64
    a = np.array([[1, 2, 3], [4, 5, 6]])
    assert_array_equal(np.median(a, axis=0), [2.5, 3.5, 4.5])
    assert_array_equal(np.median(a, axis=1), [2.0, 5.0])
    kept = np.median(np.array([[1, 3], [2, 4]]), axis=1, keepdims=True)
    assert kept.shape == (2, 1)
    assert_array_equal(kept, [[2.0], [3.0]])


def test_percentile_uses_linear_interpolation():
    assert np.percentile([1, 2, 3, 4], 50) == 2.5
    assert_array_equal(np.percentile([1, 2, 3, 4], [25, 50, 75]), [1.75, 2.5, 3.25])
    a = np.array([[1, 2, 3], [4, 5, 6]])
    assert_array_equal(np.percentile(a, 50, axis=0), [2.5, 3.5, 4.5])
    assert_array_equal(np.percentile(a, [25, 75], axis=1), [[1.5, 4.5], [2.5, 5.5]])


def test_quantile_matches_percentile_scale():
    assert np.quantile([1, 2, 3, 4], 0.25) == 1.75
    assert_array_equal(np.quantile([10, 20, 30, 40, 50], [0.0, 0.1, 1.0]), [10.0, 14.0, 50.0])
    result = np.quantile(np.array([[1, 2, 3], [4, 5, 6]]), [0.5, 1.0], axis=1)
    assert result.shape == (2, 2)
    assert_array_equal(result, [[2.0, 5.0], [3.0, 6.0]])


def test_percentile_and_quantile_reject_out_of_range_q():
    with pytest.raises(ValueError) as info:
        np.percentile([1, 2], 101)
    assert str(info.value) == "Percentiles must be in the range [0, 100]"
    with pytest.raises(ValueError) as info:
        np.quantile([1, 2], 1.5)
    assert str(info.value) == "Quantiles must be in the range [0, 1]"


def test_average_with_weights_and_returned_sum_of_weights():
    assert np.average([1, 2, 3, 4]) == 2.5
    assert np.average([1, 2, 3], weights=[3, 1, 0]) == 1.25
    average, weight_sum = np.average([1.0, 2.0], weights=[1, 3], returned=True)
    assert average == 1.75
    assert weight_sum == 4.0
    a = np.array([[1, 2, 3], [4, 5, 6]])
    assert_array_equal(np.average(a, axis=1, weights=[1, 0, 1]), [2.0, 5.0])
    averages, weight_sums = np.average(a, axis=0, weights=[1, 3], returned=True)
    assert_array_equal(averages, [3.25, 4.25, 5.25])
    assert_array_equal(weight_sums, [4.0, 4.0, 4.0])


def test_average_rejects_weights_that_sum_to_zero():
    with pytest.raises(ZeroDivisionError) as info:
        np.average([1, 2], weights=[1, -1])
    assert str(info.value) == "Weights sum to zero, can't be normalized"


def test_argmin_and_argmax_flat_and_along_axes():
    c = np.array([[3, 7, 7], [9, 1, 9]])
    assert c.argmax() == 3
    assert c.argmin() == 4
    assert type(c.argmax()) is np.int64
    assert_array_equal(np.argmax(c, axis=0), [1, 0, 1])
    assert_array_equal(np.argmax(c, axis=1), [1, 0])
    assert_array_equal(np.argmin(c, axis=1), [0, 1])
    assert np.argmax(c, axis=1, keepdims=True).shape == (2, 1)


def test_argmin_and_argmax_return_first_of_ties_and_first_nan():
    assert np.argmax([1, 5, 5, 2]) == 1
    assert np.argmin([2, 0, 0]) == 1
    assert np.argmax([1.0, np.nan, 3.0]) == 1
    assert np.argmin([1.0, np.nan, 0.0]) == 1


def test_argmax_of_empty_array_raises():
    with pytest.raises(ValueError) as info:
        np.argmax(np.array([]))
    assert str(info.value) == "attempt to get argmax of an empty sequence"


def test_any_and_all_on_ints_and_floats():
    d = np.array([[0, 1, 2], [0, 0, 3]])
    assert d.any()
    assert not d.all()
    assert type(d.any()) is np.bool_
    assert_array_equal(d.any(axis=0), [False, True, True])
    assert_array_equal(np.all(d, axis=1), [False, False])
    assert_array_equal(np.any(d, axis=1, keepdims=True), [[True], [True]])
    assert np.all([1.0, np.nan])
    assert not np.any([0.0, -0.0])
    assert np.any(np.array([0.0, 0.5]))


def test_any_and_all_of_empty_arrays_use_identities():
    assert np.all(np.array([]))
    assert not np.any(np.array([]))


def test_cumsum_and_cumprod_flatten_without_axis():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    assert_array_equal(np.cumsum(a), [1, 3, 6, 10, 15, 21])
    assert_array_equal(a.cumsum(), [1, 3, 6, 10, 15, 21])
    assert_array_equal(np.cumprod(np.array([[1, 2], [3, 4]])), [1, 2, 6, 24])
    assert_array_equal(np.cumprod([1, 2, 3, 4]), [1, 2, 6, 24])


def test_cumsum_and_cumprod_along_axes():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    assert_array_equal(np.cumsum(a, axis=0), [[1, 2, 3], [5, 7, 9]])
    assert_array_equal(np.cumsum(a, axis=1), [[1, 3, 6], [4, 9, 15]])
    assert_array_equal(np.cumsum(a, axis=-1), [[1, 3, 6], [4, 9, 15]])
    assert_array_equal(np.cumprod(a, axis=1), [[1, 2, 6], [4, 20, 120]])


def test_cumsum_promotes_small_ints_and_bools_to_int64():
    small = np.cumsum(np.array([100, 100], np.int8))
    assert small.dtype == np.int64
    assert_array_equal(small, [100, 200])
    flags = np.cumsum(np.array([True, True]))
    assert flags.dtype == np.int64
    assert_array_equal(flags, [1, 2])
    assert np.cumsum(np.array([1.5], np.float32)).dtype == np.float32


def test_count_nonzero_total_and_along_axes():
    d = np.array([[0, 1, 2], [0, 0, 3]])
    assert np.count_nonzero(d) == 3
    assert_array_equal(np.count_nonzero(d, axis=0), [0, 1, 2])
    assert_array_equal(np.count_nonzero(d, axis=1), [2, 1])
    assert np.count_nonzero([0.0, np.nan, -0.0, 2.0]) == 2


def test_nansum_nanmean_nanmin_nanmax_skip_nan():
    x = np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, 6.0]])
    assert np.isnan(np.sum(x))
    assert np.nansum(x) == 10.0
    assert_array_equal(np.nansum(x, axis=0), [1.0, 0.0, 9.0])
    assert_array_equal(np.nanmean(x, axis=1), [2.0, 6.0])
    assert_array_equal(np.nanmin(x, axis=1), [1.0, 6.0])
    assert np.nanmax(x) == 6.0
    assert np.nansum(np.array([1, 2])) == 3


def test_nanstd_nanvar_nanmedian_skip_nan():
    x = np.array([1.0, np.nan, 3.0])
    assert np.nanstd(x) == 1.0
    assert np.nanvar(x, ddof=1) == 2.0
    assert np.nanmedian(np.array([1.0, np.nan, 3.0, 10.0])) == 3.0
    assert_array_equal(np.nanmedian(np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, 6.0]]), axis=1), [2.0, 6.0])


def test_all_nan_slices_return_nan_and_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        maximum = np.nanmax(np.array([np.nan, np.nan]))
        column_minimums = np.nanmin(np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, 6.0]]), axis=0)
        mean = np.nanmean(np.array([np.nan]))
    assert np.isnan(maximum)
    assert_array_equal(column_minimums, [1.0, np.nan, 3.0])
    assert np.isnan(mean)
    messages = [str(item.message) for item in caught]
    assert all(issubclass(item.category, RuntimeWarning) for item in caught)
    assert messages.count("All-NaN slice encountered") == 2
    assert "Mean of empty slice" in messages


def test_sum_and_prod_of_empty_arrays_are_identities():
    empty = np.array([])
    assert empty.sum() == 0.0
    assert type(empty.sum()) is np.float64
    assert empty.prod() == 1.0
    assert np.sum(np.array([], np.int64)) == 0
    assert np.prod(np.array([], np.int8)) == 1
    assert_array_equal(np.zeros((0, 3)).sum(axis=0), [0.0, 0.0, 0.0])
    assert np.zeros((0, 3)).sum(axis=1).shape == (0,)


def test_max_and_min_of_empty_arrays_raise():
    with pytest.raises(ValueError) as info:
        np.max(np.array([]))
    assert str(info.value) == "zero-size array to reduction operation maximum which has no identity"
    with pytest.raises(ValueError) as info:
        np.array([]).min()
    assert str(info.value) == "zero-size array to reduction operation minimum which has no identity"
    with pytest.raises(ValueError):
        np.zeros((0, 3)).max(axis=0)
    assert np.zeros((0, 3)).max(axis=1).shape == (0,)


def test_mean_of_empty_array_is_nan_with_runtime_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = np.array([]).mean()
    assert np.isnan(result)
    assert caught
    assert all(issubclass(item.category, RuntimeWarning) for item in caught)
    assert "Mean of empty slice" in [str(item.message) for item in caught]


def test_mean_of_empty_array_is_nan_when_warnings_ignored():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isnan(np.array([]).mean())
        assert_array_equal(np.zeros((0, 3)).mean(axis=0), [np.nan, np.nan, np.nan])


@pytest.mark.parametrize(
    "input_name, result_name",
    [
        ("int8", "int64"),
        ("int16", "int64"),
        ("int32", "int64"),
        ("int64", "int64"),
        ("uint8", "uint64"),
        ("uint32", "uint64"),
        ("bool", "int64"),
        ("float32", "float32"),
        ("float64", "float64"),
        ("complex64", "complex64"),
    ],
)
def test_sum_and_prod_result_dtypes(input_name, result_name):
    values = np.ones(3, np.dtype(input_name))
    assert values.sum().dtype == np.dtype(result_name)
    assert values.prod().dtype == np.dtype(result_name)
    assert values.sum(axis=0).dtype == np.dtype(result_name)


def test_sum_of_small_ints_does_not_wrap():
    assert np.array([100, 100], np.int8).sum() == 200
    total = np.array([200, 200], np.uint8).sum()
    assert type(total) is np.uint64
    assert total == 400
    count = np.array([True, True, False]).sum()
    assert type(count) is np.int64
    assert count == 2


@pytest.mark.parametrize(
    "input_name, result_name",
    [
        ("int8", "float64"),
        ("int64", "float64"),
        ("uint8", "float64"),
        ("bool", "float64"),
        ("float32", "float32"),
        ("float64", "float64"),
        ("complex128", "complex128"),
    ],
)
def test_mean_result_dtypes(input_name, result_name):
    values = np.ones(4, np.dtype(input_name))
    assert values.mean().dtype == np.dtype(result_name)
    assert np.mean(values, axis=0).dtype == np.dtype(result_name)


def test_min_max_and_statistics_keep_or_widen_dtype():
    small = np.array([5, 1, 3], np.int8)
    assert small.max().dtype == np.int8
    assert small.min() == 1
    assert np.median(small) == 3.0
    assert np.median(small).dtype == np.float64
    assert np.var(np.array([1, 2, 3, 4], np.float32)).dtype == np.float32
    assert np.array([1, 2], np.float32).mean() == np.float32(1.5)
    assert np.array([1 + 2j, 3 + 4j]).mean() == 2 + 3j


def test_reduction_results_are_numpy_scalars():
    assert type(np.arange(3).sum()) is np.int64
    assert type(np.array([1.0, 2.0]).mean()) is np.float64
    assert type(np.array([1.0, 2.0], np.float32).sum()) is np.float32
    assert type(np.array([1, 0]).all()) is np.bool_
    assert type(np.array([3, 1]).argmin()) is np.int64
    assert type(np.array([1.0, 2.0]).std()) is np.float64


def test_method_forms_match_function_forms():
    a = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])
    assert_array_equal(a.sum(axis=0), np.sum(a, axis=0))
    assert_array_equal(a.prod(axis=1), np.prod(a, axis=1))
    assert a.mean() == np.mean(a)
    assert a.max() == 6.0
    assert_array_equal(a.min(axis=1), [1.0, 2.0])
    assert a.std() == np.std(a)
    assert a.var(ddof=1) == np.var(a, ddof=1)
    assert_array_equal(a.argmax(axis=1), [1, 2])
    assert_array_equal(a.cumprod(axis=0), [[1.0, 5.0, 3.0], [4.0, 10.0, 18.0]])
    assert a.all()
    assert_array_equal(a.any(axis=0), [True, True, True])


def test_axis_out_of_range_raises_value_error():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError) as info:
        a.sum(axis=2)
    assert str(info.value) == "axis 2 is out of bounds for array of dimension 2"
    with pytest.raises(ValueError):
        np.mean(a, axis=-3)


def test_cov_of_two_series_and_matrix_rows():
    assert_allclose(np.cov([1, 2, 3, 4], [2, 4, 6, 8]), [[5.0 / 3.0, 10.0 / 3.0], [10.0 / 3.0, 20.0 / 3.0]])
    rows = np.cov(np.array([[0, 1, 2], [2, 1, 0]]))
    assert rows.shape == (2, 2)
    assert_array_equal(rows, [[1.0, -1.0], [-1.0, 1.0]])


def test_cov_of_one_series_is_zero_dimensional_array():
    result = np.cov([1.0, 2.0, 3.0])
    assert isinstance(result, np.ndarray)
    assert result.shape == ()
    assert result == 1.0


def test_cov_ddof_and_bias():
    assert np.cov([1, 2, 3, 4], ddof=0) == 1.25
    assert_allclose(np.cov([1, 2, 3], [1, 2, 3], bias=True), [[2.0 / 3.0, 2.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]])


def test_corrcoef_of_small_exact_data():
    assert_allclose(np.corrcoef([1, 2, 3, 4], [2, 4, 6, 8]), [[1.0, 1.0], [1.0, 1.0]])
    assert_allclose(np.corrcoef([1, 2, 3], [3, 2, 1]), [[1.0, -1.0], [-1.0, 1.0]])
    assert_allclose(np.corrcoef(np.array([[1, 2, 3], [1, 3, 2]])), [[1.0, 0.5], [0.5, 1.0]])


def test_float_summation_is_accurate_within_tolerance():
    assert_allclose(np.full(1000, 0.1).sum(), 100.0, rtol=1e-13)
    assert_allclose(np.full(1000, 0.1).mean(), 0.1, rtol=1e-13)
    assert np.arange(1, 10001, dtype=np.float64).sum() == 50005000.0
    # Sequential float32 accumulation drifts to about 999.90; pairwise summation stays near 1000.
    assert_allclose(np.full(10000, 0.1, np.float32).sum(), 1000.0, rtol=1e-5)
