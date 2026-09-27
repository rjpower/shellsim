# Portable NumPy semantics. Expectations checked against NumPy 2.5.3 on CPython 3.14.4.
# Scope: floating-point error flags, np.errstate/seterr/geterr, and their RuntimeWarning reporting.

import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_equal

DEFAULT_ERROR_STATE = {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}


def _recorded(function):
    """Call ``function`` and return its result with every warning it emitted as (category, text)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = function()
    return result, [(item.category, str(item.message)) for item in caught]


def test_default_error_state():
    assert np.geterr() == DEFAULT_ERROR_STATE


def test_float_divide_by_zero_returns_signed_infinity_and_warns():
    result, caught = _recorded(lambda: np.array([1.0, -1.0]) / 0)
    assert_array_equal(result, [np.inf, -np.inf])
    assert caught == [(RuntimeWarning, "divide by zero encountered in divide")]


def test_zero_divided_by_zero_returns_nan_and_warns_invalid():
    result, caught = _recorded(lambda: np.array([0.0]) / 0)
    assert np.isnan(result[0])
    assert caught == [(RuntimeWarning, "invalid value encountered in divide")]


def test_each_flag_warns_once_per_call_with_divide_before_invalid():
    result, caught = _recorded(lambda: np.array([0.0, 1.0, 0.0, -1.0]) / 0)
    assert_array_equal(result, [np.nan, np.inf, np.nan, -np.inf])
    assert caught == [
        (RuntimeWarning, "divide by zero encountered in divide"),
        (RuntimeWarning, "invalid value encountered in divide"),
    ]
    _, caught = _recorded(lambda: np.ones(100) / 0)
    assert len(caught) == 1


def test_int_true_divide_by_zero_returns_infinity_and_warns():
    result, caught = _recorded(lambda: np.array([1, 2]) / 0)
    assert_array_equal(result, [np.inf, np.inf])
    assert caught == [(RuntimeWarning, "divide by zero encountered in divide")]


@pytest.mark.parametrize("name", ["floor_divide", "remainder"])
def test_int_division_by_zero_returns_zero_and_warns(name):
    result, caught = _recorded(lambda: getattr(np, name)(np.array([1, 2]), 0))
    assert result.dtype == np.int64
    assert_array_equal(result, [0, 0])
    assert caught == [(RuntimeWarning, "divide by zero encountered in " + name)]


def test_int_operators_and_divmod_by_zero_warn():
    result, caught = _recorded(lambda: np.array([5]) // 0)
    assert_array_equal(result, [0])
    assert caught == [(RuntimeWarning, "divide by zero encountered in floor_divide")]
    (quotient, remainder), caught = _recorded(lambda: divmod(np.array([1, 2]), 0))
    assert_array_equal(quotient, [0, 0])
    assert_array_equal(remainder, [0, 0])
    assert caught == [(RuntimeWarning, "divide by zero encountered in divmod")]


def test_float_floor_divide_and_remainder_by_zero():
    result, caught = _recorded(lambda: np.array([1.0]) // 0)
    assert_array_equal(result, [np.inf])
    assert caught == [(RuntimeWarning, "divide by zero encountered in floor_divide")]
    result, caught = _recorded(lambda: np.array([1.0]) % 0)
    assert np.isnan(result[0])
    assert caught == [(RuntimeWarning, "invalid value encountered in remainder")]


def test_float_overflow_in_multiply_warns():
    result, caught = _recorded(lambda: np.array([1e308]) * 10)
    assert_array_equal(result, [np.inf])
    assert caught == [(RuntimeWarning, "overflow encountered in multiply")]
    single, caught = _recorded(lambda: np.array([3e38], np.float32) * np.float32(10))
    assert single.dtype == np.float32
    assert_array_equal(single, [np.inf])
    assert caught == [(RuntimeWarning, "overflow encountered in multiply")]


def test_overflow_in_exp_power_and_reduce_warns():
    result, caught = _recorded(lambda: np.exp(np.array([1000.0])))
    assert_array_equal(result, [np.inf])
    assert caught == [(RuntimeWarning, "overflow encountered in exp")]
    result, caught = _recorded(lambda: np.power(np.array([10.0]), 400))
    assert_array_equal(result, [np.inf])
    assert caught == [(RuntimeWarning, "overflow encountered in power")]
    result, caught = _recorded(lambda: np.sum(np.array([1e308, 1e308])))
    assert result == np.inf
    assert caught == [(RuntimeWarning, "overflow encountered in reduce")]


def test_overflow_and_invalid_in_one_call_warn_in_flag_order():
    result, caught = _recorded(lambda: np.array([np.inf, 1e308]) * np.array([0.0, 10.0]))
    assert np.isnan(result[0])
    assert result[1] == np.inf
    assert caught == [
        (RuntimeWarning, "overflow encountered in multiply"),
        (RuntimeWarning, "invalid value encountered in multiply"),
    ]


@pytest.mark.parametrize("name", ["sqrt", "log", "log2", "log10", "log1p", "arcsin", "arccos"])
def test_invalid_domain_returns_nan_and_warns(name):
    result, caught = _recorded(lambda: getattr(np, name)(np.array([-2.0, 0.25])))
    assert np.isnan(result[0])
    assert np.isfinite(result[1])
    assert caught == [(RuntimeWarning, "invalid value encountered in " + name)]


@pytest.mark.parametrize("name", ["log", "log2", "log10"])
def test_log_of_zero_returns_negative_infinity_and_warns_divide(name):
    result, caught = _recorded(lambda: getattr(np, name)(np.array([0.0])))
    assert_array_equal(result, [-np.inf])
    assert caught == [(RuntimeWarning, "divide by zero encountered in " + name)]


def test_infinite_arithmetic_warns_invalid():
    result, caught = _recorded(lambda: np.array([np.inf]) - np.inf)
    assert np.isnan(result[0])
    assert caught == [(RuntimeWarning, "invalid value encountered in subtract")]
    result, caught = _recorded(lambda: np.array([np.inf]) * 0)
    assert np.isnan(result[0])
    assert caught == [(RuntimeWarning, "invalid value encountered in multiply")]


def test_nan_inputs_propagate_without_warning():
    result, caught = _recorded(lambda: np.array([np.nan]) + 1)
    assert np.isnan(result[0])
    assert caught == []
    result, caught = _recorded(lambda: np.array([np.nan, 1.0]) < 1)
    assert_array_equal(result, [False, False])
    assert caught == []


def test_underflow_is_ignored_by_default():
    result, caught = _recorded(lambda: np.array([1e-308]) * 1e-308)
    assert_array_equal(result, [0.0])
    assert caught == []


def test_integer_array_overflow_wraps_silently_even_when_raising():
    result, caught = _recorded(lambda: np.array([127], np.int8) + 1)
    assert_array_equal(result, [-128])
    assert caught == []
    with np.errstate(all="raise"):
        assert_array_equal(np.array([127], np.int8) + np.int8(1), [-128])
        assert_array_equal(np.array([2]) ** 70, [0])


def test_integer_min_floor_divided_by_minus_one_wraps_and_warns_overflow():
    result, caught = _recorded(lambda: np.array([-128], np.int8) // np.int8(-1))
    assert result.dtype == np.int8
    assert_array_equal(result, [-128])
    assert caught == [(RuntimeWarning, "overflow encountered in floor_divide")]


def test_scalar_arithmetic_uses_scalar_operation_names():
    result, caught = _recorded(lambda: np.float64(1.0) / np.float64(0.0))
    assert result == np.inf
    assert caught == [(RuntimeWarning, "divide by zero encountered in scalar divide")]
    with np.errstate(over="raise"):
        with pytest.raises(FloatingPointError) as info:
            np.int8(127) + np.int8(1)
    assert str(info.value) == "overflow encountered in scalar add"


def test_errstate_ignore_divide_suppresses_only_divide_warning():
    with np.errstate(divide="ignore"):
        result, caught = _recorded(lambda: np.array([1.0, 0.0]) / 0)
    assert result[0] == np.inf
    assert np.isnan(result[1])
    assert caught == [(RuntimeWarning, "invalid value encountered in divide")]


def test_errstate_ignore_all_suppresses_every_warning():
    with np.errstate(all="ignore"):
        result, caught = _recorded(lambda: np.array([1.0, 0.0, 1e308]) / np.array([0.0, 0.0, 1e-308]))
        assert np.geterr() == {"divide": "ignore", "over": "ignore", "under": "ignore", "invalid": "ignore"}
    assert result[0] == np.inf
    assert np.isnan(result[1])
    assert result[2] == np.inf
    assert caught == []


def test_errstate_raise_divide_raises_floating_point_error():
    with np.errstate(divide="raise"):
        with pytest.raises(FloatingPointError) as info:
            np.array([1.0]) / 0
        assert str(info.value) == "divide by zero encountered in divide"
        result, caught = _recorded(lambda: np.array([0.0]) / 0)
    assert np.isnan(result[0])
    assert caught == [(RuntimeWarning, "invalid value encountered in divide")]


def test_errstate_all_raise_covers_every_flag():
    with np.errstate(all="raise"):
        with pytest.raises(FloatingPointError) as info:
            np.array([1e308]) * 10
        assert str(info.value) == "overflow encountered in multiply"
        with pytest.raises(FloatingPointError) as info:
            np.array([1e-308]) * 1e-308
        assert str(info.value) == "underflow encountered in multiply"
        with pytest.raises(FloatingPointError) as info:
            np.sqrt(np.array([-1.0]))
        assert str(info.value) == "invalid value encountered in sqrt"
        with pytest.raises(FloatingPointError) as info:
            np.array([1, 2]) // 0
        assert str(info.value) == "divide by zero encountered in floor_divide"


def test_errstate_raise_reports_divide_before_invalid():
    with np.errstate(all="raise"):
        with pytest.raises(FloatingPointError) as info:
            np.array([0.0, 1.0]) / 0
    assert str(info.value) == "divide by zero encountered in divide"


def test_errstate_specific_keyword_overrides_all():
    with np.errstate(all="ignore", divide="raise"):
        assert np.geterr() == {"divide": "raise", "over": "ignore", "under": "ignore", "invalid": "ignore"}
        with pytest.raises(FloatingPointError):
            np.array([0.0, 1.0]) / 0
        result, caught = _recorded(lambda: np.array([0.0]) / 0)
    assert np.isnan(result[0])
    assert caught == []


def test_errstate_restores_settings_on_exit():
    before = np.geterr()
    with np.errstate(divide="ignore", over="raise"):
        assert np.geterr() == {"divide": "ignore", "over": "raise", "under": "ignore", "invalid": "warn"}
    assert np.geterr() == before
    _, caught = _recorded(lambda: np.array([1.0]) / 0)
    assert caught == [(RuntimeWarning, "divide by zero encountered in divide")]


def test_errstate_restores_settings_after_exception():
    with pytest.raises(FloatingPointError) as info:
        with np.errstate(divide="raise"):
            np.array([1.0]) / 0
    assert str(info.value) == "divide by zero encountered in divide"
    assert np.geterr() == DEFAULT_ERROR_STATE


def test_nested_errstate_restores_outer_settings():
    with np.errstate(divide="ignore"):
        with np.errstate(invalid="ignore"):
            assert np.geterr() == {"divide": "ignore", "over": "warn", "under": "ignore", "invalid": "ignore"}
        assert np.geterr() == {"divide": "ignore", "over": "warn", "under": "ignore", "invalid": "warn"}
    assert np.geterr() == DEFAULT_ERROR_STATE


def test_errstate_object_applies_only_inside_with_block():
    state = np.errstate(divide="ignore")
    assert np.geterr()["divide"] == "warn"
    with state:
        assert np.geterr()["divide"] == "ignore"
    assert np.geterr()["divide"] == "warn"


def test_errstate_rejects_unknown_mode():
    with pytest.raises(ValueError) as info:
        with np.errstate(divide="bogus"):
            pass
    assert str(info.value) == "invalid error mode 'bogus'"
    assert np.geterr() == DEFAULT_ERROR_STATE


def test_seterr_returns_previous_settings_and_geterr_reflects_them():
    previous = np.seterr(all="raise")
    try:
        assert previous == DEFAULT_ERROR_STATE
        assert np.geterr() == {"divide": "raise", "over": "raise", "under": "raise", "invalid": "raise"}
        with pytest.raises(FloatingPointError):
            np.array([1.0]) / 0
    finally:
        replaced = np.seterr(**previous)
    assert replaced == {"divide": "raise", "over": "raise", "under": "raise", "invalid": "raise"}
    assert np.geterr() == DEFAULT_ERROR_STATE


def test_seterr_single_category_leaves_others_unchanged():
    previous = np.seterr(divide="ignore")
    try:
        assert np.geterr() == {"divide": "ignore", "over": "warn", "under": "ignore", "invalid": "warn"}
        result, caught = _recorded(lambda: np.array([1.0]) / 0)
    finally:
        np.seterr(**previous)
    assert_array_equal(result, [np.inf])
    assert caught == []
    assert np.geterr() == DEFAULT_ERROR_STATE


def test_seterr_rejects_unknown_mode_without_changing_state():
    with pytest.raises(ValueError) as info:
        np.seterr(divide="bogus")
    assert str(info.value) == "invalid error mode 'bogus'"
    assert np.geterr() == DEFAULT_ERROR_STATE


def test_warning_and_error_categories():
    assert issubclass(RuntimeWarning, Warning)
    assert issubclass(FloatingPointError, ArithmeticError)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        np.array([1.0]) / 0
    assert len(caught) == 1
    assert caught[0].category is RuntimeWarning
    assert isinstance(caught[0].message, RuntimeWarning)
    assert isinstance(caught[0].message, Warning)


def test_warnings_error_filter_turns_numpy_warning_into_exception():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(RuntimeWarning) as info:
            np.array([1.0]) / 0
    assert str(info.value) == "divide by zero encountered in divide"


def test_warnings_ignore_filter_suppresses_numpy_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore")
        result = np.array([1.0, 0.0]) / 0
    assert result[0] == np.inf
    assert caught == []


def test_filterwarnings_by_message_ignores_only_matching_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", message="divide by zero")
        np.array([1.0, 0.0]) / 0
    assert [str(item.message) for item in caught] == ["invalid value encountered in divide"]


def test_filterwarnings_by_category_ignores_runtime_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        np.array([1.0, 0.0]) / 0
        warnings.warn("kept", UserWarning, stacklevel=2)
    assert [(item.category, str(item.message)) for item in caught] == [(UserWarning, "kept")]
