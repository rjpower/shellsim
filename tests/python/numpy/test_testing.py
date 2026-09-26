# Portable NumPy semantics. Expectations checked against NumPy 2.5.3 on CPython 3.14.4.
# Scope: numpy.testing assertion helpers, their tolerances, and failure exception types.

import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)


def test_assert_allclose_passes_for_close_values():
    assert_allclose([1.0, 2.0], [1.0, 2.0 + 1e-8])
    assert_allclose(np.arange(3), [0, 1, 2])
    assert_allclose([1 + 1j], [1 + 1j])
    assert_allclose(1.0 + 1e-8, 1.0)


def test_assert_allclose_default_rtol_rejects_larger_error():
    with pytest.raises(AssertionError):
        assert_allclose(1.0 + 1e-6, 1.0)


def test_assert_allclose_fails_on_mismatch():
    with pytest.raises(AssertionError):
        assert_allclose([1.0, 2.0], [1.0, 2.5])


def test_assert_allclose_rtol_is_relative_to_desired():
    assert_allclose(100.0, 111.0, rtol=0.1)
    with pytest.raises(AssertionError):
        assert_allclose(111.0, 100.0, rtol=0.1)


def test_assert_allclose_default_atol_is_zero():
    with pytest.raises(AssertionError):
        assert_allclose(1e-9, 0.0)
    assert_allclose(1e-9, 0.0, atol=1e-8)


def test_assert_allclose_combines_atol_and_rtol():
    assert_allclose([1.0, 0.0], [1.05, 1e-9], rtol=0.1, atol=1e-8)
    with pytest.raises(AssertionError):
        assert_allclose([1.0, 0.0], [1.05, 1e-7], rtol=0.1, atol=1e-8)


def test_assert_allclose_treats_nan_as_equal_by_default():
    assert_allclose([np.nan, 1.0], [np.nan, 1.0])
    assert_allclose([np.inf], [np.inf])
    with pytest.raises(AssertionError):
        assert_allclose([np.nan], [np.nan], equal_nan=False)


def test_assert_allclose_broadcasts_scalar_desired():
    assert_allclose([[1.0, 1.0]], 1.0)


def test_assert_allclose_shape_mismatch_fails():
    with pytest.raises(AssertionError):
        assert_allclose([1, 2], [1, 2, 3])


def test_assert_allclose_err_msg_still_raises_assertion_error():
    with pytest.raises(AssertionError):
        assert_allclose(1.0, 2.0, err_msg="context")


def test_assert_array_equal_accepts_lists_and_arrays():
    assert_array_equal([1, 2, 3], np.array([1, 2, 3]))
    assert_array_equal(np.array([1, 2]), [1.0, 2.0])
    assert_array_equal(np.array(["a", "b"]), ["a", "b"])
    assert_array_equal(np.array([True, False]), [True, False])
    assert_array_equal(np.array([True, False]), [1, 0])


def test_assert_array_equal_broadcasts_scalar():
    assert_array_equal(np.zeros(3), 0)


def test_assert_array_equal_value_mismatch_fails():
    with pytest.raises(AssertionError):
        assert_array_equal([1, 2, 3], [1, 2, 4])


def test_assert_array_equal_shape_mismatch_fails():
    with pytest.raises(AssertionError):
        assert_array_equal(np.zeros((2, 3)), np.zeros((3, 2)))


def test_assert_array_equal_nan_handling():
    assert_array_equal([np.nan, 1.0], [np.nan, 1.0])
    with pytest.raises(AssertionError):
        assert_array_equal([np.nan], [1.0])


def test_assert_array_equal_ignores_zero_sign():
    assert_array_equal(np.array([0.0]), np.array([-0.0]))


def test_assert_array_almost_equal_default_six_decimals():
    assert_array_almost_equal([1.0, 2.0], [1.0000001, 2.0])
    with pytest.raises(AssertionError):
        assert_array_almost_equal([1.0], [1.00001])


def test_assert_array_almost_equal_uses_one_and_a_half_units_threshold():
    assert_array_almost_equal([1.0], [1.0000014])
    with pytest.raises(AssertionError):
        assert_array_almost_equal([1.0], [1.0000016])


def test_assert_array_almost_equal_decimal_argument():
    assert_array_almost_equal([1.0], [1.00001], decimal=4)
    assert_array_almost_equal(np.eye(2), [[1.0, 1e-8], [0.0, 1.0]])


def test_assert_array_almost_equal_shape_mismatch_fails():
    with pytest.raises(AssertionError):
        assert_array_almost_equal([1.0, 2.0], [1.0])


def test_assert_array_almost_equal_nan_and_inf_positions():
    assert_array_almost_equal([np.nan], [np.nan])
    assert_array_almost_equal([np.inf], [np.inf])
    with pytest.raises(AssertionError):
        assert_array_almost_equal([np.inf, 1.0], [1.0, np.inf])


def test_assert_equal_scalars_and_strings():
    assert_equal(3, 3)
    assert_equal(1, 1.0)
    assert_equal("abc", "abc")
    assert_equal(np.nan, np.nan)
    assert_equal(np.inf, np.inf)
    with pytest.raises(AssertionError):
        assert_equal(3, 4)
    with pytest.raises(AssertionError):
        assert_equal(np.inf, -np.inf)


def test_assert_equal_distinguishes_signed_zero_scalars():
    with pytest.raises(AssertionError):
        assert_equal(0.0, -0.0)


def test_assert_equal_nested_lists_and_tuples():
    assert_equal([1, [2, 3]], [1, [2, 3]])
    assert_equal((1, 2), [1, 2])
    with pytest.raises(AssertionError):
        assert_equal([1, [2, 3]], [1, [2, 4]])
    with pytest.raises(AssertionError):
        assert_equal([1, 2], [1, 2, 3])


def test_assert_equal_dicts():
    assert_equal({"a": 1, "b": [1, 2]}, {"a": 1, "b": [1, 2]})
    with pytest.raises(AssertionError):
        assert_equal({"a": 1}, {"a": 2})
    with pytest.raises(AssertionError):
        assert_equal({"a": 1}, {"b": 1})


def test_assert_equal_arrays_inside_containers():
    assert_equal(np.arange(3), [0, 1, 2])
    assert_equal({"x": np.arange(2)}, {"x": np.array([0, 1])})
    assert_equal([np.arange(2), 5], [[0, 1], 5])
    with pytest.raises(AssertionError):
        assert_equal(np.zeros(2), np.zeros(3))


def test_assert_almost_equal_default_seven_decimals():
    assert_almost_equal(1.0, 1.00000001)
    assert_almost_equal(1.0, 1.00000014)
    with pytest.raises(AssertionError):
        assert_almost_equal(1.0, 1.00000016)
    with pytest.raises(AssertionError):
        assert_almost_equal(1.0, 1.000001)


def test_assert_almost_equal_decimal_argument():
    assert_almost_equal(1.0, 1.001, decimal=2)
    assert_almost_equal(1.0, 1.01, decimal=2)
    with pytest.raises(AssertionError):
        assert_almost_equal(1.0, 1.02, decimal=2)


def test_assert_almost_equal_arrays_complex_and_nan():
    assert_almost_equal(np.array([1.0, 2.0]), [1.0, 2.00000001])
    assert_almost_equal(1 + 1j, 1 + 1.00000001j)
    assert_almost_equal(np.nan, np.nan)
    with pytest.raises(AssertionError):
        assert_almost_equal(np.array([1.0, 2.0]), [1.0, 2.1])
