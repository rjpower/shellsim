# Portable NumPy semantics. Expectations checked against NumPy 2.5.3 on CPython 3.14.4.
# Scope: elementwise ufuncs, operators, result dtypes, broadcasting, out=, and numeric helpers.

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal


def test_array_arithmetic_between_int_arrays_keeps_int64():
    a = np.array([5, 7, 9])
    b = np.array([1, 2, 3])
    for result, expected in [(a + b, [6, 9, 12]), (a - b, [4, 5, 6]), (a * b, [5, 14, 27])]:
        assert result.dtype == np.int64
        assert_array_equal(result, expected)


def test_python_scalars_on_either_side_of_operators():
    a = np.array([1, 2, 4])
    assert_array_equal(a + 2, [3, 4, 6])
    assert_array_equal(2 + a, [3, 4, 6])
    assert_array_equal(a - 10, [-9, -8, -6])
    assert_array_equal(10 - a, [9, 8, 6])
    assert_array_equal(a * 3, [3, 6, 12])
    assert_array_equal(3 * a, [3, 6, 12])
    assert_array_equal(a / 2, [0.5, 1.0, 2.0])
    assert_array_equal(8 / a, [8.0, 4.0, 2.0])
    assert_array_equal(a // 2, [0, 1, 2])
    assert_array_equal(9 // a, [9, 4, 2])
    assert_array_equal(a % 3, [1, 2, 1])
    assert_array_equal(9 % a, [0, 1, 1])
    assert_array_equal(a**2, [1, 4, 16])
    assert_array_equal(2**a, [2, 4, 16])
    assert (a + 2).dtype == np.int64
    assert (2**a).dtype == np.int64


def test_true_divide_of_ints_gives_float64_and_float32_stays_float32():
    ints = np.array([1, 2], np.int32) / np.array([2, 4], np.int32)
    assert ints.dtype == np.float64
    assert_array_equal(ints, [0.5, 0.5])
    floats = np.array([1.0, 3.0], np.float32) / 2
    assert floats.dtype == np.float32
    assert_array_equal(floats, [0.5, 1.5])


def test_int_floor_divide_and_remainder_follow_python_signs():
    dividend = np.array([-7, 7, -7, 7])
    divisor = np.array([2, 2, -2, -2])
    assert_array_equal(dividend // divisor, [-4, 3, 3, -4])
    assert_array_equal(dividend % np.array([3, 3, -3, -3]), [2, 1, -1, -2])
    assert_array_equal(np.floor_divide(dividend, divisor), [-4, 3, 3, -4])
    assert_array_equal(np.remainder(dividend, 3), [2, 1, 2, 1])
    assert (dividend // divisor).dtype == np.int64


def test_float_floor_divide_and_remainder_follow_python_signs():
    dividend = np.array([-7.0, 7.0, -7.0, 7.0])
    assert_array_equal(dividend // np.array([2, 2, -2, -2]), [-4.0, 3.0, 3.0, -4.0])
    assert_array_equal(dividend % np.array([3, 3, -3, -3]), [2.0, 1.0, -1.0, -2.0])
    assert_array_equal(np.floor_divide(np.array([5.5, -5.5]), 2), [2.0, -3.0])
    assert_array_equal(np.remainder(np.array([5.5, -5.5]), 2), [1.5, 0.5])


def test_divmod_builtin_and_ufunc_return_quotient_and_remainder():
    quotient, remainder = divmod(np.array([-7, 7]), 3)
    assert_array_equal(quotient, [-3, 2])
    assert_array_equal(remainder, [2, 1])
    quotient, remainder = np.divmod(np.array([-7.5, 7.5]), 2)
    assert_array_equal(quotient, [-4.0, 3.0])
    assert_array_equal(remainder, [0.5, 1.5])


def test_integer_power_and_negative_integer_exponent_error():
    assert_array_equal(np.array([2, 3]) ** 2, [4, 9])
    assert_array_equal(np.power(np.array([2, 3]), np.array([3, 0])), [8, 1])
    with pytest.raises(ValueError) as info:
        np.array([2]) ** -1
    assert str(info.value) == "Integers to negative integer powers are not allowed."
    with pytest.raises(ValueError):
        np.power(np.array([2]), np.array([-1]))


def test_float_power_and_int_array_with_float_exponent():
    assert_array_equal(np.array([4.0, 9.0]) ** 0.5, [2.0, 3.0])
    assert_array_equal(np.array([2.0, 4.0]) ** -1, [0.5, 0.25])
    result = np.array([4]) ** 0.5
    assert result.dtype == np.float64
    assert_array_equal(result, [2.0])


def test_unary_negative_positive_and_absolute():
    a = np.array([1, -2, 0])
    assert_array_equal(-a, [-1, 2, 0])
    assert_array_equal(+a, [1, -2, 0])
    assert_array_equal(abs(a), [1, 2, 0])
    assert_array_equal(np.negative(a), [-1, 2, 0])
    assert_array_equal(np.absolute(np.array([-1.5, 2.5])), [1.5, 2.5])
    assert abs(a).dtype == np.int64


def test_invert_on_ints_and_bools():
    assert_array_equal(~np.array([0, 1, -2]), [-1, -2, 1])
    assert_array_equal(np.invert(np.array([0, 255], np.uint8)), [255, 0])
    flipped = ~np.array([True, False])
    assert flipped.dtype == np.bool_
    assert_array_equal(flipped, [False, True])


@pytest.mark.parametrize(
    "name, expected",
    [
        ("less", [True, False, False]),
        ("less_equal", [True, True, False]),
        ("equal", [False, True, False]),
        ("not_equal", [True, False, True]),
        ("greater", [False, False, True]),
        ("greater_equal", [False, True, True]),
    ],
)
def test_comparison_ufuncs_return_bool_arrays(name, expected):
    result = getattr(np, name)(np.array([1, 2, 3]), np.array([2, 2, 2]))
    assert result.dtype == np.bool_
    assert_array_equal(result, expected)


def test_comparison_operators_with_python_scalars_and_mixed_dtypes():
    a = np.array([1, 2, 3])
    assert_array_equal(a < 2, [True, False, False])
    assert_array_equal(2 < a, [False, False, True])
    assert_array_equal(a <= 2, [True, True, False])
    assert_array_equal(a == 2.0, [False, True, False])
    assert_array_equal(a != 2, [True, False, True])
    assert_array_equal(a >= np.array([1.5, 2.0, 2.5]), [False, True, True])
    assert (a > 1).dtype == np.bool_


def test_logical_ufuncs_use_truthiness_and_return_bool():
    assert_array_equal(np.logical_and([1, 0, 2], [3, 0, 0]), [True, False, False])
    assert_array_equal(np.logical_or([0.0, 0.0, 1.5], [0, 2, 0]), [False, True, True])
    assert_array_equal(np.logical_xor([True, True, False], [True, False, False]), [False, True, False])
    result = np.logical_not(np.array([0, 1, 2]))
    assert result.dtype == np.bool_
    assert_array_equal(result, [True, False, False])


def test_bitwise_operators_on_ints():
    a = np.array([12, 10])
    assert_array_equal(a & 6, [4, 2])
    assert_array_equal(a | 1, [13, 11])
    assert_array_equal(a ^ 6, [10, 12])
    assert_array_equal(np.bitwise_and(a, np.array([4, 8])), [4, 8])
    assert_array_equal(np.bitwise_or(a, 3), [15, 11])
    assert_array_equal(np.bitwise_xor(a, a), [0, 0])


def test_bitwise_operators_on_bools_stay_bool():
    left = np.array([True, True, False, False])
    right = np.array([True, False, True, False])
    for result, expected in [
        (left & right, [True, False, False, False]),
        (left | right, [True, True, True, False]),
        (left ^ right, [False, True, True, False]),
    ]:
        assert result.dtype == np.bool_
        assert_array_equal(result, expected)


def test_shift_operators_and_arithmetic_right_shift():
    assert_array_equal(np.array([1, 2]) << np.array([3, 4]), [8, 32])
    assert_array_equal(np.array([1, 2, 4]) << 2, [4, 8, 16])
    assert_array_equal(np.array([-8, 8]) >> 1, [-4, 4])
    assert_array_equal(np.right_shift(np.array([-1], np.int8), 3), [-1])
    assert_array_equal(np.left_shift(np.array([1], np.uint8), 7), [128])
    assert_array_equal(np.left_shift(np.array([1], np.int8), 7), [-128])


def test_int8_array_arithmetic_wraps():
    a = np.array([127], np.int8)
    result = a + np.int8(1)
    assert result.dtype == np.int8
    assert_array_equal(result, [-128])
    assert_array_equal(a + 1, [-128])
    assert_array_equal(np.array([-128], np.int8) - 1, [127])
    assert_array_equal(np.array([100], np.int8) * 2, [-56])
    assert_array_equal(np.power(np.array([2], np.int8), 8), [0])


def test_uint8_array_arithmetic_wraps():
    assert_array_equal(np.array([0], np.uint8) - np.uint8(1), [255])
    assert_array_equal(np.array([255], np.uint8) + 1, [0])
    result = np.array([200], np.uint8) * 2
    assert result.dtype == np.uint8
    assert_array_equal(result, [144])


def test_int8_scalar_arithmetic_wraps_when_overflow_ignored():
    with np.errstate(over="ignore"):
        total = np.int8(127) + np.int8(1)
        difference = np.uint8(0) - np.uint8(1)
    assert type(total) is np.int8
    assert total == -128
    assert type(difference) is np.uint8
    assert difference == 255


def test_python_int_out_of_range_for_array_dtype_raises_overflow_error():
    with pytest.raises(OverflowError) as info:
        np.array([1], np.int8) + 300
    assert str(info.value) == "Python integer 300 out of bounds for int8"
    with pytest.raises(OverflowError) as info:
        np.array([1], np.uint8) + (-1)
    assert str(info.value) == "Python integer -1 out of bounds for uint8"


def test_binary_result_dtypes_follow_promotion_rules():
    assert (np.array([1], np.uint8) + np.array([1], np.int8)).dtype == np.int16
    assert (np.array([1], np.int32) + np.array([1], np.float32)).dtype == np.float64
    assert (np.array([1], np.int8) + np.array([1], np.float32)).dtype == np.float32
    assert (np.array([1], np.int8) + 1.5).dtype == np.float64
    assert (np.array([1], np.float32) + np.float64(1)).dtype == np.float64
    assert (np.array([1], np.float32) * np.array([2.0])).dtype == np.float64
    assert (np.array([1, 2]) + 1j).dtype == np.complex128
    assert (np.array([1], np.float32) + 1j).dtype == np.complex64


def test_float32_operations_stay_float32():
    f = np.array([1.0, 4.0], np.float32)
    for result in [f + 1.5, f * f, f - 2, f / 2, f**2, np.sqrt(f), np.exp(f), -f, abs(f)]:
        assert result.dtype == np.float32
    assert_array_equal(np.sqrt(f), [1.0, 2.0])
    assert_allclose(np.exp(f), [2.7182817, 54.59815], rtol=1e-6)


@pytest.mark.parametrize(
    "int_name, float_name",
    [
        ("int8", "float16"),
        ("uint8", "float16"),
        ("int16", "float32"),
        ("uint16", "float32"),
        ("int32", "float64"),
        ("int64", "float64"),
    ],
)
def test_float_ufuncs_on_ints_use_smallest_sufficient_float(int_name, float_name):
    result = np.sqrt(np.array([4, 9], getattr(np, int_name)))
    assert result.dtype == getattr(np, float_name)
    assert_array_equal(result, [2.0, 3.0])


@pytest.mark.parametrize(
    "name, value, expected",
    [
        ("sqrt", 2.25, 1.5),
        ("exp", 0.5, 1.6487212707001282),
        ("expm1", 1e-10, 1.00000000005e-10),
        ("log", 2.5, 0.9162907318741551),
        ("log1p", 1e-10, 9.9999999995e-11),
        ("log2", 10.5, 3.3923174227787602),
        ("log10", 31.5, 1.4983105537896004),
        ("sin", 0.5, 0.479425538604203),
        ("cos", 0.5, 0.8775825618903728),
        ("tan", 0.5, 0.5463024898437905),
        ("arcsin", 0.5, 0.5235987755982989),
        ("arccos", 0.5, 1.0471975511965979),
        ("arctan", 0.5, 0.4636476090008061),
        ("sinh", 0.5, 0.5210953054937474),
        ("cosh", 0.5, 1.1276259652063807),
        ("tanh", 0.5, 0.46211715726000974),
    ],
)
def test_unary_math_ufuncs_match_reference_values(name, value, expected):
    result = getattr(np, name)(np.array([value, value]))
    assert result.dtype == np.float64
    assert_allclose(result, [expected, expected], rtol=1e-14)


def test_arctan2_and_hypot():
    assert_allclose(np.arctan2(1.0, -1.0), 3 * np.pi / 4, rtol=1e-15)
    assert_allclose(np.arctan2(np.array([0.0, 1.0]), np.array([1.0, 0.0])), [0.0, np.pi / 2])
    assert np.hypot(3, 4) == 5.0
    assert_array_equal(np.hypot(np.array([3.0, 5.0]), np.array([4.0, 12.0])), [5.0, 13.0])


def test_floor_ceil_trunc_rint_on_floats():
    values = np.array([-1.5, 1.5, -0.2, 2.5])
    assert_array_equal(np.floor(values), [-2.0, 1.0, -1.0, 2.0])
    assert_array_equal(np.ceil(values), [-1.0, 2.0, -0.0, 3.0])
    assert_array_equal(np.trunc(np.array([-1.7, 1.7])), [-1.0, 1.0])
    assert_array_equal(np.rint(np.array([-1.5, 0.5, 1.5, 2.5, 2.6])), [-2.0, 0.0, 2.0, 2.0, 3.0])


def test_floor_ceil_trunc_keep_integer_dtype_but_rint_does_not():
    ints = np.array([2, -3])
    for name in ["floor", "ceil", "trunc"]:
        result = getattr(np, name)(ints)
        assert result.dtype == np.int64
        assert_array_equal(result, [2, -3])
    assert np.ceil(np.array([2], np.int8)).dtype == np.int8
    assert np.rint(ints).dtype == np.float64


def test_round_uses_round_half_to_even():
    assert np.round(2.5) == 2.0
    assert type(np.round(2.5)) is np.float64
    assert np.round(1.2345, 2) == 1.23
    assert_array_equal(np.around(np.array([0.5, 1.5, 2.5, 3.5, -1.5])), [0.0, 2.0, 2.0, 4.0, -2.0])
    assert_array_equal(np.round(np.array([0.125, 0.375]), 2), [0.12, 0.38])
    assert_array_equal(np.array([1.25, 2.35]).round(1), [1.2, 2.4])


def test_round_with_negative_decimals():
    assert_array_equal(np.round(np.array([15.0, 25.0, 35.0]), -1), [20.0, 20.0, 40.0])
    result = np.round(np.array([1250, 1350, -1250]), -2)
    assert result.dtype == np.int64
    assert_array_equal(result, [1200, 1400, -1200])


def test_sign_of_ints_and_floats():
    ints = np.sign(np.array([-2, 0, 3]))
    assert ints.dtype == np.int64
    assert_array_equal(ints, [-1, 0, 1])
    assert_array_equal(np.sign(np.array([-2.5, 0.0, np.nan])), [-1.0, 0.0, np.nan])


def test_nan_and_infinity_predicates():
    values = np.array([1.0, np.nan, np.inf, -np.inf])
    assert_array_equal(np.isnan(values), [False, True, False, False])
    assert_array_equal(np.isinf(values), [False, False, True, True])
    assert_array_equal(np.isfinite(values), [True, False, False, False])
    assert_array_equal(np.isnan(np.array([1, 2])), [False, False])
    assert_array_equal(np.isfinite(np.array([1, 2])), [True, True])
    assert np.isnan(values).dtype == np.bool_


def test_clip_with_optional_bounds():
    a = np.arange(6)
    assert_array_equal(np.clip(a, 2, None), [2, 2, 2, 3, 4, 5])
    assert_array_equal(np.clip(a, None, 3), [0, 1, 2, 3, 3, 3])
    assert_array_equal(a.clip(1, 3), [1, 1, 2, 3, 3, 3])
    assert np.clip(a, 1, 3).dtype == np.int64
    assert_array_equal(np.clip(a, 1, 4.5), [1.0, 1.0, 2.0, 3.0, 4.0, 4.5])


def test_maximum_minimum_propagate_nan_but_fmax_fmin_ignore_it():
    x = np.array([1.0, np.nan, 3.0])
    y = np.array([2.0, 5.0, np.nan])
    assert_array_equal(np.maximum(x, y), [2.0, np.nan, np.nan])
    assert_array_equal(np.minimum(x, y), [1.0, np.nan, np.nan])
    assert_array_equal(np.fmax(x, y), [2.0, 5.0, 3.0])
    assert_array_equal(np.fmin(x, y), [1.0, 5.0, 3.0])
    assert np.isnan(np.fmax(np.nan, np.nan))
    assert_array_equal(np.maximum(np.array([1, 5]), 3), [3, 5])


def test_square_reciprocal_and_angle_conversions():
    squared = np.square(np.array([-3, 4], np.int8))
    assert squared.dtype == np.int8
    assert_array_equal(squared, [9, 16])
    assert_array_equal(np.reciprocal(np.array([2.0, 4.0, -0.5])), [0.5, 0.25, -2.0])
    assert_allclose(np.deg2rad(np.array([180.0, 90.0])), [np.pi, np.pi / 2])
    assert_allclose(np.rad2deg(np.array([np.pi, np.pi / 4])), [180.0, 45.0])


def test_broadcasting_column_against_row():
    column = np.array([[0], [10], [20]])
    row = np.array([[1, 2, 3, 4]])
    result = column + row
    assert result.shape == (3, 4)
    assert_array_equal(result, [[1, 2, 3, 4], [11, 12, 13, 14], [21, 22, 23, 24]])


def test_broadcasting_matrix_against_trailing_vector():
    result = np.array([[1, 2, 3], [4, 5, 6]]) * np.array([10, 0, -1])
    assert result.shape == (2, 3)
    assert_array_equal(result, [[10, 0, -3], [40, 0, -6]])
    assert (np.ones((2, 1, 3)) + np.ones((4, 1))).shape == (2, 4, 3)


def test_incompatible_broadcast_raises_value_error():
    with pytest.raises(ValueError) as info:
        np.ones((2, 3)) + np.ones((2,))
    assert "could not be broadcast together with shapes (2,3) (2,)" in str(info.value)


def test_named_ufuncs_are_ufunc_objects():
    for name in ["add", "multiply", "maximum", "sqrt", "logical_and"]:
        ufunc = getattr(np, name)
        assert isinstance(ufunc, np.ufunc)
        assert ufunc.__name__ == name
    assert np.add.nin == 2
    assert np.add.nout == 1
    assert np.sqrt.nin == 1


def test_ufunc_reduce_accumulate_and_outer():
    matrix = np.array([[1, 2], [3, 4]])
    assert_array_equal(np.add.reduce(matrix), [4, 6])
    assert_array_equal(np.add.reduce(matrix, axis=1), [3, 7])
    assert np.maximum.reduce(np.array([3, 1, 4])) == 4
    assert type(np.maximum.reduce(np.array([3, 1, 4]))) is np.int64
    assert_array_equal(np.multiply.accumulate(np.array([1, 2, 3, 4])), [1, 2, 6, 24])
    assert_array_equal(np.add.accumulate(matrix, axis=1), [[1, 3], [3, 7]])
    outer = np.add.outer(np.array([1, 2]), np.array([10, 20, 30]))
    assert outer.shape == (2, 3)
    assert_array_equal(outer, [[11, 21, 31], [12, 22, 32]])
    assert_array_equal(np.multiply.outer([1, 2], [3, 4]), [[3, 4], [6, 8]])


def test_out_argument_returns_and_fills_the_given_array():
    a = np.array([1, 2])
    target = np.zeros(2, np.int64)
    result = np.add(a, a, out=target)
    assert result is target
    assert_array_equal(target, [2, 4])
    floats = np.zeros(2)
    assert np.multiply(a, 3, out=floats) is floats
    assert floats.dtype == np.float64
    assert_array_equal(floats, [3.0, 6.0])
    assert np.sqrt(np.array([4.0, 9.0]), out=floats) is floats
    assert_array_equal(floats, [2.0, 3.0])


def test_out_argument_rejects_cast_across_kinds():
    target = np.zeros(2, np.int64)
    with pytest.raises(TypeError) as info:
        np.add(np.array([1, 2]), 1.5, out=target)
    assert str(info.value) == (
        "Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'"
    )
    assert_array_equal(target, [0, 0])


def test_inplace_operators_update_the_same_array():
    a = np.array([1, 2, 3])
    alias = a
    a += 1
    a *= 3
    a -= 2
    a //= 2
    a %= 4
    a **= 2
    assert alias is a
    assert_array_equal(a, [4, 9, 1])
    f = np.array([1.0, 2.0])
    f /= 4
    assert_array_equal(f, [0.25, 0.5])
    f += np.array([1, 1])
    assert_array_equal(f, [1.25, 1.5])


def test_inplace_operator_updates_views():
    a = np.arange(6)
    view = a[::2]
    view += 10
    assert_array_equal(a, [10, 1, 12, 3, 14, 5])


def test_inplace_int_array_rejects_float_result():
    a = np.array([1, 2])
    with pytest.raises(TypeError) as info:
        a += 1.5
    assert str(info.value) == (
        "Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'"
    )
    assert_array_equal(a, [1, 2])
    with pytest.raises(TypeError) as info:
        a /= 2
    assert "Cannot cast ufunc 'divide' output" in str(info.value)


def test_inplace_add_with_reversed_self_reads_original_values():
    a = np.arange(4)
    a += a[::-1]
    assert_array_equal(a, [3, 3, 3, 3])


def test_inplace_int8_add_wraps_and_rejects_out_of_range_python_int():
    a = np.array([100], np.int8)
    a += 100
    assert a.dtype == np.int8
    assert_array_equal(a, [-56])
    with pytest.raises(OverflowError):
        a += 300


def test_complex128_arithmetic_and_parts():
    z = np.array([3 + 4j, 1 - 1j])
    assert z.dtype == np.complex128
    assert_array_equal(z * z, [-7 + 24j, -2j])
    assert_array_equal(z + 1, [4 + 4j, 2 - 1j])
    assert_array_equal(z / 1j, [4 - 3j, -1 - 1j])
    magnitude = np.abs(z)
    assert magnitude.dtype == np.float64
    assert_allclose(magnitude, [5.0, np.sqrt(2.0)])
    assert_array_equal(z.conj(), [3 - 4j, 1 + 1j])
    assert_array_equal(np.conj(z), [3 - 4j, 1 + 1j])
    assert_array_equal(z.real, [3.0, 1.0])
    assert_array_equal(z.imag, [4.0, -1.0])
    assert_allclose(np.angle(np.array([1j, -1, 1])), [np.pi / 2, np.pi, 0.0])


def test_isclose_and_allclose_tolerances():
    assert_array_equal(np.isclose([1.0, 1.0, np.nan], [1.0 + 1e-9, 1.1, np.nan]), [True, False, False])
    assert_array_equal(np.isclose([np.nan], [np.nan], equal_nan=True), [True])
    assert np.isclose(100.0, 101.0, rtol=0.01)
    assert np.isclose(0.0, 1e-9)
    assert not np.isclose(0.0, 1e-7)
    assert np.isclose(0.0, 1e-7, atol=1e-6)
    assert np.isclose(np.inf, np.inf)
    assert not np.isclose(np.inf, -np.inf)
    assert np.allclose([1.0, 2.0], [1.0, 2.0 + 1e-10]) is True
    assert np.allclose([1.0, np.nan], [1.0, np.nan]) is False
    assert np.allclose([1.0, np.nan], [1.0, np.nan], equal_nan=True) is True


def test_array_equal_compares_shape_and_values():
    assert np.array_equal([1, 2], [1, 2]) is True
    assert np.array_equal([1, 2], [1, 2, 3]) is False
    assert np.array_equal([1, 2], [[1, 2]]) is False
    assert np.array_equal(np.array([1, 2]), np.array([1.0, 2.0])) is True
    assert np.array_equal([1.0, np.nan], [1.0, np.nan]) is False
    assert np.array_equal([1.0, np.nan], [1.0, np.nan], equal_nan=True) is True


def test_interp_linear_with_clamped_and_explicit_edges():
    xp = [1, 2, 3]
    fp = [3, 2, 0]
    assert np.interp(2.5, xp, fp) == 1.0
    assert type(np.interp(2.5, xp, fp)) is np.float64
    assert_array_equal(np.interp([0, 1.5, 5], xp, fp), [3.0, 2.5, 0.0])
    assert_array_equal(np.interp([0, 5], xp, fp, left=-1, right=99), [-1.0, 99.0])


def test_polyfit_recovers_exact_polynomials_and_polyval_evaluates():
    line = np.polyfit([0, 1, 2, 3], [1, 3, 5, 7], 1)
    assert line.shape == (2,)
    assert_allclose(line, [2.0, 1.0], atol=1e-12)
    quadratic = np.polyfit([-1, 0, 1, 2], [6, 3, 2, 3], 2)
    assert_allclose(quadratic, [1.0, -2.0, 3.0], atol=1e-12)
    assert np.polyval([2, 1], 3) == 7
    assert_array_equal(np.polyval([1, -2, 3], np.array([0, 1, 2])), [3, 2, 3])
    assert_allclose(np.polyval(quadratic, 4.0), 11.0)


def test_convolve_modes():
    signal = [1, 2, 3]
    kernel = [0, 1, 0.5]
    assert_array_equal(np.convolve(signal, kernel), [0.0, 1.0, 2.5, 4.0, 1.5])
    assert_array_equal(np.convolve(signal, kernel, "full"), [0.0, 1.0, 2.5, 4.0, 1.5])
    assert_array_equal(np.convolve(signal, kernel, mode="same"), [1.0, 2.5, 4.0])
    assert_array_equal(np.convolve(signal, kernel, mode="valid"), [2.5])
    ints = np.convolve([1, 2, 3], [1, 1])
    assert ints.dtype == np.int64
    assert_array_equal(ints, [1, 3, 5, 3])


def test_cross_product_of_three_vectors():
    assert_array_equal(np.cross([1, 0, 0], [0, 1, 0]), [0, 0, 1])
    result = np.cross([1, 2, 3], [4, 5, 6])
    assert result.dtype == np.int64
    assert_array_equal(result, [-3, 6, -3])
    assert_array_equal(np.cross([1.0, 2.0, 3.0], [4, 5, 6]), [-3.0, 6.0, -3.0])


def test_gradient_of_one_dimensional_data():
    assert_array_equal(np.gradient(np.array([1, 2, 4, 7, 11])), [1.0, 1.5, 2.5, 3.5, 4.0])
    assert_array_equal(np.gradient([1, 4, 9, 16]), [3.0, 4.0, 6.0, 7.0])
    assert_array_equal(np.gradient(np.array([1.0, 2.0, 4.0, 7.0, 11.0]), 2.0), [0.5, 0.75, 1.25, 1.75, 2.0])


def test_nan_to_num_defaults_and_overrides():
    result = np.nan_to_num(np.array([np.nan, np.inf, -np.inf, 1.0]))
    assert_array_equal(result, [0.0, 1.7976931348623157e308, -1.7976931348623157e308, 1.0])
    overridden = np.nan_to_num(np.array([np.nan, np.inf, -np.inf]), nan=-1.0, posinf=9.0, neginf=-9.0)
    assert_array_equal(overridden, [-1.0, 9.0, -9.0])
    ints = np.nan_to_num(np.array([1, 2]))
    assert ints.dtype == np.int64
    assert_array_equal(ints, [1, 2])


def test_zero_dimensional_results_are_numpy_scalars():
    assert type(np.sqrt(np.float64(4))) is np.float64
    assert type(np.sqrt(np.float32(4))) is np.float32
    assert type(np.sqrt(4.0)) is np.float64
    total = np.add(np.array(1), 2)
    assert type(total) is np.int64
    assert total == 3
    assert type(np.array([1.0, 2.0])[0] + 1) is np.float64
    assert type(np.int8(1) + 1) is np.int8
    assert type(np.maximum(np.float32(1), 2.0)) is np.float32
    assert type(np.less(np.int64(1), 2)) is np.bool_
