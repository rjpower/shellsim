# Portable NumPy semantics. Expectations checked against NumPy 2.5.3 on CPython 3.14.4.
# Scope: NumPy scalars as Python numbers, their reprs, dtype-preserving arithmetic, and 0-d results.

import json
import math
import operator

import numpy as np
import pytest


def test_numeric_conversions():
    assert int(np.int64(3)) == 3
    assert type(int(np.int64(3))) is int
    assert float(np.float64(2.5)) == 2.5
    assert type(float(np.float64(2.5))) is float
    assert float(np.int64(3)) == 3.0
    assert int(np.float64(-2.7)) == -2
    assert int(np.float32(2.9)) == 2
    assert int(np.bool_(True)) == 1
    assert float(np.float16(0.5)) == 0.5
    assert complex(np.float64(1.5)) == complex(1.5, 0)
    assert complex(np.int64(2)) == complex(2, 0)
    assert complex(np.complex128(complex(1, 2))) == complex(1, 2)


def test_bool_conversion():
    assert bool(np.float64(0.0)) is False
    assert bool(np.int64(3)) is True
    assert bool(np.bool_(False)) is False
    assert (not np.int64(0)) is True


def test_integer_scalars_work_as_indices():
    assert list(range(np.int64(3))) == [0, 1, 2]
    assert [10, 20, 30][np.int64(1)] == 20
    assert [1, 2, 3, 4][np.int8(1) : np.int64(3)] == [2, 3]
    assert "abcdef"[np.int64(2)] == "c"
    assert [0] * np.int64(3) == [0, 0, 0]
    assert "ab" * np.int64(2) == "abab"


def test_operator_index_on_integer_scalars():
    value = operator.index(np.int32(7))
    assert value == 7
    assert type(value) is int
    assert hex(np.int64(255)) == "0xff"
    assert bin(np.uint8(5)) == "0b101"


def test_float_scalar_is_not_an_index():
    with pytest.raises(TypeError) as info:
        operator.index(np.float64(1.0))
    assert str(info.value) == "'numpy.float64' object cannot be interpreted as an integer"
    with pytest.raises(TypeError):
        [1, 2][np.float64(0)]


def test_round_without_ndigits_returns_python_int():
    assert round(np.float64(2.5)) == 2
    assert round(np.float64(3.5)) == 4
    assert type(round(np.float64(2.5))) is int
    assert type(round(np.float32(2.5))) is int
    assert type(round(np.int64(7))) is int


def test_round_with_ndigits_keeps_numpy_type():
    r = round(np.float64(2.567), 2)
    assert type(r) is np.float64
    assert r == np.float64(2.57)
    assert type(round(np.float32(2.567), 1)) is np.float32
    i = round(np.int64(1234), -2)
    assert type(i) is np.int64
    assert i == 1200


def test_math_module_accepts_scalars():
    assert math.floor(np.float64(2.7)) == 2
    assert type(math.floor(np.float64(2.7))) is int
    assert math.ceil(np.float64(2.1)) == 3
    assert math.trunc(np.float64(-2.5)) == -2
    assert math.sqrt(np.int64(16)) == 4.0
    assert math.isnan(np.float64(np.nan))


@pytest.mark.parametrize(
    "template, expected",
    [
        ("{:.2f}", "2.50"),
        ("{:e}", "2.500000e+00"),
        ("{:>6}", "   2.5"),
        ("{}", "2.5"),
    ],
)
def test_format_spec_float64(template, expected):
    assert template.format(np.float64(2.5)) == expected


@pytest.mark.parametrize(
    "template, expected",
    [
        ("{:03d}", "005"),
        ("{:+d}", "+5"),
        ("{:x}", "5"),
        ("{:>4}", "   5"),
        ("{}", "5"),
    ],
)
def test_format_spec_int64(template, expected):
    assert template.format(np.int64(5)) == expected


def test_format_spec_other_scalar_types():
    assert f"{np.float32(0.5):.3f}" == "0.500"
    assert f"{np.int8(-5):+d}" == "-5"
    assert f"{np.uint8(255):x}" == "ff"
    assert f"{np.float64(1234.5):,.1f}" == "1,234.5"


def test_float32_format_uses_python_float_value():
    # Formatting goes through float(), so the float32 rounding error shows, while str() does not.
    assert f"{np.float32(0.1)}" == "0.10000000149011612"
    assert f"{np.float32(0.1):.10f}" == "0.1000000015"
    assert str(np.float32(0.1)) == "0.1"


def test_percent_formatting():
    assert "%d" % np.int64(3) == "3"
    assert "%.1f" % np.float64(2.25) == "2.2"
    assert "%s" % np.float64(2.5) == "2.5"
    assert "%r" % np.float64(2.5) == "np.float64(2.5)"
    assert "%s|%s" % (np.int8(3), np.True_) == "3|True"


def test_hash_matches_equal_python_numbers_and_dict_keys_interchange():
    assert hash(np.int64(3)) == hash(3)
    assert hash(np.float64(2.5)) == hash(2.5)
    assert hash(np.float64(2.0)) == hash(2)
    assert hash(np.float32(0.5)) == hash(0.5)
    assert hash(np.bool_(True)) == hash(True)
    assert {np.int64(1): "a"}[1] == "a"
    assert {1: "a"}[np.int64(1)] == "a"
    assert {2.5: "x"}[np.float64(2.5)] == "x"


def test_comparisons_with_python_numbers():
    assert np.int64(3) == 3
    assert 3 == np.int64(3)
    assert np.int64(5) == 5.0
    assert np.int64(3) < 3.5
    assert np.float64(2.5) > 2
    assert 2 < np.int8(3)
    assert np.int64(2) != 3
    assert np.uint8(255) == 255
    assert np.uint64(2**64 - 1) > 2**63
    assert np.int8(-1) < np.uint64(0)


def test_nan_scalar_is_not_equal_to_itself():
    nan = np.float64(np.nan)
    assert not (nan == nan)
    assert nan != nan


def test_float32_equals_python_float_after_weak_cast():
    # The Python float is weak, so it is cast to float32 before comparing.
    assert np.float32(0.1) == 0.1
    assert not (np.float32(0.1) == np.float64(0.1))
    assert float(np.float32(0.1)) != 0.1


def test_builtin_sum_max_min_sorted_over_arrays():
    a = np.arange(4)
    total = sum(a)
    assert total == 6
    assert type(total) is np.int64
    assert sum(np.array([1, 2]), 10) == 13
    assert max(a) == 3
    assert type(max(a)) is np.int64
    assert min(np.array([3.5, 1.5])) == 1.5
    ordered = sorted(np.array([3, 1, 2]))
    assert ordered == [1, 2, 3]
    assert type(ordered[0]) is np.int64


def test_iteration_yields_numpy_scalars():
    items = list(np.array([1, 2]))
    assert items == [1, 2]
    assert type(items[0]) is np.int64
    assert type(list(np.array([1.5]))[0]) is np.float64


def test_isinstance_with_python_types():
    assert isinstance(np.float64(1.0), float)
    assert not isinstance(np.float32(1.0), float)
    assert not isinstance(np.int64(1), int)
    assert not isinstance(np.bool_(True), bool)
    assert isinstance(np.complex128(1j), complex)


def test_isinstance_with_numpy_abstract_types():
    assert isinstance(np.int64(1), np.integer)
    assert isinstance(np.int8(1), np.signedinteger)
    assert isinstance(np.uint8(1), np.unsignedinteger)
    assert isinstance(np.float64(1), np.floating)
    assert isinstance(np.float32(1), np.generic)
    assert not isinstance(np.float64(1), np.integer)


def test_json_serializes_float64_but_not_int64():
    assert json.dumps(np.float64(1.5)) == "1.5"
    assert json.dumps([np.float64(0.25)]) == "[0.25]"
    with pytest.raises(TypeError) as info:
        json.dumps(np.int64(1))
    assert str(info.value) == "Object of type int64 is not JSON serializable"


@pytest.mark.parametrize(
    "dtype_name, value, expected_repr, expected_str",
    [
        ("float64", 1.5, "np.float64(1.5)", "1.5"),
        ("float64", 1e-5, "np.float64(1e-05)", "1e-05"),
        ("float32", 0.1, "np.float32(0.1)", "0.1"),
        ("float16", 0.1, "np.float16(0.1)", "0.1"),
        ("int64", 3, "np.int64(3)", "3"),
        ("int16", 7, "np.int16(7)", "7"),
        ("uint8", 200, "np.uint8(200)", "200"),
        ("uint64", 18446744073709551615, "np.uint64(18446744073709551615)", "18446744073709551615"),
        ("bool", True, "np.True_", "True"),
        ("bool", False, "np.False_", "False"),
        ("complex128", 1j, "np.complex128(1j)", "1j"),
    ],
)
def test_scalar_repr_and_str(dtype_name, value, expected_repr, expected_str):
    scalar = np.dtype(dtype_name).type(value)
    assert repr(scalar) == expected_repr
    assert str(scalar) == expected_str


def test_float_repr_edge_cases():
    assert repr(np.float64(2.0)) == "np.float64(2.0)"
    assert str(np.float64(2.0)) == "2.0"
    assert repr(np.float64(1e20)) == "np.float64(1e+20)"
    assert str(np.float64(1e20)) == "1e+20"
    assert repr(np.float64(-0.0)) == "np.float64(-0.0)"
    assert repr(np.float64(np.nan)) == "np.float64(nan)"
    assert repr(np.float64(np.inf)) == "np.float64(inf)"
    assert str(np.float64(-np.inf)) == "-inf"


def test_negative_integer_scalar_repr():
    assert repr(np.int8(-5)) == "np.int8(-5)"
    assert str(np.int8(-5)) == "-5"
    assert repr(np.int16(-7)) == "np.int16(-7)"


def test_complex_scalar_repr_and_str():
    assert repr(np.complex128(complex(1, 2))) == "np.complex128(1+2j)"
    assert str(np.complex128(complex(1, 2))) == "(1+2j)"
    assert repr(np.complex64(complex(1, 2))) == "np.complex64(1+2j)"
    assert repr(np.complex128(complex(-1.5, 0))) == "np.complex128(-1.5+0j)"


def test_integer_scalar_overflow_wraps():
    with np.errstate(over="ignore"):
        wrapped = np.int8(127) + np.int8(1)
        assert type(wrapped) is np.int8
        assert wrapped == -128
        assert np.int8(127) + 1 == -128
        assert np.uint8(0) - np.uint8(1) == 255
        assert np.uint8(200) * np.uint8(2) == 144
        assert np.int16(32767) + 1 == -32768


def test_integer_scalar_overflow_raises_under_errstate():
    with np.errstate(over="raise"):
        with pytest.raises(FloatingPointError) as info:
            np.int8(127) + np.int8(1)
    assert str(info.value) == "overflow encountered in scalar add"


def test_mixed_integer_scalars_promote():
    widened = np.int8(5) + np.int16(1)
    assert type(widened) is np.int16
    assert widened == 6
    mixed_sign = np.uint8(3) + np.int8(1)
    assert type(mixed_sign) is np.int16
    assert mixed_sign == 4
    to_float = np.int64(1) + np.uint64(1)
    assert type(to_float) is np.float64
    assert to_float == 2.0


def test_integer_scalar_with_python_int_keeps_dtype():
    product = np.int32(7) * 3
    assert type(product) is np.int32
    assert product == 21
    assert type(np.bool_(True) + 1) is np.int64


def test_integer_scalar_with_python_float_gives_float64():
    assert type(np.int64(3) + 1.5) is np.float64
    assert np.int64(3) + 1.5 == 4.5
    assert type(np.int8(3) * 2.0) is np.float64
    assert type(np.int64(7) / 2) is np.float64
    assert np.int64(7) / 2 == 3.5


def test_scalar_floor_division_and_modulo_round_toward_negative_infinity():
    assert np.int64(7) // 2 == 3
    assert np.int64(-7) // 2 == -4
    assert np.int64(-7) % 3 == 2
    assert type(np.int64(-7) // 2) is np.int64
    floored = np.float64(-7.5) // 2
    assert type(floored) is np.float64
    assert floored == -4.0


def test_scalar_power_and_unary_operators_keep_dtype():
    power = np.int64(2) ** 3
    assert type(power) is np.int64
    assert power == 8
    negated = -np.int8(5)
    assert type(negated) is np.int8
    assert negated == -5
    absolute = abs(np.int8(-5))
    assert type(absolute) is np.int8
    assert absolute == 5
    half = np.float16(1) + np.float16(0.5)
    assert type(half) is np.float16
    assert half == 1.5


def test_bool_scalar_addition_is_logical_or():
    value = np.bool_(True) + np.bool_(True)
    assert value is np.True_


def test_float32_scalar_arithmetic_stays_float32():
    total = np.float32(1.5) + np.float32(2.25)
    assert type(total) is np.float32
    assert total == 3.75
    assert type(np.float32(0.1) * 3) is np.float32
    assert type(np.float32(1) + 1.0) is np.float32
    assert type(np.float32(1) + np.float64(1)) is np.float64
    assert np.float32(16777216) + np.float32(1) == 16777216
    assert np.float16(2048) + np.float16(1) == 2048


def test_divmod_on_scalars():
    q, r = divmod(np.int64(7), 2)
    assert (q, r) == (3, 1)
    assert type(q) is np.int64
    fq, fr = divmod(np.float64(7.5), 2)
    assert (fq, fr) == (3.0, 1.5)
    assert type(fr) is np.float64


def test_ufunc_on_zero_dimensional_array_returns_scalar():
    value = np.add(np.array(1), 2)
    assert type(value) is np.int64
    assert value == 3
    assert type(np.array(2.0) * 3) is np.float64
    assert type(np.sqrt(np.array(4.0))) is np.float64
    assert type(np.negative(np.array(2))) is np.int64
    assert type(np.array(1) + np.array(2)) is np.int64


def test_full_index_returns_scalar():
    value = np.array(2)[()]
    assert type(value) is np.int64
    assert value == 2
    assert type(np.array([[1, 2]])[0, 1]) is np.int64


def test_reductions_return_scalars():
    assert type(np.array([1, 2, 3]).sum()) is np.int64
    assert type(np.array([1.0, 2.0], np.float32).sum()) is np.float32
    assert type(np.array([[1, 2], [3, 4]]).max()) is np.int64
    assert type(np.array([1, 2]).argmax()) is np.int64
    assert type(np.array([1, 2, 3]).mean()) is np.float64
    assert np.array([True, False]).any() is np.True_


@pytest.mark.parametrize(
    "values, dtype_name, expected_type",
    [
        ([1, 2], "int64", "int64"),
        ([1, 2], "int8", "int8"),
        ([1, 2], "uint16", "uint16"),
        ([1.5], "float64", "float64"),
        ([1.5], "float32", "float32"),
        ([True], "bool", "bool"),
        ([1j], "complex128", "complex128"),
    ],
)
def test_integer_index_returns_numpy_scalar(values, dtype_name, expected_type):
    a = np.array(values, dtype=np.dtype(dtype_name))
    assert type(a[0]) is getattr(np, expected_type)


@pytest.mark.parametrize(
    "values, dtype_name, expected_type",
    [
        ([1], "int64", "int"),
        ([1], "int8", "int"),
        ([1], "uint64", "int"),
        ([1.5], "float64", "float"),
        ([1.5], "float32", "float"),
        ([True], "bool", "bool"),
        ([1j], "complex128", "complex"),
    ],
)
def test_item_returns_python_types(values, dtype_name, expected_type):
    item = np.array(values, dtype=np.dtype(dtype_name)).item()
    assert type(item).__name__ == expected_type


def test_item_with_index_and_tolist():
    a = np.array([[1, 2], [3, 4]])
    assert a.item(3) == 4
    assert a.item((1, 0)) == 3
    assert type(np.array(5).item()) is int
    assert np.int64(5).item() == 5
    assert type(np.float32(1.5).item()) is float
    assert type(np.array([1.5]).tolist()[0]) is float
    assert type(np.array([1], np.int8).tolist()[0]) is int


def test_uint64_above_int64_max_round_trips():
    big = np.uint64(2**64 - 1)
    assert int(big) == 2**64 - 1
    assert big == 2**64 - 1
    assert np.uint64(2**63) == 2**63
    assert np.uint64(2**63 + 7).item() == 9223372036854775815
    assert int(np.array([2**63], dtype=np.uint64)[0]) == 9223372036854775808
    assert np.array([2**63 + 1], dtype=np.uint64).tolist() == [9223372036854775809]


def test_float_methods_are_inherited():
    assert np.float64(2.0).is_integer()
    assert not np.float64(2.5).is_integer()
    assert np.float64(0.5).as_integer_ratio() == (1, 2)


def test_scalar_real_imag_and_dtype_attributes():
    assert np.float64(2.5).real == 2.5
    assert np.float64(2.5).imag == 0.0
    assert np.float64(1.5).dtype == np.dtype("float64")
    assert np.int8(1).itemsize == 1
    assert np.float64(1).ndim == 0
    assert np.float64(1).shape == ()


def test_complex_scalars_from_arrays():
    z = np.array([complex(1, 2), complex(3, -4)])
    assert type(z[0]) is np.complex128
    assert z[0] == complex(1, 2)
    assert z[1] == complex(3, -4)
    assert abs(z[1]) == 5.0
    assert type(abs(z[1])) is np.float64
    assert z[0].real == 1.0
    assert z[0].imag == 2.0
    assert complex(z[0]) == complex(1, 2)
    assert np.complex64(complex(1, 2)) == complex(1, 2)


def test_complex_scalar_arithmetic():
    doubled = np.complex128(complex(1, 2)) * 2
    assert type(doubled) is np.complex128
    assert doubled == complex(2, 4)
    shifted = np.complex64(complex(1, 2)) + 1
    assert type(shifted) is np.complex64
    assert shifted == complex(2, 2)
    assert np.complex128(complex(1, 2)).conjugate() == complex(1, -2)
