# Portable NumPy semantics. Expectations checked against NumPy 2.5.3 on CPython 3.14.4.
# Scope: dtype objects, the abstract type hierarchy, casting, and NEP 50 result-type rules.

import numpy as np
import pytest


@pytest.mark.parametrize(
    "spec, name, kind, itemsize",
    [
        ("i4", "int32", "i", 4),
        ("int8", "int8", "i", 1),
        ("<i8", "int64", "i", 8),
        ("u1", "uint8", "u", 1),
        ("uint16", "uint16", "u", 2),
        ("u8", "uint64", "u", 8),
        ("f2", "float16", "f", 2),
        ("float32", "float32", "f", 4),
        ("f8", "float64", "f", 8),
        ("double", "float64", "f", 8),
        ("c8", "complex64", "c", 8),
        ("complex128", "complex128", "c", 16),
        ("?", "bool", "b", 1),
        ("bool", "bool", "b", 1),
        ("O", "object", "O", 8),
        ("int", "int64", "i", 8),
        ("float", "float64", "f", 8),
        ("complex", "complex128", "c", 16),
    ],
)
def test_dtype_from_string(spec, name, kind, itemsize):
    d = np.dtype(spec)
    assert d.name == name
    assert d.kind == kind
    assert d.itemsize == itemsize


@pytest.mark.parametrize(
    "spec, char",
    [
        ("bool", "?"),
        ("int8", "b"),
        ("uint8", "B"),
        ("int16", "h"),
        ("uint16", "H"),
        ("int32", "i"),
        ("float16", "e"),
        ("float32", "f"),
        ("float64", "d"),
        ("complex64", "F"),
        ("complex128", "D"),
        ("object", "O"),
    ],
)
def test_dtype_char_codes(spec, char):
    assert np.dtype(spec).char == char
    assert np.dtype(char) == np.dtype(spec)


def test_dtype_from_python_types():
    assert np.dtype(int) == np.dtype("int64")
    assert np.dtype(float) == np.dtype("float64")
    assert np.dtype(complex) == np.dtype("complex128")
    assert np.dtype(bool) == np.dtype("bool")
    assert np.dtype(object) == np.dtype("O")
    assert np.dtype(str).kind == "U"


def test_dtype_from_numpy_scalar_types():
    assert np.dtype(np.float32) == np.dtype("f4")
    assert np.dtype(np.int16).name == "int16"
    assert np.dtype(np.float32).type is np.float32
    assert np.dtype("f8").type is np.float64


def test_unicode_dtype_width():
    d = np.dtype("U5")
    assert d.kind == "U"
    assert d.itemsize == 20
    assert d == np.dtype("<U5")
    assert d != np.dtype("U6")
    assert str(d) == "<U5"
    assert repr(d) == "dtype('<U5')"


def test_dtype_equality_with_aliases_and_types():
    assert np.dtype("i4") == np.int32
    assert np.dtype("f8") == "float64"
    assert np.dtype("int32") == np.dtype("i4")
    assert np.dtype("float32") != np.dtype("float64")


def test_dtype_str_and_repr():
    d = np.dtype("float32")
    assert str(d) == "float32"
    assert repr(d) == "dtype('float32')"
    assert repr(np.dtype("int64")) == "dtype('int64')"
    assert repr(np.dtype("bool")) == "dtype('bool')"


@pytest.mark.parametrize("spec", ["bogus", "i3"])
def test_unknown_dtype_string_raises_type_error(spec):
    with pytest.raises(TypeError) as info:
        np.dtype(spec)
    assert str(info.value) == f"data type '{spec}' not understood"


@pytest.mark.parametrize(
    "dtype_name, abstract_name, expected",
    [
        ("int8", "integer", True),
        ("int8", "signedinteger", True),
        ("uint8", "signedinteger", False),
        ("uint8", "unsignedinteger", True),
        ("int64", "unsignedinteger", False),
        ("float16", "floating", True),
        ("float64", "integer", False),
        ("complex64", "complexfloating", True),
        ("complex64", "floating", False),
        ("float32", "inexact", True),
        ("complex128", "inexact", True),
        ("float32", "number", True),
        ("uint32", "number", True),
        ("bool", "number", False),
        ("bool", "integer", False),
        ("bool", "generic", True),
        ("object", "generic", True),
        ("int64", "generic", True),
    ],
)
def test_issubdtype_abstract_hierarchy(dtype_name, abstract_name, expected):
    abstract = getattr(np, abstract_name)
    assert np.issubdtype(np.dtype(dtype_name), abstract) is expected


def test_issubdtype_accepts_scalar_types_and_strings():
    assert np.issubdtype(np.int8, np.integer)
    assert np.issubdtype("i4", np.integer)
    assert np.issubdtype(np.array([1.5]).dtype, np.floating)
    assert np.issubdtype(np.int64, np.int64)
    assert not np.issubdtype(np.int32, np.int64)


def test_issubdtype_python_types_map_to_default_dtypes():
    assert np.issubdtype(np.float64, float)
    assert np.issubdtype(np.int64, int)


def test_abstract_types_nest():
    assert np.issubdtype(np.integer, np.number)
    assert np.issubdtype(np.floating, np.inexact)
    assert np.issubdtype(np.signedinteger, np.integer)
    assert np.issubdtype(np.number, np.generic)


@pytest.mark.parametrize(
    "dtype_name",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ],
)
def test_astype_round_trips_small_integers(dtype_name):
    dtype = np.dtype(dtype_name)
    a = np.array([0, 1, 2, 3]).astype(dtype)
    assert a.dtype == dtype
    assert a.tolist() == [0, 1, 2, 3]
    # .real keeps complex inputs from emitting ComplexWarning on the cast back.
    back = a.real.astype(np.int64)
    assert back.dtype == np.dtype("int64")
    assert back.tolist() == [0, 1, 2, 3]


def test_astype_float_to_int_truncates_toward_zero():
    a = np.array([1.9, -1.9, 2.5])
    assert a.astype(np.int64).tolist() == [1, -1, 2]
    assert a.astype(np.int8).tolist() == [1, -1, 2]


def test_astype_int_to_narrower_int_wraps():
    assert np.array([300, -1, 256, 255]).astype(np.uint8).tolist() == [44, 255, 0, 255]
    assert np.array([200, 128, -129]).astype(np.int8).tolist() == [-56, -128, 127]
    assert np.array([2**32 + 5]).astype(np.uint32).tolist() == [5]
    assert np.array([-1]).astype(np.uint64).tolist() == [18446744073709551615]


def test_astype_to_and_from_bool():
    assert np.array([1, 0, 2]).astype(bool).tolist() == [True, False, True]
    assert np.array([0.0, 0.5]).astype(bool).tolist() == [False, True]
    assert np.array([True, False]).astype(float).tolist() == [1.0, 0.0]


def test_astype_float16_rounds_to_half_precision():
    a = np.array([0.1, 1 / 3, 65504.0, 2049.0, 0.5]).astype(np.float16)
    assert a.dtype == np.dtype("float16")
    assert a.tolist() == [0.0999755859375, 0.333251953125, 65504.0, 2048.0, 0.5]


def test_astype_float32_rounds_to_single_precision():
    a = np.array([0.1]).astype(np.float32)
    assert a.tolist() == [0.10000000149011612]
    assert a.astype(np.float64).tolist() == [0.10000000149011612]


def test_astype_returns_copy_unless_copy_false():
    a = np.array([1, 2])
    assert a.astype(np.int64) is not a
    assert a.astype(np.int64, copy=False) is a


@pytest.mark.parametrize(
    "left, right, expected",
    [
        ("int8", "uint8", "int16"),
        ("int16", "uint16", "int32"),
        ("uint32", "int32", "int64"),
        ("uint8", "int64", "int64"),
        ("int8", "int16", "int16"),
        ("uint8", "uint16", "uint16"),
        ("int64", "uint64", "float64"),
        ("uint64", "int8", "float64"),
        ("bool", "int8", "int8"),
        ("bool", "float16", "float16"),
        ("int8", "float16", "float16"),
        ("uint8", "float16", "float16"),
        ("int16", "float16", "float32"),
        ("int8", "float32", "float32"),
        ("int16", "float32", "float32"),
        ("int32", "float32", "float64"),
        ("uint64", "float32", "float64"),
        ("float16", "float32", "float32"),
        ("float32", "complex64", "complex64"),
        ("int16", "complex64", "complex64"),
        ("int32", "complex64", "complex128"),
        ("int64", "complex64", "complex128"),
        ("float64", "complex64", "complex128"),
    ],
)
def test_strong_array_promotion(left, right, expected):
    a = np.array([1], dtype=np.dtype(left))
    b = np.array([1], dtype=np.dtype(right))
    assert (a + b).dtype == np.dtype(expected)
    assert (b + a).dtype == np.dtype(expected)
    assert np.result_type(a, b) == np.dtype(expected)
    assert np.promote_types(left, right) == np.dtype(expected)


def test_promote_types_for_strings_and_objects():
    assert np.promote_types("U3", "U5") == np.dtype("U5")
    assert np.promote_types("int64", "O") == np.dtype("O")


@pytest.mark.parametrize(
    "dtype_name, scalar_kind, expected",
    [
        ("int8", "int", "int8"),
        ("uint8", "int", "uint8"),
        ("int8", "bool", "int8"),
        ("int8", "float", "float64"),
        ("int8", "complex", "complex128"),
        ("float16", "float", "float16"),
        ("float32", "float", "float32"),
        ("float32", "complex", "complex64"),
        ("complex64", "float", "complex64"),
        ("bool", "int", "int64"),
        ("bool", "float", "float64"),
    ],
)
def test_python_scalars_are_weak(dtype_name, scalar_kind, expected):
    scalar = {"int": 1, "bool": True, "float": 1.5, "complex": 1j}[scalar_kind]
    a = np.array([1], dtype=np.dtype(dtype_name))
    assert (a + scalar).dtype == np.dtype(expected)
    assert (scalar + a).dtype == np.dtype(expected)


def test_numpy_scalars_are_strong():
    assert (np.array([1], np.float32) + np.float64(1.5)).dtype == np.dtype("float64")
    assert (np.array([1], np.int8) + np.int64(1)).dtype == np.dtype("int64")


def test_weak_int_wraps_within_array_dtype():
    assert (np.array([1], np.uint8) - 2).tolist() == [255]
    assert (np.array([1], np.uint8) + 255).tolist() == [0]


@pytest.mark.parametrize(
    "dtype_name, value",
    [
        ("int8", 300),
        ("int8", 128),
        ("uint8", 256),
        ("int16", 40000),
    ],
)
def test_out_of_range_python_int_with_array_raises_overflow(dtype_name, value):
    a = np.array([1], dtype=np.dtype(dtype_name))
    with pytest.raises(OverflowError) as info:
        a + value
    assert str(info.value) == f"Python integer {value} out of bounds for {dtype_name}"


def test_negative_python_int_out_of_range_raises_overflow():
    with pytest.raises(OverflowError) as info:
        np.array([1], np.uint8) + (-1)
    assert str(info.value) == "Python integer -1 out of bounds for uint8"
    with pytest.raises(OverflowError) as info:
        np.array([1], np.int8) - (-129)
    assert str(info.value) == "Python integer -129 out of bounds for int8"


def test_out_of_range_python_int_with_scalar_raises_overflow():
    with pytest.raises(OverflowError) as info:
        np.int8(1) + 300
    assert str(info.value) == "Python integer 300 out of bounds for int8"


@pytest.mark.parametrize(
    "dtype_name, expected",
    [
        ("bool", "float16"),
        ("int8", "float16"),
        ("uint8", "float16"),
        ("int16", "float32"),
        ("uint16", "float32"),
        ("int32", "float64"),
        ("uint32", "float64"),
        ("int64", "float64"),
        ("float16", "float16"),
        ("float32", "float32"),
    ],
)
def test_float_ufuncs_on_integers_pick_smallest_safe_float(dtype_name, expected):
    a = np.array([4], dtype=np.dtype(dtype_name))
    assert np.sqrt(a).dtype == np.dtype(expected)
    assert np.exp(a).dtype == np.dtype(expected)


@pytest.mark.parametrize(
    "dtype_name, expected",
    [
        ("int8", "float64"),
        ("int16", "float64"),
        ("int64", "float64"),
        ("float16", "float16"),
        ("float32", "float32"),
    ],
)
def test_true_division_dtype(dtype_name, expected):
    a = np.array([3, 4], dtype=np.dtype(dtype_name))
    assert (a / 2).dtype == np.dtype(expected)
    assert (a / a).dtype == np.dtype(expected)
    assert (a / 2).tolist() == [1.5, 2.0]


def test_floor_divide_remainder_and_comparison_dtypes():
    a = np.array([7], np.int8)
    assert (a // 2).dtype == np.dtype("int8")
    assert (a % 2).dtype == np.dtype("int8")
    assert (a < 2).dtype == np.dtype("bool")


@pytest.mark.parametrize(
    "dtype_name, sum_dtype, mean_dtype",
    [
        ("bool", "int64", "float64"),
        ("int8", "int64", "float64"),
        ("int32", "int64", "float64"),
        ("int64", "int64", "float64"),
        ("uint8", "uint64", "float64"),
        ("uint32", "uint64", "float64"),
        ("float16", "float16", "float16"),
        ("float32", "float32", "float32"),
        ("float64", "float64", "float64"),
        ("complex64", "complex64", "complex64"),
    ],
)
def test_reduction_dtypes(dtype_name, sum_dtype, mean_dtype):
    a = np.array([1, 0, 1], dtype=np.dtype(dtype_name))
    assert a.sum().dtype == np.dtype(sum_dtype)
    assert a.prod().dtype == np.dtype(sum_dtype)
    assert np.cumsum(a).dtype == np.dtype(sum_dtype)
    assert a.mean().dtype == np.dtype(mean_dtype)
    assert a.max().dtype == np.dtype(dtype_name)


def test_small_int_sum_does_not_wrap():
    assert np.array([100, 100], np.int8).sum() == 200
    assert np.array([True, True, False]).sum() == 2


def test_sum_dtype_argument_overrides_accumulator():
    assert np.array([1, 2], np.int8).sum(dtype=np.int8).dtype == np.dtype("int8")


def test_in_place_add_of_float_to_int_array_raises():
    a = np.array([1, 2, 3])
    with pytest.raises(TypeError) as info:
        a += 1.5
    assert str(info.value) == (
        "Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'"
    )
    assert a.tolist() == [1, 2, 3]


def test_in_place_add_of_complex_to_float_array_raises():
    a = np.array([1.0, 2.0])
    with pytest.raises(TypeError) as info:
        a += 1j
    assert str(info.value) == (
        "Cannot cast ufunc 'add' output from dtype('complex128') to dtype('float64') with casting rule 'same_kind'"
    )


def test_out_argument_uses_same_kind_casting():
    out = np.zeros(3, dtype=np.int64)
    with pytest.raises(TypeError) as info:
        np.multiply(np.array([1, 2, 3]), 2.5, out=out)
    assert str(info.value) == (
        "Cannot cast ufunc 'multiply' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'"
    )


def test_in_place_same_kind_downcast_is_allowed():
    a = np.array([1, 2, 3], np.int8)
    a += np.array([1, 2, 3])
    assert a.dtype == np.dtype("int8")
    assert a.tolist() == [2, 4, 6]
    f = np.array([1.0, 2.0], np.float32)
    f += 1.5
    assert f.dtype == np.dtype("float32")
    assert f.tolist() == [2.5, 3.5]


def test_assignment_truncates_float_into_int_array():
    a = np.array([1, 2, 3])
    a[0] = 2.7
    a[1] = -2.7
    assert a.tolist() == [2, -2, 3]


def test_assignment_converts_into_float_and_bool_arrays():
    f = np.array([1.5, 2.5])
    f[0] = 7
    assert f.tolist() == [7.0, 2.5]
    b = np.array([True, False])
    b[1] = 5
    assert b.tolist() == [True, True]


def test_assignment_of_out_of_range_int_raises_overflow():
    a = np.array([1, 2], np.int8)
    with pytest.raises(OverflowError) as info:
        a[0] = 300
    assert str(info.value) == "Python integer 300 out of bounds for int8"
    wide = np.array([1, 2])
    with pytest.raises(OverflowError):
        wide[0] = 2**70


def test_result_type_with_python_scalars():
    assert np.result_type(np.int8, 1) == np.dtype("int8")
    assert np.result_type(np.array([1], np.int8), 1.0) == np.dtype("float64")
    assert np.result_type(np.float32, 1j) == np.dtype("complex64")
    assert np.result_type(3) == np.dtype("int64")
    assert np.result_type(3.0) == np.dtype("float64")


@pytest.mark.parametrize(
    "source, target, casting, expected",
    [
        ("int8", "int16", "safe", True),
        ("int16", "int8", "safe", False),
        ("uint8", "int8", "safe", False),
        ("uint8", "int16", "safe", True),
        ("int64", "float64", "safe", True),
        ("int32", "float64", "safe", True),
        ("int64", "complex128", "safe", True),
        ("float64", "int64", "safe", False),
        ("float64", "float32", "safe", False),
        ("float64", "float32", "same_kind", True),
        ("float64", "int64", "same_kind", False),
        ("float64", "int64", "unsafe", True),
    ],
)
def test_can_cast(source, target, casting, expected):
    assert np.can_cast(np.dtype(source), np.dtype(target), casting=casting) is expected
