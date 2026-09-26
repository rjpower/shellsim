# Portable NumPy semantics. Expectations checked against NumPy 2.5.3 on CPython 3.14.4.
# Scope: fixed-width unicode (Str) arrays and dtype=object arrays holding Python objects.

import operator

import numpy as np
import pytest


def test_string_array_dtype_uses_longest_element():
    a = np.array(["a", "bc"])
    assert a.dtype == np.dtype("<U2")
    assert a.itemsize == 8
    assert a.shape == (2,)


def test_string_dtype_counts_characters_not_bytes():
    a = np.array(["héllo", "ω"])
    assert a.dtype == np.dtype("<U5")
    assert a.tolist() == ["héllo", "ω"]


def test_two_dimensional_string_array():
    a = np.array([["a", "bb"], ["ccc", "d"]])
    assert a.dtype == np.dtype("<U3")
    assert a.shape == (2, 2)
    assert a.T.tolist() == [["a", "ccc"], ["bb", "d"]]


def test_string_element_access_and_tolist_return_str():
    a = np.array(["ab", "c"])
    first = a[0]
    assert isinstance(first, str)
    assert first == "ab"
    assert len(first) == 2
    assert first.upper() == "AB"
    assert first + "z" == "abz"
    items = np.array(["ab", "c"]).tolist()
    assert items == ["ab", "c"]
    assert type(items[0]) is str


def test_explicit_string_width():
    assert np.array(["a", "bc"], dtype="U5").dtype == np.dtype("<U5")
    assert np.array(["abcdef"], dtype="U3").tolist() == ["abc"]
    assert np.zeros(2, dtype="U3").tolist() == ["", ""]


def test_assignment_truncates_to_array_width():
    a = np.array(["ab", "c"])
    a[1] = "xyz"
    assert a.tolist() == ["ab", "xy"]


def test_empty_string_arrays_have_width_one():
    empty = np.array([], dtype=str)
    assert empty.dtype == np.dtype("<U1")
    assert empty.shape == (0,)
    assert empty.tolist() == []
    blank = np.array([""])
    assert blank.dtype == np.dtype("<U1")
    assert blank.tolist() == [""]


def test_full_with_string_fill():
    f = np.full(2, "hi")
    assert f.dtype == np.dtype("<U2")
    assert f.tolist() == ["hi", "hi"]


@pytest.mark.parametrize(
    "op_name, other, expected",
    [
        ("eq", "a", [True, False]),
        ("ne", "a", [False, True]),
        ("lt", "b", [True, False]),
        ("ge", "a", [True, True]),
    ],
)
def test_string_comparison_with_scalar(op_name, other, expected):
    result = getattr(operator, op_name)(np.array(["a", "bc"]), other)
    assert result.dtype == np.dtype("bool")
    assert result.tolist() == expected


def test_string_comparison_between_arrays_is_lexicographic():
    assert (np.array(["a", "bb"]) == np.array(["a", "b"])).tolist() == [True, False]
    assert (np.array(["b", "a"]) < np.array(["a", "b"])).tolist() == [False, True]
    assert (np.array(["abc"]) > np.array(["abd"])).tolist() == [False]
    assert (np.array(["a"]) < np.array(["aa"])).tolist() == [True]


def test_string_equality_with_number_is_false():
    assert (np.array(["a", "b"]) == 1).tolist() == [False, False]


def test_string_concatenation():
    a = np.array(["ab", "c"])
    joined = a + np.array(["x", "y"])
    assert joined.dtype == np.dtype("<U3")
    assert joined.tolist() == ["abx", "cy"]
    assert (a + "z").tolist() == ["abz", "cz"]
    assert ("z" + a).tolist() == ["zab", "zc"]
    assert (a + "zz").dtype == np.dtype("<U4")


def test_string_repeat_with_strings_multiply():
    repeated = np.strings.multiply(np.array(["ab", "c"]), 2)
    assert repeated.dtype == np.dtype("<U4")
    assert repeated.tolist() == ["abab", "cc"]


def test_string_arithmetic_without_a_loop_raises_type_error():
    a = np.array(["a", "b"])
    with pytest.raises(TypeError) as info:
        a - a
    assert str(info.value) == (
        "ufunc 'subtract' did not contain a loop with signature matching types (dtype('<U1'), dtype('<U1')) -> None"
    )
    with pytest.raises(TypeError) as info:
        np.array(["a"]) + 1
    assert str(info.value) == (
        "ufunc 'add' did not contain a loop with signature matching types (dtype('<U1'), dtype('int64')) -> None"
    )


def test_sort_strings():
    assert np.sort(np.array(["pear", "apple", "fig"])).tolist() == ["apple", "fig", "pear"]
    assert np.sort(np.array(["b", "B", "a", "A"])).tolist() == ["A", "B", "a", "b"]
    assert np.sort(np.array(["bb", "a"])).dtype == np.dtype("<U2")
    assert np.argsort(np.array(["c", "a", "b"])).tolist() == [1, 2, 0]


def test_unique_strings():
    values = np.array(["b", "a", "c", "a"])
    assert np.unique(values).tolist() == ["a", "b", "c"]
    assert np.unique(values).dtype == np.dtype("<U1")
    values, index, inverse, counts = np.unique(
        np.array(["b", "a", "b"]), return_index=True, return_inverse=True, return_counts=True
    )
    assert values.tolist() == ["a", "b"]
    assert index.tolist() == [1, 0]
    assert inverse.tolist() == [1, 0, 1]
    assert counts.tolist() == [1, 2]


def test_searchsorted_and_where_on_strings():
    assert np.searchsorted(np.array(["a", "c", "e"]), "d") == 2
    assert np.where(np.array(["a", "b", "a"]) == "a")[0].tolist() == [0, 2]


def test_builtin_max_and_min_over_string_array():
    a = np.array(["b", "a", "c"])
    assert max(a) == "c"
    assert min(a) == "a"


def test_string_indexing_and_slicing():
    a = np.array(["x", "y", "z"])
    assert a[::-1].tolist() == ["z", "y", "x"]
    assert a[np.array([True, False, True])].tolist() == ["x", "z"]
    assert a[[2, 0]].tolist() == ["z", "x"]


@pytest.mark.parametrize(
    "values, dtype_name, expected",
    [
        ([1, 22, 333], "<U21", ["1", "22", "333"]),
        ([1.5, 2.25, 0.1], "<U32", ["1.5", "2.25", "0.1"]),
        ([True, False], "<U5", ["True", "False"]),
    ],
)
def test_astype_str_formats_with_python_str(values, dtype_name, expected):
    converted = np.array(values).astype(str)
    assert converted.dtype == np.dtype(dtype_name)
    assert converted.tolist() == expected


def test_astype_str_edge_cases():
    assert np.array([-3]).astype(str).tolist() == ["-3"]
    assert np.array([2.0]).astype(str).tolist() == ["2.0"]
    assert np.array([123]).astype("U2").tolist() == ["12"]
    assert np.array([1.5]).astype("U3").tolist() == ["1.5"]


def test_astype_numeric_parses_strings():
    ints = np.array(["1", "22", "-3"]).astype(int)
    assert ints.dtype == np.dtype("int64")
    assert ints.tolist() == [1, 22, -3]
    floats = np.array(["1.5", "2", "1e3"]).astype(float)
    assert floats.dtype == np.dtype("float64")
    assert floats.tolist() == [1.5, 2.0, 1000.0]
    assert np.array(["10", "20"]).astype(np.int8).tolist() == [10, 20]
    assert np.array(["1.5"]).astype(np.float32).dtype == np.dtype("float32")
    with pytest.raises(ValueError):
        np.array(["abc"]).astype(int)


def test_string_array_to_object_keeps_python_strings():
    converted = np.array(["ab", "c"]).astype(object)
    assert converted.dtype == np.dtype("O")
    assert converted.tolist() == ["ab", "c"]
    assert type(converted[0]) is str


def test_object_array_holds_mixed_python_objects():
    a = np.array([1, "a", None, [1, 2]], dtype=object)
    assert a.dtype == np.dtype("object")
    assert a.shape == (4,)
    assert a.tolist() == [1, "a", None, [1, 2]]
    assert type(a[0]) is int
    assert a[2] is None
    assert type(a[3]) is list


def test_object_inference_without_dtype():
    assert np.array([None, None]).dtype == np.dtype("O")
    assert np.array([None]).tolist() == [None]
    big = np.array([2**100])
    assert big.dtype == np.dtype("O")
    assert big[0] == 2**100
    assert type(big[0]) is int


def test_object_array_nests_equal_sequences_and_keeps_ragged_ones():
    assert np.array([[1, 2], [3, 4]], dtype=object).shape == (2, 2)
    a = np.array([[1, 2], [3]], dtype=object)
    assert a.shape == (2,)
    assert a[1] == [3]


def test_object_indexing_returns_the_stored_object():
    items = [1, 2]
    a = np.array([items, "y", 3], dtype=object)
    assert a.shape == (3,)
    assert a[0] is items


def test_object_assignment_stores_references():
    items = [1]
    a = np.empty(1, dtype=object)
    a[0] = items
    items.append(2)
    assert a[0] == [1, 2]
    assert a[0] is items
    assert a.copy()[0] is items


def test_object_assignment_accepts_any_python_value():
    a = np.array([1, 2, 3], dtype=object)
    a[1] = "x"
    assert a.dtype == np.dtype("O")
    assert a.tolist() == [1, "x", 3]


def test_object_constructors():
    zeros = np.zeros(2, dtype=object)
    assert zeros.tolist() == [0, 0]
    assert type(zeros[0]) is int
    assert np.empty(2, dtype=object).tolist() == [None, None]
    assert np.ones(2, dtype=object).tolist() == [1, 1]
    assert np.full(2, "x", dtype=object).tolist() == ["x", "x"]
    empty = np.array([], dtype=object)
    assert empty.shape == (0,)
    assert empty.tolist() == []


def test_object_arithmetic_uses_python_operators():
    doubled = np.array([1, "a"], dtype=object) * 2
    assert doubled.dtype == np.dtype("O")
    assert doubled.tolist() == [2, "aa"]
    assert (np.array([1, 2], dtype=object) + np.array([10, 20], dtype=object)).tolist() == [11, 22]
    assert (np.array(["a", "b"], dtype=object) + "c").tolist() == ["ac", "bc"]


def test_object_arithmetic_on_tuples_concatenates():
    a = np.empty(2, dtype=object)
    a[0] = (1, 2)
    a[1] = (3,)
    assert (a + a).tolist() == [(1, 2, 1, 2), (3, 3)]
    assert (a * 2).tolist() == [(1, 2, 1, 2), (3, 3)]


def test_object_arithmetic_keeps_python_result_types():
    shifted = np.array([1, 2], dtype=object) + 1.5
    assert shifted.tolist() == [2.5, 3.5]
    assert type(shifted[0]) is float
    big = np.array([2**100, 3], dtype=object) * 2
    assert big[0] == 2**101
    assert type(big[0]) is int


def test_object_power_division_modulo_and_unary_operators():
    assert (np.array([2, 3], dtype=object) ** 2).tolist() == [4, 9]
    assert (np.array([7, 8], dtype=object) // 2).tolist() == [3, 4]
    assert (np.array([7, 8], dtype=object) % 3).tolist() == [1, 2]
    assert (-np.array([1, -2], dtype=object)).tolist() == [-1, 2]
    assert abs(np.array([-1, 2], dtype=object)).tolist() == [1, 2]


def test_object_arithmetic_propagates_python_type_error():
    with pytest.raises(TypeError) as info:
        np.array([1, "a"], dtype=object) + 1
    assert str(info.value) == 'can only concatenate str (not "int") to str'
    with pytest.raises(TypeError) as info:
        np.array([1, None], dtype=object) + 1
    assert str(info.value) == "unsupported operand type(s) for +: 'NoneType' and 'int'"


def test_object_equality_returns_bool_array():
    result = np.array([1, "a", None], dtype=object) == 1
    assert result.dtype == np.dtype("bool")
    assert result.tolist() == [True, False, False]
    pairwise = np.array([1, "a", None], dtype=object) == np.array([1, "b", None], dtype=object)
    assert pairwise.dtype == np.dtype("bool")
    assert pairwise.tolist() == [True, False, True]


def test_object_ordering_returns_bool_array():
    result = np.array([1, 2], dtype=object) < np.array([2, 2], dtype=object)
    assert result.dtype == np.dtype("bool")
    assert result.tolist() == [True, False]


def test_object_sum_folds_with_python_add():
    total = np.array([1, 2, 3], dtype=object).sum()
    assert total == 6
    assert type(total) is int
    assert np.array([2**70, 1], dtype=object).sum() == 2**70 + 1
    assert np.array(["a", "b", "c"], dtype=object).sum() == "abc"


def test_object_sum_of_mixed_types_raises():
    with pytest.raises(TypeError) as info:
        np.array([1, "a"], dtype=object).sum()
    assert str(info.value) == "unsupported operand type(s) for +: 'int' and 'str'"


def test_object_prod_and_cumsum():
    assert np.array([1, 2, 3], dtype=object).prod() == 6
    running = np.cumsum(np.array([1, 2, 3], dtype=object))
    assert running.dtype == np.dtype("O")
    assert running.tolist() == [1, 3, 6]


def test_object_min_max_and_sort_use_python_ordering():
    a = np.array([3, 1, 2], dtype=object)
    assert a.max() == 3
    assert type(a.max()) is int
    assert np.array(["b", "a"], dtype=object).min() == "a"
    assert np.sort(a).tolist() == [1, 2, 3]
    assert np.sort(np.array(["b", "a", "c"], dtype=object)).tolist() == ["a", "b", "c"]


def test_object_astype_numeric_and_str():
    floats = np.array([1, 2], dtype=object).astype(float)
    assert floats.dtype == np.dtype("float64")
    assert floats.tolist() == [1.0, 2.0]
    assert np.array(["1", "2"], dtype=object).astype(int).tolist() == [1, 2]
    text = np.array([1, "ab"], dtype=object).astype(str)
    assert text.dtype == np.dtype("<U2")
    assert text.tolist() == ["1", "ab"]


def test_numeric_astype_object_boxes_python_numbers():
    ints = np.array([1, 2]).astype(object)
    assert ints.tolist() == [1, 2]
    assert type(ints[0]) is int
    assert type(np.array([1.5]).astype(object)[0]) is float


def test_object_array_indexing_forms():
    a = np.array([1, "a", None], dtype=object)
    assert a[::-1].tolist() == [None, "a", 1]
    assert a[np.array([True, False, True])].tolist() == [1, None]
    grid = np.array([[1, "a"], [None, 2.5]], dtype=object)
    assert grid.T.tolist() == [[1, None], ["a", 2.5]]


def test_object_itemsize_and_nbytes():
    a = np.array([1, None], dtype=object)
    assert a.itemsize == 8
    assert a.nbytes == 16
