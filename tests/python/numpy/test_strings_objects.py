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


def test_strings_multiply_broadcasts_an_array_of_repeat_counts():
    repeated = np.strings.multiply(np.array(["ab", "cd", "ef"]), np.array([2, 0, 3]))
    # NumPy sizes this from the actual longest result, the same as constructing an array from
    # the computed text; an all-empty result still floors at width 1.
    assert repeated.dtype == np.dtype("<U6")
    assert repeated.tolist() == ["abab", "", "efefef"]


def test_strings_case_functions_keep_the_input_width():
    a = np.array(["Hello World", "foo BAR"])
    assert a.dtype == np.dtype("<U11")
    for result, expected in (
        (np.strings.capitalize(a), ["Hello world", "Foo bar"]),
        (np.strings.lower(a), ["hello world", "foo bar"]),
        (np.strings.upper(a), ["HELLO WORLD", "FOO BAR"]),
        (np.strings.swapcase(a), ["hELLO wORLD", "FOO bar"]),
        (np.strings.title(a), ["Hello World", "Foo Bar"]),
    ):
        # NumPy's fixed-width buffer keeps exactly the input's width for every case function,
        # even though none of these particular results is as long as "Hello World".
        assert result.dtype == np.dtype("<U11")
        assert result.tolist() == expected


def test_strings_upper_truncates_when_case_folding_grows_past_the_input_width():
    # German sharp s upper-cases to "SS", which is one character longer than "straße"; NumPy's
    # `upper` does not grow the array to fit, it truncates to the original width instead.
    a = np.array(["straße"])
    assert a.dtype == np.dtype("<U6")
    result = np.strings.upper(a)
    assert result.dtype == np.dtype("<U6")
    assert result.tolist() == ["STRASS"]


def test_strings_padding_functions():
    a = np.array(["ab", "c"])
    assert np.strings.center(np.array(["ab", "cd"]), 6, "*").tolist() == ["**ab**", "**cd**"]
    # An odd padding unit goes on the left only when the requested width is odd.
    assert np.strings.center(np.array(["ab", "a", "abc"]), 5, "*").tolist() == ["**ab*", "**a**", "*abc*"]
    assert np.strings.center(np.array(["a", "ab"]), 4, "*").tolist() == ["*a**", "*ab*"]
    assert np.strings.ljust(a, 4, "-").tolist() == ["ab--", "c---"]
    assert np.strings.rjust(a, 4, "-").tolist() == ["--ab", "---c"]
    assert np.strings.zfill(np.array(["7", "-3", "42"]), 4).tolist() == ["0007", "-003", "0042"]


def test_strings_padding_functions_grow_to_the_wider_of_input_and_requested_width():
    a = np.array(["Hello World"])  # <U11
    # A requested width narrower than the input leaves the array's width unchanged.
    small = np.strings.ljust(a, 3)
    assert small.dtype == np.dtype("<U11")
    assert small.tolist() == ["Hello World"]
    big = np.strings.rjust(a, 15, "-")
    assert big.dtype == np.dtype("<U15")
    assert big.tolist() == ["----Hello World"]
    narrow = np.strings.zfill(np.array(["7", "-3", "42"]), 1)
    assert narrow.dtype == np.dtype("<U2")
    assert narrow.tolist() == ["7", "-3", "42"]


def test_strings_padding_functions_broadcast_an_array_of_widths():
    result = np.strings.center(np.array(["ab", "cd"]), np.array([2, 8]), "*")
    # The dtype fits the largest requested width even though only one row needs it.
    assert result.dtype == np.dtype("<U8")
    assert result.tolist() == ["ab", "***cd***"]


def test_strings_trimming_functions_keep_the_input_width():
    a = np.array(["  hi  ", "yo   "])
    assert a.dtype == np.dtype("<U6")
    for result, expected in (
        (np.strings.strip(a), ["hi", "yo"]),
        (np.strings.lstrip(a), ["hi  ", "yo   "]),
        (np.strings.rstrip(a), ["  hi", "yo"]),
    ):
        assert result.dtype == np.dtype("<U6")
        assert result.tolist() == expected
    assert np.strings.strip(np.array(["xxhixx"]), "x").tolist() == ["hi"]


def test_strings_strip_broadcasts_an_array_of_character_sets():
    result = np.strings.strip(np.array(["xxhixx", "yyhoyy"]), np.array(["x", "y"]))
    assert result.dtype == np.dtype("<U6")
    assert result.tolist() == ["hi", "ho"]


def test_strings_search_and_replace_functions():
    a = np.array(["Hello World", "foo bar"])
    assert np.strings.count(a, "o").tolist() == [2, 2]
    assert np.strings.find(a, "o").tolist() == [4, 1]
    assert np.strings.rfind(a, "o").tolist() == [7, 2]
    assert np.strings.index(a, "o").tolist() == [4, 1]
    assert np.strings.rindex(a, "o").tolist() == [7, 2]
    assert np.strings.find(a, "z").tolist() == [-1, -1]
    with pytest.raises(ValueError):
        np.strings.index(a, "z")
    assert np.strings.replace(a, "o", "0").tolist() == ["Hell0 W0rld", "f00 bar"]
    assert np.strings.replace(np.array(["aaaa"]), "a", "bb").tolist() == ["bbbbbbbb"]
    # Content-based, not preserved: replacing a two-character run with one character shrinks
    # below the input's own <U4 width.
    shrunk = np.strings.replace(np.array(["aaaa"]), "aa", "b")
    assert shrunk.dtype == np.dtype("<U2")
    assert shrunk.tolist() == ["bb"]


def test_strings_count_and_replace_broadcast_their_array_arguments():
    a = np.array(["Hello World", "foo bar"])
    assert np.strings.count(a, "o", np.array([0, 2])).tolist() == [2, 1]
    replaced = np.strings.replace(np.array(["aaa", "bbb"]), np.array(["a", "b"]), "Z")
    assert replaced.dtype == np.dtype("<U3")
    assert replaced.tolist() == ["ZZZ", "ZZZ"]


def test_strings_partition_and_rpartition_return_three_arrays():
    a = np.array(["Hello World", "foobar"])
    before, sep, after = np.strings.partition(a, " ")
    assert (before.tolist(), sep.tolist(), after.tolist()) == (
        ["Hello", "foobar"],
        [" ", ""],
        ["World", ""],
    )
    before, sep, after = np.strings.rpartition(a, " ")
    assert (before.tolist(), sep.tolist(), after.tolist()) == (
        ["Hello", ""],
        [" ", ""],
        ["World", "foobar"],
    )


def test_strings_partition_broadcasts_an_array_of_separators():
    before, sep, after = np.strings.partition(np.array(["a-b", "c:d"]), np.array(["-", ":"]))
    assert before.tolist() == ["a", "c"]
    assert sep.tolist() == ["-", ":"]
    assert after.tolist() == ["b", "d"]


def test_strings_slice_matches_python_slice_semantics():
    a = np.array(["Hello World", "foo bar"])
    # A single positional argument is `stop`, as with the builtin `slice(stop)`.
    assert np.strings.slice(a, 5).tolist() == ["Hello", "foo b"]
    assert np.strings.slice(a, 1, 5).tolist() == ["ello", "oo b"]
    assert np.strings.slice(a, None, None, 2).tolist() == ["HloWrd", "fobr"]
    # `slice` keeps the input's width, like `strip` and the case functions.
    small = np.strings.slice(a, 3)
    assert small.dtype == np.dtype("<U11")
    assert small.tolist() == ["Hel", "foo"]


def test_strings_slice_broadcasts_arrays_of_start_and_stop():
    result = np.strings.slice(np.array(["Hello World", "foo bar"]), np.array([1, 0]), np.array([5, 3]))
    assert result.tolist() == ["ello", "foo"]


def test_strings_translate_maps_characters_elementwise():
    table = {ord("a"): "A", ord("b"): "B"}
    result = np.strings.translate(np.array(["abc", "cab"]), table)
    assert result.dtype == np.dtype("<U3")
    assert result.tolist() == ["ABc", "cAB"]


def test_strings_mod_broadcasts_values_per_element():
    result = np.strings.mod(np.array(["n=%d", "m=%d"]), np.array([7, 9]))
    assert result.dtype == np.dtype("<U3")
    assert result.tolist() == ["n=7", "m=9"]
    # A one-element `values` broadcasts against every row, like any other array argument.
    assert np.strings.mod(np.array(["n=%d", "m=%d"]), (7,)).tolist() == ["n=7", "m=7"]


def test_strings_predicate_functions():
    assert np.strings.str_len(np.array(["ab", "c"])).tolist() == [2, 1]
    assert np.strings.isalpha(np.array(["abc", "ab1"])).tolist() == [True, False]
    assert np.strings.isdigit(np.array(["123", "12a"])).tolist() == [True, False]
    assert np.strings.isdecimal(np.array(["123", "½"])).tolist() == [True, False]
    assert np.strings.isnumeric(np.array(["123", "½"])).tolist() == [True, True]
    assert np.strings.isalnum(np.array(["abc123", "ab 1"])).tolist() == [True, False]
    assert np.strings.isspace(np.array(["   ", "a"])).tolist() == [True, False]
    assert np.strings.islower(np.array(["abc", "Abc"])).tolist() == [True, False]
    assert np.strings.isupper(np.array(["ABC", "Abc"])).tolist() == [True, False]
    assert np.strings.istitle(np.array(["Hello World", "hello world"])).tolist() == [True, False]
    assert np.strings.startswith(np.array(["Hello", "foo"]), "He").tolist() == [True, False]
    assert np.strings.endswith(np.array(["Hello", "foo"]), "lo").tolist() == [True, False]


def test_strings_startswith_and_endswith_broadcast_an_array_of_patterns():
    a = np.array(["Hello", "foo", "bar"])
    assert np.strings.startswith(a, np.array(["He", "fo", "z"])).tolist() == [True, True, False]
    assert np.strings.endswith(a, np.array(["lo", "o", "r"])).tolist() == [True, True, True]


def test_strings_expandtabs_broadcasts_tabsize_but_keeps_content_based_width():
    # `expandtabs` broadcasts `tabsize` like every other parameter; its result dtype is the one
    # documented exception that does not match NumPy's own (unpinnable) width exactly -- values
    # still match NumPy exactly, only `.dtype.itemsize` can differ.
    result = np.strings.expandtabs(np.array(["a\tb", "cd"]), np.array([2, 4]))
    assert result.tolist() == ["a b", "cd"]


def test_strings_comparison_functions():
    a = np.array(["b", "a"])
    b = np.array(["a", "a"])
    assert np.strings.equal(a, b).tolist() == [False, True]
    assert np.strings.not_equal(a, b).tolist() == [True, False]
    assert np.strings.greater(a, b).tolist() == [True, False]
    assert np.strings.greater_equal(a, b).tolist() == [True, True]
    assert np.strings.less(a, b).tolist() == [False, False]
    assert np.strings.less_equal(a, b).tolist() == [False, True]


def test_strings_split_family_is_absent():
    # `numpy.strings` has no `split`/`rsplit`/`splitlines`/`join`: ragged, per-element results
    # do not fit a rectangular array, and NumPy 2.5.3 does not define them either.
    for name in ("split", "rsplit", "splitlines", "join"):
        assert not hasattr(np.strings, name)


def test_string_arithmetic_without_a_loop_raises_type_error():
    a = np.array(["a", "b"])
    with pytest.raises(TypeError):
        a - a
    with pytest.raises(TypeError):
        np.array(["a"]) + 1


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
    with pytest.raises(TypeError):
        np.array([1, "a"], dtype=object) + 1
    with pytest.raises(TypeError):
        np.array([1, None], dtype=object) + 1


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


def test_object_reductions_start_from_initial_or_the_first_element():
    assert np.array([1, 2], dtype=object).sum(initial=10) == 13
    assert np.array(["a", "b"], dtype=object).sum(initial="x") == "xab"
    assert np.add.reduce(np.array([], dtype=object)) == 0
    assert np.multiply.reduce(np.array([], dtype=object)) == 1
    with pytest.raises(ValueError):
        np.array(["a"], dtype=object).sum(where=[False])


def test_object_sum_of_mixed_types_raises():
    with pytest.raises(TypeError):
        np.array([1, "a"], dtype=object).sum()


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


class _FloatOnly:
    def __float__(self):
        return 2.5


def test_object_astype_numeric_uses_number_protocols():
    zero_d = np.empty(2, dtype=object)
    zero_d[0] = np.array(10.0)
    zero_d[1] = np.array(3)
    assert type(zero_d[0]) is np.ndarray
    assert zero_d.astype(float).tolist() == [10.0, 3.0]
    assert zero_d.astype(np.int64).tolist() == [10, 3]
    assert zero_d.astype(complex).tolist() == [10 + 0j, 3 + 0j]
    assert np.array([_FloatOnly()], dtype=object).astype(float).tolist() == [2.5]
    with pytest.raises(TypeError):
        np.array([_FloatOnly()], dtype=object).astype(np.int64)
    nested = np.empty(1, dtype=object)
    nested[0] = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        nested.astype(float)


def test_vectorize_converts_zero_d_results_to_otypes():
    doubled = np.vectorize(lambda x: np.where(x < 3, x, 2 * x), otypes="d")
    result = doubled([1.0, 5.0])
    assert result.dtype == np.dtype("float64")
    assert result.tolist() == [1.0, 10.0]


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
