# Portable NumPy semantics. Expectations checked against NumPy 2.5.3 on CPython 3.14.4.
# Scope: exact str/repr text for arrays and scalars, print options, and array2string.

import numpy as np


def test_int_array_one_dimensional():
    values = np.array([0, 1, 2])
    assert str(values) == "[0 1 2]"
    assert repr(values) == "array([0, 1, 2])"


def test_int_array_pads_to_widest_element():
    values = np.array([-1, 10, 100])
    assert str(values) == "[ -1  10 100]"
    assert repr(values) == "array([ -1,  10, 100])"


def test_int_array_two_dimensional():
    values = np.arange(6).reshape(2, 3)
    assert str(values) == "[[0 1 2]\n [3 4 5]]"
    assert repr(values) == "array([[0, 1, 2],\n       [3, 4, 5]])"


def test_int_column_vector():
    assert str(np.array([[1], [2], [3]])) == "[[1]\n [2]\n [3]]"


def test_int_array_three_dimensional_separates_blocks_with_blank_line():
    values = np.arange(8).reshape(2, 2, 2)
    assert str(values) == "[[[0 1]\n  [2 3]]\n\n [[4 5]\n  [6 7]]]"
    assert repr(values) == "array([[[0, 1],\n        [2, 3]],\n\n       [[4, 5],\n        [6, 7]]])"


def test_float_array_pads_fraction_digits():
    values = np.array([0.1, 0.25])
    assert str(values) == "[0.1  0.25]"
    assert repr(values) == "array([0.1 , 0.25])"


def test_integral_floats_print_trailing_point():
    values = np.array([1.0, 2.0])
    assert str(values) == "[1. 2.]"
    assert repr(values) == "array([1., 2.])"
    assert repr(np.zeros(3)) == "array([0., 0., 0.])"


def test_float_array_with_negative_values():
    values = np.array([-1.5, 2.0, 3.25])
    assert str(values) == "[-1.5   2.    3.25]"
    assert repr(values) == "array([-1.5 ,  2.  ,  3.25])"
    assert repr(np.array([-0.0, 1.0])) == "array([-0.,  1.])"


def test_float_array_rounds_to_eight_digits():
    assert str(np.array([1 / 3, 2 / 3])) == "[0.33333333 0.66666667]"
    assert repr(np.array([0.123456789012, 1.0])) == "array([0.12345679, 1.        ])"


def test_float_array_two_dimensional():
    values = np.array([[1.5, 2.0], [3.0, -4.25]])
    assert str(values) == "[[ 1.5   2.  ]\n [ 3.   -4.25]]"
    assert repr(values) == "array([[ 1.5 ,  2.  ],\n       [ 3.  , -4.25]])"


def test_float_array_three_dimensional():
    values = np.arange(8.0).reshape(2, 2, 2) / 2
    assert str(values) == "[[[0.  0.5]\n  [1.  1.5]]\n\n [[2.  2.5]\n  [3.  3.5]]]"
    assert repr(values) == "array([[[0. , 0.5],\n        [1. , 1.5]],\n\n       [[2. , 2.5],\n        [3. , 3.5]]])"


def test_small_magnitude_switches_to_scientific():
    values = np.array([1e-5, 1.0])
    assert str(values) == "[1.e-05 1.e+00]"
    assert repr(values) == "array([1.e-05, 1.e+00])"
    assert repr(np.array([0.0001, 1.0])) == "array([1.e-04, 1.e+00])"
    assert repr(np.array([0.00011, 1.0])) == "array([1.1e-04, 1.0e+00])"
    assert repr(np.array([-1e-5, 1.0])) == "array([-1.e-05,  1.e+00])"


def test_large_magnitude_switches_to_scientific():
    values = np.array([1e10, 1.0])
    assert str(values) == "[1.e+10 1.e+00]"
    assert repr(values) == "array([1.e+10, 1.e+00])"
    assert repr(np.array([1e8, 1.0])) == "array([1.e+08, 1.e+00])"
    assert repr(np.array([99999999.0, 1.0])) == "array([9.9999999e+07, 1.0000000e+00])"
    assert repr(np.array([123456789.0, 0.5])) == "array([1.23456789e+08, 5.00000000e-01])"


def test_magnitude_ratio_above_thousand_switches_to_scientific():
    assert repr(np.array([1.0, 1000.0])) == "array([   1., 1000.])"
    assert repr(np.array([1.0, 1001.0])) == "array([1.000e+00, 1.001e+03])"
    assert repr(np.array([1e7, 1.5])) == "array([1.0e+07, 1.5e+00])"
    assert repr(np.array([100.0, 2000.0, 30000.0])) == "array([  100.,  2000., 30000.])"


def test_nan_and_inf():
    values = np.array([np.nan, np.inf, -np.inf, 1.0])
    assert str(values) == "[ nan  inf -inf   1.]"
    assert repr(values) == "array([ nan,  inf, -inf,   1.])"


def test_bool_arrays():
    values = np.array([True, False])
    assert str(values) == "[ True False]"
    assert repr(values) == "array([ True, False])"
    grid = np.array([[True, False], [False, True]])
    assert str(grid) == "[[ True False]\n [False  True]]"
    assert repr(grid) == "array([[ True, False],\n       [False,  True]])"


def test_complex_arrays():
    values = np.array([1 + 2j, 3 - 4j])
    assert str(values) == "[1.+2.j 3.-4.j]"
    assert repr(values) == "array([1.+2.j, 3.-4.j])"
    assert repr(np.array([1j])) == "array([0.+1.j])"


def test_complex_array_pads_real_and_imaginary_parts():
    values = np.array([0.5 + 1j, 2 - 1.25j])
    assert str(values) == "[0.5+1.j   2. -1.25j]"
    assert repr(values) == "array([0.5+1.j  , 2. -1.25j])"


def test_string_arrays():
    values = np.array(["a", "bc"])
    assert str(values) == "['a' 'bc']"
    assert repr(values) == "array(['a', 'bc'], dtype='<U2')"
    grid = np.array([["ab", "c"], ["d", "efg"]])
    assert str(grid) == "[['ab' 'c']\n ['d' 'efg']]"
    assert repr(grid) == "array([['ab', 'c'],\n       ['d', 'efg']], dtype='<U3')"
    assert repr(np.array(["hello world"])) == "array(['hello world'], dtype='<U11')"


def test_object_arrays():
    values = np.array([1, "a", None], dtype=object)
    assert str(values) == "[1 'a' None]"
    assert repr(values) == "array([1, 'a', None], dtype=object)"
    assert repr(np.array([1, 2, 3], dtype=object)) == "array([1, 2, 3], dtype=object)"
    grid = np.array([[1, "a"], [None, 2.5]], dtype=object)
    assert repr(grid) == "array([[1, 'a'],\n       [None, 2.5]], dtype=object)"


def test_empty_arrays():
    assert str(np.array([])) == "[]"
    assert repr(np.array([])) == "array([], dtype=float64)"
    assert repr(np.array([], dtype=int)) == "array([], dtype=int64)"
    assert repr(np.array([], dtype=np.int8)) == "array([], dtype=int8)"
    assert repr(np.array([], dtype=bool)) == "array([], dtype=bool)"


def test_empty_multidimensional_arrays_show_shape():
    assert str(np.zeros((0, 3))) == "[]"
    assert repr(np.zeros((0, 3))) == "array([], shape=(0, 3), dtype=float64)"
    assert repr(np.zeros((0, 3), dtype=int)) == "array([], shape=(0, 3), dtype=int64)"
    assert repr(np.zeros((1, 0))) == "array([], shape=(1, 0), dtype=float64)"


def test_non_default_dtypes_show_dtype_in_repr_only():
    assert repr(np.array([1, 2], dtype=np.int8)) == "array([1, 2], dtype=int8)"
    assert repr(np.array([1, 2], dtype=np.uint8)) == "array([1, 2], dtype=uint8)"
    assert repr(np.array([1, 2], dtype=np.int32)) == "array([1, 2], dtype=int32)"
    assert repr(np.array([-128, 0, 127], dtype=np.int8)) == "array([-128,    0,  127], dtype=int8)"
    assert str(np.array([1, 2], dtype=np.int8)) == "[1 2]"


def test_float32_arrays_use_float32_shortest_digits():
    values = np.array([0.1, 1.0], dtype=np.float32)
    assert str(values) == "[0.1 1. ]"
    assert repr(values) == "array([0.1, 1. ], dtype=float32)"
    assert repr(np.array([1 / 3], dtype=np.float32)) == "array([0.33333334], dtype=float32)"


def test_float32_linspace_wraps_with_dtype_suffix():
    assert repr(np.linspace(0, 1, 12, dtype=np.float32)) == (
        "array([0.        , 0.09090909, 0.18181819, 0.27272728, 0.36363637,\n"
        "       0.45454547, 0.54545456, 0.6363636 , 0.72727275, 0.8181818 ,\n"
        "       0.90909094, 1.        ], dtype=float32)"
    )


def test_long_array_is_summarized():
    values = np.arange(2000)
    assert str(values) == "[   0    1    2 ... 1997 1998 1999]"
    assert repr(values) == "array([   0,    1,    2, ..., 1997, 1998, 1999], shape=(2000,))"


def test_summarization_threshold_is_exclusive():
    assert "..." not in repr(np.arange(1000))
    assert repr(np.arange(1001)) == "array([   0,    1,    2, ...,  998,  999, 1000], shape=(1001,))"


def test_long_two_dimensional_array_is_summarized_on_both_axes():
    values = np.arange(2000).reshape(100, 20)
    assert str(values) == (
        "[[   0    1    2 ...   17   18   19]\n"
        " [  20   21   22 ...   37   38   39]\n"
        " [  40   41   42 ...   57   58   59]\n"
        " ...\n"
        " [1940 1941 1942 ... 1957 1958 1959]\n"
        " [1960 1961 1962 ... 1977 1978 1979]\n"
        " [1980 1981 1982 ... 1997 1998 1999]]"
    )
    assert repr(values) == (
        "array([[   0,    1,    2, ...,   17,   18,   19],\n"
        "       [  20,   21,   22, ...,   37,   38,   39],\n"
        "       [  40,   41,   42, ...,   57,   58,   59],\n"
        "       ...,\n"
        "       [1940, 1941, 1942, ..., 1957, 1958, 1959],\n"
        "       [1960, 1961, 1962, ..., 1977, 1978, 1979],\n"
        "       [1980, 1981, 1982, ..., 1997, 1998, 1999]], shape=(100, 20))"
    )


def test_int_array_wraps_at_75_characters():
    values = np.arange(30)
    assert str(values) == (
        "[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23\n 24 25 26 27 28 29]"
    )
    assert repr(values) == (
        "array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n"
        "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])"
    )
    assert repr(np.arange(30, dtype=np.uint8)) == (
        "array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n"
        "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], dtype=uint8)"
    )


def test_float_array_wraps_at_75_characters():
    values = np.linspace(0, 1, 12)
    assert str(values) == (
        "[0.         0.09090909 0.18181818 0.27272727 0.36363636 0.45454545\n"
        " 0.54545455 0.63636364 0.72727273 0.81818182 0.90909091 1.        ]"
    )
    assert repr(values) == (
        "array([0.        , 0.09090909, 0.18181818, 0.27272727, 0.36363636,\n"
        "       0.45454545, 0.54545455, 0.63636364, 0.72727273, 0.81818182,\n"
        "       0.90909091, 1.        ])"
    )


def test_two_dimensional_rows_wrap_independently():
    values = np.linspace(0, 1, 12).reshape(2, 6)
    assert str(values) == (
        "[[0.         0.09090909 0.18181818 0.27272727 0.36363636 0.45454545]\n"
        " [0.54545455 0.63636364 0.72727273 0.81818182 0.90909091 1.        ]]"
    )
    assert repr(values) == (
        "array([[0.        , 0.09090909, 0.18181818, 0.27272727, 0.36363636,\n"
        "        0.45454545],\n"
        "       [0.54545455, 0.63636364, 0.72727273, 0.81818182, 0.90909091,\n"
        "        1.        ]])"
    )


def test_bool_array_wraps():
    assert repr(np.array([True, False] * 10)) == (
        "array([ True, False,  True, False,  True, False,  True, False,  True,\n"
        "       False,  True, False,  True, False,  True, False,  True, False,\n"
        "        True, False])"
    )


def test_zero_dimensional_arrays():
    assert str(np.array(5)) == "5"
    assert repr(np.array(5)) == "array(5)"
    assert str(np.array(2.5)) == "2.5"
    assert repr(np.array(2.5)) == "array(2.5)"
    assert str(np.array(True)) == "True"
    assert repr(np.array(True)) == "array(True)"
    assert repr(np.array(5, dtype=np.int8)) == "array(5, dtype=int8)"
    assert str(np.array("hi")) == "hi"
    assert repr(np.array("hi")) == "array('hi', dtype='<U2')"


def test_float_scalar_repr_and_str():
    assert repr(np.float64(1.5)) == "np.float64(1.5)"
    assert str(np.float64(1.5)) == "1.5"
    assert repr(np.float64(0.1)) == "np.float64(0.1)"
    assert str(np.float64(1 / 3)) == "0.3333333333333333"
    assert str(np.float64(5.0)) == "5.0"
    assert repr(np.float64(100.0)) == "np.float64(100.0)"
    assert repr(np.float64(1e16)) == "np.float64(1e+16)"
    assert repr(np.float64(1e-5)) == "np.float64(1e-05)"
    assert repr(np.float64(np.nan)) == "np.float64(nan)"
    assert repr(np.float64(np.inf)) == "np.float64(inf)"
    assert repr(np.float64(-0.0)) == "np.float64(-0.0)"


def test_integer_scalar_repr_and_str():
    assert repr(np.int64(3)) == "np.int64(3)"
    assert str(np.int64(3)) == "3"
    assert repr(np.int8(-3)) == "np.int8(-3)"
    assert repr(np.int32(-7)) == "np.int32(-7)"
    assert repr(np.uint8(7)) == "np.uint8(7)"
    assert repr(np.uint64(18446744073709551615)) == "np.uint64(18446744073709551615)"


def test_float32_scalar_repr_uses_shortest_float32_digits():
    assert repr(np.float32(0.1)) == "np.float32(0.1)"
    assert str(np.float32(0.1)) == "0.1"
    assert repr(np.float32(1.5)) == "np.float32(1.5)"
    assert repr(np.float32(1 / 3)) == "np.float32(0.33333334)"


def test_bool_scalar_repr_and_str():
    assert repr(np.bool_(True)) == "np.True_"
    assert repr(np.False_) == "np.False_"
    assert str(np.bool_(False)) == "False"
    assert repr(np.array([True, False])[1]) == "np.False_"


def test_scalars_from_reductions_and_indexing():
    assert repr(np.array([1, 2, 3]).sum()) == "np.int64(6)"
    assert repr(np.array([1, 2, 3]).mean()) == "np.float64(2.0)"
    assert repr(np.array([1.5, 2.5])[0]) == "np.float64(1.5)"
    assert repr([np.int64(1), np.float64(2.0)]) == "[np.int64(1), np.float64(2.0)]"
    assert f"{np.float64(1 / 3):.3f}" == "0.333"


def test_printoptions_precision():
    with np.printoptions(precision=3):
        assert repr(np.array([1 / 3, 2 / 3, 1.0])) == "array([0.333, 0.667, 1.   ])"
        assert str(np.array([1 / 3, 1.23456789])) == "[0.333 1.235]"
        assert repr(np.array([1e-5, 1 / 3])) == "array([1.000e-05, 3.333e-01])"
    assert repr(np.array([1 / 3])) == "array([0.33333333])"


def test_printoptions_precision_does_not_change_scalar_repr():
    with np.printoptions(precision=3):
        assert repr(np.float64(1 / 3)) == "np.float64(0.3333333333333333)"


def test_printoptions_suppress():
    with np.printoptions(suppress=True):
        assert repr(np.array([1e-5, 1.0])) == "array([0.00001, 1.     ])"
        assert repr(np.array([1e-10, 1.0, 1000.5])) == "array([   0. ,    1. , 1000.5])"
        assert repr(np.array([1e10, 1.0])) == "array([1.e+10, 1.e+00])"
    with np.printoptions(precision=2, suppress=True):
        assert repr(np.array([1e-5, 1.23456, 100.0])) == "array([  0.  ,   1.23, 100.  ])"
    assert repr(np.array([1e-5, 1.0])) == "array([1.e-05, 1.e+00])"


def test_printoptions_threshold_and_edgeitems():
    with np.printoptions(threshold=5):
        assert repr(np.arange(10)) == "array([0, 1, 2, ..., 7, 8, 9], shape=(10,))"
        assert str(np.arange(10)) == "[0 1 2 ... 7 8 9]"
        assert repr(np.arange(5)) == "array([0, 1, 2, 3, 4])"
    with np.printoptions(threshold=5, edgeitems=2):
        assert str(np.arange(10)) == "[0 1 ... 8 9]"


def test_printoptions_linewidth():
    with np.printoptions(linewidth=20):
        assert str(np.arange(12)) == "[ 0  1  2  3  4  5\n  6  7  8  9 10 11]"
        assert repr(np.arange(12)) == "array([ 0,  1,  2,\n        3,  4,  5,\n        6,  7,  8,\n        9, 10, 11])"


def test_printoptions_nanstr_and_infstr():
    with np.printoptions(nanstr="NaN", infstr="Inf"):
        assert str(np.array([np.nan, np.inf, 1.0])) == "[NaN Inf  1.]"


def test_printoptions_context_restores_defaults():
    with np.printoptions(precision=2, threshold=10, linewidth=40, suppress=True):
        assert np.get_printoptions()["precision"] == 2
    options = np.get_printoptions()
    assert options["precision"] == 8
    assert options["threshold"] == 1000
    assert options["linewidth"] == 75
    assert options["edgeitems"] == 3
    assert options["suppress"] is False


def test_set_printoptions_changes_global_state():
    try:
        np.set_printoptions(precision=4)
        assert np.get_printoptions()["precision"] == 4
        assert repr(np.array([1 / 3])) == "array([0.3333])"
    finally:
        np.set_printoptions(
            edgeitems=3,
            infstr="inf",
            linewidth=75,
            nanstr="nan",
            precision=8,
            suppress=False,
            threshold=1000,
            formatter=None,
        )
    assert repr(np.array([1 / 3])) == "array([0.33333333])"


def test_array2string_default_matches_str():
    assert np.array2string(np.array([1.5, 2.0])) == "[1.5 2. ]"
    assert np.array2string(np.arange(12), max_line_width=20) == "[ 0  1  2  3  4  5\n  6  7  8  9 10 11]"


def test_array2string_separator():
    assert np.array2string(np.array([1, 2, 3]), separator=", ") == "[1, 2, 3]"
    assert np.array2string(np.array([0.1, 0.25]), separator=", ") == "[0.1 , 0.25]"
    assert np.array2string(np.array([[1, 2], [3, 4]]), separator=", ") == "[[1, 2],\n [3, 4]]"


def test_array2string_options():
    assert np.array2string(np.array([1 / 3, 2 / 3]), precision=2) == "[0.33 0.67]"
    assert np.array2string(np.array([1e-6, 1.0]), suppress_small=True) == "[0.000001 1.      ]"
    assert np.array2string(np.arange(12), threshold=5) == "[ 0  1  2 ...  9 10 11]"
    assert np.array2string(np.array([[1, 2], [3, 4]]), separator=",", prefix="x = ") == "[[1,2],\n     [3,4]]"
