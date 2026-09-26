# Portable NumPy semantics. Expectations checked against NumPy 2.5.3 on CPython 3.14.4.
# Scope: array construction, dtype inference, copy semantics, constructors and basic attributes.

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.mark.parametrize(
    "values, dtype_name",
    [
        ([True, False], "bool"),
        ([1, 2, 3], "int64"),
        ([0.5, 2.5], "float64"),
        ([1, 2.5], "float64"),
        ([True, 1], "int64"),
        ([True, 2.5], "float64"),
        ([1, 1j], "complex128"),
        ([1.5, 2j], "complex128"),
        (["a", "bc"], "<U2"),
        ([[1], [2.5]], "float64"),
    ],
)
def test_array_infers_dtype_from_python_values(values, dtype_name):
    assert np.array(values).dtype == np.dtype(dtype_name)


def test_array_from_nested_lists_has_shape_and_values():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    assert a.shape == (2, 3)
    assert a.tolist() == [[1, 2, 3], [4, 5, 6]]


def test_array_accepts_tuples_and_mixed_sequence_types():
    assert np.array((1, 2, 3)).tolist() == [1, 2, 3]
    assert np.array(((1, 2), (3, 4))).shape == (2, 2)
    assert np.array([(1, 2), [3, 4]]).tolist() == [[1, 2], [3, 4]]
    assert np.asarray((1.5, 2)).dtype == np.dtype("float64")


def test_array_of_numpy_scalars_keeps_their_dtype():
    assert np.array([np.int8(1), np.int8(2)]).dtype == np.dtype("int8")
    assert np.array([np.float32(1), np.float64(2)]).dtype == np.dtype("float64")


def test_array_from_list_of_arrays_stacks_them():
    a = np.array([np.array([1, 2]), np.array([3, 4])])
    assert a.shape == (2, 2)
    assert a.tolist() == [[1, 2], [3, 4]]


def test_mixed_python_objects_need_dtype_object():
    a = np.array([1, "a", None], dtype=object)
    assert a.dtype == np.dtype("object")
    assert a.tolist() == [1, "a", None]


@pytest.mark.parametrize(
    "values, dtype_name, expected",
    [
        ([1, 2], "float64", [1.0, 2.0]),
        ([1.7, 2.9], "int64", [1, 2]),
        ([0, 1, 2], "bool", [False, True, True]),
        ([1, 2], "uint8", [1, 2]),
        ([1, 2], "float32", [1.0, 2.0]),
    ],
)
def test_array_with_explicit_dtype_converts_values(values, dtype_name, expected):
    a = np.array(values, dtype=np.dtype(dtype_name))
    assert a.dtype == np.dtype(dtype_name)
    assert a.tolist() == expected


def test_array_with_complex64_dtype_holds_complex_values():
    a = np.array([1, 2], dtype=np.complex64)
    assert a.dtype == np.dtype("complex64")
    assert a.tolist() == [complex(1, 0), complex(2, 0)]
    assert type(a.tolist()[0]) is complex


def test_explicit_dtype_rejects_out_of_range_python_int():
    with pytest.raises(OverflowError) as info:
        np.array([300], dtype=np.int8)
    assert str(info.value) == "Python integer 300 out of bounds for int8"


def test_explicit_float_dtype_rejects_non_numeric_string():
    with pytest.raises(ValueError) as info:
        np.array([1.5, "a"], dtype=float)
    assert str(info.value) == "could not convert string to float: 'a'"


def test_array_copies_existing_array():
    a = np.array([1, 2, 3])
    c = np.array(a)
    assert c is not a
    c[0] = 99
    assert a.tolist() == [1, 2, 3]


def test_asarray_returns_same_array_when_dtype_matches():
    a = np.array([1, 2, 3])
    b = np.asarray(a)
    assert b is a
    assert np.asarray(a, dtype=np.int64) is a
    b[1] = 42
    assert a.tolist() == [1, 42, 3]


def test_asarray_converts_when_dtype_differs():
    a = np.array([1, 2, 3])
    b = np.asarray(a, dtype=np.float64)
    assert b is not a
    assert b.dtype == np.dtype("float64")


def test_asarray_of_list_does_not_alias_list():
    values = [1, 2]
    a = np.asarray(values)
    a[0] = 5
    assert values == [1, 2]


def test_array_copy_false_refuses_to_copy():
    a = np.array([1, 2, 3])
    assert np.array(a, copy=False) is a
    with pytest.raises(ValueError):
        np.array(a, dtype=float, copy=False)


@pytest.mark.parametrize(
    "rows",
    [
        [[1, 2], [3]],
        [[1, 2], 3],
        [[[1], [2]], [[3]]],
    ],
)
def test_ragged_input_raises_value_error(rows):
    with pytest.raises(ValueError) as info:
        np.array(rows)
    assert "inhomogeneous shape" in str(info.value)


def test_zeros_and_ones_default_to_float64():
    z = np.zeros((2, 3))
    assert z.dtype == np.dtype("float64")
    assert z.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert np.ones(3).tolist() == [1.0, 1.0, 1.0]


def test_zeros_and_ones_accept_dtype():
    assert np.zeros(3, dtype=bool).tolist() == [False, False, False]
    o = np.ones((2,), dtype=int)
    assert o.dtype == np.dtype("int64")
    assert o.tolist() == [1, 1]


def test_empty_has_requested_shape_and_dtype():
    e = np.empty((2, 3), dtype=np.int32)
    assert e.shape == (2, 3)
    assert e.dtype == np.dtype("int32")


def test_negative_dimension_raises_value_error():
    with pytest.raises(ValueError) as info:
        np.zeros(-1)
    assert str(info.value) == "negative dimensions are not allowed"


@pytest.mark.parametrize(
    "fill, dtype_name",
    [
        (7, "int64"),
        (7.5, "float64"),
        (True, "bool"),
        (1j, "complex128"),
    ],
)
def test_full_infers_dtype_from_fill_value(fill, dtype_name):
    f = np.full((2, 2), fill)
    assert f.dtype == np.dtype(dtype_name)
    assert f.tolist() == [[fill, fill], [fill, fill]]


def test_full_with_dtype_converts_fill_value():
    f = np.full(2, 7, dtype=np.int8)
    assert f.dtype == np.dtype("int8")
    assert f.tolist() == [7, 7]


def test_like_constructors_preserve_dtype_and_shape():
    src = np.array([[1, 2, 3]], dtype=np.int16)
    for made in (np.zeros_like(src), np.ones_like(src), np.empty_like(src), np.full_like(src, 5)):
        assert made.dtype == np.dtype("int16")
        assert made.shape == (1, 3)
    assert np.zeros_like(src).tolist() == [[0, 0, 0]]
    assert np.ones_like(src).tolist() == [[1, 1, 1]]
    assert np.full_like(src, 5).tolist() == [[5, 5, 5]]


def test_like_constructors_accept_dtype_and_shape_overrides():
    src = np.array([1, 2, 3])
    assert np.full_like(src, 2.7).tolist() == [2, 2, 2]
    f = np.full_like(src, 2.7, dtype=float)
    assert f.dtype == np.dtype("float64")
    assert f.tolist() == [2.7, 2.7, 2.7]
    z = np.zeros_like(src, shape=(2, 2))
    assert z.shape == (2, 2)
    assert z.dtype == np.dtype("int64")


@pytest.mark.parametrize(
    "args, expected",
    [
        ((5,), [0, 1, 2, 3, 4]),
        ((2, 6), [2, 3, 4, 5]),
        ((1, 10, 3), [1, 4, 7]),
        ((0,), []),
        ((5, 1), []),
    ],
)
def test_arange_integer_steps(args, expected):
    a = np.arange(*args)
    assert a.dtype == np.dtype("int64")
    assert a.tolist() == expected


def test_arange_negative_bounds_and_steps():
    assert np.arange(-3, 3).tolist() == [-3, -2, -1, 0, 1, 2]
    assert np.arange(5, 0, -2).tolist() == [5, 3, 1]
    assert np.arange(10, 0, -3).tolist() == [10, 7, 4, 1]


def test_arange_float_steps():
    a = np.arange(1, 2, 0.25)
    assert a.dtype == np.dtype("float64")
    assert a.tolist() == [1.0, 1.25, 1.5, 1.75]
    assert np.arange(0.5, 3).tolist() == [0.5, 1.5, 2.5]
    assert np.arange(3.0).dtype == np.dtype("float64")
    inexact = np.arange(0.0, 1.0, 0.3)
    assert inexact.shape == (4,)
    assert_allclose(inexact, [0.0, 0.3, 0.6, 0.9])


def test_arange_accepts_dtype():
    assert np.arange(3, dtype=np.int8).dtype == np.dtype("int8")
    a = np.arange(1, 3, dtype=float)
    assert a.dtype == np.dtype("float64")
    assert a.tolist() == [1.0, 2.0]


def test_arange_zero_step_raises():
    with pytest.raises(ZeroDivisionError):
        np.arange(0, 5, 0)


def test_linspace_includes_endpoint_by_default():
    a = np.linspace(0, 1, 5)
    assert a.dtype == np.dtype("float64")
    assert a.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert np.linspace(5, 1, 3).tolist() == [5.0, 3.0, 1.0]


def test_linspace_without_endpoint():
    assert np.linspace(0, 1, 4, endpoint=False).tolist() == [0.0, 0.25, 0.5, 0.75]


def test_linspace_small_counts():
    assert np.linspace(1, 1, 1).tolist() == [1.0]
    assert np.linspace(0, 1, 1).tolist() == [0.0]
    assert np.linspace(0, 1, 0).shape == (0,)


def test_linspace_retstep_returns_step():
    values, step = np.linspace(0, 10, 5, retstep=True)
    assert values.tolist() == [0.0, 2.5, 5.0, 7.5, 10.0]
    assert step == 2.5
    assert type(step) is np.float64


def test_linspace_with_dtype():
    assert np.linspace(0, 10, 5, dtype=int).tolist() == [0, 2, 5, 7, 10]
    assert np.linspace(2, 3, 5, dtype=np.float32).dtype == np.dtype("float32")


def test_logspace_powers_of_base():
    assert np.logspace(0, 3, 4).tolist() == [1.0, 10.0, 100.0, 1000.0]
    assert np.logspace(0, 3, 4, base=2).tolist() == [1.0, 2.0, 4.0, 8.0]
    assert_allclose(np.logspace(0, 2, 3, endpoint=False), [1.0, 4.641588833612778, 21.544346900318832])


def test_eye_and_identity():
    assert np.eye(2, 3).tolist() == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert np.identity(2).tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert np.identity(2).dtype == np.dtype("float64")
    assert np.identity(3, dtype=int).tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert np.eye(2, dtype=bool).tolist() == [[True, False], [False, True]]


@pytest.mark.parametrize(
    "k, expected",
    [
        (1, [[0, 1, 0], [0, 0, 1], [0, 0, 0]]),
        (2, [[0, 0, 1], [0, 0, 0], [0, 0, 0]]),
        (3, [[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
    ],
)
def test_eye_diagonal_offset(k, expected):
    assert np.eye(3, k=k).tolist() == expected


def test_eye_negative_diagonal_offset():
    assert np.eye(3, k=-1).tolist() == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def test_diag_builds_matrix_from_vector():
    assert np.diag([1, 2, 3]).tolist() == [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    assert np.diag([1, 2], k=1).tolist() == [[0, 1, 0], [0, 0, 2], [0, 0, 0]]
    assert np.diag([1.5, 2]).dtype == np.dtype("float64")


@pytest.mark.parametrize(
    "k, expected",
    [
        (0, [0, 4, 8]),
        (1, [1, 5]),
        (2, [2]),
        (5, []),
    ],
)
def test_diag_extracts_diagonal_from_matrix(k, expected):
    m = np.arange(9).reshape(3, 3)
    assert np.diag(m, k=k).tolist() == expected


def test_diag_extracts_below_main_diagonal():
    assert np.diag(np.arange(9).reshape(3, 3), k=-1).tolist() == [3, 7]


def test_diag_extracts_from_non_square_matrix():
    assert np.diag(np.array([[1, 2], [3, 4], [5, 6]])).tolist() == [1, 4]


def test_meshgrid_xy_indexing():
    x, y = np.meshgrid([1, 2, 3], [4, 5])
    assert x.shape == (2, 3)
    assert x.tolist() == [[1, 2, 3], [1, 2, 3]]
    assert y.tolist() == [[4, 4, 4], [5, 5, 5]]


def test_meshgrid_ij_indexing():
    x, y = np.meshgrid([1, 2, 3], [4, 5], indexing="ij")
    assert x.shape == (3, 2)
    assert x.tolist() == [[1, 1], [2, 2], [3, 3]]
    assert y.tolist() == [[4, 5], [4, 5], [4, 5]]


def test_meshgrid_three_inputs_and_dtypes():
    grids = np.meshgrid([1, 2], [3, 4, 5], [6.5])
    assert len(grids) == 3
    assert grids[0].shape == (3, 2, 1)
    assert grids[0].dtype == np.dtype("int64")
    assert grids[2].dtype == np.dtype("float64")


def test_fromiter_consumes_iterable():
    a = np.fromiter((x * x for x in range(4)), dtype=float)
    assert a.dtype == np.dtype("float64")
    assert a.tolist() == [0.0, 1.0, 4.0, 9.0]
    assert np.fromiter([], dtype=float).shape == (0,)


def test_fromiter_count_limits_items():
    a = np.fromiter(range(10), dtype=np.int8, count=3)
    assert a.dtype == np.dtype("int8")
    assert a.tolist() == [0, 1, 2]


def test_fromiter_short_iterator_raises():
    with pytest.raises(ValueError) as info:
        np.fromiter(range(2), dtype=int, count=5)
    assert str(info.value) == "iterator too short: Expected 5 but iterator had only 2 items."


@pytest.mark.parametrize(
    "dtype_name, itemsize",
    [
        ("bool", 1),
        ("int8", 1),
        ("int16", 2),
        ("float16", 2),
        ("int32", 4),
        ("float32", 4),
        ("int64", 8),
        ("float64", 8),
        ("complex64", 8),
        ("complex128", 16),
        ("object", 8),
    ],
)
def test_size_attributes(dtype_name, itemsize):
    a = np.zeros((2, 3), dtype=np.dtype(dtype_name))
    assert a.ndim == 2
    assert a.shape == (2, 3)
    assert a.size == 6
    assert a.itemsize == itemsize
    assert a.nbytes == 6 * itemsize


def test_zero_dimensional_arrays():
    for a in (np.array(5), np.zeros(())):
        assert a.shape == ()
        assert a.ndim == 0
        assert a.size == 1


def test_empty_dimension_arrays():
    a = np.zeros((0, 3))
    assert a.shape == (0, 3)
    assert a.size == 0
    assert a.nbytes == 0
    assert len(a) == 0
    assert a.tolist() == []
    assert np.zeros((3, 0)).tolist() == [[], [], []]


def test_empty_list_gives_float64():
    a = np.array([])
    assert a.dtype == np.dtype("float64")
    assert a.shape == (0,)
    assert np.array([[]]).shape == (1, 0)
