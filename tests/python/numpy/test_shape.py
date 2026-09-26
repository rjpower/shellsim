# Portable NumPy semantics. Expectations checked against NumPy 2.5.3 on CPython 3.14.4.
# Scope: reshaping, transposition, joining, splitting, tiling, padding, triangles and diffs.

import numpy as np
import pytest


def test_reshape_preserves_c_order_values():
    a = np.arange(6)
    assert a.reshape(2, 3).tolist() == [[0, 1, 2], [3, 4, 5]]
    assert a.reshape((3, 2)).tolist() == [[0, 1], [2, 3], [4, 5]]
    assert np.reshape(a, (1, 6)).shape == (1, 6)


def test_reshape_infers_one_unknown_dimension():
    a = np.arange(6)
    assert a.reshape(-1, 2).shape == (3, 2)
    assert a.reshape((2, -1)).shape == (2, 3)
    assert a.reshape(-1).shape == (6,)
    assert a.reshape(1, -1, 3).shape == (1, 2, 3)
    assert a.reshape(2, -1).tolist() == [[0, 1, 2], [3, 4, 5]]


def test_reshape_single_element_to_zero_d():
    z = np.array([7]).reshape(())
    assert z.shape == ()
    assert z[()] == 7


@pytest.mark.parametrize("shape", [(4,), (5, 2), (6, 0)])
def test_reshape_rejects_size_mismatch(shape):
    with pytest.raises(ValueError):
        np.arange(6).reshape(shape)


def test_reshape_rejects_unknown_dimension_that_does_not_divide():
    with pytest.raises(ValueError):
        np.arange(6).reshape(4, -1)


def test_reshape_rejects_two_unknown_dimensions():
    with pytest.raises(ValueError):
        np.arange(6).reshape(-1, -1)


def test_reshape_of_contiguous_array_is_view():
    a = np.arange(6)
    b = a.reshape(2, 3)
    assert np.shares_memory(a, b)
    b[0, 0] = 100
    assert a[0] == 100


def test_reshape_of_transposed_array_reads_logical_order():
    t = np.arange(6).reshape(2, 3).T
    assert t.reshape(6).tolist() == [0, 3, 1, 4, 2, 5]
    assert t.reshape(3, 2).tolist() == [[0, 3], [1, 4], [2, 5]]
    assert t.reshape(2, 3).tolist() == [[0, 3, 1], [4, 2, 5]]


def test_reshape_of_transposed_array_copies():
    a = np.arange(6).reshape(2, 3)
    r = a.T.reshape(6)
    assert not np.shares_memory(a, r)
    r[0] = 99
    assert a[0, 0] == 0


def test_flatten_returns_copy():
    a = np.arange(6).reshape(2, 3)
    f = a.flatten()
    assert f.tolist() == [0, 1, 2, 3, 4, 5]
    assert not np.shares_memory(a, f)
    f[0] = 99
    assert a[0, 0] == 0


def test_ravel_of_contiguous_array_is_view():
    a = np.arange(6).reshape(2, 3)
    r = a.ravel()
    assert r.tolist() == [0, 1, 2, 3, 4, 5]
    assert np.shares_memory(a, r)
    r[0] = 99
    assert a[0, 0] == 99


def test_ravel_of_transposed_array_copies_in_c_order():
    a = np.arange(6).reshape(2, 3)
    r = np.ravel(a.T)
    assert r.tolist() == [0, 3, 1, 4, 2, 5]
    assert not np.shares_memory(a, r)


def test_transpose_default_reverses_axes():
    a = np.arange(24).reshape(2, 3, 4)
    assert a.T.shape == (4, 3, 2)
    assert np.transpose(a).shape == (4, 3, 2)
    assert a.T[3, 2, 1] == a[1, 2, 3]


def test_transpose_of_1d_is_unchanged():
    assert np.arange(3).T.tolist() == [0, 1, 2]


def test_transpose_with_explicit_axes():
    a = np.arange(24).reshape(2, 3, 4)
    t = np.transpose(a, (1, 0, 2))
    assert t.shape == (3, 2, 4)
    assert t[2, 1, 3] == a[1, 2, 3]
    assert a.transpose(2, 0, 1).shape == (4, 2, 3)
    assert a.transpose((0, -1, 1)).shape == (2, 4, 3)


def test_transpose_is_view():
    a = np.arange(6).reshape(2, 3)
    a.T[0, 1] = 50
    assert a.tolist() == [[0, 1, 2], [50, 4, 5]]


@pytest.mark.parametrize("axes", [(0, 0), (0, 2), (0,)])
def test_transpose_rejects_invalid_axes(axes):
    a = np.arange(6).reshape(2, 3)
    with pytest.raises(ValueError):
        a.transpose(axes)


def test_swapaxes_exchanges_two_axes():
    a = np.arange(24).reshape(2, 3, 4)
    s = np.swapaxes(a, 0, 2)
    assert s.shape == (4, 3, 2)
    assert s[3, 1, 0] == a[0, 1, 3]
    assert a.swapaxes(-1, 1).shape == (2, 4, 3)


def test_moveaxis_moves_single_and_multiple_axes():
    a = np.arange(24).reshape(2, 3, 4)
    assert np.moveaxis(a, 0, -1).shape == (3, 4, 2)
    assert np.moveaxis(a, [0, 1], [-1, -2]).shape == (4, 3, 2)
    m = np.moveaxis(a, 2, 0)
    assert m.shape == (4, 2, 3)
    assert m[3, 1, 2] == a[1, 2, 3]


def test_squeeze_removes_all_length_one_axes():
    a = np.zeros((1, 3, 1, 2))
    assert np.squeeze(a).shape == (3, 2)
    assert np.squeeze(np.array([[5]])).shape == ()


def test_squeeze_with_axis_argument():
    a = np.zeros((1, 3, 1, 2))
    assert a.squeeze(axis=0).shape == (3, 1, 2)
    assert a.squeeze(axis=(0, 2)).shape == (3, 2)
    assert np.squeeze(a, axis=-2).shape == (1, 3, 2)


def test_squeeze_rejects_axis_longer_than_one():
    with pytest.raises(ValueError):
        np.zeros((1, 3)).squeeze(axis=1)


def test_expand_dims_inserts_axes():
    a = np.arange(3)
    assert np.expand_dims(a, 0).shape == (1, 3)
    assert np.expand_dims(a, 1).tolist() == [[0], [1], [2]]
    assert np.expand_dims(a, -1).shape == (3, 1)
    assert np.expand_dims(a, (0, 2)).shape == (1, 3, 1)


def test_concatenate_along_first_and_second_axis():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6]])
    assert np.concatenate((a, b)).tolist() == [[1, 2], [3, 4], [5, 6]]
    assert np.concatenate([a, b.T], axis=1).tolist() == [[1, 2, 5], [3, 4, 6]]


def test_concatenate_with_axis_none_flattens_inputs():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6]])
    assert np.concatenate([a, b], axis=None).tolist() == [1, 2, 3, 4, 5, 6]


def test_concatenate_promotes_dtype():
    r = np.concatenate([np.array([1, 2]), np.array([0.5])])
    assert r.dtype == np.dtype("float64")
    assert r.tolist() == [1.0, 2.0, 0.5]


def test_concatenate_returns_copy():
    a = np.arange(3)
    r = np.concatenate([a, a])
    assert not np.shares_memory(a, r)


def test_concatenate_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        np.concatenate([np.zeros((2, 2)), np.zeros((2, 3))], axis=0)
    with pytest.raises(ValueError):
        np.concatenate([np.zeros(2), np.zeros((2, 2))])


def test_concatenate_rejects_zero_d_inputs():
    with pytest.raises(ValueError):
        np.concatenate([np.array(1), np.array(2)])


@pytest.mark.parametrize("axis, shape", [(0, (2, 3)), (1, (3, 2))])
def test_stack_inserts_new_axis(axis, shape):
    assert np.stack([np.arange(3), np.arange(3, 6)], axis=axis).shape == shape


def test_stack_values_along_last_axis():
    r = np.stack([np.array([1, 2, 3]), np.array([4, 5, 6])], axis=-1)
    assert r.shape == (3, 2)
    assert r.tolist() == [[1, 4], [2, 5], [3, 6]]


def test_stack_rejects_different_shapes():
    with pytest.raises(ValueError):
        np.stack([np.zeros(2), np.zeros(3)])


def test_hstack_joins_1d_end_to_end_and_2d_by_columns():
    assert np.hstack([np.array([1, 2]), np.array([3])]).tolist() == [1, 2, 3]
    left = np.ones((2, 1), dtype=np.int64)
    right = np.zeros((2, 2), dtype=np.int64)
    assert np.hstack([left, right]).tolist() == [[1, 0, 0], [1, 0, 0]]


def test_vstack_treats_1d_inputs_as_rows():
    r = np.vstack([np.array([1, 2]), np.array([3, 4])])
    assert r.tolist() == [[1, 2], [3, 4]]
    assert np.vstack([np.zeros((2, 2)), np.ones((1, 2))]).shape == (3, 2)


def test_column_stack_turns_1d_inputs_into_columns():
    r = np.column_stack((np.array([1, 2, 3]), np.array([4, 5, 6])))
    assert r.tolist() == [[1, 4], [2, 5], [3, 6]]


def test_dstack_joins_along_third_axis():
    r = np.dstack((np.array([1, 2]), np.array([3, 4])))
    assert r.shape == (1, 2, 2)
    assert r.tolist() == [[[1, 3], [2, 4]]]


def test_split_into_equal_sections():
    parts = np.split(np.arange(6), 3)
    assert [p.tolist() for p in parts] == [[0, 1], [2, 3], [4, 5]]


def test_split_at_indices_allows_empty_tail():
    parts = np.split(np.arange(8), [3, 5])
    assert [p.tolist() for p in parts] == [[0, 1, 2], [3, 4], [5, 6, 7]]
    parts = np.split(np.arange(5), [2, 10])
    assert [p.tolist() for p in parts] == [[0, 1], [2, 3, 4], []]


def test_split_rejects_unequal_sections():
    with pytest.raises(ValueError):
        np.split(np.arange(7), 3)


def test_split_along_second_axis():
    left, right = np.split(np.arange(8).reshape(2, 4), 2, axis=1)
    assert left.tolist() == [[0, 1], [4, 5]]
    assert right.tolist() == [[2, 3], [6, 7]]


def test_split_returns_views():
    a = np.arange(4)
    first, _ = np.split(a, 2)
    first[0] = 9
    assert a[0] == 9


def test_array_split_puts_extra_elements_first():
    parts = np.array_split(np.arange(7), 3)
    assert [p.tolist() for p in parts] == [[0, 1, 2], [3, 4], [5, 6]]
    parts = np.array_split(np.arange(2), 3)
    assert [p.tolist() for p in parts] == [[0], [1], []]


def test_tile_repeats_whole_array():
    a = np.array([1, 2])
    assert np.tile(a, 3).tolist() == [1, 2, 1, 2, 1, 2]
    assert np.tile(a, (2, 2)).tolist() == [[1, 2, 1, 2], [1, 2, 1, 2]]
    assert np.tile(np.array([[1], [2]]), (1, 3)).tolist() == [[1, 1, 1], [2, 2, 2]]


def test_repeat_scalar_count_flattens_input():
    a = np.array([[1, 2], [3, 4]])
    assert np.repeat(a, 2).tolist() == [1, 1, 2, 2, 3, 3, 4, 4]


def test_repeat_per_element_counts():
    assert np.repeat(np.array([1, 2, 3]), [2, 0, 1]).tolist() == [1, 1, 3]


def test_repeat_along_axis():
    a = np.array([[1, 2], [3, 4]])
    assert np.repeat(a, 2, axis=0).tolist() == [[1, 2], [1, 2], [3, 4], [3, 4]]
    assert a.repeat([1, 2], axis=1).tolist() == [[1, 2, 2], [3, 4, 4]]


def test_repeat_rejects_bad_counts():
    with pytest.raises(ValueError):
        np.repeat(np.array([1, 2]), [1, 2, 3])
    with pytest.raises(ValueError):
        np.repeat(np.array([1, 2]), -1)


def test_flip_without_axis_reverses_every_axis():
    a = np.arange(6).reshape(2, 3)
    assert np.flip(a).tolist() == [[5, 4, 3], [2, 1, 0]]
    assert np.flip(a, axis=(0, 1)).tolist() == [[5, 4, 3], [2, 1, 0]]


def test_flip_along_one_axis():
    a = np.arange(6).reshape(2, 3)
    assert np.flip(a, 0).tolist() == [[3, 4, 5], [0, 1, 2]]
    assert np.flip(a, axis=1).tolist() == [[2, 1, 0], [5, 4, 3]]


def test_fliplr_and_flipud():
    a = np.arange(6).reshape(2, 3)
    assert np.fliplr(a).tolist() == [[2, 1, 0], [5, 4, 3]]
    assert np.flipud(a).tolist() == [[3, 4, 5], [0, 1, 2]]


def test_flip_returns_view():
    a = np.arange(4)
    f = np.flip(a)
    assert np.shares_memory(a, f)
    f[0] = 9
    assert a.tolist() == [0, 1, 2, 9]


def test_roll_1d_wraps_shifts():
    a = np.arange(5)
    assert np.roll(a, 1).tolist() == [4, 0, 1, 2, 3]
    assert np.roll(a, -2).tolist() == [2, 3, 4, 0, 1]
    assert np.roll(a, 7).tolist() == [3, 4, 0, 1, 2]
    assert np.roll(a, 0).tolist() == [0, 1, 2, 3, 4]


def test_roll_without_axis_uses_flattened_order():
    a = np.arange(6).reshape(2, 3)
    assert np.roll(a, 1).tolist() == [[5, 0, 1], [2, 3, 4]]


def test_roll_along_axis():
    a = np.arange(6).reshape(2, 3)
    assert np.roll(a, 1, axis=1).tolist() == [[2, 0, 1], [5, 3, 4]]
    assert np.roll(a, -1, axis=0).tolist() == [[3, 4, 5], [0, 1, 2]]
    assert np.roll(a, (1, 1), axis=(0, 1)).tolist() == [[5, 3, 4], [2, 0, 1]]


def test_pad_constant_defaults_to_zero():
    r = np.pad(np.array([1, 2, 3]), 2)
    assert r.dtype == np.dtype("int64")
    assert r.tolist() == [0, 0, 1, 2, 3, 0, 0]


def test_pad_constant_with_before_after_widths_and_value():
    r = np.pad(np.array([1, 2, 3]), (1, 2), constant_values=9)
    assert r.tolist() == [9, 1, 2, 3, 9, 9]


def test_pad_constant_values_per_side():
    r = np.pad(np.array([1, 2]), 1, constant_values=(7, 8))
    assert r.tolist() == [7, 1, 2, 8]


def test_pad_constant_with_per_axis_widths():
    a = np.array([[1, 2], [3, 4]])
    r = np.pad(a, ((1, 0), (0, 2)), constant_values=-1)
    assert r.tolist() == [[-1, -1, -1, -1], [1, 2, -1, -1], [3, 4, -1, -1]]


def test_pad_constant_value_is_cast_to_array_dtype():
    r = np.pad(np.array([1, 2]), 1, constant_values=2.7)
    assert r.dtype == np.dtype("int64")
    assert r.tolist() == [2, 1, 2, 2]


def test_pad_edge_repeats_border_values():
    assert np.pad(np.array([1, 2, 3]), (2, 1), mode="edge").tolist() == [1, 1, 1, 2, 3, 3]
    a = np.array([[1, 2], [3, 4]])
    assert np.pad(a, 1, mode="edge").tolist() == [
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ]


def test_broadcast_to_expands_length_one_axes():
    b = np.broadcast_to(np.array([[1], [2]]), (2, 3))
    assert b.tolist() == [[1, 1, 1], [2, 2, 2]]
    assert np.broadcast_to(5, (2,)).tolist() == [5, 5]
    assert np.broadcast_to(np.arange(2), (3, 2)).shape == (3, 2)


@pytest.mark.parametrize("shape", [(4,), (3, 1), (2,)])
def test_broadcast_to_rejects_incompatible_shape(shape):
    with pytest.raises(ValueError):
        np.broadcast_to(np.arange(3), shape)


def test_broadcast_arrays_returns_common_shape():
    x, y = np.broadcast_arrays(np.array([[1], [2]]), np.array([10, 20, 30]))
    assert x.shape == (2, 3)
    assert y.shape == (2, 3)
    assert x.tolist() == [[1, 1, 1], [2, 2, 2]]
    assert y.tolist() == [[10, 20, 30], [10, 20, 30]]


def test_broadcast_arrays_rejects_incompatible_shapes():
    with pytest.raises(ValueError):
        np.broadcast_arrays(np.zeros(2), np.zeros(3))


def test_atleast_1d():
    assert np.atleast_1d(5).shape == (1,)
    assert np.atleast_1d(np.arange(3)).shape == (3,)
    assert np.atleast_1d(np.zeros((2, 2))).shape == (2, 2)


def test_atleast_1d_with_several_inputs_returns_tuple():
    result = np.atleast_1d(1, [2, 3])
    assert isinstance(result, tuple)
    assert [r.tolist() for r in result] == [[1], [2, 3]]


def test_atleast_2d():
    assert np.atleast_2d(5).shape == (1, 1)
    assert np.atleast_2d(np.arange(3)).tolist() == [[0, 1, 2]]
    assert np.atleast_2d(np.zeros((2, 2, 2))).shape == (2, 2, 2)


def test_tril_and_triu_main_diagonal():
    a = np.arange(1, 10).reshape(3, 3)
    assert np.tril(a).tolist() == [[1, 0, 0], [4, 5, 0], [7, 8, 9]]
    assert np.triu(a).tolist() == [[1, 2, 3], [0, 5, 6], [0, 0, 9]]


def test_tril_with_offsets():
    a = np.arange(1, 10).reshape(3, 3)
    assert np.tril(a, k=-1).tolist() == [[0, 0, 0], [4, 0, 0], [7, 8, 0]]
    assert np.tril(a, k=1).tolist() == [[1, 2, 0], [4, 5, 6], [7, 8, 9]]


def test_triu_with_offsets():
    a = np.arange(1, 10).reshape(3, 3)
    assert np.triu(a, 1).tolist() == [[0, 2, 3], [0, 0, 6], [0, 0, 0]]
    assert np.triu(a, -1).tolist() == [[1, 2, 3], [4, 5, 6], [0, 8, 9]]


def test_triu_on_non_square_matrix():
    r = np.triu(np.ones((2, 4), dtype=np.int64), 1)
    assert r.tolist() == [[0, 1, 1, 1], [0, 0, 1, 1]]


def test_diagonal_with_offsets():
    a = np.arange(12).reshape(3, 4)
    assert a.diagonal().tolist() == [0, 5, 10]
    assert np.diagonal(a, 1).tolist() == [1, 6, 11]
    assert np.diagonal(a, offset=-1).tolist() == [4, 9]
    assert np.diagonal(a, 5).shape == (0,)


def test_diagonal_returns_read_only_view():
    a = np.arange(9).reshape(3, 3)
    d = a.diagonal()
    assert d.flags.writeable is False
    assert np.shares_memory(a, d)
    with pytest.raises(ValueError):
        d[0] = 1


def test_trace_sums_diagonal():
    a = np.arange(12).reshape(3, 4)
    assert np.trace(a) == 15
    assert a.trace(1) == 18
    assert np.trace(a, offset=-1) == 13


def test_trace_of_3d_sums_over_first_two_axes():
    a = np.arange(8).reshape(2, 2, 2)
    assert np.trace(a).tolist() == [6, 8]


def test_diff_first_order():
    r = np.diff(np.array([1, 4, 9, 16]))
    assert r.dtype == np.dtype("int64")
    assert r.tolist() == [3, 5, 7]


@pytest.mark.parametrize("n, expected", [(0, [1, 4, 9, 16]), (2, [2, 2]), (3, [0]), (5, [])])
def test_diff_higher_order(n, expected):
    assert np.diff(np.array([1, 4, 9, 16]), n=n).tolist() == expected


def test_diff_along_axis():
    a = np.array([[1, 3, 6], [10, 15, 21]])
    assert np.diff(a).tolist() == [[2, 3], [5, 6]]
    assert np.diff(a, axis=0).tolist() == [[9, 12, 15]]


def test_diff_of_bool_uses_not_equal():
    r = np.diff(np.array([True, False, False, True]))
    assert r.dtype == np.dtype("bool")
    assert r.tolist() == [True, False, True]


def test_diff_of_unsigned_wraps():
    r = np.diff(np.array([5, 3], dtype=np.uint8))
    assert r.dtype == np.dtype("uint8")
    assert r.tolist() == [254]


def test_diff_rejects_negative_order():
    with pytest.raises(ValueError):
        np.diff(np.arange(3), n=-1)
