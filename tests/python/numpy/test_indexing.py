# Portable NumPy semantics. Expectations checked against NumPy 2.5.3 on CPython 3.14.4.
# Scope: basic, advanced and boolean indexing, assignment through each form, views and aliasing.

import numpy as np
import pytest


def test_integer_index_returns_numpy_scalar():
    a = np.arange(5) * 10
    value = a[2]
    assert isinstance(value, np.int64)
    assert value == 20


def test_negative_integer_index_counts_from_end():
    a = np.arange(5) * 10
    assert a[-1] == 40
    assert a[-5] == 0


def test_numpy_integer_scalar_is_accepted_as_index():
    a = np.arange(5) * 10
    assert a[np.int64(3)] == 30


@pytest.mark.parametrize("index", [5, 100])
def test_integer_index_past_end_raises_index_error(index):
    a = np.arange(5)
    with pytest.raises(IndexError):
        a[index]


def test_negative_index_before_start_raises_index_error():
    a = np.arange(5)
    with pytest.raises(IndexError):
        a[-6]


def test_out_of_range_index_on_second_axis_raises_index_error():
    a = np.arange(6).reshape(2, 3)
    with pytest.raises(IndexError):
        a[0, 3]
    with pytest.raises(IndexError):
        a[1, -4]


def test_too_many_indices_raises_index_error():
    a = np.arange(6).reshape(2, 3)
    with pytest.raises(IndexError):
        a[0, 0, 0]


def test_float_index_raises_index_error():
    a = np.arange(4)
    with pytest.raises(IndexError):
        a[1.0]


def test_row_index_on_2d_returns_writable_view():
    a = np.arange(6).reshape(2, 3)
    row = a[1]
    assert row.shape == (3,)
    assert row.tolist() == [3, 4, 5]
    row[0] = 99
    assert a.tolist() == [[0, 1, 2], [99, 4, 5]]


@pytest.mark.parametrize(
    "start, stop, step, expected",
    [
        (None, None, 2, [0, 2, 4, 6, 8]),
        (1, 7, 3, [1, 4]),
        (5, 5, None, []),
        (20, 30, None, []),
        (None, 3, None, [0, 1, 2]),
    ],
)
def test_one_dimensional_forward_slices(start, stop, step, expected):
    a = np.arange(10)
    assert a[start:stop:step].tolist() == expected


def test_one_dimensional_slices_with_negative_bounds():
    a = np.arange(10)
    assert a[-3:].tolist() == [7, 8, 9]
    assert a[-100:3].tolist() == [0, 1, 2]
    assert a[2:-5].tolist() == [2, 3, 4]


def test_one_dimensional_slices_with_negative_steps():
    a = np.arange(10)
    assert a[::-1].tolist() == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert a[8:2:-2].tolist() == [8, 6, 4]
    assert a[:-7:-3].tolist() == [9, 6]
    assert a[2:8:-1].tolist() == []


def test_slice_with_zero_step_raises_value_error():
    a = np.arange(5)
    with pytest.raises(ValueError):
        a[::0]


def test_slices_with_steps_on_each_axis():
    a = np.arange(20).reshape(4, 5)
    assert a[::2, ::-2].tolist() == [[4, 2, 0], [14, 12, 10]]
    assert a[::-1, 1:4].tolist() == [[16, 17, 18], [11, 12, 13], [6, 7, 8], [1, 2, 3]]
    assert a[-1:0:-2, ::3].tolist() == [[15, 18], [5, 8]]


def test_tuple_index_mixes_integers_and_slices():
    a = np.arange(12).reshape(3, 4)
    assert a[1, ::-1].tolist() == [7, 6, 5, 4]
    assert a[:, 1].tolist() == [1, 5, 9]
    assert a[:, 1].shape == (3,)
    assert a[1:, -1].tolist() == [7, 11]
    assert a[2, 3] == 11


def test_ellipsis_fills_remaining_axes():
    a = np.arange(24).reshape(2, 3, 4)
    assert a[..., 0].tolist() == [[0, 4, 8], [12, 16, 20]]
    assert a[0, ...].shape == (3, 4)
    assert a[..., 1, 2].tolist() == [6, 18]
    assert a[1, ..., 3].tolist() == [15, 19, 23]
    assert a[...].shape == (2, 3, 4)


def test_more_than_one_ellipsis_raises_index_error():
    a = np.arange(8).reshape(2, 2, 2)
    with pytest.raises(IndexError):
        a[..., 0, ...]


def test_none_and_newaxis_insert_length_one_axes():
    a = np.arange(3)
    assert np.newaxis is None
    assert a[None].shape == (1, 3)
    assert a[:, np.newaxis].shape == (3, 1)
    assert a[:, None].tolist() == [[0], [1], [2]]
    b = np.arange(6).reshape(2, 3)
    assert b[:, None, :].shape == (2, 1, 3)
    assert b[None, ..., None].shape == (1, 2, 3, 1)


def test_integer_array_gathers_one_dimensional():
    a = np.arange(10) * 10
    result = a[[3, 0, -1, 3]]
    assert result.dtype == np.dtype("int64")
    assert result.tolist() == [30, 0, 90, 30]


def test_integer_array_index_shape_sets_result_shape():
    a = np.arange(10) * 10
    idx = np.array([[1, 2], [3, 4]])
    assert a[idx].shape == (2, 2)
    assert a[idx].tolist() == [[10, 20], [30, 40]]


def test_empty_integer_list_selects_nothing():
    a = np.arange(4)
    assert a[[]].shape == (0,)


def test_integer_array_out_of_range_raises_index_error():
    a = np.arange(4)
    with pytest.raises(IndexError):
        a[[0, 4]]


def test_integer_array_gathers_rows():
    a = np.arange(12).reshape(3, 4)
    assert a[[2, 0]].tolist() == [[8, 9, 10, 11], [0, 1, 2, 3]]


def test_paired_integer_arrays_select_points():
    a = np.arange(12).reshape(3, 4)
    assert a[[0, 1], [1, 2]].tolist() == [1, 6]
    assert a[[0, 2, 2], [-1, 0, 3]].tolist() == [3, 8, 11]


def test_integer_index_arrays_broadcast_together():
    a = np.arange(12).reshape(3, 4)
    result = a[[[0], [1]], [0, 2]]
    assert result.shape == (2, 2)
    assert result.tolist() == [[0, 2], [4, 6]]
    rows = np.arange(3)[:, None]
    cols = np.array([3, 0])
    assert a[rows, cols].tolist() == [[3, 0], [7, 4], [11, 8]]


def test_index_arrays_that_do_not_broadcast_raise_index_error():
    a = np.arange(12).reshape(3, 4)
    with pytest.raises(IndexError):
        a[[0, 1, 2], [0, 1]]


def test_integer_array_mixed_with_slice_and_integer():
    a = np.arange(12).reshape(3, 4)
    assert a[1:, [0, 3]].tolist() == [[4, 7], [8, 11]]
    assert a[[2, 0], ::2].tolist() == [[8, 10], [0, 2]]
    assert a[[0, 2], 1].tolist() == [1, 9]


def test_adjacent_advanced_indices_keep_their_position():
    a = np.arange(24).reshape(2, 3, 4)
    result = a[:, [0, 2], [1, 3]]
    assert result.shape == (2, 2)
    assert result.tolist() == [[1, 11], [13, 23]]


def test_advanced_indices_separated_by_slice_move_to_front():
    a = np.arange(24).reshape(2, 3, 4)
    result = a[[0, 1], :, [1, 2]]
    assert result.shape == (2, 3)
    assert result.tolist() == [[1, 5, 9], [14, 18, 22]]


def test_boolean_mask_one_dimensional():
    a = np.arange(6)
    assert a[a % 2 == 0].tolist() == [0, 2, 4]
    assert a[a > 3].tolist() == [4, 5]
    assert a[np.zeros(6, dtype=bool)].shape == (0,)


def test_boolean_list_is_a_mask_not_integers():
    a = np.arange(3) * 10
    assert a[[True, False, True]].tolist() == [0, 20]


def test_full_shape_boolean_mask_selects_in_c_order():
    a = np.arange(12).reshape(3, 4)
    result = a[a % 5 == 0]
    assert result.shape == (3,)
    assert result.tolist() == [0, 5, 10]


def test_boolean_mask_on_first_axis_selects_rows():
    a = np.arange(12).reshape(3, 4)
    mask = np.array([True, False, True])
    assert a[mask].tolist() == [[0, 1, 2, 3], [8, 9, 10, 11]]
    assert a[mask, :].tolist() == [[0, 1, 2, 3], [8, 9, 10, 11]]
    assert a[mask, 1].tolist() == [1, 9]


def test_boolean_mask_on_second_axis_selects_columns():
    a = np.arange(12).reshape(3, 4)
    mask = np.array([True, False, False, True])
    assert a[:, mask].tolist() == [[0, 3], [4, 7], [8, 11]]


def test_boolean_mask_with_wrong_length_raises_index_error():
    a = np.arange(4)
    with pytest.raises(IndexError):
        a[np.array([True, False])]


def test_scalar_assignment_broadcasts_over_strided_slice():
    a = np.zeros((3, 4), dtype=np.int64)
    a[1:, ::2] = 7
    assert a.tolist() == [[0, 0, 0, 0], [7, 0, 7, 0], [7, 0, 7, 0]]


def test_row_assignment_broadcasts_over_rows():
    a = np.zeros((3, 3), dtype=np.int64)
    a[:] = [1, 2, 3]
    assert a.tolist() == [[1, 2, 3], [1, 2, 3], [1, 2, 3]]


def test_column_assignment_broadcasts_over_columns():
    a = np.zeros((3, 3), dtype=np.int64)
    a[:, 1:] = [[10], [20], [30]]
    assert a.tolist() == [[0, 10, 10], [0, 20, 20], [0, 30, 30]]


def test_ellipsis_assignment():
    a = np.zeros((2, 2, 2), dtype=np.int64)
    a[..., 0] = 1
    assert a.tolist() == [[[1, 0], [1, 0]], [[1, 0], [1, 0]]]


def test_assignment_with_mismatched_shape_raises_value_error():
    a = np.zeros(4)
    with pytest.raises(ValueError):
        a[:] = [1.0, 2.0, 3.0]


def test_float_assignment_into_int_array_truncates_toward_zero():
    a = np.zeros(3, dtype=np.int64)
    a[:] = [1.9, -1.9, 2.5]
    assert a.tolist() == [1, -1, 2]


def test_fancy_assignment_with_repeated_index_last_write_wins():
    a = np.zeros(5, dtype=np.int64)
    a[[0, 2, 0, 4]] = [1, 2, 3, 4]
    assert a.tolist() == [3, 0, 2, 0, 4]


def test_fancy_augmented_assignment_applies_once_per_repeated_index():
    a = np.zeros(3, dtype=np.int64)
    a[[0, 0, 1]] += 1
    assert a.tolist() == [1, 1, 0]


def test_fancy_assignment_to_points_in_2d():
    a = np.zeros((3, 3), dtype=np.int64)
    a[[0, 1, 2], [2, 1, 0]] = 5
    assert a.tolist() == [[0, 0, 5], [0, 5, 0], [5, 0, 0]]


def test_fancy_assignment_to_rows_broadcasts_value():
    a = np.zeros((3, 2), dtype=np.int64)
    a[[2, 0]] = [7, 8]
    assert a.tolist() == [[7, 8], [0, 0], [7, 8]]


def test_boolean_mask_assignment_with_scalar():
    a = np.arange(6)
    a[a % 2 == 1] = -1
    assert a.tolist() == [0, -1, 2, -1, 4, -1]


def test_boolean_mask_assignment_with_matching_values():
    a = np.arange(6)
    a[a > 2] = [30, 40, 50]
    assert a.tolist() == [0, 1, 2, 30, 40, 50]


def test_boolean_row_mask_assignment_broadcasts_row():
    a = np.zeros((3, 2), dtype=np.int64)
    a[np.array([True, False, True])] = [1, 2]
    assert a.tolist() == [[1, 2], [0, 0], [1, 2]]


def test_basic_slice_is_a_view():
    a = np.arange(6)
    s = a[1:4]
    assert s.base is a
    assert np.shares_memory(a, s)
    s[:] = 0
    assert a.tolist() == [0, 0, 0, 0, 4, 5]
    a[1] = 9
    assert s.tolist() == [9, 0, 0]


def test_column_view_augmented_assignment_writes_back():
    a = np.arange(6).reshape(2, 3)
    col = a[:, 1]
    col += 100
    assert a.tolist() == [[0, 101, 2], [3, 104, 5]]


def test_integer_array_index_returns_copy():
    a = np.arange(6)
    c = a[[1, 2, 3]]
    assert not np.shares_memory(a, c)
    c[:] = 0
    assert a.tolist() == [0, 1, 2, 3, 4, 5]


def test_boolean_mask_index_returns_copy():
    a = np.arange(6)
    c = a[a > 2]
    assert not np.shares_memory(a, c)
    c[:] = 0
    assert a.tolist() == [0, 1, 2, 3, 4, 5]


def test_zero_d_empty_tuple_index_returns_scalar():
    z = np.array(5)
    assert z.shape == ()
    assert z.ndim == 0
    value = z[()]
    assert not isinstance(value, np.ndarray)
    assert isinstance(value, np.int64)
    assert value == 5


def test_zero_d_ellipsis_index_returns_zero_d_view():
    z = np.array(5)
    view = z[...]
    assert isinstance(view, np.ndarray)
    assert view.shape == ()
    view[...] = 7
    assert z[()] == 7


def test_zero_d_integer_index_raises_index_error():
    with pytest.raises(IndexError):
        np.array(5)[0]


def test_ellipsis_after_full_integer_index_returns_zero_d_array():
    a = np.arange(6.0).reshape(2, 3)
    assert isinstance(a[1, 2], np.float64)
    result = a[1, 2, ...]
    assert isinstance(result, np.ndarray)
    assert result.shape == ()
    assert result[()] == 5.0


def test_item_returns_python_scalar():
    a = np.arange(6).reshape(2, 3)
    assert a.item(4) == 4
    assert type(a.item(4)) is int
    assert a.item(1, 2) == 5
    assert type(np.array(2.5).item()) is float


def test_flat_iterates_in_c_order_of_logical_layout():
    a = np.arange(6).reshape(2, 3).T
    assert [int(v) for v in a.flat] == [0, 3, 1, 4, 2, 5]


def test_flat_indexing_reads_by_flat_position():
    a = np.arange(6).reshape(2, 3) * 10
    assert a.flat[4] == 40
    assert a.flat[-1] == 50
    assert a.flat[[1, 5]].tolist() == [10, 50]
    assert a.flat[1:4].tolist() == [10, 20, 30]


def test_flat_assignment_writes_by_flat_position():
    a = np.zeros((2, 3), dtype=np.int64)
    a.flat[[0, 5]] = -1
    a.flat[2] = 9
    assert a.tolist() == [[-1, 0, 9], [0, 0, -1]]


def test_iteration_yields_first_axis_subarrays():
    a = np.arange(6).reshape(3, 2)
    assert len(a) == 3
    assert [row.tolist() for row in a] == [[0, 1], [2, 3], [4, 5]]


def test_iteration_over_1d_yields_numpy_scalars():
    values = list(np.array([1.5, 2.5]))
    assert values == [1.5, 2.5]
    assert all(isinstance(v, np.float64) for v in values)


def test_zero_d_array_has_no_len_or_iteration():
    z = np.array(1)
    with pytest.raises(TypeError):
        len(z)
    with pytest.raises(TypeError):
        iter(z)


def test_take_uses_flat_positions_without_axis():
    a = np.arange(12).reshape(3, 4)
    assert np.take(a, [0, 5, 11]).tolist() == [0, 5, 11]


def test_take_along_axis_argument():
    a = np.arange(12).reshape(3, 4)
    assert np.take(a, [3, 0], axis=1).tolist() == [[3, 0], [7, 4], [11, 8]]
    assert a.take([2], axis=0).tolist() == [[8, 9, 10, 11]]


def test_put_writes_flat_positions_and_cycles_values():
    a = np.zeros(5, dtype=np.int64)
    np.put(a, [0, 3, -1], [10, 30, 40])
    assert a.tolist() == [10, 0, 0, 30, 40]
    b = np.zeros(4, dtype=np.int64)
    b.put([0, 1, 2, 3], [1, 2])
    assert b.tolist() == [1, 2, 1, 2]


def test_where_with_condition_only_returns_index_tuple():
    a = np.array([[0, 3], [5, 0]])
    rows, cols = np.where(a > 0)
    assert rows.tolist() == [0, 1]
    assert cols.tolist() == [1, 0]
    assert a[np.where(a > 0)].tolist() == [3, 5]


def test_broadcast_to_returns_read_only_view():
    a = np.arange(3)
    b = np.broadcast_to(a, (2, 3))
    assert b.tolist() == [[0, 1, 2], [0, 1, 2]]
    assert b.flags.writeable is False
    assert np.shares_memory(a, b)
    a[0] = 9
    assert b[1, 0] == 9


def test_assigning_into_broadcast_view_raises_value_error():
    b = np.broadcast_to(np.arange(3), (2, 3))
    with pytest.raises(ValueError) as info:
        b[0, 0] = 5
    assert str(info.value) == "assignment destination is read-only"


def test_overlapping_forward_shift_uses_original_values():
    a = np.arange(6)
    a[1:] = a[:-1]
    assert a.tolist() == [0, 0, 1, 2, 3, 4]


def test_overlapping_backward_shift_uses_original_values():
    a = np.arange(6)
    a[:-1] = a[1:]
    assert a.tolist() == [1, 2, 3, 4, 5, 5]


def test_assigning_reversed_self_reverses():
    a = np.arange(5)
    a[:] = a[::-1]
    assert a.tolist() == [4, 3, 2, 1, 0]


def test_in_place_add_of_reversed_self_uses_original_values():
    a = np.array([1, 2, 4, 8, 16])
    a += a[::-1]
    assert a.tolist() == [17, 10, 8, 10, 17]


def test_complex_real_and_imag_read_components():
    z = np.array([1 + 2j, 3 - 4j])
    assert z.real.dtype == np.dtype("float64")
    assert z.real.tolist() == [1.0, 3.0]
    assert z.imag.tolist() == [2.0, -4.0]
    assert np.array([1 + 2j], dtype=np.complex64).imag.dtype == np.dtype("float32")


def test_complex_real_and_imag_are_writable_views():
    z = np.array([1 + 2j, 3 - 4j])
    assert np.shares_memory(z, z.imag)
    z.imag[:] = 0
    assert z.tolist() == [1 + 0j, 3 + 0j]
    z.real[1] = 5
    assert z.tolist() == [1 + 0j, 5 + 0j]
