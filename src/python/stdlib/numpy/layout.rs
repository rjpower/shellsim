//! Memory order of new arrays: NumPy's `order` argument, and arrays laid out in C order,
//! Fortran order, or a prototype's stride order.
//!
//! A layout is a permutation of axes from the slowest-varying to the fastest, so C order is
//! the identity and Fortran order its reverse. A buffer for a layout holds the elements in that
//! memory order, which is C order over the permuted axes. As in NumPy, a newly allocated array
//! with no elements has all-zero strides.

use super::super::super::native::{PyArrayBuffer, PyError, PyResult, PyRuntime, PyValue};
use super::array::{self, Array};
use super::dtype::DType;

/// NumPy's `order` argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Order {
    C,
    F,
    /// Fortran order when the prototype is Fortran- but not C-contiguous, otherwise C order.
    A,
    /// The prototype's memory order, as closely as possible.
    K,
}

impl Order {
    /// Parse `order` as `PyArray_OrderConverter` does: one letter in either case, with a
    /// missing or `None` value leaving the caller's `default`.
    pub(super) fn parse(
        runtime: &mut dyn PyRuntime,
        value: Option<PyValue>,
        default: Order,
    ) -> PyResult<Order> {
        let Some(value) = value.filter(|value| !value.is_none()) else {
            return Ok(default);
        };
        let Some(text) = runtime.string_value(&value)? else {
            return Err(PyError::type_error(format!(
                "order must be str, not {}",
                runtime.type_name(&value)?
            )));
        };
        Ok(match text.as_str() {
            "C" | "c" => Order::C,
            "F" | "f" => Order::F,
            "A" | "a" => Order::A,
            "K" | "k" => Order::K,
            _ => {
                return Err(PyError::value_error(format!(
                    "order must be one of 'C', 'F', 'A', or 'K' (got '{text}')"
                )))
            }
        })
    }

    /// Parse the `order` of a constructor without a prototype, which permits only C and F.
    pub(super) fn parse_new(
        runtime: &mut dyn PyRuntime,
        value: Option<PyValue>,
    ) -> PyResult<Order> {
        match Order::parse(runtime, value, Order::C)? {
            Order::A | Order::K => Err(PyError::value_error("only 'C' or 'F' order is permitted")),
            order => Ok(order),
        }
    }
}

/// Whether `array` is Fortran-contiguous: C-contiguous with its axes reversed.
pub(super) fn is_f_contiguous(array: &Array) -> bool {
    let shape = array.shape().iter().rev().copied().collect::<Vec<_>>();
    let strides = array.strides().iter().rev().copied().collect::<Vec<_>>();
    array::is_c_contiguous(&shape, &strides, array.itemsize())
}

/// NumPy's `PyArray_ISFORTRAN`: Fortran-contiguous but not C-contiguous.
pub(super) fn is_fortran(array: &Array) -> bool {
    is_f_contiguous(array) && !array.is_c_contiguous()
}

/// Whether `array` already has the layout `order` asks for, so a request that copies only
/// when needed can return it unchanged (NumPy's `STRIDING_OK`).
pub(super) fn satisfies(array: &Array, order: Order) -> bool {
    match order {
        Order::C => array.is_c_contiguous(),
        Order::F => is_f_contiguous(array),
        Order::A | Order::K => true,
    }
}

/// The layout of a new rank-`ndim` array in C or Fortran order.
pub(super) fn axes(order: Order, ndim: usize) -> Vec<usize> {
    match order {
        Order::F => (0..ndim).rev().collect(),
        Order::C | Order::A | Order::K => (0..ndim).collect(),
    }
}

/// The layout `PyArray_NewLikeArrayWithShape` picks for a new rank-`ndim` array modeled on
/// `prototype`. `A` follows the prototype's Fortran order. `K` keeps a C- or Fortran-contiguous
/// prototype's order and otherwise sorts its axes by decreasing absolute stride, ties in axis
/// order; it means C order when the ranks differ.
pub(super) fn axes_like(prototype: &Array, order: Order, ndim: usize) -> Vec<usize> {
    let order = match order {
        Order::A if is_fortran(prototype) => Order::F,
        Order::A => Order::C,
        Order::K if ndim != prototype.ndim() || ndim <= 1 || prototype.is_c_contiguous() => {
            Order::C
        }
        Order::K if is_f_contiguous(prototype) => Order::F,
        order => order,
    };
    if order != Order::K {
        return axes(order, ndim);
    }
    let mut axes = (0..ndim).collect::<Vec<_>>();
    axes.sort_by_key(|axis| std::cmp::Reverse(prototype.strides()[*axis].unsigned_abs()));
    axes
}

/// The memory order NumPy's iterator (`npyiter_find_best_axis_ordering`) picks for an
/// allocated output of `shape` when `order='K'`. `strides` holds each array operand's strides
/// broadcast to `shape`. Starting from C order, a stable insertion sort moves each axis
/// outward while the operands agree that its strides are larger. Axes of length one or with
/// zero stride carry no information, and a conflict between operands keeps C order.
pub(super) fn iteration_axes(strides: &[Vec<isize>], shape: &[usize]) -> Vec<usize> {
    let stride = |operand: &[isize], axis: usize| {
        if shape[axis] == 1 {
            0
        } else {
            operand[axis]
        }
    };
    // `order` lists axes from the fastest-varying to the slowest while sorting.
    let mut order = (0..shape.len()).rev().collect::<Vec<_>>();
    for position in 1..order.len() {
        let axis = order[position];
        let mut insert = position;
        for earlier in (0..position).rev() {
            let other = order[earlier];
            let mut ambiguous = true;
            let mut swap = false;
            for operand in strides {
                let (mine, theirs) = (stride(operand, axis), stride(operand, other));
                if mine != 0 && theirs != 0 {
                    // Any operand that keeps the current order wins over one that would swap.
                    if theirs.unsigned_abs() <= mine.unsigned_abs() {
                        swap = false;
                    } else if ambiguous {
                        swap = true;
                    }
                    ambiguous = false;
                }
            }
            if !ambiguous {
                if !swap {
                    break;
                }
                insert = earlier;
            }
        }
        if insert != position {
            order.remove(position);
            order.insert(insert, axis);
        }
    }
    order.reverse();
    order
}

/// `array` broadcast to `shape` and read in `axes` order; see [`reading_order`].
pub(super) fn broadcast_reading_order(
    array: &Array,
    shape: &[usize],
    axes: &[usize],
) -> PyResult<Array> {
    let mut broadcast = array.clone();
    broadcast.view.strides = array::broadcast_strides(&array.view, shape)?;
    broadcast.view.shape = shape.to_vec();
    Ok(reading_order(&broadcast, axes))
}

/// Byte strides for `shape` laid out in `axes` order; all zero when there are no elements.
pub(super) fn strides(shape: &[usize], itemsize: usize, axes: &[usize]) -> Vec<isize> {
    let mut strides = vec![0isize; shape.len()];
    if shape.contains(&0) {
        return strides;
    }
    // Views are validated against the storage, whose size bounds every product here.
    let mut stride = itemsize;
    for axis in axes.iter().rev() {
        strides[*axis] = stride as isize;
        stride = stride.saturating_mul(shape[*axis]);
    }
    strides
}

/// Wrap `buffer`, which holds the elements in `axes` memory order, in a new array of `shape`.
pub(super) fn new_array(
    runtime: &mut dyn PyRuntime,
    buffer: PyArrayBuffer,
    dtype: DType,
    shape: Vec<usize>,
    axes: &[usize],
) -> PyResult<Array> {
    let strides = strides(&shape, dtype.itemsize(), axes);
    array::new_array_with_strides(runtime, buffer, dtype, shape, strides)
}

/// `array`'s storage read with its axes permuted to `axes`, so that gathering it in C order
/// visits the elements in the memory order of a new array laid out in `axes`. The result
/// shares `array`'s handle and serves only as a source to read from.
pub(super) fn reading_order(array: &Array, axes: &[usize]) -> Array {
    let mut permuted = array.clone();
    permuted.view.shape = axes.iter().map(|axis| array.shape()[*axis]).collect();
    permuted.view.strides = axes.iter().map(|axis| array.strides()[*axis]).collect();
    permuted
}

/// A copy of `array` laid out in `axes` order.
pub(super) fn copy(runtime: &mut dyn PyRuntime, array: &Array, axes: &[usize]) -> PyResult<Array> {
    let buffer = array::contiguous_buffer(runtime, &reading_order(array, axes))?;
    new_array(runtime, buffer, array.dtype, array.shape().to_vec(), axes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_follow_the_axis_permutation() {
        assert_eq!(strides(&[2, 3, 4], 8, &[0, 1, 2]), [96, 32, 8]);
        assert_eq!(strides(&[2, 3, 4], 8, &[2, 1, 0]), [8, 16, 48]);
        assert_eq!(strides(&[2, 3, 4], 1, &[1, 2, 0]), [1, 8, 2]);
        assert_eq!(strides(&[2, 0, 4], 8, &[0, 1, 2]), [0, 0, 0]);
    }

    #[test]
    fn iteration_axes_follow_numpy_operand_strides() {
        let fortran = vec![8, 16];
        let c_order = vec![24, 8];
        assert_eq!(
            iteration_axes(std::slice::from_ref(&fortran), &[2, 3]),
            [1, 0]
        );
        assert_eq!(
            iteration_axes(&[fortran.clone(), vec![0, 0]], &[2, 3]),
            [1, 0]
        );
        // Conflicting operands keep C order.
        assert_eq!(iteration_axes(&[fortran, c_order], &[2, 3]), [0, 1]);
        // `np.arange(24).reshape(2, 3, 4).transpose(1, 2, 0)`.
        assert_eq!(iteration_axes(&[vec![32, 8, 96]], &[3, 4, 2]), [2, 0, 1]);
        // Negative strides sort by magnitude, and length-one axes stay put.
        assert_eq!(iteration_axes(&[vec![24, -8]], &[2, 3]), [0, 1]);
        assert_eq!(iteration_axes(&[vec![8, 8]], &[1, 3]), [0, 1]);
    }
}
