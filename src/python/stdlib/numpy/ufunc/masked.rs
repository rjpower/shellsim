//! Ufunc calls with a `where=` mask.
//!
//! Only the selected elements are computed: each array operand's selected elements are
//! gathered into one run, the ufunc runs on those runs, and the results are scattered back.
//! Masked-off elements therefore raise no floating-point errors, as in NumPy, where
//! `np.divide(a, b, out=c, where=b != 0)` never divides by zero, and they keep their values in
//! `out=`. Without `out=`, NumPy leaves them uninitialized; here they are zero.
//!
//! The mask converts like NumPy's `PyArray_FromAny(where, bool)`: an array must already be
//! boolean, and other values convert element by element with Python truth.

use super::super::super::super::native::{PyError, PyNativeKind, PyResult, PyRuntime, PyValue};
use super::super::array::{self, broadcast_shapes, format_shape, new_array, Array};
use super::super::convert;
use super::super::dtype::{Casting, DType};
use super::super::index;
use super::super::select::broadcast_truth;
use super::{check_output_cast, evaluate as evaluate_all, Evaluated, Options, UFUNCS};

/// The mask as a boolean array.
fn mask_array<'s>(runtime: &mut dyn PyRuntime<'s>, mask: PyValue<'s>) -> PyResult<'s, Array<'s>> {
    if runtime.native_kind(&mask)? != Some(PyNativeKind::Array) {
        return convert::array_from_python(runtime, mask, Some(DType::BOOL), false);
    }
    let mask = Array::from_value(runtime, mask)?;
    if mask.dtype != DType::BOOL {
        return Err(PyError::type_error(format!(
            "Cannot cast array data from {} to {} according to the rule 'safe'",
            mask.dtype.repr(),
            DType::BOOL.repr()
        )));
    }
    Ok(mask)
}

/// Byte offsets of the elements of `array` where `truth`, which is in C order, holds.
fn selected_offsets<'s>(array: &Array<'s>, truth: &[bool]) -> Vec<usize> {
    array
        .offsets()
        .zip(truth)
        .filter_map(|(offset, selected)| selected.then_some(offset))
        .collect()
}

/// Run ufunc `index` on `inputs` where `mask` is true.
pub(super) fn evaluate<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    index: usize,
    inputs: &[PyValue<'s>],
    options: &Options<'s>,
    mask: PyValue<'s>,
) -> PyResult<'s, Evaluated<'s>> {
    let ufunc = &UFUNCS[index];
    let mask = mask_array(runtime, mask)?;
    // Python scalars stay as they are, so they promote as weak scalars in the inner call.
    let mut operands = Vec::with_capacity(inputs.len());
    for value in inputs {
        operands.push(match convert::weak_scalar(runtime, value)? {
            Some(_) => None,
            None => Some(convert::as_array(runtime, *value)?),
        });
    }
    let mut shapes = operands
        .iter()
        .map(|operand| operand.as_ref().map_or(&[][..], Array::shape))
        .collect::<Vec<_>>();
    // NumPy's iterator broadcasts the inputs, `out`, and the mask together.
    if let Some(out) = &options.out {
        shapes.push(out.shape());
    }
    shapes.push(mask.shape());
    let shape = broadcast_shapes(&shapes)?;
    if let Some(out) = &options.out {
        if out.shape() != shape.as_slice() {
            return Err(PyError::value_error(format!(
                "non-broadcastable output operand with shape {} doesn't match the broadcast \
                 shape {}",
                format_shape(out.shape()),
                format_shape(&shape)
            )));
        }
    }
    let truth = broadcast_truth(runtime, &mask, &shape)?;
    let count = truth.iter().filter(|selected| **selected).count();
    let mut gathered = Vec::with_capacity(inputs.len());
    for (value, operand) in inputs.iter().zip(&operands) {
        let Some(operand) = operand else {
            gathered.push(*value);
            continue;
        };
        let strides = array::broadcast_strides(&operand.view, &shape)?;
        let view = array::new_view(
            runtime,
            operand,
            operand.dtype,
            shape.clone(),
            strides,
            operand.view.offset,
        )?;
        let offsets = selected_offsets(&view, &truth);
        gathered.push(index::gather(runtime, &view, &offsets, vec![count])?.value());
    }
    let inner = Options {
        dtype: options.dtype,
        casting: options.casting,
        keep_array: true,
        ..Options::default()
    };
    let evaluated = evaluate_all(runtime, index, &gathered, &inner)?;
    let values = Array::from_value(runtime, evaluated.value)?;
    let target = match &options.out {
        Some(out) => {
            let casting = options.casting.unwrap_or(Casting::SameKind);
            check_output_cast(ufunc, values.dtype, out, casting)?;
            out.clone()
        }
        None => {
            let size = array::element_count(&shape)?;
            let buffer = array::zeroed_buffer(runtime, values.dtype, size)?;
            new_array(runtime, buffer, values.dtype, shape.clone())?
        }
    };
    let offsets = selected_offsets(&target, &truth);
    let buffer = array::broadcast_buffer(runtime, &values, target.dtype, &[count])?;
    array::scatter(runtime, &target, &offsets, &buffer)?;
    let value = if options.out.is_none() && target.ndim() == 0 && !options.keep_array {
        convert::element_to_scalar(runtime, &target, target.view.offset)?
    } else {
        target.value()
    };
    Ok(Evaluated {
        value,
        flags: evaluated.flags,
        scalar_math: false,
    })
}
