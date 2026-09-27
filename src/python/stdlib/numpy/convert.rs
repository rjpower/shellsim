//! Conversion between Python values and arrays, and casts between dtypes.
//!
//! `np.array` discovers the shape breadth-first: a level is an array axis when every node is a
//! list, tuple, or ndarray and all have one length. Nodes that disagree are *ragged*: an error
//! for typed arrays, and the leaf level for `dtype=object`, as in NumPy. Leaves infer a dtype
//! by promotion (Python `int` → `int64`, `float` → `float64`, `str` → `<U{n}`, anything else →
//! `object`), then each leaf is written with the target dtype's conversion rules. Python ints
//! that do not fit an integer dtype raise NumPy's `OverflowError`.

use super::super::super::native::{
    PyArrayBuffer, PyArrayData, PyArrayDtype, PyError, PyKind, PyNativeKind, PyResult, PyRuntime,
    PyValue, PyValueCast,
};
use super::super::super::protocol::quote_string;
use super::array::{
    buffer_with_capacity, contiguous_buffer, element_count, gather_into, new_array,
    reserve_elements, Array,
};
use super::dtype::{self, Category, DType, Kind};
use super::element::{self, Number};
use super::scalar;

/// Python ints are exact up to this width; wider values only fit `object` arrays.
type WideInt = i128;

/// One scalar leaf seen while converting a Python value.
#[derive(Clone, Debug)]
pub(in crate::python) enum Leaf {
    Bool(bool),
    Int(WideInt),
    /// A Python int outside `i128`, kept as decimal text for error messages.
    BigInt(String),
    Float(f64),
    Complex(f64, f64),
    NumPy(DType, Number),
    Str(String),
    Object,
}

/// Classify one non-sequence Python value.
pub(in crate::python) fn leaf(runtime: &dyn PyRuntime, value: &PyValue) -> PyResult<Leaf> {
    if let Some((dtype, number)) = scalar::unbox_number(runtime, value) {
        return Ok(Leaf::NumPy(dtype, number));
    }
    Ok(match runtime.kind(value)? {
        PyKind::Bool => Leaf::Bool(value.bool_value().unwrap_or(false)),
        PyKind::Int => match runtime.int_value(value) {
            Some(value) => Leaf::Int(WideInt::from(value)),
            None => {
                let text = runtime
                    .integer_text(value)?
                    .ok_or_else(|| PyError::runtime_error("integer lost its value"))?;
                text.parse::<WideInt>()
                    .map(Leaf::Int)
                    .unwrap_or(Leaf::BigInt(text))
            }
        },
        PyKind::Float => Leaf::Float(value.float_value().unwrap_or(f64::NAN)),
        PyKind::Complex => match runtime.number(value) {
            Some(super::super::super::number::NumberRef::Complex(real, imag)) => {
                Leaf::Complex(real, imag)
            }
            _ => Leaf::Object,
        },
        PyKind::String => Leaf::Str(runtime.string_value(value)?.unwrap_or_default()),
        _ => Leaf::Object,
    })
}

impl Leaf {
    /// The dtype NumPy infers for this leaf on its own.
    fn dtype(&self) -> PyResult<DType> {
        Ok(match self {
            Self::Bool(_) => DType::BOOL,
            Self::Int(value) if i64::try_from(*value).is_ok() => DType::INT64,
            Self::Int(value) if u64::try_from(*value).is_ok() => DType::UINT64,
            Self::Int(_) | Self::BigInt(_) | Self::Object => DType::OBJECT,
            Self::Float(_) => DType::FLOAT64,
            Self::Complex(..) => DType::COMPLEX128,
            Self::NumPy(dtype, _) => *dtype,
            Self::Str(text) => DType::str(text.chars().count())?,
        })
    }

    /// The weak category of a Python scalar operand, or `None` for strong and non-numeric
    /// values.
    pub(in crate::python) fn weak(&self) -> Option<dtype::Weak> {
        match self {
            Self::Bool(_) => Some(dtype::Weak::Bool),
            Self::Int(_) | Self::BigInt(_) => Some(dtype::Weak::Int),
            Self::Float(_) => Some(dtype::Weak::Float),
            Self::Complex(..) => Some(dtype::Weak::Complex),
            _ => None,
        }
    }
}

/// Combine two inferred dtypes the way array construction does: strings absorb numbers at
/// their printed width instead of failing.
pub(in crate::python) fn infer_promote(left: DType, right: DType) -> PyResult<DType> {
    match (left.category(), right.category()) {
        (Category::Str, Category::Str) | (Category::Object, _) | (_, Category::Object) => {
            dtype::promote(left, right)
        }
        (Category::Str, _) => DType::str(left.chars().max(dtype::str_width_for(right))),
        (_, Category::Str) => DType::str(right.chars().max(dtype::str_width_for(left))),
        _ => dtype::promote(left, right),
    }
}

/// Nodes of one breadth-first level while discovering an array's shape.
enum Level {
    Values(Vec<PyValue>),
    /// Every node was an ndarray of the same shape; their elements finish the shape.
    Blocks(Vec<Array>),
}

fn sequence_items(runtime: &mut dyn PyRuntime, value: &PyValue) -> PyResult<Option<Vec<PyValue>>> {
    match runtime.kind(value)? {
        PyKind::List => {
            let list = value.cast(runtime)?;
            runtime.list_items(list).map(Some)
        }
        PyKind::Tuple => {
            let tuple = value.cast(runtime)?;
            runtime.tuple_items(tuple).map(Some)
        }
        _ if runtime.type_name(value)? == "range" => {
            let iterator = runtime.iterator(*value)?;
            let mut items = Vec::new();
            while let Some(item) = runtime.iterator_next(iterator)? {
                runtime.charge_cpu(1)?;
                items.push(item);
            }
            Ok(Some(items))
        }
        _ => Ok(None),
    }
}

fn inhomogeneous(shape: &[usize], depth: usize) -> PyError {
    PyError::value_error(format!(
        "setting an array element with a sequence. The requested array has an inhomogeneous \
         shape after {depth} dimensions. The detected shape was {} + inhomogeneous part.",
        super::array::format_shape(shape)
    ))
}

/// Discover the shape and leaves of a nested Python value.
fn discover(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
    object_dtype: bool,
) -> PyResult<(Vec<usize>, Level)> {
    let mut shape = Vec::new();
    let mut nodes = vec![value];
    loop {
        let arrays = nodes
            .iter()
            .map(|node| {
                (runtime.native_kind(node).ok().flatten() == Some(PyNativeKind::Array))
                    .then(|| Array::from_value(runtime, *node))
                    .transpose()
            })
            .collect::<PyResult<Vec<_>>>()?;
        if !nodes.is_empty() && arrays.iter().all(Option::is_some) {
            let blocks = arrays.into_iter().map(Option::unwrap).collect::<Vec<_>>();
            if blocks
                .iter()
                .all(|block| block.shape() == blocks[0].shape())
            {
                shape.extend_from_slice(blocks[0].shape());
                super::array::element_count(&shape)?;
                return Ok((shape, Level::Blocks(blocks)));
            }
        }
        let mut children = Vec::new();
        let mut length = None;
        let mut sequences = 0usize;
        for node in &nodes {
            runtime.charge_cpu(1)?;
            let items = if runtime.native_kind(node)? == Some(PyNativeKind::Array) {
                let array = Array::from_value(runtime, *node)?;
                if array.ndim() == 0 {
                    None
                } else {
                    Some(array_rows(runtime, &array)?)
                }
            } else {
                sequence_items(runtime, node)?
            };
            let Some(items) = items else {
                continue;
            };
            sequences += 1;
            if *length.get_or_insert(items.len()) != items.len() {
                length = Some(usize::MAX);
            }
            runtime.reserve_memory(items.len().saturating_mul(16))?;
            children.extend(items);
        }
        if sequences == 0 {
            return Ok((shape, Level::Values(nodes)));
        }
        if sequences != nodes.len() || length == Some(usize::MAX) {
            if object_dtype {
                return Ok((shape, Level::Values(nodes)));
            }
            return Err(inhomogeneous(&shape, shape.len()));
        }
        shape.push(length.unwrap_or(0));
        super::array::element_count(&shape)?;
        if children.is_empty() {
            return Ok((shape, Level::Values(Vec::new())));
        }
        nodes = children;
    }
}

/// Split an array into Python values along its first axis: sub-array views, or scalars for
/// 1-d arrays.
fn array_rows(runtime: &mut dyn PyRuntime, array: &Array) -> PyResult<Vec<PyValue>> {
    let length = array.shape()[0];
    let mut rows = Vec::with_capacity(length);
    for index in 0..length {
        let offset = (array.view.offset as isize + index as isize * array.strides()[0]) as usize;
        if array.ndim() == 1 {
            rows.push(element_to_scalar(runtime, array, offset)?);
        } else {
            let view = super::array::new_view(
                runtime,
                array,
                array.dtype,
                array.shape()[1..].to_vec(),
                array.strides()[1..].to_vec(),
                offset,
            )?;
            rows.push(view.value());
        }
    }
    Ok(rows)
}

/// `np.array(value, dtype=dtype)`. An ndarray input is copied, or returned unchanged when
/// `copy` is false and no cast is needed.
pub(in crate::python) fn array_from_python(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
    dtype: Option<DType>,
    copy: bool,
) -> PyResult<Array> {
    if runtime.native_kind(&value)? == Some(PyNativeKind::Array) {
        let array = Array::from_value(runtime, value)?;
        let target = dtype.unwrap_or(array.dtype);
        return cast_array(runtime, &array, target, copy);
    }
    let object_dtype = dtype.is_some_and(|dtype| dtype.kind() == Kind::Object);
    let (shape, level) = discover(runtime, value, object_dtype)?;
    match level {
        Level::Blocks(blocks) => {
            let target = match dtype {
                Some(dtype) => dtype,
                None => blocks
                    .iter()
                    .skip(1)
                    .try_fold(blocks[0].dtype, |acc, block| {
                        infer_promote(acc, block.dtype)
                    })?,
            };
            let target = widen_unsized_str(runtime, target, &blocks)?;
            let count = element_count(&shape)?;
            reserve_elements(runtime, target, count)?;
            let mut output = buffer_with_capacity(target, count);
            for block in &blocks {
                let block = cast_array(runtime, block, target, false)?;
                runtime.read_arrays(&[block.handle], &mut |arrays| {
                    gather_into(&arrays[0], block.offsets(), &mut output);
                    Ok(())
                })?;
            }
            new_array(runtime, output, target, shape)
        }
        Level::Values(values) => {
            let leaves = values
                .iter()
                .map(|value| leaf(runtime, value))
                .collect::<PyResult<Vec<_>>>()?;
            let target = match dtype {
                Some(dtype) if dtype.kind() == Kind::Str && dtype.chars() == 0 => {
                    let mut widest = 1;
                    for (leaf, value) in leaves.iter().zip(&values) {
                        widest = widest.max(leaf_text(runtime, leaf, value)?.chars().count());
                    }
                    DType::str(widest)?
                }
                Some(dtype) => dtype,
                None if leaves.is_empty() => DType::FLOAT64,
                None => {
                    let mut target = leaves[0].dtype()?;
                    for leaf in &leaves[1..] {
                        target = infer_promote(target, leaf.dtype()?)?;
                    }
                    if target.kind() == Kind::Str && target.chars() == 0 {
                        DType::str(1)?
                    } else {
                        target
                    }
                }
            };
            let buffer = write_leaves(runtime, &values, &leaves, target)?;
            new_array(runtime, buffer, target, shape)
        }
    }
}

/// `np.array(arrays, dtype=str)` sizes the strings from the widest block.
fn widen_unsized_str(
    runtime: &mut dyn PyRuntime,
    target: DType,
    blocks: &[Array],
) -> PyResult<DType> {
    if target.kind() != Kind::Str || target.chars() != 0 {
        return Ok(target);
    }
    let mut widest = 1;
    for block in blocks {
        widest = widest.max(cast_array(runtime, block, target, false)?.dtype.chars());
    }
    DType::str(widest)
}

/// `np.asarray(value)`: the array itself, or a new array.
pub(in crate::python) fn as_array(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<Array> {
    array_from_python(runtime, value, None, false)
}

fn overflow(value: &dyn std::fmt::Display, dtype: DType) -> PyError {
    PyError::overflow_error(format!(
        "Python integer {value} out of bounds for {}",
        dtype.kind().name()
    ))
}

/// Check that a Python int fits an integer dtype, as NumPy 2 does for weak ints.
pub(in crate::python) fn checked_int(value: WideInt, dtype: DType) -> PyResult<Number> {
    let bits = dtype.kind().bits();
    let fits = match dtype.category() {
        Category::Signed => {
            let limit = 1i128 << (bits - 1);
            (-limit..limit).contains(&value)
        }
        Category::Unsigned => (0..(1i128 << bits)).contains(&value),
        _ => true,
    };
    if !fits {
        return Err(overflow(&value, dtype));
    }
    Ok(match u64::try_from(value) {
        Ok(value) if value > i64::MAX as u64 => Number::UInt(value),
        _ => i64::try_from(value)
            .map(Number::Int)
            .unwrap_or(Number::Float(value as f64)),
    })
}

/// The number a leaf stands for when stored into a numeric dtype.
pub(in crate::python) fn leaf_number(leaf: &Leaf, target: DType) -> PyResult<Number> {
    let integer_target = target.is_integer();
    Ok(match leaf {
        Leaf::Bool(value) => Number::Bool(*value),
        Leaf::Int(value) if integer_target => checked_int(*value, target)?,
        Leaf::Int(value) => match i64::try_from(*value) {
            Ok(value) => Number::Int(value),
            Err(_) => Number::Float(*value as f64),
        },
        Leaf::BigInt(text) if integer_target => return Err(overflow(text, target)),
        Leaf::BigInt(text) => Number::Float(text.parse::<f64>().unwrap_or(f64::INFINITY)),
        Leaf::Float(value) if integer_target => {
            if value.is_nan() {
                return Err(PyError::value_error("cannot convert float NaN to integer"));
            }
            if value.is_infinite() {
                return Err(PyError::overflow_error(
                    "cannot convert float infinity to integer",
                ));
            }
            Number::Float(*value)
        }
        Leaf::Float(value) => Number::Float(*value),
        Leaf::Complex(real, imag) => {
            if !matches!(target.category(), Category::Complex | Category::Bool) {
                return Err(PyError::type_error(format!(
                    "{}() argument must be a string or a real number, not 'complex'",
                    if integer_target { "int" } else { "float" }
                )));
            }
            Number::Complex(*real, *imag)
        }
        Leaf::NumPy(_, number) => *number,
        Leaf::Str(text) => parse_number(text, target)?,
        Leaf::Object => {
            return Err(PyError::type_error(format!(
                "{}() argument must be a string or a real number",
                if integer_target { "int" } else { "float" }
            )))
        }
    })
}

/// Parse text as NumPy does when casting strings to numbers.
pub(in crate::python) fn parse_number(text: &str, target: DType) -> PyResult<Number> {
    let trimmed = text.trim();
    match target.category() {
        Category::Bool => Ok(Number::Bool(!text.is_empty())),
        Category::Signed | Category::Unsigned => {
            let cleaned = trimmed.replace('_', "");
            let value = cleaned.parse::<WideInt>().map_err(|_| {
                PyError::value_error(format!(
                    "invalid literal for int() with base 10: {}",
                    quote_string(text)
                ))
            })?;
            checked_int(value, target)
        }
        Category::Complex => {
            let parsed = parse_float(trimmed)
                .map_err(|_| PyError::value_error("complex() arg is a malformed string"))?;
            Ok(Number::Complex(parsed, 0.0))
        }
        _ => parse_float(trimmed).map(Number::Float).map_err(|_| {
            PyError::value_error(format!(
                "could not convert string to float: {}",
                quote_string(text)
            ))
        }),
    }
}

fn parse_float(text: &str) -> Result<f64, ()> {
    let lower = text.to_ascii_lowercase();
    match lower.trim_start_matches(['+', '-']) {
        "inf" | "infinity" | "nan" => {}
        body if body
            .chars()
            .all(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | '+' | '-' | '_')) => {}
        _ => return Err(()),
    }
    lower.replace('_', "").parse::<f64>().map_err(|_| ())
}

/// Text a leaf becomes in a string array: Python `str()` of the value.
fn leaf_text(runtime: &mut dyn PyRuntime, leaf: &Leaf, value: &PyValue) -> PyResult<String> {
    Ok(match leaf {
        Leaf::Bool(value) => if *value { "True" } else { "False" }.to_string(),
        Leaf::Int(value) => value.to_string(),
        Leaf::BigInt(text) => text.clone(),
        Leaf::Float(value) => super::format::float_repr(*value, super::format::Precision::Double),
        Leaf::Complex(real, imag) => {
            super::format::complex_repr(*real, *imag, super::format::Precision::Double)
        }
        Leaf::NumPy(dtype, number) => scalar::scalar_str(*dtype, *number),
        Leaf::Str(text) => text.clone(),
        Leaf::Object => runtime.display(value)?,
    })
}

/// Write UCS-4 code points of `text`, truncated or zero-padded to `chars`.
pub(in crate::python) fn write_str(text: &str, chars: usize, output: &mut Vec<u8>) {
    let mut written = 0;
    for character in text.chars().take(chars) {
        output.extend_from_slice(&u32::from(character).to_le_bytes());
        written += 1;
    }
    output.resize(output.len() + (chars - written) * 4, 0);
}

/// Decode one UCS-4 element, dropping trailing NULs as NumPy does.
pub(in crate::python) fn read_str(bytes: &[u8]) -> String {
    let mut text: String = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .map(|code| char::from_u32(code).unwrap_or('\u{fffd}'))
        .collect();
    let trimmed = text.trim_end_matches('\0').len();
    text.truncate(trimmed);
    text
}

/// Write converted leaves into new storage of `target`.
fn write_leaves(
    runtime: &mut dyn PyRuntime,
    values: &[PyValue],
    leaves: &[Leaf],
    target: DType,
) -> PyResult<PyArrayBuffer> {
    reserve_elements(runtime, target, leaves.len())?;
    runtime.charge_cpu(leaves.len() as u64 + 1)?;
    Ok(match target.kind() {
        Kind::Object => PyArrayBuffer::Values(values.to_vec()),
        Kind::Str => {
            let mut bytes = Vec::with_capacity(leaves.len() * target.itemsize());
            for (leaf, value) in leaves.iter().zip(values) {
                let text = leaf_text(runtime, leaf, value)?;
                write_str(&text, target.chars(), &mut bytes);
            }
            PyArrayBuffer::Bytes(bytes)
        }
        kind => {
            let itemsize = target.itemsize();
            let mut bytes = vec![0u8; leaves.len() * itemsize];
            for (leaf, chunk) in leaves.iter().zip(bytes.chunks_exact_mut(itemsize)) {
                element::write_number(kind, leaf_number(leaf, target)?, chunk);
            }
            PyArrayBuffer::Bytes(bytes)
        }
    })
}

/// Encode one Python value as an element of `target`, for item assignment and `fill`.
pub(in crate::python) fn value_to_buffer(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
    target: DType,
) -> PyResult<PyArrayBuffer> {
    let leaf = leaf(runtime, &value)?;
    write_leaves(runtime, &[value], &[leaf], target)
}

/// Box the element at byte `offset` of `array` as the value indexing returns: a NumPy scalar
/// for numbers, `str` for strings, and the stored object for `object` arrays.
pub(in crate::python) fn element_to_scalar(
    runtime: &mut dyn PyRuntime,
    array: &Array,
    offset: usize,
) -> PyResult<PyValue> {
    match read_element(runtime, array, offset)? {
        Element::Number(bytes) => scalar::box_bytes(runtime, array.dtype, &bytes),
        Element::Str(text) => runtime.new_string(text),
        Element::Object(value) => Ok(value),
    }
}

/// The builtin Python value `item()` and `tolist()` return for the element at `offset`.
pub(in crate::python) fn element_to_python(
    runtime: &mut dyn PyRuntime,
    array: &Array,
    offset: usize,
) -> PyResult<PyValue> {
    match read_element(runtime, array, offset)? {
        Element::Number(bytes) => {
            let number = element::read_number(array.dtype.kind(), &bytes);
            scalar::number_to_python(runtime, number)
        }
        Element::Str(text) => runtime.new_string(text),
        Element::Object(value) => Ok(value),
    }
}

/// One element copied out of array storage.
pub(in crate::python) enum Element {
    Number([u8; 16]),
    Str(String),
    Object(PyValue),
}

pub(in crate::python) fn read_element(
    runtime: &mut dyn PyRuntime,
    array: &Array,
    offset: usize,
) -> PyResult<Element> {
    let itemsize = array.itemsize();
    let kind = array.dtype.kind();
    let mut element = None;
    runtime.read_arrays(&[array.handle], &mut |arrays| {
        element = Some(match &arrays[0].data {
            PyArrayData::Values(values) => {
                Element::Object(values[offset / PyArrayDtype::VALUE_ITEMSIZE])
            }
            PyArrayData::Bytes(bytes) if kind == Kind::Str => {
                Element::Str(read_str(&bytes[offset..offset + itemsize]))
            }
            PyArrayData::Bytes(bytes) => {
                let mut copy = [0u8; 16];
                copy[..itemsize].copy_from_slice(&bytes[offset..offset + itemsize]);
                Element::Number(copy)
            }
        });
        Ok(())
    })?;
    Ok(element.expect("read callback ran"))
}

/// `array.astype(target)`: a converted copy, or `array` itself when no conversion or copy is
/// needed. Casting is NumPy's `unsafe` rule: integers wrap, floats truncate toward zero.
pub(in crate::python) fn cast_array(
    runtime: &mut dyn PyRuntime,
    array: &Array,
    target: DType,
    copy: bool,
) -> PyResult<Array> {
    let target =
        if target.kind() == Kind::Str && target.chars() == 0 && array.dtype.kind() != Kind::Str {
            DType::str(dtype::str_width_for(array.dtype).max(1))?
        } else if target.kind() == Kind::Str && target.chars() == 0 {
            array.dtype
        } else {
            target
        };
    if target == array.dtype {
        return if copy {
            super::array::copy_array(runtime, array)
        } else {
            Ok(array.clone())
        };
    }
    let buffer = cast_buffer(runtime, array, target)?;
    new_array(runtime, buffer, target, array.shape().to_vec())
}

/// Convert every element of `array`, in C order, into new storage of `target`.
fn cast_buffer(
    runtime: &mut dyn PyRuntime,
    array: &Array,
    target: DType,
) -> PyResult<PyArrayBuffer> {
    let count = array.size();
    let source = array.dtype;
    if source.is_numeric() && target.is_numeric() {
        reserve_elements(runtime, target, count)?;
        runtime.charge_cpu(count as u64 + 1)?;
        let mut output = vec![0u8; count * target.itemsize()];
        let (from, to, size) = (source.kind(), target.kind(), target.itemsize());
        runtime.read_arrays(&[array.handle], &mut |arrays| {
            let PyArrayData::Bytes(bytes) = arrays[0].data else {
                return Err(PyError::runtime_error("numeric array has object storage"));
            };
            for (offset, chunk) in array.offsets().zip(output.chunks_exact_mut(size)) {
                element::write_number(to, element::read_number(from, &bytes[offset..]), chunk);
            }
            Ok(())
        })?;
        return Ok(PyArrayBuffer::Bytes(output));
    }
    // Strings and objects go through Python values one element at a time.
    let contiguous = contiguous_buffer(runtime, array)?;
    let values = match &contiguous {
        PyArrayBuffer::Values(values) => values.clone(),
        PyArrayBuffer::Bytes(bytes) => {
            let mut values = Vec::with_capacity(count);
            runtime.reserve_memory(count.saturating_mul(24))?;
            for chunk in bytes.chunks_exact(source.itemsize().max(1)).take(count) {
                values.push(if source.kind() == Kind::Str {
                    runtime.new_string(read_str(chunk))?
                } else if target.kind() == Kind::Object {
                    scalar::number_to_python(runtime, element::read_number(source.kind(), chunk))?
                } else {
                    scalar::box_bytes(runtime, source, chunk)?
                });
            }
            if source.itemsize() == 0 {
                for _ in 0..count {
                    values.push(runtime.new_string(String::new())?);
                }
            }
            values
        }
    };
    if source.kind() == Kind::Object && target.kind() == Kind::Bool {
        // NumPy's `BOOL_setitem` takes the truth of any object, so `None` and `[]` are False.
        reserve_elements(runtime, target, count)?;
        runtime.charge_cpu(count as u64 + 1)?;
        let mut bytes = Vec::with_capacity(count);
        for value in &values {
            bytes.push(u8::from(runtime.truth(value)?));
        }
        return Ok(PyArrayBuffer::Bytes(bytes));
    }
    let leaves = values
        .iter()
        .map(|value| leaf(runtime, value))
        .collect::<PyResult<Vec<_>>>()?;
    write_leaves(runtime, &values, &leaves, target)
}

/// Whether `value` is a Python number usable as a weak scalar, and which kind.
pub(in crate::python) fn weak_scalar(
    runtime: &dyn PyRuntime,
    value: &PyValue,
) -> PyResult<Option<(dtype::Weak, Leaf)>> {
    if scalar::unbox(runtime, value).is_some() {
        return Ok(None);
    }
    match runtime.kind(value)? {
        PyKind::Bool | PyKind::Int | PyKind::Float | PyKind::Complex => {
            let leaf = leaf(runtime, value)?;
            Ok(leaf.weak().map(|weak| (weak, leaf)))
        }
        _ => Ok(None),
    }
}

/// A weak Python scalar as a 0-d array of the dtype chosen by promotion, checking that Python
/// ints fit an integer target.
pub(in crate::python) fn weak_array(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
    leaf: &Leaf,
    target: DType,
) -> PyResult<Array> {
    let buffer = write_leaves(runtime, &[value], std::slice::from_ref(leaf), target)?;
    new_array(runtime, buffer, target, Vec::new())
}
