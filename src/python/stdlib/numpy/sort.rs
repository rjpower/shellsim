//! Sorting, searching, and counting: `sort`, `argsort`, `lexsort`, `partition`,
//! `argpartition`, `searchsorted`, and `bincount`. Set operations, `histogram`, `digitize`,
//! `median`, and the percentiles are Python on top of these, as in NumPy.
//!
//! Functions are exported through the native module `_numpy_sort`, which the frozen `numpy`
//! package re-exports.
//!
//! Every sort here is a stable merge sort of a permutation, whatever `kind` requests. NumPy's
//! `quicksort` and `heapsort` are not stable and its SIMD sorts depend on the CPU, so the order
//! it gives equal elements is unspecified, and the stable order is one NumPy may produce.
//! `partition` sorts each lane completely, which satisfies its contract: the kth element is in
//! sorted position, with no larger element before it and no smaller one after.
//!
//! The order is NumPy's `Tag::less`: NaN sorts after every number; complex values compare by
//! real part, then imaginary part, with values that have NaN parts last; strings compare by
//! code point; objects compare with Python `<`. `descending=True` reverses the order of the
//! non-NaN values and keeps NaN last, as NumPy does.

mod bincount;
mod search;

use std::cmp::Ordering;

use super::super::super::ast::ComparisonOperator;
use super::super::super::native::{
    CallArgs, FunctionDef, MethodDef, ModuleDef, NativeTypeDef, PyArrayBuffer, PyArrayData,
    PyError, PyKind, PyNativeKind, PyOperator, PyResult, PyRuntime, PyValue,
};
use super::super::super::Value;
use super::args::{self, Bound, Signature};
use super::array::{self, contiguous_strides, new_array, reserve_elements, Array, Offsets};
use super::convert;
use super::dtype::{Category, DType, Kind};
use super::element::{self, Number};

pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_numpy_sort",
    functions: FUNCTIONS,
    values: &[],
};

const fn function(
    name: &'static str,
    call: fn(&mut dyn PyRuntime, CallArgs) -> PyResult,
) -> FunctionDef {
    FunctionDef {
        module: "numpy",
        name,
        call,
    }
}

static FUNCTIONS: &[FunctionDef] = &[
    function("sort", module_sort),
    function("argsort", module_argsort),
    function("lexsort", module_lexsort),
    function("partition", module_partition),
    function("argpartition", module_argpartition),
    function("searchsorted", search::module_searchsorted),
    function("bincount", bincount::module_bincount),
];

const fn method(
    name: &'static str,
    call: fn(&mut dyn PyRuntime, PyValue, CallArgs) -> PyResult,
) -> MethodDef {
    MethodDef {
        type_name: "numpy.ndarray",
        name,
        call,
    }
}

/// Methods this area installs on `numpy.ndarray`.
pub(in crate::python) static ARRAY_METHODS: NativeTypeDef = NativeTypeDef {
    name: "numpy.ndarray",
    methods: &[
        method("sort", method_sort),
        method("argsort", method_argsort),
        method("partition", method_partition),
        method("argpartition", method_argpartition),
        method("searchsorted", search::method_searchsorted),
    ],
    getters: &[],
};

/// Elements read in C order into a form that carries NumPy's sort order.
enum Keys {
    /// Bool and integer kinds; `i128` holds every `int64` and `uint64` value.
    Integers(Vec<i128>),
    /// Real floats, widened exactly to `f64`.
    Reals(Vec<f64>),
    Complex(Vec<(f64, f64)>),
    /// Fixed-width strings of `width` code points each, padded with NUL.
    Strings {
        chars: Vec<u32>,
        width: usize,
    },
    Objects(Vec<PyValue>),
}

impl Keys {
    /// Read every element of `arrays`, which share one dtype, one array after another.
    fn read(runtime: &mut dyn PyRuntime, arrays: &[&Array]) -> PyResult<Self> {
        let dtype = arrays[0].dtype;
        let count = arrays.iter().map(|array| array.size()).sum::<usize>();
        runtime.charge_cpu(count as u64 + 1)?;
        if dtype.kind() == Kind::Object {
            let mut values = Vec::with_capacity(count);
            for array in arrays {
                values.extend(array::read_objects(runtime, array)?);
            }
            return Ok(Self::Objects(values));
        }
        runtime.reserve_memory(count.saturating_mul(dtype.itemsize().max(16)))?;
        let kind = dtype.kind();
        let width = dtype.chars();
        let mut keys = match dtype.category() {
            Category::Str => Self::Strings {
                chars: Vec::with_capacity(count.saturating_mul(width)),
                width,
            },
            Category::Float => Self::Reals(Vec::with_capacity(count)),
            Category::Complex => Self::Complex(Vec::with_capacity(count)),
            _ => Self::Integers(Vec::with_capacity(count)),
        };
        for array in arrays {
            runtime.read_arrays(&[array.handle], &mut |views| {
                let PyArrayData::Bytes(bytes) = views[0].data else {
                    return Err(PyError::runtime_error("numeric array has object storage"));
                };
                for offset in array.offsets() {
                    match &mut keys {
                        Self::Strings { chars, width } => {
                            let text = &bytes[offset..offset + *width * 4];
                            chars.extend(text.chunks_exact(4).map(|unit| {
                                u32::from_le_bytes(unit.try_into().expect("four bytes"))
                            }));
                        }
                        Self::Reals(values) => {
                            values.push(element::read_number(kind, &bytes[offset..]).as_f64());
                        }
                        Self::Complex(values) => {
                            values.push(element::read_number(kind, &bytes[offset..]).as_complex());
                        }
                        Self::Integers(values) => {
                            values.push(match element::read_number(kind, &bytes[offset..]) {
                                Number::UInt(value) => i128::from(value),
                                other => i128::from(other.wrapping_i64()),
                            });
                        }
                        Self::Objects(_) => unreachable!("object arrays return early"),
                    }
                }
                Ok(())
            })?;
        }
        Ok(keys)
    }

    /// NumPy's order between elements `left` and `right`, for every kind but objects.
    /// NaN-carrying values rank after all others, and `descending` reverses only the rest.
    fn order(&self, left: usize, right: usize, descending: bool) -> Ordering {
        let (rank, values) = match self {
            Self::Integers(values) => (Ordering::Equal, values[left].cmp(&values[right])),
            Self::Reals(values) => {
                let (a, b) = (values[left], values[right]);
                (
                    a.is_nan().cmp(&b.is_nan()),
                    a.partial_cmp(&b).unwrap_or(Ordering::Equal),
                )
            }
            Self::Complex(values) => {
                let (a, b) = (values[left], values[right]);
                let within = match complex_class(a) {
                    0 => (a.0, a.1).partial_cmp(&(b.0, b.1)),
                    1 => a.0.partial_cmp(&b.0),
                    2 => a.1.partial_cmp(&b.1),
                    _ => Some(Ordering::Equal),
                };
                (
                    complex_class(a).cmp(&complex_class(b)),
                    within.unwrap_or(Ordering::Equal),
                )
            }
            Self::Strings { chars, width } => (
                Ordering::Equal,
                chars[left * width..(left + 1) * width]
                    .cmp(&chars[right * width..(right + 1) * width]),
            ),
            Self::Objects(_) => unreachable!("objects compare through Python"),
        };
        rank.then(if descending { values.reverse() } else { values })
    }

    /// Whether element `left` sorts before element `right`.
    fn less(
        &self,
        runtime: &mut dyn PyRuntime,
        left: usize,
        right: usize,
        descending: bool,
    ) -> PyResult<bool> {
        match self {
            Self::Objects(values) => {
                let (a, b) = if descending {
                    (values[right], values[left])
                } else {
                    (values[left], values[right])
                };
                python_less(runtime, a, b)
            }
            _ => Ok(self.order(left, right, descending) == Ordering::Less),
        }
    }

    /// NumPy's three-way `compare` function, which binary search on strings and objects
    /// uses: objects try `<`, then `>`.
    fn compare(
        &self,
        runtime: &mut dyn PyRuntime,
        left: usize,
        right: usize,
    ) -> PyResult<Ordering> {
        let Self::Objects(values) = self else {
            return Ok(self.order(left, right, false));
        };
        let (a, b) = (values[left], values[right]);
        if python_less(runtime, a, b)? {
            return Ok(Ordering::Less);
        }
        if python_less(runtime, b, a)? {
            return Ok(Ordering::Greater);
        }
        Ok(Ordering::Equal)
    }

    /// Whether NumPy searches this kind with its generic `compare`-based binary search.
    fn is_generic(&self) -> bool {
        matches!(self, Self::Strings { .. } | Self::Objects(_))
    }
}

/// The NaN class of a complex value in NumPy's order: `R + Rj`, `R + nanj`, `nan + Rj`,
/// `nan + nanj`.
fn complex_class((real, imag): (f64, f64)) -> u8 {
    u8::from(real.is_nan()) * 2 + u8::from(imag.is_nan())
}

fn python_less(runtime: &mut dyn PyRuntime, left: PyValue, right: PyValue) -> PyResult<bool> {
    let result = runtime.apply_operator(
        PyOperator::Compare(ComparisonOperator::Less),
        &[left, right],
    )?;
    runtime.truth(&result)
}

/// Stable merge sort of `items` by `less`, using `scratch` of the same length. Comparisons
/// may fail, since object comparisons run Python code; the first failure ends the sort.
fn merge_sort(
    items: &mut [usize],
    scratch: &mut [usize],
    less: &mut dyn FnMut(usize, usize) -> PyResult<bool>,
) -> PyResult<()> {
    const RUN: usize = 16;
    let count = items.len();
    for start in (0..count).step_by(RUN) {
        let end = (start + RUN).min(count);
        for next in start + 1..end {
            let mut position = next;
            while position > start && less(items[position], items[position - 1])? {
                items.swap(position, position - 1);
                position -= 1;
            }
        }
    }
    let mut width = RUN;
    while width < count {
        for start in (0..count).step_by(2 * width) {
            let middle = (start + width).min(count);
            let end = (start + 2 * width).min(count);
            let (mut left, mut right, mut output) = (start, middle, start);
            while left < middle && right < end {
                // Take from the right run only when strictly smaller, which keeps ties stable.
                if less(items[right], items[left])? {
                    scratch[output] = items[right];
                    right += 1;
                } else {
                    scratch[output] = items[left];
                    left += 1;
                }
                output += 1;
            }
            let rest = middle - left;
            scratch[output..output + rest].copy_from_slice(&items[left..middle]);
            scratch[output + rest..end].copy_from_slice(&items[right..end]);
        }
        items.copy_from_slice(scratch);
        width *= 2;
    }
    Ok(())
}

/// The lanes of a C-contiguous shape along one axis, addressed in elements.
struct Lanes {
    /// First element of each lane, lanes in C order of the other axes.
    bases: Vec<usize>,
    length: usize,
    step: usize,
}

impl Lanes {
    fn new(runtime: &mut dyn PyRuntime, shape: &[usize], axis: usize) -> PyResult<Self> {
        let mut strides = contiguous_strides(shape, 1);
        let step = strides.remove(axis) as usize;
        let mut lane_shape = shape.to_vec();
        let length = lane_shape.remove(axis);
        let lanes = array::element_count(&lane_shape)?;
        runtime.reserve_memory(lanes.saturating_mul(8))?;
        Ok(Self {
            bases: Offsets::new(&lane_shape, &strides, 0).collect(),
            length,
            step,
        })
    }

    /// C-order index of element `position` of lane `lane`.
    fn index(&self, lane: usize, position: usize) -> usize {
        self.bases[lane] + position * self.step
    }

    /// The identity permutation of every lane, lane after lane.
    fn identity(&self, runtime: &mut dyn PyRuntime) -> PyResult<Vec<usize>> {
        let count = self.bases.len().saturating_mul(self.length);
        runtime.reserve_memory(count.saturating_mul(8))?;
        Ok((0..self.bases.len()).flat_map(|_| 0..self.length).collect())
    }
}

/// Stably reorder each lane's positions in `permutation` by `keys`.
fn sort_lanes(
    runtime: &mut dyn PyRuntime,
    keys: &Keys,
    lanes: &Lanes,
    permutation: &mut [usize],
    descending: bool,
) -> PyResult<()> {
    let length = lanes.length;
    if length < 2 {
        return Ok(());
    }
    runtime.reserve_memory(length.saturating_mul(8))?;
    let mut scratch = vec![0; length];
    let depth = u64::from(usize::BITS - length.leading_zeros());
    let comparisons = (length as u64).saturating_mul(depth);
    let step = lanes.step;
    for (lane, base) in lanes.bases.iter().enumerate() {
        runtime.charge_cpu(comparisons)?;
        merge_sort(
            &mut permutation[lane * length..(lane + 1) * length],
            &mut scratch,
            &mut |left, right| {
                keys.less(runtime, base + left * step, base + right * step, descending)
            },
        )?;
    }
    Ok(())
}

/// The stable sorting permutation of every lane of `array` along `axis`.
fn argsort_lanes(
    runtime: &mut dyn PyRuntime,
    array: &Array,
    axis: usize,
    descending: bool,
) -> PyResult<(Lanes, Vec<usize>)> {
    if descending && array.dtype.kind() == Kind::Object {
        return Err(PyError::type_error(
            "no current sort function meets the requirements",
        ));
    }
    let keys = Keys::read(runtime, &[array])?;
    let lanes = Lanes::new(runtime, array.shape(), axis)?;
    let mut permutation = lanes.identity(runtime)?;
    sort_lanes(runtime, &keys, &lanes, &mut permutation, descending)?;
    Ok((lanes, permutation))
}

/// A sorted C-contiguous copy of `array`, each lane along `axis` in order.
fn sorted_copy(
    runtime: &mut dyn PyRuntime,
    array: &Array,
    axis: usize,
    descending: bool,
) -> PyResult<Array> {
    let (lanes, permutation) = argsort_lanes(runtime, array, axis, descending)?;
    let source = array::contiguous_buffer(runtime, array)?;
    reserve_elements(runtime, array.dtype, array.size())?;
    let length = lanes.length;
    let moves = (0..lanes.bases.len()).flat_map(|lane| {
        let (lanes, permutation) = (&lanes, &permutation);
        (0..length).map(move |position| {
            (
                lanes.index(lane, position),
                lanes.index(lane, permutation[lane * length + position]),
            )
        })
    });
    let buffer = match &source {
        PyArrayBuffer::Bytes(bytes) => {
            let size = array.itemsize();
            let mut output = vec![0u8; bytes.len()];
            for (target, origin) in moves {
                output[target * size..(target + 1) * size]
                    .copy_from_slice(&bytes[origin * size..(origin + 1) * size]);
            }
            PyArrayBuffer::Bytes(output)
        }
        PyArrayBuffer::Values(values) => {
            let mut output = vec![Value::None; values.len()];
            for (target, origin) in moves {
                output[target] = values[origin];
            }
            PyArrayBuffer::Values(output)
        }
    };
    new_array(runtime, buffer, array.dtype, array.shape().to_vec())
}

/// Sorting permutations as an `int64` array of `shape`.
fn permutation_array(
    runtime: &mut dyn PyRuntime,
    shape: Vec<usize>,
    lanes: &Lanes,
    permutation: &[usize],
) -> PyResult<Array> {
    let mut indices = vec![0i64; permutation.len()];
    for lane in 0..lanes.bases.len() {
        for position in 0..lanes.length {
            indices[lanes.index(lane, position)] =
                permutation[lane * lanes.length + position] as i64;
        }
    }
    array::array_from_elements(runtime, DType::INT64, shape, &indices)
}

static SORT: Signature = Signature::new("sort", &["a", "axis", "kind", "order"], 1)
    .keyword_only(&["stable", "descending"]);
static ARGSORT: Signature = Signature::new("argsort", &["a", "axis", "kind", "order"], 1)
    .keyword_only(&["stable", "descending"]);
static SORT_METHOD: Signature =
    Signature::new("sort", &["axis", "kind", "order"], 0).keyword_only(&["stable", "descending"]);
static ARGSORT_METHOD: Signature = Signature::new("argsort", &["axis", "kind", "order"], 0)
    .keyword_only(&["stable", "descending"]);

/// Check `kind`, `order`, `stable`, and `descending` as `array_sort` does, returning whether
/// to sort in descending order.
fn sort_options(runtime: &mut dyn PyRuntime, bound: &Bound) -> PyResult<bool> {
    let mut optional_flag = |name| -> PyResult<Option<bool>> {
        bound
            .value(name)
            .map(|value| runtime.truth(&value))
            .transpose()
    };
    let stable = optional_flag("stable")?;
    let descending = optional_flag("descending")?;
    if let Some(kind) = bound.value("kind") {
        parse_option(
            runtime,
            &kind,
            "sort kind",
            "must be one of 'quick', 'heap', or 'stable'",
            |text| {
                matches!(
                    text.chars().next(),
                    Some('q' | 'Q' | 'h' | 'H' | 'm' | 'M' | 's' | 'S')
                )
            },
        )?;
        if stable.is_some() || descending.is_some() {
            return Err(PyError::value_error(
                "`kind` and keyword parameters can't be provided at the same time. Use only one \
                 of them.",
            ));
        }
    }
    reject_order(bound)?;
    Ok(descending.unwrap_or(false))
}

/// Check a string option as NumPy's `string_converter_helper` does.
fn parse_option(
    runtime: &mut dyn PyRuntime,
    value: &PyValue,
    name: &str,
    message: &str,
    accept: impl Fn(&str) -> bool,
) -> PyResult<()> {
    let Some(text) = runtime.string_value(value)? else {
        return Err(PyError::type_error(format!(
            "{name} must be str, not {}",
            runtime.type_name(value)?
        )));
    };
    if accept(&text) {
        return Ok(());
    }
    let repr = runtime.repr(value)?;
    Err(PyError::value_error(format!(
        "{name} {message} (got {repr})"
    )))
}

/// `order=` names structured fields, which shellsim's dtypes do not have.
fn reject_order(bound: &Bound) -> PyResult<()> {
    if bound.value("order").is_some() {
        return Err(PyError::value_error(
            "Cannot specify order when the array has no fields.",
        ));
    }
    Ok(())
}

/// `array` and the axis to sort along: `axis=None` sorts the flattened array.
fn sort_axis(
    runtime: &mut dyn PyRuntime,
    array: Array,
    axis: Option<PyValue>,
) -> PyResult<(Array, usize)> {
    match axis {
        Some(axis) if axis.is_none() => Ok((array::ravel(runtime, &array)?, 0)),
        axis => {
            let axis = axis.map_or(Ok(-1), |axis| args::index_int(runtime, &axis))?;
            let axis = array::normalize_axis(axis, array.ndim())?;
            Ok((array, axis))
        }
    }
}

/// `np.sort(a, axis=-1, kind=None, order=None, *, stable=None, descending=None)`.
fn module_sort(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = SORT.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    let descending = sort_options(runtime, &bound)?;
    let (array, axis) = sort_axis(runtime, array, bound.get("axis"))?;
    Ok(sorted_copy(runtime, &array, axis, descending)?.value())
}

/// `ndarray.sort(axis=-1, kind=None, order=None, *, stable=None, descending=None)`: sort in
/// place, writing through views.
fn method_sort(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let bound = SORT_METHOD.bind(&args)?;
    let array = Array::from_value(runtime, receiver)?;
    let descending = sort_options(runtime, &bound)?;
    let axis = match bound.value("axis") {
        Some(axis) => args::index_int(runtime, &axis)?,
        None => -1,
    };
    let axis = array::normalize_axis(axis, array.ndim())?;
    if !array.view.writeable {
        return Err(PyError::value_error("sort array is read-only"));
    }
    let sorted = sorted_copy(runtime, &array, axis, descending)?;
    array::assign(runtime, &array, &sorted)?;
    Ok(Value::None)
}

/// Shared by `np.argsort` and `ndarray.argsort`. A 0-d array argsorts as one element, as
/// `PyArray_CheckAxis` makes it.
fn argsort(runtime: &mut dyn PyRuntime, array: Array, bound: &Bound) -> PyResult {
    let descending = sort_options(runtime, bound)?;
    let array = if array.ndim() == 0 {
        array::ravel(runtime, &array)?
    } else {
        array
    };
    let (array, axis) = sort_axis(runtime, array, bound.get("axis"))?;
    let (lanes, permutation) = argsort_lanes(runtime, &array, axis, descending)?;
    Ok(permutation_array(runtime, array.shape().to_vec(), &lanes, &permutation)?.value())
}

/// `np.argsort(a, axis=-1, kind=None, order=None, *, stable=None, descending=None)`.
fn module_argsort(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = ARGSORT.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    argsort(runtime, array, &bound)
}

fn method_argsort(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let bound = ARGSORT_METHOD.bind(&args)?;
    let array = Array::from_value(runtime, receiver)?;
    argsort(runtime, array, &bound)
}

/// `np.lexsort(keys, axis=-1)`: the last key is the primary one. Following `PyArray_LexSort`,
/// the permutation is sorted stably by each key in turn, from the first key to the last.
fn module_lexsort(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("lexsort", &["keys", "axis"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let keys = lexsort_keys(runtime, bound.required("keys"))?;
    let shape = keys[0].shape().to_vec();
    if keys[1..].iter().any(|key| key.shape() != shape) {
        return Err(PyError::value_error("all keys need to be the same shape"));
    }
    let axis = match bound.value("axis") {
        Some(axis) => args::index_int(runtime, &axis)?,
        None => -1,
    };
    // A 0-d key accepts axis 0 or -1 for backwards compatibility.
    if shape.is_empty() && (axis == 0 || axis == -1) {
        return Ok(array::array_from_elements(runtime, DType::INT64, shape, &[0i64])?.value());
    }
    let axis = array::normalize_axis(axis, shape.len())?;
    let lanes = Lanes::new(runtime, &shape, axis)?;
    let mut permutation = lanes.identity(runtime)?;
    for key in &keys {
        let key = Keys::read(runtime, &[key])?;
        sort_lanes(runtime, &key, &lanes, &mut permutation, false)?;
    }
    Ok(permutation_array(runtime, shape, &lanes, &permutation)?.value())
}

/// The keys of `lexsort`: the items of a sequence, or the rows of an array.
fn lexsort_keys(runtime: &mut dyn PyRuntime, keys: PyValue) -> PyResult<Vec<Array>> {
    let items = match runtime.kind(&keys)? {
        PyKind::Tuple | PyKind::List => {
            super::shape::sequence_items(runtime, &keys)?.unwrap_or_default()
        }
        _ if runtime.native_kind(&keys)? == Some(PyNativeKind::Array) => {
            let array = Array::from_value(runtime, keys)?;
            let rows = array.shape().first().copied().unwrap_or(0);
            (0..rows)
                .map(|row| super::index::get_item(runtime, &array, Value::Int(row as i64)))
                .collect::<PyResult<Vec<_>>>()?
        }
        _ => Vec::new(),
    };
    if items.is_empty() {
        return Err(PyError::type_error(
            "need sequence of keys with len > 0 in lexsort",
        ));
    }
    items
        .into_iter()
        .map(|item| convert::as_array(runtime, item))
        .collect()
}

static PARTITION: Signature =
    Signature::new("partition", &["a", "kth", "axis", "kind", "order"], 2);
static ARGPARTITION: Signature =
    Signature::new("argpartition", &["a", "kth", "axis", "kind", "order"], 2);
static PARTITION_METHOD: Signature =
    Signature::new("partition", &["kth", "axis", "kind", "order"], 1);
static ARGPARTITION_METHOD: Signature =
    Signature::new("argpartition", &["kth", "axis", "kind", "order"], 1);

/// Validate `kth`, `kind`, and `order` as `partition_prep_kth_array` and `array_partition`
/// do. The positions are only checked, since the lanes are sorted completely.
fn check_partition(
    runtime: &mut dyn PyRuntime,
    bound: &Bound,
    array: &Array,
    axis: usize,
) -> PyResult<()> {
    if let Some(kind) = bound.value("kind") {
        parse_option(
            runtime,
            &kind,
            "select kind",
            "must be 'introselect'",
            |text| text == "introselect",
        )?;
    }
    reject_order(bound)?;
    let kth = bound.required("kth");
    let is_array = runtime.native_kind(&kth)? == Some(PyNativeKind::Array);
    let kth = convert::as_array(runtime, kth)?;
    if kth.ndim() > 1 {
        return Err(too_deep(is_array));
    }
    match kth.dtype.category() {
        Category::Bool => {
            return Err(PyError::value_error(
                "Booleans unacceptable as partition index",
            ))
        }
        Category::Signed | Category::Unsigned => {}
        _ => return Err(PyError::type_error("Partition index must be integer")),
    }
    let kth = convert::cast_array(runtime, &kth, DType::INT64, false)?;
    let length = array.shape()[axis] as i64;
    for position in array::read_elements::<i64>(runtime, &kth)? {
        let position = if position < 0 {
            position + length
        } else {
            position
        };
        if array.size() != 0 && !(0..length).contains(&position) {
            return Err(PyError::value_error(format!(
                "kth(={position}) out of bounds ({length})"
            )));
        }
    }
    Ok(())
}

/// The error `PyArray_FromAny` gives a value with more than one dimension when at most one is
/// allowed: a nested sequence fails while its shape is discovered, an array afterwards.
fn too_deep(is_array: bool) -> PyError {
    PyError::value_error(if is_array {
        "object too deep for desired array"
    } else {
        "setting an array element with a sequence. The requested array would exceed the \
         maximum number of dimension of 1."
    })
}

/// The array to partition and its axis, after validating the arguments.
fn partition_input(
    runtime: &mut dyn PyRuntime,
    array: Array,
    bound: &Bound,
) -> PyResult<(Array, usize)> {
    let (array, axis) = sort_axis(runtime, array, bound.get("axis"))?;
    check_partition(runtime, bound, &array, axis)?;
    Ok((array, axis))
}

/// `np.partition(a, kth, axis=-1, kind='introselect', order=None)`.
fn module_partition(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = PARTITION.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    let (array, axis) = partition_input(runtime, array, &bound)?;
    Ok(sorted_copy(runtime, &array, axis, false)?.value())
}

/// `ndarray.partition(kth, axis=-1, kind='introselect', order=None)`, in place.
fn method_partition(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let bound = PARTITION_METHOD.bind(&args)?;
    let array = Array::from_value(runtime, receiver)?;
    if bound.get("axis").is_some_and(|axis| axis.is_none()) {
        return Err(PyError::type_error(
            "'NoneType' object cannot be interpreted as an integer",
        ));
    }
    let (array, axis) = partition_input(runtime, array, &bound)?;
    if !array.view.writeable {
        return Err(PyError::value_error("partition array is read-only"));
    }
    let sorted = sorted_copy(runtime, &array, axis, false)?;
    array::assign(runtime, &array, &sorted)?;
    Ok(Value::None)
}

fn argpartition(runtime: &mut dyn PyRuntime, array: Array, bound: &Bound) -> PyResult {
    let array = if array.ndim() == 0 {
        array::ravel(runtime, &array)?
    } else {
        array
    };
    let (array, axis) = partition_input(runtime, array, bound)?;
    let (lanes, permutation) = argsort_lanes(runtime, &array, axis, false)?;
    Ok(permutation_array(runtime, array.shape().to_vec(), &lanes, &permutation)?.value())
}

/// `np.argpartition(a, kth, axis=-1, kind='introselect', order=None)`.
fn module_argpartition(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = ARGPARTITION.bind(&args)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    argpartition(runtime, array, &bound)
}

fn method_argpartition(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    let bound = ARGPARTITION_METHOD.bind(&args)?;
    let array = Array::from_value(runtime, receiver)?;
    argpartition(runtime, array, &bound)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sort_keys(keys: &Keys, count: usize, descending: bool) -> Vec<usize> {
        let mut items = (0..count).collect::<Vec<_>>();
        let mut scratch = vec![0; count];
        merge_sort(&mut items, &mut scratch, &mut |left, right| {
            Ok(keys.order(left, right, descending) == Ordering::Less)
        })
        .expect("numeric comparisons cannot fail");
        items
    }

    #[test]
    fn merge_sort_is_stable_across_runs() {
        // 40 elements span three insertion-sorted runs and two merge passes.
        let values = (0..40)
            .map(|index| f64::from(index % 3))
            .collect::<Vec<_>>();
        let order = sort_keys(&Keys::Reals(values), 40, false);
        let expected = (0..40)
            .filter(|index| index % 3 == 0)
            .chain((0..40).filter(|index| index % 3 == 1))
            .chain((0..40).filter(|index| index % 3 == 2))
            .collect::<Vec<_>>();
        assert_eq!(order, expected);
    }

    #[test]
    fn nan_sorts_last_in_both_directions() {
        let keys = Keys::Reals(vec![1.0, f64::NAN, 3.0, 1.0, -0.0, 0.0]);
        assert_eq!(sort_keys(&keys, 6, false), [4, 5, 0, 3, 2, 1]);
        assert_eq!(sort_keys(&keys, 6, true), [2, 0, 3, 4, 5, 1]);
    }

    #[test]
    fn complex_values_with_nan_parts_follow_numpy_classes() {
        let nan = f64::NAN;
        let keys = Keys::Complex(vec![
            (1.0, 1.0),
            (nan, 0.0),
            (1.0, nan),
            (0.0, 0.0),
            (nan, nan),
            (1.0, -1.0),
            (nan, -1.0),
        ]);
        // NumPy 2.5: np.argsort(c, stable=True) and np.argsort(c, descending=True, stable=True).
        assert_eq!(sort_keys(&keys, 7, false), [3, 5, 0, 2, 6, 1, 4]);
        assert_eq!(sort_keys(&keys, 7, true), [0, 5, 3, 2, 1, 6, 4]);
    }
}
