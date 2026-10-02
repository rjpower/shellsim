//! Sorting, partial sorting, and lexicographic ordering: `sort`, `argsort`, `lexsort`,
//! `partition`, `argpartition`, and the matching `numpy.ndarray` methods. `numpy.searchsorted`
//! and `numpy.bincount` live in the [`search`] and [`bincount`] submodules.
//!
//! NumPy documents three sort algorithms (`quicksort`/`heapsort`/`stable`, aliased as
//! `mergesort`) plus `introselect` for `partition`, and calls the first three "unstable": NumPy
//! does not promise how they order equal elements. A stable sort is therefore a valid result
//! for every `kind`, so every `kind` here runs the same algorithm — sort an index permutation
//! with one comparator per dtype family, using [`slice::sort_by`] (a stable sort) for byte-backed
//! dtypes and a hand-written stable merge sort ([`merge_sort_by`]) where the comparator must call
//! back into Python (`object` elements). `partition`/`argpartition` fully sort each lane too,
//! which satisfies their weaker contract (only position `kth` need be in its sorted place).
//!
//! The sort order follows NumPy's documentation: NaN sorts after every non-NaN value of the
//! same dtype; complex values compare by real part then imaginary part, and (derived from
//! black-box probing of NumPy 2.5.3, since the docs only say "NaN-containing values sort last")
//! any value with NaN in either part sorts after every NaN-free value, then among NaN-containing
//! values again by real part then imaginary part with NaN standing in for "greater than any
//! number" in that part; strings compare by code point; `object` elements compare with Python
//! `<`. `descending=True` reverses the non-NaN run and leaves the NaN suffix (already in input
//! order) in place, matching NumPy's observed output.
//!
//! Every sort and partial sort charges CPU proportional to `n·log2(n)` per lane before running;
//! `searchsorted` charges `m·log2(n)` for `m` binary searches against a lane of length `n`. See
//! `sort/search.rs` and `sort/bincount.rs` for their own strategy notes.

mod bincount;
mod search;

use std::cmp::Ordering;

use super::super::super::ast::ComparisonOperator;
use super::super::super::native::{
    CallArgs, FunctionDef, MethodDef, ModuleDef, NativeFn, NativeMethodFn, NativeTypeDef,
    PyArrayBuffer, PyArrayData, PyArrayDtype, PyError, PyKind, PyNativeKind, PyOperator, PyResult,
    PyRuntime, PyValue,
};
use super::args::{self, Signature};
use super::array::{self, Array};
use super::convert;
use super::dtype::{DType, Kind};
use super::element::{dispatch_numeric, Element, Number, C128, C64, F16};
use super::index;
use super::ops::{ComplexParts, Numeric};

pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_numpy_sort",
    functions: FUNCTIONS,
    values: &[],
};

const fn function(name: &'static str, call: NativeFn) -> FunctionDef {
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

const fn method(name: &'static str, call: NativeMethodFn) -> MethodDef {
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

fn receiver<'s>(runtime: &dyn PyRuntime<'s>, value: PyValue<'s>) -> PyResult<'s, Array<'s>> {
    Array::from_value(runtime, value)
}

/// One unit of work per comparison, `ceil(log2(n+1))` comparisons per element: a conservative
/// bound on a stable `n log n` sort or merge sort of `n` items.
fn cost_n_log_n(n: usize) -> u64 {
    let depth = usize::BITS - n.leading_zeros();
    (n as u64).saturating_mul(u64::from(depth) + 1)
}

/// `ceil(log2(n+1))`, the depth of a binary search over `n` items.
pub(in crate::python) fn cost_log_n(n: usize) -> u64 {
    u64::from(usize::BITS - n.leading_zeros()) + 1
}

/// NumPy's sort order for one real (bool/integer/float) dtype: IEEE order among non-NaN values,
/// NaN after every non-NaN value, NaN equal to NaN (so a stable sort keeps NaNs in input order).
fn real_order<T: Numeric>(a: T, b: T) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (false, false) => a
            .compare(b)
            .expect("non-NaN values of one dtype always compare"),
        (false, true) => Ordering::Less,
        (true, false) => Ordering::Greater,
        (true, true) => Ordering::Equal,
    }
}

/// NumPy's sort order for `complex`, derived by probing NumPy 2.5.3's `argsort`: NaN-free
/// values order by real part then imaginary part; a value with NaN in either part sorts after
/// every NaN-free value; among NaN-containing values, real part then imaginary part again,
/// with NaN acting as "greater than any number" in that part.
fn complex_order(a: (f64, f64), b: (f64, f64)) -> Ordering {
    let has_nan = |v: (f64, f64)| v.0.is_nan() || v.1.is_nan();
    match (has_nan(a), has_nan(b)) {
        (false, true) => Ordering::Less,
        (true, false) => Ordering::Greater,
        _ => real_order(a.0, b.0).then_with(|| real_order(a.1, b.1)),
    }
}

/// NumPy's sort order, one comparator per byte-backed element type so [`dispatch_numeric!`]
/// picks it monomorphically. Bool/integer/float route to [`real_order`]; complex routes to
/// [`complex_order`] through its real and imaginary parts.
trait SortKey: Copy {
    fn sort_cmp(self, other: Self) -> Ordering;
    fn sort_is_nan(self) -> bool;
}

macro_rules! real_sort_key {
    ($($t:ty),* $(,)?) => {
        $(impl SortKey for $t {
            fn sort_cmp(self, other: Self) -> Ordering {
                real_order(self, other)
            }
            fn sort_is_nan(self) -> bool {
                Numeric::is_nan(self)
            }
        })*
    };
}

real_sort_key!(bool, i8, i16, i32, i64, u8, u16, u32, u64, F16, f32, f64);

macro_rules! complex_sort_key {
    ($($t:ty),* $(,)?) => {
        $(impl SortKey for $t {
            fn sort_cmp(self, other: Self) -> Ordering {
                complex_order(self.parts(), other.parts())
            }
            fn sort_is_nan(self) -> bool {
                Numeric::is_nan(self)
            }
        })*
    };
}

complex_sort_key!(C64, C128);

/// Reverse the non-NaN prefix of an ascending order, leaving the trailing `nan_suffix`
/// positions (already in input order, since equal keys sort stably) in place — NumPy's
/// `descending=True`, which keeps NaN last either way.
fn apply_descending(mut order: Vec<usize>, nan_suffix: usize, descending: bool) -> Vec<usize> {
    if descending {
        let split = order.len() - nan_suffix;
        order[..split].reverse();
    }
    order
}

/// Byte offsets, relative to an array's own `view.offset`, of every element of `shape` at
/// `strides`, enumerated lane by lane: every axis but `axis`, in C order, outermost first, then
/// `0..shape[axis]` along `axis`. [`lane_orders`] and every function that builds a same-shaped
/// output from its result use this same enumeration, so their flat positions line up.
fn lane_offsets(shape: &[usize], strides: &[isize], axis: usize) -> Vec<isize> {
    let n = shape[axis];
    let axis_stride = strides[axis];
    let (outer_shape, outer_strides): (Vec<usize>, Vec<isize>) = shape
        .iter()
        .zip(strides)
        .enumerate()
        .filter(|(a, _)| *a != axis)
        .map(|(_, (&d, &s))| (d, s))
        .unzip();
    let outer = index::relative_offsets(&outer_shape, &outer_strides);
    let mut offsets = Vec::with_capacity(outer.len() * n);
    for lane in outer {
        for k in 0..n {
            offsets.push(lane + k as isize * axis_stride);
        }
    }
    offsets
}

/// [`lane_offsets`], as absolute byte offsets from `base`.
fn absolute_lane_offsets(
    shape: &[usize],
    strides: &[isize],
    axis: usize,
    base: usize,
) -> Vec<usize> {
    lane_offsets(shape, strides, axis)
        .into_iter()
        .map(|relative| (base as isize + relative) as usize)
        .collect()
}

/// A stable sort of `items` by a fallible `less` predicate, ties keeping their original
/// relative order: a plain top-down merge sort, so a Python-level comparator (`object`
/// elements) runs `O(n log n)` times, matching the CPU already charged for the lane.
fn merge_sort_by<'s, T: Copy>(
    items: &mut [T],
    less: &mut impl FnMut(&T, &T) -> PyResult<'s, bool>,
) -> PyResult<'s, ()> {
    let n = items.len();
    if n <= 1 {
        return Ok(());
    }
    let mid = n / 2;
    let mut left = items[..mid].to_vec();
    let mut right = items[mid..].to_vec();
    merge_sort_by(&mut left, less)?;
    merge_sort_by(&mut right, less)?;
    let (mut i, mut j, mut k) = (0, 0, 0);
    while i < left.len() && j < right.len() {
        if less(&right[j], &left[i])? {
            items[k] = right[j];
            j += 1;
        } else {
            items[k] = left[i];
            i += 1;
        }
        k += 1;
    }
    items[k..k + (left.len() - i)].copy_from_slice(&left[i..]);
    let k = k + (left.len() - i);
    items[k..k + (right.len() - j)].copy_from_slice(&right[j..]);
    Ok(())
}

/// `a < b` for `object` elements, NumPy's sort comparator for that dtype.
fn less_than<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    a: PyValue<'s>,
    b: PyValue<'s>,
) -> PyResult<'s, bool> {
    let result = runtime.apply_operator(PyOperator::Compare(ComparisonOperator::Less), &[a, b])?;
    runtime.truth(&result)
}

/// Ascending-stable order of one numeric or `float16`/complex lane at relative offsets `lane`.
fn numeric_lane_order<T: SortKey + Element>(
    bytes: &[u8],
    base: usize,
    lane: &[isize],
    descending: bool,
) -> Vec<usize> {
    let mut keyed: Vec<(T, usize)> = lane
        .iter()
        .enumerate()
        .map(|(k, &relative)| (T::read(&bytes[(base as isize + relative) as usize..]), k))
        .collect();
    keyed.sort_by(|a, b| a.0.sort_cmp(b.0));
    let nan_suffix = keyed
        .iter()
        .rev()
        .take_while(|(v, _)| v.sort_is_nan())
        .count();
    let order = keyed.into_iter().map(|(_, k)| k).collect();
    apply_descending(order, nan_suffix, descending)
}

/// Compare two elements' UCS-4 code points in order; equal-length runs since both come from one
/// array's fixed-width `str` storage.
fn compare_code_points(a: &[u8], b: &[u8]) -> Ordering {
    a.chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .cmp(
            b.chunks_exact(4)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four bytes"))),
        )
}

fn str_lane_order(
    bytes: &[u8],
    base: usize,
    lane: &[isize],
    width: usize,
    descending: bool,
) -> Vec<usize> {
    let read = |relative: isize| {
        let offset = (base as isize + relative) as usize;
        &bytes[offset..offset + width]
    };
    let mut order: Vec<usize> = (0..lane.len()).collect();
    order.sort_by(|&a, &b| compare_code_points(read(lane[a]), read(lane[b])));
    apply_descending(order, 0, descending)
}

fn object_lane_orders<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    array: &Array<'s>,
    offsets: &[isize],
    n: usize,
    descending: bool,
) -> PyResult<'s, Vec<usize>> {
    let base = array.view.offset;
    let mut snapshots: Vec<Vec<PyValue<'s>>> = Vec::new();
    runtime.read_arrays(&[array.handle], &mut |refs, arrays| {
        let PyArrayData::Values(values) = arrays[0].data else {
            return Err(PyError::runtime_error(
                "sort saw byte storage for an object dtype",
            ));
        };
        for lane in offsets.chunks(n.max(1)) {
            snapshots.push(
                lane.iter()
                    .map(|&relative| {
                        refs.handle(
                            &values[((base as isize + relative) as usize)
                                / PyArrayDtype::VALUE_ITEMSIZE],
                        )
                    })
                    .collect(),
            );
        }
        Ok(())
    })?;
    let mut result = Vec::with_capacity(offsets.len());
    for lane in snapshots {
        let mut order: Vec<usize> = (0..lane.len()).collect();
        merge_sort_by(&mut order, &mut |&a, &b| {
            less_than(runtime, lane[a], lane[b])
        })?;
        result.extend(apply_descending(order, 0, descending));
    }
    Ok(result)
}

/// The ascending- (or, if `descending`, mostly-descending-) stable permutation of every lane of
/// `array` along `axis`: for each lane, the `n` local positions `0..n` in sorted order, lanes
/// concatenated in the [`lane_offsets`] enumeration.
fn lane_orders<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    array: &Array<'s>,
    axis: usize,
    descending: bool,
) -> PyResult<'s, Vec<usize>> {
    let n = array.shape()[axis];
    let offsets = lane_offsets(array.shape(), array.strides(), axis);
    let lanes = offsets.len().checked_div(n).unwrap_or(0);
    runtime.charge_cpu(
        cost_n_log_n(n)
            .saturating_mul(lanes as u64)
            .saturating_add(1),
    )?;
    runtime.reserve_memory(offsets.len().saturating_mul(std::mem::size_of::<usize>()))?;
    if array.dtype.kind() == Kind::Object {
        return object_lane_orders(runtime, array, &offsets, n, descending);
    }
    let base = array.view.offset;
    let kind = array.dtype.kind();
    let width = array.dtype.itemsize();
    let mut result = Vec::with_capacity(offsets.len());
    runtime.read_arrays(&[array.handle], &mut |_refs, arrays| {
        let PyArrayData::Bytes(bytes) = arrays[0].data else {
            return Err(PyError::runtime_error(
                "sort saw object storage for a numeric dtype",
            ));
        };
        if kind == Kind::Str {
            for lane in offsets.chunks(n.max(1)) {
                result.extend(str_lane_order(bytes, base, lane, width, descending));
            }
        } else {
            dispatch_numeric!(kind, T => {
                for lane in offsets.chunks(n.max(1)) {
                    result.extend(numeric_lane_order::<T>(bytes, base, lane, descending));
                }
            }, _ => unreachable!("Str and Object are handled separately"));
        }
        Ok(())
    })?;
    Ok(result)
}

/// A buffer holding, for every output lane position (in the [`lane_offsets`] enumeration), the
/// element `source` had at the `chosen` local position of that same lane.
fn gather_chosen<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    source: &Array<'s>,
    source_offsets: &[isize],
    n: usize,
    chosen: &[usize],
) -> PyResult<'s, PyArrayBuffer<'s>> {
    let count = chosen.len();
    array::reserve_elements(runtime, source.dtype, count)?;
    runtime.charge_cpu(count as u64 + 1)?;
    let mut buffer = array::buffer_with_capacity(source.dtype, count);
    let base = source.view.offset;
    runtime.read_arrays(&[source.handle], &mut |refs, arrays| {
        match (&arrays[0].data, &mut buffer) {
            (PyArrayData::Bytes(bytes), PyArrayBuffer::Bytes(out)) => {
                let itemsize = source.itemsize();
                for (i, &k) in chosen.iter().enumerate() {
                    let lane = i / n.max(1);
                    let relative = source_offsets[lane * n + k];
                    let offset = (base as isize + relative) as usize;
                    out.extend_from_slice(&bytes[offset..offset + itemsize]);
                }
            }
            (PyArrayData::Values(values), PyArrayBuffer::Values(out)) => {
                for (i, &k) in chosen.iter().enumerate() {
                    let lane = i / n.max(1);
                    let relative = source_offsets[lane * n + k];
                    let offset = (base as isize + relative) as usize;
                    out.push(refs.handle(&values[offset / PyArrayDtype::VALUE_ITEMSIZE]));
                }
            }
            _ => unreachable!("gather copies between storages of one element kind"),
        }
        Ok(())
    })?;
    Ok(buffer)
}

/// Write each of `source`'s lanes, permuted by `chosen`, into `target` (of the same shape and
/// dtype — `source` itself for an in-place sort, or a fresh array for a copying one).
fn write_sorted<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    source: &Array<'s>,
    target: &Array<'s>,
    axis: usize,
    chosen: &[usize],
) -> PyResult<'s, ()> {
    let n = source.shape()[axis];
    let source_offsets = lane_offsets(source.shape(), source.strides(), axis);
    let buffer = gather_chosen(runtime, source, &source_offsets, n, chosen)?;
    let target_offsets =
        absolute_lane_offsets(target.shape(), target.strides(), axis, target.view.offset);
    array::scatter(runtime, target, &target_offsets, &buffer)
}

/// A fresh array of `source`'s shape and dtype holding its elements sorted along `axis` by
/// `chosen`.
fn sorted_copy<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    source: &Array<'s>,
    axis: usize,
    chosen: &[usize],
) -> PyResult<'s, Array<'s>> {
    let buffer = array::zeroed_buffer(runtime, source.dtype, source.size())?;
    let target = array::new_array(runtime, buffer, source.dtype, source.shape().to_vec())?;
    write_sorted(runtime, source, &target, axis, chosen)?;
    Ok(target)
}

/// A fresh int64 array of `shape` holding `chosen` (local axis positions) placed along `axis`
/// the same way [`sorted_copy`] places values — `argsort`'s and `lexsort`'s result.
fn indices_array<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    shape: Vec<usize>,
    axis: usize,
    chosen: &[usize],
) -> PyResult<'s, Array<'s>> {
    let count = chosen.len();
    array::reserve_elements(runtime, DType::INT64, count)?;
    runtime.charge_cpu(count as u64 + 1)?;
    let indices: Vec<i64> = chosen.iter().map(|&k| k as i64).collect();
    let buffer = array::zeroed_buffer(runtime, DType::INT64, count)?;
    let target = array::new_array(runtime, buffer, DType::INT64, shape.clone())?;
    let contiguous = array::contiguous_strides(&shape, DType::INT64.itemsize());
    let offsets = absolute_lane_offsets(&shape, &contiguous, axis, 0);
    array::scatter(
        runtime,
        &target,
        &offsets,
        &PyArrayBuffer::Bytes(array::pack_elements(&indices)),
    )?;
    Ok(target)
}

/// `axis=None` flattens (`np.sort`/`np.argsort`/`np.partition`/`np.argpartition`, module and
/// method forms except in-place `sort`/`partition`); an omitted axis defaults to -1; otherwise
/// the axis is normalized.
fn resolve_axis_flatten<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    array: &Array<'s>,
    axis: Option<PyValue<'s>>,
) -> PyResult<'s, (Array<'s>, usize)> {
    match axis {
        None => Ok((array.clone(), array::normalize_axis(-1, array.ndim())?)),
        Some(value) if value.is_none() => {
            let flat = array::ravel(runtime, array)?;
            Ok((flat, 0))
        }
        Some(value) => {
            let axis = array::normalize_axis(args::index_int(runtime, &value)?, array.ndim())?;
            Ok((array.clone(), axis))
        }
    }
}

/// The axis for in-place `ndarray.sort`/`ndarray.partition`, which (unlike every other form)
/// does not accept `axis=None`: an omitted axis defaults to -1, and any explicit value —
/// including `None` — goes through the ordinary integer conversion, so `None` raises NumPy's
/// "cannot be interpreted as an integer" `TypeError` instead of flattening.
fn resolve_axis_strict<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    array: &Array<'s>,
    axis: Option<PyValue<'s>>,
) -> PyResult<'s, usize> {
    let raw = match axis {
        None => -1,
        Some(value) => args::index_int(runtime, &value)?,
    };
    array::normalize_axis(raw, array.ndim())
}

fn kind_type_error<'s>(runtime: &dyn PyRuntime<'s>, value: &PyValue<'s>, noun: &str) -> PyError {
    let type_name = runtime
        .type_name(value)
        .unwrap_or_else(|_| "object".to_string());
    PyError::type_error(format!("{noun} kind must be str, not {type_name}"))
}

/// `kind=` for `sort`/`argsort`: NumPy's `PyArray_SortkindConverter` matches by lowercase first
/// letter, so any spelling of `quicksort`, `heapsort`, `mergesort`, or `stable` (and anything
/// else starting the same way) is accepted; shellsim always sorts stably, which is a valid
/// result for every one of them.
fn check_sort_kind<'s>(runtime: &mut dyn PyRuntime<'s>, value: PyValue<'s>) -> PyResult<'s, ()> {
    let text = runtime
        .string_value(&value)?
        .ok_or_else(|| kind_type_error(runtime, &value, "sort"))?;
    match text.chars().next().map(|c| c.to_ascii_lowercase()) {
        Some('q' | 'h' | 'm' | 's') => Ok(()),
        _ => Err(PyError::value_error(format!(
            "sort kind must be one of 'quick', 'heap', or 'stable' (got '{text}')"
        ))),
    }
}

/// `kind=` for `partition`/`argpartition`: NumPy accepts only the exact string `'introselect'`
/// (its default), so an explicit `None` is a `TypeError` rather than "use the default".
fn check_select_kind<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    value: Option<PyValue<'s>>,
) -> PyResult<'s, ()> {
    let Some(value) = value else { return Ok(()) };
    let text = runtime
        .string_value(&value)?
        .ok_or_else(|| kind_type_error(runtime, &value, "select"))?;
    if text == "introselect" {
        Ok(())
    } else {
        Err(PyError::value_error(format!(
            "select kind must be 'introselect' (got '{text}')"
        )))
    }
}

fn reject_order<'s>(bound: &args::Bound<'s>) -> PyResult<'s, ()> {
    if bound.value("order").is_some() {
        return Err(PyError::value_error(
            "Cannot specify order when the array has no fields.",
        ));
    }
    Ok(())
}

/// `kind=`/`stable=`/`descending=` for `sort`/`argsort`: NumPy rejects `kind` together with
/// either keyword parameter (whether that parameter is `True` or `False`; an explicit `None` on
/// either side counts as "not given"), and otherwise applies whichever of `stable`/`descending`
/// was given (both default to ascending, stably — this implementation's only mode).
fn sort_direction<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    bound: &args::Bound<'s>,
) -> PyResult<'s, bool> {
    let kind = bound.value("kind");
    let stable = bound.value("stable");
    let descending = bound.value("descending");
    if kind.is_some() && (stable.is_some() || descending.is_some()) {
        return Err(PyError::value_error(
            "`kind` and keyword parameters can't be provided at the same time. Use only one of \
             them.",
        ));
    }
    if let Some(kind) = kind {
        check_sort_kind(runtime, kind)?;
    }
    match descending {
        Some(value) => runtime.truth(&value),
        None => Ok(false),
    }
}

static SORT_SIGNATURE: Signature = Signature::new("sort", &["a", "axis", "kind", "order"], 1)
    .keyword_only(&["stable", "descending"]);
static SORT_METHOD_SIGNATURE: Signature =
    Signature::new("sort", &["axis", "kind", "order"], 0).keyword_only(&["stable", "descending"]);
static ARGSORT_SIGNATURE: Signature = Signature::new("argsort", &["a", "axis", "kind", "order"], 1)
    .keyword_only(&["stable", "descending"]);
static ARGSORT_METHOD_SIGNATURE: Signature =
    Signature::new("argsort", &["axis", "kind", "order"], 0)
        .keyword_only(&["stable", "descending"]);
static PARTITION_SIGNATURE: Signature =
    Signature::new("partition", &["a", "kth", "axis", "kind", "order"], 2);
static PARTITION_METHOD_SIGNATURE: Signature =
    Signature::new("partition", &["kth", "axis", "kind", "order"], 1);
static ARGPARTITION_SIGNATURE: Signature =
    Signature::new("argpartition", &["a", "kth", "axis", "kind", "order"], 2);
static ARGPARTITION_METHOD_SIGNATURE: Signature =
    Signature::new("argpartition", &["kth", "axis", "kind", "order"], 1);

fn module_sort<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    let bound = SORT_SIGNATURE.bind(&args)?;
    reject_order(&bound)?;
    let descending = sort_direction(runtime, &bound)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    let (target, axis) = resolve_axis_flatten(runtime, &array, bound.get("axis"))?;
    let order = lane_orders(runtime, &target, axis, descending)?;
    Ok(sorted_copy(runtime, &target, axis, &order)?.value())
}

fn method_sort<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver_value: PyValue<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    let bound = SORT_METHOD_SIGNATURE.bind(&args)?;
    reject_order(&bound)?;
    let descending = sort_direction(runtime, &bound)?;
    let array = receiver(runtime, receiver_value)?;
    let axis = resolve_axis_strict(runtime, &array, bound.get("axis"))?;
    let order = lane_orders(runtime, &array, axis, descending)?;
    write_sorted(runtime, &array, &array, axis, &order)?;
    Ok(super::super::super::Value::None)
}

fn module_argsort<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    let bound = ARGSORT_SIGNATURE.bind(&args)?;
    reject_order(&bound)?;
    let descending = sort_direction(runtime, &bound)?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    let (target, axis) = resolve_axis_flatten(runtime, &array, bound.get("axis"))?;
    let order = lane_orders(runtime, &target, axis, descending)?;
    Ok(indices_array(runtime, target.shape().to_vec(), axis, &order)?.value())
}

fn method_argsort<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver_value: PyValue<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    let bound = ARGSORT_METHOD_SIGNATURE.bind(&args)?;
    reject_order(&bound)?;
    let descending = sort_direction(runtime, &bound)?;
    let array = receiver(runtime, receiver_value)?;
    let (target, axis) = resolve_axis_flatten(runtime, &array, bound.get("axis"))?;
    let order = lane_orders(runtime, &target, axis, descending)?;
    Ok(indices_array(runtime, target.shape().to_vec(), axis, &order)?.value())
}

/// `kth=`: one `int`, or a sequence/array of them (each checked, since our full sort already
/// puts every position in its final place regardless of which `kth` NumPy would have picked).
fn kth_values<'s>(runtime: &mut dyn PyRuntime<'s>, value: PyValue<'s>) -> PyResult<'s, Vec<i64>> {
    if runtime.native_kind(&value)? == Some(PyNativeKind::Array) {
        let array = Array::from_value(runtime, value)?;
        if array.ndim() == 0 {
            return Ok(vec![kth_int(runtime, &value)?]);
        }
        let cast = convert::cast_array(runtime, &array, DType::INT64, false)?;
        return array::read_elements::<i64>(runtime, &cast);
    }
    match runtime.kind(&value)? {
        PyKind::List => {
            let list = super::super::super::native::PyValueCast::cast(value, runtime)?;
            runtime
                .list_items(list)?
                .iter()
                .map(|item| kth_int(runtime, item))
                .collect()
        }
        PyKind::Tuple => {
            let tuple = super::super::super::native::PyValueCast::cast(value, runtime)?;
            runtime
                .tuple_items(tuple)?
                .iter()
                .map(|item| kth_int(runtime, item))
                .collect()
        }
        _ => Ok(vec![kth_int(runtime, &value)?]),
    }
}

fn kth_int<'s>(runtime: &mut dyn PyRuntime<'s>, value: &PyValue<'s>) -> PyResult<'s, i64> {
    if let Some(value) = runtime.int_value(value) {
        return Ok(value);
    }
    if let Some((dtype, number)) = super::scalar::unbox_number(runtime, value) {
        if super::scalar::is_index_dtype(dtype) {
            return Ok(number.wrapping_i64());
        }
    }
    Err(PyError::type_error("Partition index must be integer"))
}

fn check_kth<'s>(kth: PyValue<'s>, runtime: &mut dyn PyRuntime<'s>, n: usize) -> PyResult<'s, ()> {
    for k in kth_values(runtime, kth)? {
        let normalized = if k < 0 { k + n as i64 } else { k };
        if !(0..n as i64).contains(&normalized) {
            return Err(PyError::value_error(format!(
                "kth(={normalized}) out of bounds ({n})"
            )));
        }
    }
    Ok(())
}

fn module_partition<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    let bound = PARTITION_SIGNATURE.bind(&args)?;
    reject_order(&bound)?;
    check_select_kind(runtime, bound.get("kind"))?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    let (target, axis) = resolve_axis_flatten(runtime, &array, bound.get("axis"))?;
    check_kth(bound.required("kth"), runtime, target.shape()[axis])?;
    let order = lane_orders(runtime, &target, axis, false)?;
    Ok(sorted_copy(runtime, &target, axis, &order)?.value())
}

fn method_partition<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver_value: PyValue<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    let bound = PARTITION_METHOD_SIGNATURE.bind(&args)?;
    reject_order(&bound)?;
    check_select_kind(runtime, bound.get("kind"))?;
    let array = receiver(runtime, receiver_value)?;
    let axis = resolve_axis_strict(runtime, &array, bound.get("axis"))?;
    check_kth(bound.required("kth"), runtime, array.shape()[axis])?;
    let order = lane_orders(runtime, &array, axis, false)?;
    write_sorted(runtime, &array, &array, axis, &order)?;
    Ok(super::super::super::Value::None)
}

fn module_argpartition<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    let bound = ARGPARTITION_SIGNATURE.bind(&args)?;
    reject_order(&bound)?;
    check_select_kind(runtime, bound.get("kind"))?;
    let array = convert::as_array(runtime, bound.required("a"))?;
    let (target, axis) = resolve_axis_flatten(runtime, &array, bound.get("axis"))?;
    check_kth(bound.required("kth"), runtime, target.shape()[axis])?;
    let order = lane_orders(runtime, &target, axis, false)?;
    Ok(indices_array(runtime, target.shape().to_vec(), axis, &order)?.value())
}

fn method_argpartition<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver_value: PyValue<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    let bound = ARGPARTITION_METHOD_SIGNATURE.bind(&args)?;
    reject_order(&bound)?;
    check_select_kind(runtime, bound.get("kind"))?;
    let array = receiver(runtime, receiver_value)?;
    let (target, axis) = resolve_axis_flatten(runtime, &array, bound.get("axis"))?;
    check_kth(bound.required("kth"), runtime, target.shape()[axis])?;
    let order = lane_orders(runtime, &target, axis, false)?;
    Ok(indices_array(runtime, target.shape().to_vec(), axis, &order)?.value())
}

/// One `lexsort` key element, tagged with enough of its dtype family to compare correctly.
/// Columns are homogeneous (each key array has one dtype), so mixing variants never compares.
enum Key<'s> {
    Number(Number),
    Str(Vec<u32>),
    Object(PyValue<'s>),
}

fn number_sort_cmp(a: Number, b: Number) -> Ordering {
    match (a, b) {
        (Number::Bool(a), Number::Bool(b)) => a.cmp(&b),
        (Number::Int(a), Number::Int(b)) => a.cmp(&b),
        (Number::UInt(a), Number::UInt(b)) => a.cmp(&b),
        (Number::Float(a), Number::Float(b)) => real_order(a, b),
        (Number::Complex(ar, ai), Number::Complex(br, bi)) => complex_order((ar, ai), (br, bi)),
        _ => real_order(a.as_f64(), b.as_f64()),
    }
}

fn key_ordering<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    a: &Key<'s>,
    b: &Key<'s>,
) -> PyResult<'s, Ordering> {
    match (a, b) {
        (Key::Number(a), Key::Number(b)) => Ok(number_sort_cmp(*a, *b)),
        (Key::Str(a), Key::Str(b)) => Ok(a.cmp(b)),
        (Key::Object(a), Key::Object(b)) => {
            if less_than(runtime, *a, *b)? {
                Ok(Ordering::Less)
            } else if less_than(runtime, *b, *a)? {
                Ok(Ordering::Greater)
            } else {
                Ok(Ordering::Equal)
            }
        }
        _ => unreachable!("one dtype per key column"),
    }
}

fn read_key_column<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    key: &Array<'s>,
    axis: usize,
) -> PyResult<'s, Vec<Key<'s>>> {
    let offsets = lane_offsets(key.shape(), key.strides(), axis);
    let base = key.view.offset;
    let kind = key.dtype.kind();
    let mut values = Vec::with_capacity(offsets.len());
    if kind == Kind::Object {
        runtime.read_arrays(&[key.handle], &mut |refs, arrays| {
            let PyArrayData::Values(data) = arrays[0].data else {
                return Err(PyError::runtime_error(
                    "lexsort saw byte storage for an object dtype",
                ));
            };
            values.extend(offsets.iter().map(|&relative| {
                Key::Object(refs.handle(
                    &data[((base as isize + relative) as usize) / PyArrayDtype::VALUE_ITEMSIZE],
                ))
            }));
            Ok(())
        })?;
        return Ok(values);
    }
    runtime.read_arrays(&[key.handle], &mut |_refs, arrays| {
        let PyArrayData::Bytes(bytes) = arrays[0].data else {
            return Err(PyError::runtime_error(
                "lexsort saw object storage for a numeric dtype",
            ));
        };
        if kind == Kind::Str {
            let width = key.dtype.itemsize();
            values.extend(offsets.iter().map(|&relative| {
                let offset = (base as isize + relative) as usize;
                Key::Str(
                    bytes[offset..offset + width]
                        .chunks_exact(4)
                        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four bytes")))
                        .collect(),
                )
            }));
        } else {
            dispatch_numeric!(kind, T => {
                values.extend(offsets.iter().map(|&relative| {
                    let offset = (base as isize + relative) as usize;
                    Key::Number(T::read(&bytes[offset..]).to_number())
                }));
            }, _ => unreachable!("Str and Object are handled separately"));
        }
        Ok(())
    })?;
    Ok(values)
}

/// `np.lexsort(keys, axis=-1)`: an ascending-stable order along `axis`, comparing the last key
/// first (most significant) down to the first (a final tie-break).
fn lexicographic_lane_orders<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    keys: &[Array<'s>],
    axis: usize,
) -> PyResult<'s, Vec<usize>> {
    let shape = keys[0].shape();
    let n = shape[axis];
    let lanes = array::element_count(shape)?.checked_div(n).unwrap_or(0);
    runtime.charge_cpu(
        cost_n_log_n(n)
            .saturating_mul(lanes as u64)
            .saturating_mul(keys.len() as u64)
            + 1,
    )?;
    let columns = keys
        .iter()
        .map(|key| read_key_column(runtime, key, axis))
        .collect::<PyResult<'s, Vec<_>>>()?;
    let mut result = Vec::with_capacity(lanes * n);
    for lane in 0..lanes {
        let base = lane * n;
        let mut order: Vec<usize> = (0..n).collect();
        merge_sort_by(&mut order, &mut |&a, &b| {
            for column in columns.iter().rev() {
                match key_ordering(runtime, &column[base + a], &column[base + b])? {
                    Ordering::Less => return Ok(true),
                    Ordering::Greater => return Ok(false),
                    Ordering::Equal => {}
                }
            }
            Ok(false)
        })?;
        result.extend(order);
    }
    Ok(result)
}

fn key_arrays<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    value: PyValue<'s>,
) -> PyResult<'s, Vec<Array<'s>>> {
    match runtime.kind(&value)? {
        PyKind::List => {
            let list = super::super::super::native::PyValueCast::cast(value, runtime)?;
            runtime
                .list_items(list)?
                .into_iter()
                .map(|item| convert::as_array(runtime, item))
                .collect()
        }
        PyKind::Tuple => {
            let tuple = super::super::super::native::PyValueCast::cast(value, runtime)?;
            runtime
                .tuple_items(tuple)?
                .into_iter()
                .map(|item| convert::as_array(runtime, item))
                .collect()
        }
        _ => {
            let array = convert::as_array(runtime, value)?;
            if array.ndim() == 0 {
                return Ok(vec![array]);
            }
            super::ndarray::rows(runtime, &array)?
                .into_iter()
                .map(|row| Array::from_value(runtime, row))
                .collect()
        }
    }
}

fn module_lexsort<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    static SIGNATURE: Signature = Signature::new("lexsort", &["keys", "axis"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let keys = key_arrays(runtime, bound.required("keys"))?;
    let Some(first) = keys.first() else {
        return Err(PyError::type_error(
            "need sequence of keys with len > 0 in lexsort",
        ));
    };
    for key in &keys[1..] {
        if key.shape() != first.shape() {
            return Err(PyError::value_error("all keys need to be the same shape"));
        }
    }
    let axis = match bound.get("axis") {
        None => array::normalize_axis(-1, first.ndim())?,
        Some(value) => array::normalize_axis(args::index_int(runtime, &value)?, first.ndim())?,
    };
    let order = lexicographic_lane_orders(runtime, &keys, axis)?;
    Ok(indices_array(runtime, first.shape().to_vec(), axis, &order)?.value())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_order_puts_nan_after_every_number_and_treats_nan_as_equal() {
        assert_eq!(real_order(1.0f64, 2.0), Ordering::Less);
        assert_eq!(real_order(f64::NAN, 1.0), Ordering::Greater);
        assert_eq!(real_order(1.0, f64::NAN), Ordering::Less);
        assert_eq!(real_order(f64::NAN, f64::NAN), Ordering::Equal);
        assert_eq!(real_order(3i8, -1i8), Ordering::Greater);
    }

    #[test]
    fn complex_order_groups_nan_containing_values_last_by_component() {
        let nan = f64::NAN;
        assert_eq!(complex_order((1.0, 2.0), (2.0, 1.0)), Ordering::Less);
        // A NaN-free value always sorts before a NaN-containing one, even with a smaller real
        // part on the NaN-containing side.
        assert_eq!(complex_order((3.0, 0.0), (2.0, nan)), Ordering::Less);
        assert_eq!(complex_order((2.0, nan), (nan, 0.0)), Ordering::Less);
        assert_eq!(complex_order((nan, 0.0), (nan, 1.0)), Ordering::Less);
        assert_eq!(complex_order((nan, 2.0), (nan, nan)), Ordering::Less);
    }

    #[test]
    fn apply_descending_keeps_nan_suffix_in_place() {
        let order = apply_descending(vec![2, 0, 1, 3, 4], 2, true);
        assert_eq!(order, vec![1, 0, 2, 3, 4]);
        let order = apply_descending(vec![2, 0, 1], 0, false);
        assert_eq!(order, vec![2, 0, 1]);
    }

    #[test]
    fn merge_sort_by_is_stable_and_supports_a_fallible_comparator() {
        let mut items = vec![3usize, 1, 4, 1, 5, 9, 2, 6];
        let source = items.clone();
        merge_sort_by(&mut items, &mut |&a, &b| Ok(a < b)).unwrap();
        let mut expected = source;
        expected.sort();
        assert_eq!(items, expected);
    }

    #[test]
    fn lane_offsets_enumerate_lane_then_position() {
        // shape (2,3), C strides, axis=0: two lanes (one per column) of 2 elements each.
        let offsets = lane_offsets(&[2, 3], &[24, 8], 0);
        assert_eq!(offsets, vec![0, 24, 8, 32, 16, 40]);
        // axis=1: three lanes (one per row) of 3 elements each, in row order.
        let offsets = lane_offsets(&[2, 3], &[24, 8], 1);
        assert_eq!(offsets, vec![0, 8, 16, 24, 32, 40]);
    }
}
