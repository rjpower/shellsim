//! The native module `_scipy_linalg`, which plays the parts of SciPy's compiled linear algebra
//! modules for the frozen `scipy.linalg` package:
//!
//! - `_solve`, `_inv`, `_det`, `_lu`, `_cholesky`, `_qr` and `_bandwidth` follow SciPy 1.18's
//!   C++ `_batched_linalg` module (see [`batched`]), returning results and per-slice statuses
//!   for the Python layer to turn into SciPy's errors and warnings.
//! - `matrix_exponential` is `_internal_matfuncs.matrix_exponential` (see [`expm`]).
//! - `getrf`, `getrs`, `gecon`, `getri`, `trtrs`, `trtri`, `potrf`, `potrs`, `potri`, `gtsv`,
//!   `gbsv`, `lange` and `nrm2` are the f2py LAPACK and BLAS wrappers that SciPy's Python code
//!   and `scipy.linalg.lapack` users call. The frozen `lapack` module gives them f2py's keyword
//!   interface and casts their inputs; here every argument is positional.
//!
//! Inputs arrive as `float32` or `float64` arrays, which select the precision the kernels run
//! in, as SciPy selects its `s` or `d` LAPACK routine. The f2py wrappers return Fortran-ordered
//! arrays, as f2py does, and the batched functions C-ordered ones, as SciPy's C++ module does.
//! Every function charges cubic work per matrix, and reserves its working copies, before running
//! a kernel.

mod batched;
mod expm;
mod lapack;
mod openblas;

use super::super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyError, PyResult, PyRuntime, PyValue,
};
use super::super::super::Value;
use super::super::numpy::{
    array_from_elements, as_array, flag, float_arg, fortran_array_from_elements, index_int,
    read_elements, Array, Bound, DType, Element, Signature,
};
use batched::{QrMode, SliceStatus, Structure};
use lapack::{MatrixNorm, Real};

pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_scipy_linalg",
    functions: FUNCTIONS,
    values: &[],
};

const fn function(
    name: &'static str,
    call: fn(&mut dyn PyRuntime, CallArgs) -> PyResult,
) -> FunctionDef {
    FunctionDef {
        module: "scipy.linalg._batched_linalg",
        name,
        call,
    }
}

static FUNCTIONS: &[FunctionDef] = &[
    function("_solve", solve),
    function("_inv", inv),
    function("_det", det),
    function("_lu", lu),
    function("_cholesky", cholesky),
    function("_qr", qr),
    function("_bandwidth", bandwidth),
    function("matrix_exponential", matrix_exponential),
    function("getrf", getrf),
    function("getrs", getrs),
    function("gecon", gecon),
    function("getri", getri),
    function("trtrs", trtrs),
    function("trtri", trtri),
    function("potrf", potrf),
    function("potrs", potrs),
    function("potri", potri),
    function("gtsv", gtsv),
    function("gbsv", gbsv),
    function("lange", lange),
    function("nrm2", nrm2),
];

/// A precision the kernels compute in.
trait Precision: Real + Element + expm::PadeCoefficients {
    const DTYPE: DType;
}

impl Precision for f64 {
    const DTYPE: DType = DType::FLOAT64;
}

impl Precision for f32 {
    const DTYPE: DType = DType::FLOAT32;
}

/// Run `$call::<T>(...)` for the precision of `$array`.
macro_rules! by_precision {
    ($array:expr, $call:ident($($argument:expr),* $(,)?)) => {{
        let dtype = $array.dtype;
        if dtype == DType::FLOAT64 {
            $call::<f64>($($argument),*)
        } else if dtype == DType::FLOAT32 {
            $call::<f32>($($argument),*)
        } else {
            Err(PyError::type_error(format!(
                "scipy.linalg kernels need float32 or float64 input, not {}",
                dtype.repr()
            )))
        }
    }};
}

/// An operand read as `T`: the batch axes, then a `rows × columns` core in C order. A 1-d
/// operand read with one core axis has a single column.
struct Stack<T> {
    batch: Vec<usize>,
    rows: usize,
    columns: usize,
    data: Vec<T>,
}

impl<T: Copy> Stack<T> {
    fn count(&self) -> usize {
        self.batch.iter().product()
    }

    fn slice(&self, index: usize) -> &[T] {
        let size = self.rows * self.columns;
        &self.data[index * size..(index + 1) * size]
    }
}

/// Read `array` as a stack with `core` (1 or 2) core axes, in precision `T`.
fn stack<T: Precision>(
    runtime: &mut dyn PyRuntime,
    array: &Array,
    core: usize,
    name: &str,
) -> PyResult<Stack<T>> {
    if array.dtype != T::DTYPE {
        return Err(PyError::type_error(format!(
            "{name} must have dtype {}, not {}",
            T::DTYPE.repr(),
            array.dtype.repr()
        )));
    }
    let shape = array.shape();
    if shape.len() < core {
        return Err(PyError::value_error(format!(
            "{name} must have at least {core} dimensions"
        )));
    }
    let batch = shape[..shape.len() - core].to_vec();
    let (rows, columns) = if core == 2 {
        (shape[shape.len() - 2], shape[shape.len() - 1])
    } else {
        (shape[shape.len() - 1], 1)
    };
    let data = read_elements::<T>(runtime, array)?;
    Ok(Stack {
        batch,
        rows,
        columns,
        data,
    })
}

/// A new array of `values` with `shape`.
fn new_array<T: Element>(
    runtime: &mut dyn PyRuntime,
    dtype: DType,
    shape: Vec<usize>,
    values: &[T],
) -> PyResult {
    Ok(array_from_elements(runtime, dtype, shape, values)?.value())
}

/// `batch` followed by the `core` axes.
fn with_core(batch: &[usize], core: &[usize]) -> Vec<usize> {
    batch.iter().chain(core).copied().collect()
}

fn cube(n: usize) -> u64 {
    (n as u64).saturating_pow(3)
}

fn square(n: usize) -> u64 {
    (n as u64).saturating_pow(2)
}

/// Charge `per_matrix` CPU units for each of `count` matrices and reserve `elements` working
/// elements of `T`.
fn charge<T>(
    runtime: &mut dyn PyRuntime,
    count: usize,
    per_matrix: u64,
    elements: usize,
) -> PyResult<()> {
    runtime.reserve_memory(elements.saturating_mul(std::mem::size_of::<T>()))?;
    runtime.charge_cpu((count as u64).saturating_mul(per_matrix).saturating_add(1))
}

fn int(value: usize) -> PyValue {
    Value::Int(i64::try_from(value).unwrap_or(i64::MAX))
}

/// SciPy's list of status dicts, with keys in `convert_vec_status`'s order.
fn statuses(runtime: &mut dyn PyRuntime, statuses: &[SliceStatus]) -> PyResult {
    let mut items = Vec::with_capacity(statuses.len());
    for status in statuses {
        let entries = [
            ("num", int(status.num)),
            ("structure", Value::Int(status.structure.code())),
            ("is_singular", Value::Int(i64::from(status.is_singular))),
            (
                "is_ill_conditioned",
                Value::Int(i64::from(status.is_ill_conditioned)),
            ),
            ("rcond", Value::Float(status.rcond)),
            ("lapack_info", Value::Int(status.lapack_info)),
        ];
        let mut pairs = Vec::with_capacity(entries.len());
        for (key, value) in entries {
            pairs.push((runtime.new_string(key.to_string())?, value));
        }
        items.push(runtime.new_dict(pairs)?);
    }
    runtime.new_list(items)
}

fn int_argument(runtime: &mut dyn PyRuntime, bound: &Bound, name: &str) -> PyResult<i64> {
    index_int(runtime, &bound.required(name))
}

fn flag_argument(runtime: &mut dyn PyRuntime, bound: &Bound, name: &str) -> PyResult<bool> {
    flag(runtime, Some(bound.required(name)), false)
}

/// Charge the work of `count` direct factorizations of order `n` with `nrhs` right-hand
/// sides and the dozen triangular solves a condition estimate takes, reserving `elements`.
fn charge_factorization<T>(
    runtime: &mut dyn PyRuntime,
    count: usize,
    n: usize,
    nrhs: usize,
    elements: usize,
) -> PyResult<()> {
    let per_matrix = cube(n).saturating_add(square(n).saturating_mul(nrhs as u64 + 12));
    charge::<T>(runtime, count, per_matrix, elements)
}

// ---------------------------------------------------------------------------------------------
// The batched functions
// ---------------------------------------------------------------------------------------------

static SOLVE: Signature =
    Signature::new("_solve", &["a", "b", "structure", "lower", "transposed"], 5);

/// `_solve(a, b, structure, lower, transposed)`: `(x, statuses)` for square `a` of shape
/// `(..., n, n)` and `b` of shape `(..., n, nrhs)` with the same batch shape. Banded structure
/// is not supported.
fn solve(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = SOLVE.bind(&args)?;
    let a = as_array(runtime, bound.required("a"))?;
    by_precision!(a, solve_typed(runtime, &bound, &a))
}

fn solve_typed<T: Precision>(runtime: &mut dyn PyRuntime, bound: &Bound, a: &Array) -> PyResult {
    let b = as_array(runtime, bound.required("b"))?;
    let code = int_argument(runtime, bound, "structure")?;
    let lower = flag_argument(runtime, bound, "lower")?;
    let transposed = flag_argument(runtime, bound, "transposed")?;
    let structure = match Structure::from_code(code) {
        Some(Structure::Banded) => {
            return Err(PyError::exception(
                "NotImplementedError",
                "solve(..., assume_a='banded') is not supported by shellsim's SciPy",
            ))
        }
        Some(structure) => structure,
        None => return Err(PyError::value_error(format!("unknown structure {code}"))),
    };
    let a = stack::<T>(runtime, a, 2, "a")?;
    let b = stack::<T>(runtime, &b, 2, "b")?;
    let n = a.rows;
    if a.columns != n || b.rows != n || a.batch != b.batch {
        return Err(PyError::value_error("_solve: incompatible shapes"));
    }
    let (count, nrhs) = (a.count(), b.columns);
    charge_factorization::<T>(runtime, count, n, nrhs, 2 * n * n + n * nrhs)?;
    let (x, found) = batched::solve(
        &a.data, &b.data, count, n, nrhs, structure, lower, transposed,
    );
    let x = new_array(runtime, T::DTYPE, with_core(&b.batch, &[n, nrhs]), &x)?;
    let found = statuses(runtime, &found)?;
    runtime.new_tuple(vec![x, found])
}

static INV: Signature = Signature::new("_inv", &["a", "structure", "lower"], 3);

/// `_inv(a, structure, lower)`: `(inverse, statuses)` for square `a` of shape `(..., n, n)`.
fn inv(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = INV.bind(&args)?;
    let a = as_array(runtime, bound.required("a"))?;
    by_precision!(a, inv_typed(runtime, &bound, &a))
}

fn inv_typed<T: Precision>(runtime: &mut dyn PyRuntime, bound: &Bound, a: &Array) -> PyResult {
    let code = int_argument(runtime, bound, "structure")?;
    let lower = flag_argument(runtime, bound, "lower")?;
    let structure = Structure::from_code(code)
        .filter(|structure| !matches!(structure, Structure::Tridiagonal | Structure::Banded))
        .ok_or_else(|| PyError::value_error(format!("unknown structure {code}")))?;
    let a = stack::<T>(runtime, a, 2, "a")?;
    let n = a.rows;
    if a.columns != n {
        return Err(PyError::value_error("_inv: expected square matrices"));
    }
    let count = a.count();
    charge_factorization::<T>(runtime, count, n, n, 2 * n * n)?;
    let (inverse, found) = batched::inv(&a.data, count, n, structure, lower);
    let inverse = new_array(runtime, T::DTYPE, with_core(&a.batch, &[n, n]), &inverse)?;
    let found = statuses(runtime, &found)?;
    runtime.new_tuple(vec![inverse, found])
}

static ONE_MATRIX: Signature = Signature::new("_scipy_linalg", &["a"], 1);

/// `_det(a)`: the determinants of `a`'s `(n, n)` slices, as an array of the batch shape.
fn det(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = ONE_MATRIX.bind(&args)?;
    let a = as_array(runtime, bound.required("a"))?;
    by_precision!(a, det_typed(runtime, &a))
}

fn det_typed<T: Precision>(runtime: &mut dyn PyRuntime, a: &Array) -> PyResult {
    let a = stack::<T>(runtime, a, 2, "a")?;
    let n = a.rows;
    if a.columns != n {
        return Err(PyError::value_error("_det: expected square matrices"));
    }
    let count = a.count();
    charge::<T>(runtime, count, cube(n), n * n)?;
    let values = (0..count)
        .map(|index| batched::det(a.slice(index), n))
        .collect::<Vec<T>>();
    new_array(runtime, T::DTYPE, a.batch.clone(), &values)
}

static LU: Signature = Signature::new("_lu", &["a", "permute_l"], 2);

/// `_lu(a, permute_l)`: `(perm, L, U)` for `a` of shape `(..., m, n)`, with `perm` an `int32`
/// array of shape `(..., m)`.
fn lu(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = LU.bind(&args)?;
    let a = as_array(runtime, bound.required("a"))?;
    by_precision!(a, lu_typed(runtime, &bound, &a))
}

fn lu_typed<T: Precision>(runtime: &mut dyn PyRuntime, bound: &Bound, a: &Array) -> PyResult {
    let permute_l = flag_argument(runtime, bound, "permute_l")?;
    let a = stack::<T>(runtime, a, 2, "a")?;
    let (m, n) = (a.rows, a.columns);
    let k = m.min(n);
    let count = a.count();
    let per_matrix = (m as u64).saturating_mul(n as u64).saturating_mul(k as u64);
    charge::<T>(runtime, count, per_matrix, 2 * m * n)?;
    let (mut perms, mut lowers, mut uppers) = (Vec::new(), Vec::new(), Vec::new());
    for index in 0..count {
        let factors = batched::lu(a.slice(index), m, n, permute_l);
        perms.extend(factors.perm.iter().map(|row| *row as i32));
        lowers.extend(factors.l);
        uppers.extend(factors.u);
    }
    let perm = new_array(runtime, DType::INT32, with_core(&a.batch, &[m]), &perms)?;
    let l = new_array(runtime, T::DTYPE, with_core(&a.batch, &[m, k]), &lowers)?;
    let u = new_array(runtime, T::DTYPE, with_core(&a.batch, &[k, n]), &uppers)?;
    runtime.new_tuple(vec![perm, l, u])
}

static CHOLESKY: Signature = Signature::new("_cholesky", &["a", "lower", "clean"], 3);

/// `_cholesky(a, lower, clean)`: `(c, statuses)`, where `c` holds each factor with zeros in the
/// other triangle. The first slice that is not positive definite stops the loop.
fn cholesky(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = CHOLESKY.bind(&args)?;
    let a = as_array(runtime, bound.required("a"))?;
    by_precision!(a, cholesky_typed(runtime, &bound, &a))
}

fn cholesky_typed<T: Precision>(runtime: &mut dyn PyRuntime, bound: &Bound, a: &Array) -> PyResult {
    let lower = flag_argument(runtime, bound, "lower")?;
    let a = stack::<T>(runtime, a, 2, "a")?;
    let n = a.rows;
    if a.columns != n {
        return Err(PyError::value_error("_cholesky: expected square matrices"));
    }
    let count = a.count();
    charge::<T>(runtime, count, cube(n), n * n)?;
    let mut values = vec![T::ZERO; a.data.len()];
    let mut found = Vec::new();
    for index in 0..count {
        match batched::cholesky(a.slice(index), n, lower) {
            Ok(factor) => values[index * n * n..(index + 1) * n * n].copy_from_slice(&factor),
            Err(info) => {
                found.push(SliceStatus {
                    num: index,
                    structure: Structure::Detect,
                    is_singular: false,
                    is_ill_conditioned: false,
                    rcond: 0.0,
                    lapack_info: info as i64,
                });
                break;
            }
        }
    }
    let c = new_array(runtime, T::DTYPE, with_core(&a.batch, &[n, n]), &values)?;
    let found = statuses(runtime, &found)?;
    runtime.new_tuple(vec![c, found])
}

static QR: Signature = Signature::new("_qr", &["a", "mode", "pivoting"], 3);

/// `_qr(a, mode, pivoting)`: `(Q, R, tau, jpvt, statuses)` with SciPy's mode codes (1 full,
/// 11 r, 21 raw, 31 economic). `Q` is `None` in mode r, `tau` is `None` outside raw mode and
/// `jpvt` is `None` without pivoting.
fn qr(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = QR.bind(&args)?;
    let a = as_array(runtime, bound.required("a"))?;
    by_precision!(a, qr_typed(runtime, &bound, &a))
}

fn qr_typed<T: Precision>(runtime: &mut dyn PyRuntime, bound: &Bound, a: &Array) -> PyResult {
    let code = int_argument(runtime, bound, "mode")?;
    let pivoting = flag_argument(runtime, bound, "pivoting")?;
    let mode = match code {
        1 => QrMode::Full,
        11 => QrMode::R,
        21 => QrMode::Raw,
        31 => QrMode::Economic,
        _ => return Err(PyError::value_error(format!("unknown QR mode {code}"))),
    };
    let a = stack::<T>(runtime, a, 2, "a")?;
    let (m, n) = (a.rows, a.columns);
    let k = m.min(n);
    let count = a.count();
    let per_matrix = (m as u64)
        .saturating_mul(m.max(n) as u64)
        .saturating_mul(k.max(1) as u64)
        .saturating_mul(2);
    charge::<T>(runtime, count, per_matrix, m * m.max(n) * 2)?;
    let (mut qs, mut rs, mut taus, mut pivots) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for index in 0..count {
        let factors = batched::qr(a.slice(index), m, n, mode, pivoting);
        qs.extend(factors.q.unwrap_or_default());
        rs.extend(factors.r);
        taus.extend(factors.tau.unwrap_or_default());
        let columns = factors.pivots.unwrap_or_default();
        pivots.extend(columns.iter().map(|column| *column as i32));
    }
    let q_columns = match mode {
        QrMode::Full => m,
        QrMode::Economic => k,
        QrMode::Raw => n,
        QrMode::R => 0,
    };
    let q = if mode == QrMode::R {
        Value::None
    } else {
        new_array(runtime, T::DTYPE, with_core(&a.batch, &[m, q_columns]), &qs)?
    };
    let r_rows = if matches!(mode, QrMode::Full | QrMode::R) {
        m
    } else {
        k
    };
    let r = new_array(runtime, T::DTYPE, with_core(&a.batch, &[r_rows, n]), &rs)?;
    let tau = if mode == QrMode::Raw {
        new_array(runtime, T::DTYPE, with_core(&a.batch, &[k]), &taus)?
    } else {
        Value::None
    };
    let jpvt = if pivoting {
        new_array(runtime, DType::INT32, with_core(&a.batch, &[n]), &pivots)?
    } else {
        Value::None
    };
    let found = statuses(runtime, &[])?;
    runtime.new_tuple(vec![q, r, tau, jpvt, found])
}

/// `_bandwidth(nonzero)`: the lower and upper bandwidths of each `(n, m)` slice of a boolean
/// array that marks nonzero elements; Python ints for 2-d input, else `int64` arrays.
fn bandwidth(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = ONE_MATRIX.bind(&args)?;
    let array = as_array(runtime, bound.required("a"))?;
    if array.dtype != DType::BOOL || array.ndim() < 2 {
        return Err(PyError::type_error(
            "_bandwidth expects a boolean array of at least two dimensions",
        ));
    }
    let shape = array.shape().to_vec();
    let (n, m) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    let batch = shape[..shape.len() - 2].to_vec();
    let count: usize = batch.iter().product();
    let values = read_elements::<bool>(runtime, &array)?;
    let (mut lowers, mut uppers) = (Vec::with_capacity(count), Vec::with_capacity(count));
    for index in 0..count {
        let slice = &values[index * n * m..(index + 1) * n * m];
        let (lower, upper) = batched::bandwidth(n, m, |r, c| slice[r * m + c]);
        lowers.push(lower as i64);
        uppers.push(upper as i64);
    }
    if batch.is_empty() {
        return runtime.new_tuple(vec![Value::Int(lowers[0]), Value::Int(uppers[0])]);
    }
    let lower = new_array(runtime, DType::INT64, batch.clone(), &lowers)?;
    let upper = new_array(runtime, DType::INT64, batch, &uppers)?;
    runtime.new_tuple(vec![lower, upper])
}

/// `matrix_exponential(a)`: `(exp(a), info)` for each `(n, n)` slice of `a`; a nonzero `info`
/// reports an exactly singular Padé denominator.
fn matrix_exponential(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = ONE_MATRIX.bind(&args)?;
    let a = as_array(runtime, bound.required("a"))?;
    by_precision!(a, matrix_exponential_typed(runtime, &a))
}

fn matrix_exponential_typed<T: Precision>(runtime: &mut dyn PyRuntime, a: &Array) -> PyResult {
    let a = stack::<T>(runtime, a, 2, "a")?;
    let n = a.rows;
    if a.columns != n {
        return Err(PyError::value_error(
            "matrix_exponential: expected square matrices",
        ));
    }
    let count = a.count();
    charge::<T>(runtime, count, expm::cost(n), 9 * n * n)?;
    let mut values = Vec::with_capacity(a.data.len());
    let mut info = 0;
    for index in 0..count {
        let mut meter = |units: u64| runtime.charge_cpu(units);
        match expm::expm(a.slice(index), n, &mut meter)? {
            Ok(result) => values.extend(result),
            Err(singular) => {
                info = singular;
                break;
            }
        }
    }
    values.resize(a.data.len(), T::ZERO);
    let result = new_array(runtime, T::DTYPE, with_core(&a.batch, &[n, n]), &values)?;
    runtime.new_tuple(vec![result, int(info)])
}

// ---------------------------------------------------------------------------------------------
// The f2py LAPACK and BLAS wrappers
// ---------------------------------------------------------------------------------------------

/// A matrix argument of an f2py wrapper, in column-major order.
struct Matrix<T> {
    rows: usize,
    columns: usize,
    data: Vec<T>,
    /// Whether the argument was 1-d, as a right-hand side may be.
    vector: bool,
}

/// Read `bound[name]` as a matrix of precision `T`; a 1-d argument is a single column when
/// `vector` allows it.
fn matrix<T: Precision>(
    runtime: &mut dyn PyRuntime,
    bound: &Bound,
    name: &str,
    vector: bool,
) -> PyResult<Matrix<T>> {
    let array = as_array(runtime, bound.required(name))?;
    let ndim = array.ndim();
    if ndim != 2 && !(vector && ndim == 1) {
        let expected = if vector { "1-d or 2-d" } else { "2-d" };
        return Err(PyError::value_error(format!(
            "{name} must be a {expected} array"
        )));
    }
    let stack = stack::<T>(runtime, &array, ndim, name)?;
    Ok(Matrix {
        rows: stack.rows,
        columns: stack.columns,
        data: batched::to_column_major(&stack.data, stack.rows, stack.columns),
        vector: ndim == 1,
    })
}

impl<T: Precision> Matrix<T> {
    /// The matrix as f2py returns an output argument: a Fortran-ordered array, or 1-d when the
    /// argument was.
    fn value(&self, runtime: &mut dyn PyRuntime) -> PyResult {
        let shape = if self.vector {
            vec![self.rows]
        } else {
            vec![self.rows, self.columns]
        };
        Ok(fortran_array_from_elements(runtime, T::DTYPE, shape, &self.data)?.value())
    }

    fn square(&self, function: &str) -> PyResult<usize> {
        if self.rows != self.columns {
            return Err(PyError::value_error(format!(
                "{function}: expected a square matrix"
            )));
        }
        Ok(self.rows)
    }
}

/// 0-based pivots from an `int32` or `int64` array.
fn pivots(runtime: &mut dyn PyRuntime, bound: &Bound, n: usize) -> PyResult<Vec<usize>> {
    let array = as_array(runtime, bound.required("piv"))?;
    let values: Vec<i64> = if array.dtype == DType::INT32 {
        read_elements::<i32>(runtime, &array)?
            .into_iter()
            .map(i64::from)
            .collect()
    } else if array.dtype == DType::INT64 {
        read_elements::<i64>(runtime, &array)?
    } else {
        return Err(PyError::type_error("piv must be an integer array"));
    };
    if values.len() < n {
        return Err(PyError::value_error("piv is too short"));
    }
    values[..n]
        .iter()
        .map(|pivot| {
            usize::try_from(*pivot)
                .ok()
                .filter(|pivot| *pivot < n)
                .ok_or_else(|| PyError::value_error("piv holds an out-of-range row index"))
        })
        .collect()
}

/// Define an f2py wrapper: bind its signature and run the typed body for the precision of the
/// named array argument, which the Python wrapper has already cast to the routine's dtype.
macro_rules! f2py {
    ($name:ident, $signature:ident, $precision:literal, $typed:ident) => {
        fn $name(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
            let bound = $signature.bind(&args)?;
            let array = as_array(runtime, bound.required($precision))?;
            by_precision!(array, $typed(runtime, &bound))
        }
    };
}

static GETRF: Signature = Signature::new("getrf", &["a"], 1);
f2py!(getrf, GETRF, "a", getrf_typed);

/// `getrf(a)`: `(lu, piv, info)` with 0-based `int32` pivots.
fn getrf_typed<T: Precision>(runtime: &mut dyn PyRuntime, bound: &Bound) -> PyResult {
    let mut a = matrix::<T>(runtime, bound, "a", false)?;
    let (m, n) = (a.rows, a.columns);
    let k = m.min(n);
    let per_matrix = (m as u64).saturating_mul(n as u64).saturating_mul(k as u64);
    charge::<T>(runtime, 1, per_matrix, 0)?;
    let (pivots, info) = lapack::getf2(m, n, &mut a.data, m);
    let lu = a.value(runtime)?;
    let pivots = pivots.iter().map(|row| *row as i32).collect::<Vec<_>>();
    let piv = new_array(runtime, DType::INT32, vec![pivots.len()], &pivots)?;
    runtime.new_tuple(vec![lu, piv, int(info)])
}

static GETRS: Signature = Signature::new("getrs", &["lu", "piv", "b", "trans"], 4);
f2py!(getrs, GETRS, "lu", getrs_typed);

/// `getrs(lu, piv, b, trans)`: `(x, info)`; `trans` 1 or 2 solves with `Aᵀ`.
fn getrs_typed<T: Precision>(runtime: &mut dyn PyRuntime, bound: &Bound) -> PyResult {
    let lu = matrix::<T>(runtime, bound, "lu", false)?;
    let n = lu.square("getrs")?;
    let mut b = matrix::<T>(runtime, bound, "b", true)?;
    let trans = int_argument(runtime, bound, "trans")?;
    if b.rows != n {
        return Err(PyError::value_error("getrs: incompatible shapes"));
    }
    let pivots = pivots(runtime, bound, n)?;
    charge::<T>(
        runtime,
        1,
        square(n).saturating_mul(b.columns as u64 + 1),
        0,
    )?;
    lapack::getrs(
        trans != 0,
        n,
        b.columns,
        &lu.data,
        n,
        &pivots,
        &mut b.data,
        n,
    );
    let x = b.value(runtime)?;
    runtime.new_tuple(vec![x, Value::Int(0)])
}

static GECON: Signature = Signature::new("gecon", &["a", "anorm", "norm"], 3);
f2py!(gecon, GECON, "a", gecon_typed);

/// A LAPACK norm letter.
fn norm_argument(runtime: &mut dyn PyRuntime, bound: &Bound) -> PyResult<MatrixNorm> {
    let text = runtime
        .string_value(&bound.required("norm"))?
        .ok_or_else(|| PyError::type_error("norm must be a string"))?;
    match text.as_str() {
        "M" | "m" => Ok(MatrixNorm::Max),
        "1" | "O" | "o" => Ok(MatrixNorm::One),
        "I" | "i" => Ok(MatrixNorm::Infinity),
        _ => Err(PyError::exception(
            "NotImplementedError",
            format!("norm={text:?} is not supported by shellsim's SciPy LAPACK wrappers"),
        )),
    }
}

/// `gecon(lu, anorm, norm)`: `(rcond, info)` from `getrf`'s factors, for `norm` `'1'`, `'O'`
/// or `'I'`.
fn gecon_typed<T: Precision>(runtime: &mut dyn PyRuntime, bound: &Bound) -> PyResult {
    let a = matrix::<T>(runtime, bound, "a", false)?;
    let n = a.square("gecon")?;
    let anorm = T::from_f64(float_arg(runtime, &bound.required("anorm"))?);
    let one_norm = match norm_argument(runtime, bound)? {
        MatrixNorm::One => true,
        MatrixNorm::Infinity => false,
        MatrixNorm::Max => return Err(PyError::value_error("gecon: norm must be '1' or 'I'")),
    };
    charge::<T>(runtime, 1, square(n).saturating_mul(12), 2 * n)?;
    let condition = lapack::gecon(one_norm, n, &a.data, n, anorm);
    runtime.new_tuple(vec![
        Value::Float(condition.rcond.to_f64()),
        Value::Int(condition.info),
    ])
}

static GETRI: Signature = Signature::new("getri", &["lu", "piv"], 2);
f2py!(getri, GETRI, "lu", getri_typed);

/// `getri(lu, piv)`: `(inv_a, info)` from `getrf`'s factors.
fn getri_typed<T: Precision>(runtime: &mut dyn PyRuntime, bound: &Bound) -> PyResult {
    let mut lu = matrix::<T>(runtime, bound, "lu", false)?;
    let n = lu.square("getri")?;
    let pivots = pivots(runtime, bound, n)?;
    charge::<T>(runtime, 1, cube(n), n)?;
    let info = lapack::getri(n, &mut lu.data, n, &pivots);
    let inverse = lu.value(runtime)?;
    runtime.new_tuple(vec![inverse, int(info)])
}

static TRTRS: Signature = Signature::new("trtrs", &["a", "b", "lower", "trans", "unitdiag"], 5);
f2py!(trtrs, TRTRS, "a", trtrs_typed);

/// `trtrs(a, b, lower, trans, unitdiag)`: `(x, info)` for triangular `a`.
fn trtrs_typed<T: Precision>(runtime: &mut dyn PyRuntime, bound: &Bound) -> PyResult {
    let a = matrix::<T>(runtime, bound, "a", false)?;
    let n = a.square("trtrs")?;
    let mut b = matrix::<T>(runtime, bound, "b", true)?;
    let lower = flag_argument(runtime, bound, "lower")?;
    let trans = int_argument(runtime, bound, "trans")?;
    let unit = flag_argument(runtime, bound, "unitdiag")?;
    if b.rows != n {
        return Err(PyError::value_error("trtrs: incompatible shapes"));
    }
    charge::<T>(
        runtime,
        1,
        square(n).saturating_mul(b.columns as u64 + 1),
        0,
    )?;
    let columns = b.columns;
    let info = lapack::trtrs(
        !lower,
        trans != 0,
        unit,
        n,
        columns,
        &a.data,
        n,
        &mut b.data,
        n,
    );
    let x = b.value(runtime)?;
    runtime.new_tuple(vec![x, int(info)])
}

static TRTRI: Signature = Signature::new("trtri", &["c", "lower", "unitdiag"], 3);
f2py!(trtri, TRTRI, "c", trtri_typed);

/// `trtri(c, lower, unitdiag)`: `(inv_c, info)` for triangular `c`.
fn trtri_typed<T: Precision>(runtime: &mut dyn PyRuntime, bound: &Bound) -> PyResult {
    let mut c = matrix::<T>(runtime, bound, "c", false)?;
    let n = c.square("trtri")?;
    let lower = flag_argument(runtime, bound, "lower")?;
    let unit = flag_argument(runtime, bound, "unitdiag")?;
    charge::<T>(runtime, 1, cube(n), 0)?;
    let info = lapack::trti2_checked(!lower, unit, n, &mut c.data, n);
    let inverse = c.value(runtime)?;
    runtime.new_tuple(vec![inverse, int(info)])
}

static POTRF: Signature = Signature::new("potrf", &["a", "lower", "clean"], 3);
f2py!(potrf, POTRF, "a", potrf_typed);

/// `potrf(a, lower, clean)`: `(c, info)`; `clean` zeroes the triangle the factor does not use.
fn potrf_typed<T: Precision>(runtime: &mut dyn PyRuntime, bound: &Bound) -> PyResult {
    let mut a = matrix::<T>(runtime, bound, "a", false)?;
    let n = a.square("potrf")?;
    let lower = flag_argument(runtime, bound, "lower")?;
    let clean = flag_argument(runtime, bound, "clean")?;
    charge::<T>(runtime, 1, cube(n), 0)?;
    let info = lapack::potf2(!lower, n, &mut a.data, n);
    if clean {
        for j in 0..n {
            for i in j + 1..n {
                let other = if lower { j + i * n } else { i + j * n };
                a.data[other] = T::ZERO;
            }
        }
    }
    let c = a.value(runtime)?;
    runtime.new_tuple(vec![c, int(info)])
}

static POTRS: Signature = Signature::new("potrs", &["c", "b", "lower"], 3);
f2py!(potrs, POTRS, "c", potrs_typed);

/// `potrs(c, b, lower)`: `(x, info)` from `potrf`'s factor.
fn potrs_typed<T: Precision>(runtime: &mut dyn PyRuntime, bound: &Bound) -> PyResult {
    let c = matrix::<T>(runtime, bound, "c", false)?;
    let n = c.square("potrs")?;
    let mut b = matrix::<T>(runtime, bound, "b", true)?;
    let lower = flag_argument(runtime, bound, "lower")?;
    if b.rows != n {
        return Err(PyError::value_error("potrs: incompatible shapes"));
    }
    charge::<T>(
        runtime,
        1,
        square(n).saturating_mul(b.columns as u64 + 1),
        0,
    )?;
    lapack::potrs(!lower, n, b.columns, &c.data, n, &mut b.data, n);
    let x = b.value(runtime)?;
    runtime.new_tuple(vec![x, Value::Int(0)])
}

static POTRI: Signature = Signature::new("potri", &["c", "lower"], 2);
f2py!(potri, POTRI, "c", potri_typed);

/// `potri(c, lower)`: `(inv_a, info)`, the matching triangle of `A⁻¹` from `potrf`'s factor.
fn potri_typed<T: Precision>(runtime: &mut dyn PyRuntime, bound: &Bound) -> PyResult {
    let mut c = matrix::<T>(runtime, bound, "c", false)?;
    let n = c.square("potri")?;
    let lower = flag_argument(runtime, bound, "lower")?;
    charge::<T>(runtime, 1, cube(n), 0)?;
    let info = lapack::potri(!lower, n, &mut c.data, n);
    let inverse = c.value(runtime)?;
    runtime.new_tuple(vec![inverse, int(info)])
}

static GTSV: Signature = Signature::new("gtsv", &["dl", "d", "du", "b"], 4);
f2py!(gtsv, GTSV, "d", gtsv_typed);

/// `gtsv(dl, d, du, b)`: `(du2, d, du, x, info)` for the tridiagonal system.
fn gtsv_typed<T: Precision>(runtime: &mut dyn PyRuntime, bound: &Bound) -> PyResult {
    let mut diagonals = Vec::with_capacity(3);
    for name in ["dl", "d", "du"] {
        let array = as_array(runtime, bound.required(name))?;
        if array.ndim() != 1 {
            return Err(PyError::value_error(format!("{name} must be a 1-d array")));
        }
        diagonals.push(stack::<T>(runtime, &array, 1, name)?.data);
    }
    let [mut dl, mut d, mut du]: [Vec<T>; 3] = diagonals
        .try_into()
        .map_err(|_| PyError::runtime_error("gtsv: missing diagonal"))?;
    let n = d.len();
    let mut b = matrix::<T>(runtime, bound, "b", true)?;
    if n == 0 || dl.len() + 1 != n || du.len() + 1 != n || b.rows != n {
        return Err(PyError::value_error("gtsv: incompatible shapes"));
    }
    charge::<T>(
        runtime,
        1,
        (n as u64).saturating_mul(b.columns as u64 + 4),
        0,
    )?;
    let info = lapack::gtsv(&mut dl, &mut d, &mut du, b.columns, &mut b.data, n);
    // `dgtsv` leaves the second superdiagonal of U in `dl`, which f2py returns as `du2`.
    let du2 = new_array(runtime, T::DTYPE, vec![dl.len()], &dl)?;
    let d = new_array(runtime, T::DTYPE, vec![n], &d)?;
    let du = new_array(runtime, T::DTYPE, vec![du.len()], &du)?;
    let x = b.value(runtime)?;
    runtime.new_tuple(vec![du2, d, du, x, int(info)])
}

static GBSV: Signature = Signature::new("gbsv", &["kl", "ku", "ab", "b"], 4);
f2py!(gbsv, GBSV, "ab", gbsv_typed);

/// `gbsv(kl, ku, ab, b)`: `(lu, piv, x, info)` for a band matrix in LAPACK band storage with
/// `kl` rows of workspace on top.
fn gbsv_typed<T: Precision>(runtime: &mut dyn PyRuntime, bound: &Bound) -> PyResult {
    let kl = int_argument(runtime, bound, "kl")?;
    let ku = int_argument(runtime, bound, "ku")?;
    let (Ok(kl), Ok(ku)) = (usize::try_from(kl), usize::try_from(ku)) else {
        return Err(PyError::value_error("gbsv: kl and ku must be nonnegative"));
    };
    let mut ab = matrix::<T>(runtime, bound, "ab", false)?;
    let mut b = matrix::<T>(runtime, bound, "b", true)?;
    let (ldab, n) = (ab.rows, ab.columns);
    if Some(ldab) != kl.checked_mul(2).and_then(|rows| rows.checked_add(ku + 1)) || b.rows != n {
        return Err(PyError::value_error("gbsv: incompatible shapes"));
    }
    let per_matrix = (n as u64)
        .saturating_mul(kl as u64 + 1)
        .saturating_mul((kl + ku + 1 + b.columns) as u64);
    charge::<T>(runtime, 1, per_matrix, 0)?;
    let (pivots, info) = lapack::gbtf2(n, kl, ku, &mut ab.data, ldab);
    if info == 0 {
        let columns = b.columns;
        lapack::gbtrs(n, kl, ku, columns, &ab.data, ldab, &pivots, &mut b.data, n);
    }
    let lu = ab.value(runtime)?;
    let pivots = pivots.iter().map(|row| *row as i32).collect::<Vec<_>>();
    let piv = new_array(runtime, DType::INT32, vec![pivots.len()], &pivots)?;
    let x = b.value(runtime)?;
    runtime.new_tuple(vec![lu, piv, x, int(info)])
}

static LANGE: Signature = Signature::new("lange", &["norm", "a"], 2);
f2py!(lange, LANGE, "a", lange_typed);

/// `lange(norm, a)`: the matrix norm `'M'`, `'1'` (or `'O'`) or `'I'` of `a`.
fn lange_typed<T: Precision>(runtime: &mut dyn PyRuntime, bound: &Bound) -> PyResult {
    let norm = norm_argument(runtime, bound)?;
    let a = matrix::<T>(runtime, bound, "a", false)?;
    charge::<T>(
        runtime,
        1,
        (a.rows as u64).saturating_mul(a.columns as u64),
        a.rows,
    )?;
    let value = lapack::lange(norm, a.rows, a.columns, &a.data);
    Ok(Value::Float(value.to_f64()))
}

static NRM2: Signature = Signature::new("nrm2", &["x"], 1);
f2py!(nrm2, NRM2, "x", nrm2_typed);

/// `nrm2(x)`: the Euclidean norm of a vector, as OpenBLAS computes it.
fn nrm2_typed<T: Precision>(runtime: &mut dyn PyRuntime, bound: &Bound) -> PyResult {
    let x = as_array(runtime, bound.required("x"))?;
    if x.ndim() != 1 {
        return Err(PyError::value_error("x must be a 1-d array"));
    }
    let values = stack::<T>(runtime, &x, 1, "x")?.data;
    runtime.charge_cpu(values.len() as u64 + 1)?;
    Ok(Value::Float(T::nrm2(&values).to_f64()))
}
