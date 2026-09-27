//! The native module `_numpy_linalg`, which plays the part of NumPy's `_umath_linalg`: one
//! function per LAPACK-backed generalized ufunc that `numpy/linalg.py` calls, such as `inv`,
//! `solve1`, `eigh_lo`, `svd_f`, and `lstsq`.
//!
//! Each function converts its operands to float64, loops over the leading (batch) axes, runs a
//! kernel from [`dense`] on each matrix, and returns float64 results that the Python layer
//! casts back to the input's precision, as NumPy computes `float32` input in double precision.
//! A 0-d result is a NumPy scalar, as a ufunc returns. Operand checks and their messages
//! follow NumPy's gufunc machinery; the Python layer checks squareness and rank first, so most
//! callers see the `numpy.linalg` messages instead.
//!
//! Complex operands are rejected explicitly for now, and `eig` and `eigvals`, which return
//! complex results, are not provided. Kernels are charged before they run: cubic work per
//! matrix for the direct methods, and per sweep inside the Jacobi methods. Working copies are
//! reserved once per call, since each batch element reuses the same amount.

mod dense;

pub(in crate::python) use dense::norm2;

use super::super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyError, PyResult, PyRuntime, PyValue,
};
use super::args::{self, Signature};
use super::array;
use super::convert;
use super::dtype::{self, Casting, Category, DType};
use super::element::Element;
use super::ops::FpFlags;
use dense::{Matrix, SvdVectors};

pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_numpy_linalg",
    functions: FUNCTIONS,
    values: &[],
};

const fn function(
    name: &'static str,
    call: fn(&mut dyn PyRuntime, CallArgs) -> PyResult,
) -> FunctionDef {
    FunctionDef {
        module: "numpy.linalg._umath_linalg",
        name,
        call,
    }
}

static FUNCTIONS: &[FunctionDef] = &[
    function("inv", inv),
    function("solve", solve),
    function("solve1", solve1),
    function("det", det),
    function("slogdet", slogdet),
    function("cholesky_lo", cholesky_lo),
    function("cholesky_up", cholesky_up),
    function("eigh_lo", eigh_lo),
    function("eigh_up", eigh_up),
    function("eigvalsh_lo", eigvalsh_lo),
    function("eigvalsh_up", eigvalsh_up),
    function("svd", svd_values),
    function("svd_s", svd_s),
    function("svd_f", svd_f),
    function("qr_r_raw", qr_r_raw),
    function("qr_reduced", qr_reduced),
    function("qr_complete", qr_complete),
    function("lstsq", lstsq),
];

/// A generalized ufunc's name, input count, and the signature its errors print.
struct Gufunc {
    name: &'static str,
    inputs: usize,
    signature: &'static str,
}

const fn gufunc(name: &'static str, inputs: usize, signature: &'static str) -> Gufunc {
    Gufunc {
        name,
        inputs,
        signature,
    }
}

static INV: Gufunc = gufunc("inv", 1, "(m, m)->(m, m)");
static SOLVE: Gufunc = gufunc("solve", 2, "(m,m),(m,n)->(m,n)");
static SOLVE1: Gufunc = gufunc("solve1", 2, "(m,m),(m)->(m)");
static DET: Gufunc = gufunc("det", 1, "(m,m)->()");
static SLOGDET: Gufunc = gufunc("slogdet", 1, "(m,m)->(),()");
static CHOLESKY_LO: Gufunc = gufunc("cholesky_lo", 1, "(m,m)->(m,m)");
static CHOLESKY_UP: Gufunc = gufunc("cholesky_up", 1, "(m,m)->(m,m)");
static EIGH_LO: Gufunc = gufunc("eigh_lo", 1, "(m,m)->(m),(m,m)");
static EIGH_UP: Gufunc = gufunc("eigh_up", 1, "(m,m)->(m),(m,m)");
static EIGVALSH_LO: Gufunc = gufunc("eigvalsh_lo", 1, "(m,m)->(m)");
static EIGVALSH_UP: Gufunc = gufunc("eigvalsh_up", 1, "(m,m)->(m)");
static SVD: Gufunc = gufunc("svd", 1, "(m,n)->(p)");
static SVD_S: Gufunc = gufunc("svd_s", 1, "(m,n)->(m,p),(p),(p,n)");
static SVD_F: Gufunc = gufunc("svd_f", 1, "(m,n)->(m,m),(p),(n,n)");
static QR_R_RAW: Gufunc = gufunc("qr_r_raw", 1, "(m,n)->(p)");
static QR_REDUCED: Gufunc = gufunc("qr_reduced", 2, "(m,n),(k)->(m,k)");
static QR_COMPLETE: Gufunc = gufunc("qr_complete", 2, "(m,n),(n)->(m,m)");
static LSTSQ: Gufunc = gufunc("lstsq", 3, "(m,n),(m,nrhs),()->(n,nrhs),(nrhs),(),(p)");

static ONE_ARGUMENT: Signature = Signature::new("_umath_linalg", &["a"], 1);
static TWO_ARGUMENTS: Signature = Signature::new("_umath_linalg", &["a", "b"], 2);
static LSTSQ_ARGUMENTS: Signature = Signature::new("lstsq", &["a", "b", "rcond"], 3);

/// One operand read as float64 in C order: a stack of matrices over its batch axes. A 1-d core
/// operand, such as `solve1`'s right-hand side, is a stack of single-column matrices.
struct Stack {
    shape: Vec<usize>,
    batch: Vec<usize>,
    rows: usize,
    columns: usize,
    data: Vec<f64>,
}

impl Stack {
    fn count(&self) -> usize {
        self.batch.iter().product()
    }

    fn values(&self, index: usize) -> &[f64] {
        let size = self.rows * self.columns;
        &self.data[index * size..(index + 1) * size]
    }

    fn matrix(&self, index: usize) -> Matrix {
        Matrix::new(self.rows, self.columns, self.values(index).to_vec())
    }
}

/// Read input `position` of `gufunc`, which has `core` core axes, as float64.
fn operand(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
    gufunc: &Gufunc,
    position: usize,
    core: usize,
) -> PyResult<Stack> {
    let array = convert::as_array(runtime, value)?;
    let ndim = array.ndim();
    if ndim < core {
        return Err(PyError::value_error(format!(
            "{}: Input operand {position} does not have enough dimensions (has {ndim}, gufunc \
             core with signature {} requires {core})",
            gufunc.name, gufunc.signature
        )));
    }
    if array.dtype.category() == Category::Complex {
        return Err(PyError::unsupported(format!(
            "numpy.linalg does not support complex arrays yet ({} input)",
            gufunc.name
        )));
    }
    if !dtype::can_cast(array.dtype, DType::FLOAT64, Casting::SameKind) {
        let position = if gufunc.inputs == 1 {
            String::new()
        } else {
            format!("{position} ")
        };
        return Err(PyError::exception(
            "UFuncTypeError",
            format!(
                "Cannot cast ufunc '{}' input {position}from {} to {} with casting rule \
                 'same_kind'",
                gufunc.name,
                array.dtype.repr(),
                DType::FLOAT64.repr()
            ),
        ));
    }
    let cast = convert::cast_array(runtime, &array, DType::FLOAT64, false)?;
    let data = array::read_elements::<f64>(runtime, &cast)?;
    let shape = array.shape().to_vec();
    let (rows, columns) = match core {
        2 => (shape[ndim - 2], shape[ndim - 1]),
        1 => (shape[ndim - 1], 1),
        _ => (1, 1),
    };
    Ok(Stack {
        batch: shape[..ndim - core].to_vec(),
        shape,
        rows,
        columns,
        data,
    })
}

/// NumPy's error when two core axes with the same name have different sizes.
fn core_mismatch(
    gufunc: &Gufunc,
    position: usize,
    axis: usize,
    size: usize,
    expected: usize,
) -> PyError {
    PyError::value_error(format!(
        "{}: Input operand {position} has a mismatch in its core dimension {axis}, with gufunc \
         signature {} (size {size} is different from {expected})",
        gufunc.name, gufunc.signature
    ))
}

/// Read a stack of square matrices, the `(m,m)` operand every square gufunc takes first.
fn square_operand(runtime: &mut dyn PyRuntime, value: PyValue, gufunc: &Gufunc) -> PyResult<Stack> {
    let stack = operand(runtime, value, gufunc, 0, 2)?;
    if stack.rows != stack.columns {
        return Err(core_mismatch(gufunc, 0, 1, stack.columns, stack.rows));
    }
    Ok(stack)
}

/// Charge `per_matrix` CPU units for each of `count` matrices and reserve `working` bytes.
fn charge(
    runtime: &mut dyn PyRuntime,
    count: usize,
    per_matrix: u64,
    working: usize,
) -> PyResult<()> {
    runtime.reserve_memory(working)?;
    runtime.charge_cpu((count as u64).saturating_mul(per_matrix).saturating_add(1))
}

fn cube(size: usize) -> u64 {
    (size as u64).saturating_pow(3)
}

/// Bytes of `copies` float64 matrices of `rows × columns`.
fn matrices(rows: usize, columns: usize, copies: usize) -> usize {
    rows.saturating_mul(columns)
        .saturating_mul(copies)
        .saturating_mul(std::mem::size_of::<f64>())
}

/// A result of shape `batch + core` built from typed values; a 0-d result is a NumPy scalar.
fn output<T: Element>(
    runtime: &mut dyn PyRuntime,
    dtype: DType,
    batch: &[usize],
    core: &[usize],
    values: &[T],
) -> PyResult {
    let mut shape = batch.to_vec();
    shape.extend_from_slice(core);
    let result = array::array_from_elements(runtime, dtype, shape, values)?;
    if result.ndim() == 0 {
        return convert::element_to_scalar(runtime, &result, result.view.offset);
    }
    Ok(result.value())
}

fn float_output(
    runtime: &mut dyn PyRuntime,
    batch: &[usize],
    core: &[usize],
    values: &[f64],
) -> PyResult {
    output(runtime, DType::FLOAT64, batch, core, values)
}

fn linalg_error(message: &str) -> PyError {
    PyError::exception("LinAlgError", message)
}

/// `inv(a)`: each matrix's inverse, from `dgesv` against the identity.
fn inv(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = ONE_ARGUMENT.bind(&args)?;
    let a = square_operand(runtime, bound.required("a"), &INV)?;
    let n = a.rows;
    let per_matrix = cube(n).saturating_mul(2);
    charge(runtime, a.count(), per_matrix, matrices(n, n, 3))?;
    let mut values = Vec::with_capacity(a.data.len());
    for index in 0..a.count() {
        let factors = dense::lu(a.matrix(index));
        if factors.singular {
            return Err(linalg_error("Singular matrix"));
        }
        let mut inverse = Matrix::identity(n);
        factors.solve(&mut inverse);
        values.extend(inverse.data);
    }
    float_output(runtime, &a.batch, &[n, n], &values)
}

/// `solve(a, b)` for a stack of right-hand-side matrices.
fn solve(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    solve_with(runtime, args, &SOLVE, 2)
}

/// `solve1(a, b)` for a stack of right-hand-side vectors.
fn solve1(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    solve_with(runtime, args, &SOLVE1, 1)
}

fn solve_with(
    runtime: &mut dyn PyRuntime,
    args: CallArgs,
    gufunc: &Gufunc,
    b_core: usize,
) -> PyResult {
    let bound = TWO_ARGUMENTS.bind(&args)?;
    let a = square_operand(runtime, bound.required("a"), gufunc)?;
    let b = operand(runtime, bound.required("b"), gufunc, 1, b_core)?;
    if b.rows != a.rows {
        return Err(core_mismatch(gufunc, 1, 0, b.rows, a.rows));
    }
    let output_core = if b_core == 2 {
        vec![b.rows, b.columns]
    } else {
        vec![b.rows]
    };
    let batch = broadcast_batches(&a, &b, &output_core)?;
    let count = batch.iter().product::<usize>();
    let n = a.rows;
    let per_matrix = cube(n).saturating_add((n as u64).pow(2).saturating_mul(b.columns as u64));
    let working = matrices(n, n + b.columns, 1).saturating_add(matrices(n, b.columns, count));
    charge(runtime, count, per_matrix, working)?;
    let mut values = Vec::with_capacity(count.saturating_mul(n * b.columns));
    for index in 0..count {
        let factors = dense::lu(a.matrix(batch_index(index, &batch, &a.batch)));
        if factors.singular {
            return Err(linalg_error("Singular matrix"));
        }
        let mut solution = b.matrix(batch_index(index, &batch, &b.batch));
        factors.solve(&mut solution);
        values.extend(solution.data);
    }
    float_output(runtime, &batch, &output_core, &values)
}

/// Broadcast the batch axes of two operands, with NumPy's gufunc error on a mismatch. The
/// error lists each operand's batch axes followed by one `newaxis` per output core axis.
fn broadcast_batches(a: &Stack, b: &Stack, output_core: &[usize]) -> PyResult<Vec<usize>> {
    array::broadcast_shapes(&[&a.batch, &b.batch]).map_err(|_| {
        let remapped = |stack: &Stack| {
            let parts = stack
                .batch
                .iter()
                .map(ToString::to_string)
                .chain(std::iter::repeat_n(
                    "newaxis".to_string(),
                    output_core.len(),
                ))
                .collect::<Vec<_>>();
            format!(
                "{}->({})",
                array::format_shape(&stack.shape),
                parts.join(",")
            )
        };
        let requested = output_core
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        PyError::value_error(format!(
            "operands could not be broadcast together with remapped shapes \
             [original->remapped]: {} {}  and requested shape ({})",
            remapped(a),
            remapped(b),
            requested.join(",")
        ))
    })
}

/// The index into `operand`'s batch of the `flat`-th position of the broadcast `batch`.
fn batch_index(mut flat: usize, batch: &[usize], operand: &[usize]) -> usize {
    let leading = batch.len() - operand.len();
    let (mut index, mut stride) = (0, 1);
    for axis in (0..batch.len()).rev() {
        let coordinate = flat % batch[axis];
        flat /= batch[axis];
        if axis >= leading {
            let size = operand[axis - leading];
            if size != 1 {
                index += coordinate * stride;
            }
            stride *= size;
        }
    }
    index
}

/// `(sign, log|det|)` of each matrix. Input containing NaN reports an invalid operation under
/// the gufunc's name, as LAPACK's pivot search raises the hardware flag NumPy checks.
fn signed_logdets(
    runtime: &mut dyn PyRuntime,
    args: CallArgs,
    gufunc: &Gufunc,
) -> PyResult<(Vec<usize>, Vec<f64>, Vec<f64>)> {
    let bound = ONE_ARGUMENT.bind(&args)?;
    let a = square_operand(runtime, bound.required("a"), gufunc)?;
    let n = a.rows;
    charge(runtime, a.count(), cube(n), matrices(n, n, 1))?;
    let count = a.count();
    let (mut signs, mut logdets) = (Vec::with_capacity(count), Vec::with_capacity(count));
    for index in 0..count {
        let (sign, logdet) = dense::lu(a.matrix(index)).slogdet();
        signs.push(sign);
        logdets.push(logdet);
    }
    let flags = FpFlags {
        invalid: a.data.iter().any(|value| value.is_nan()),
        ..FpFlags::default()
    };
    super::errstate::report(runtime, gufunc.name, flags)?;
    Ok((a.batch, signs, logdets))
}

/// `det(a)`: `sign * exp(log|det|)`, which is where NumPy's rounding comes from.
fn det(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let (batch, signs, logdets) = signed_logdets(runtime, args, &DET)?;
    let values = signs
        .iter()
        .zip(&logdets)
        .map(|(sign, logdet)| sign * logdet.exp())
        .collect::<Vec<_>>();
    float_output(runtime, &batch, &[], &values)
}

fn slogdet(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let (batch, signs, logdets) = signed_logdets(runtime, args, &SLOGDET)?;
    let sign = float_output(runtime, &batch, &[], &signs)?;
    let logdet = float_output(runtime, &batch, &[], &logdets)?;
    runtime.new_tuple(vec![sign, logdet])
}

fn cholesky_lo(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    cholesky(runtime, args, &CHOLESKY_LO, false)
}

fn cholesky_up(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    cholesky(runtime, args, &CHOLESKY_UP, true)
}

/// `L` with `A = L Lᵀ` from the lower triangle, or `U = Lᵀ` with `A = Uᵀ U` from the upper
/// triangle; the other triangle of the result is zero.
fn cholesky(runtime: &mut dyn PyRuntime, args: CallArgs, gufunc: &Gufunc, upper: bool) -> PyResult {
    let bound = ONE_ARGUMENT.bind(&args)?;
    let a = square_operand(runtime, bound.required("a"), gufunc)?;
    let n = a.rows;
    charge(runtime, a.count(), cube(n), matrices(n, n, 2))?;
    let mut values = Vec::with_capacity(a.data.len());
    for index in 0..a.count() {
        let matrix = if upper {
            a.matrix(index).transpose()
        } else {
            a.matrix(index)
        };
        let lower = dense::cholesky(&matrix)
            .ok_or_else(|| linalg_error("Matrix is not positive definite"))?;
        values.extend(if upper { lower.transpose() } else { lower }.data);
    }
    float_output(runtime, &a.batch, &[n, n], &values)
}

fn eigh_lo(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    eigh(runtime, args, &EIGH_LO, false, true)
}

fn eigh_up(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    eigh(runtime, args, &EIGH_UP, true, true)
}

fn eigvalsh_lo(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    eigh(runtime, args, &EIGVALSH_LO, false, false)
}

fn eigvalsh_up(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    eigh(runtime, args, &EIGVALSH_UP, true, false)
}

/// Eigenvalues, and eigenvectors if `vectors`, of symmetric matrices read from one triangle.
/// A matrix with a non-finite element in that triangle gives NaN eigenvalues and identity
/// eigenvectors, as NumPy's `dsyevd` call does for infinite input.
fn eigh(
    runtime: &mut dyn PyRuntime,
    args: CallArgs,
    gufunc: &Gufunc,
    upper: bool,
    vectors: bool,
) -> PyResult {
    let bound = ONE_ARGUMENT.bind(&args)?;
    let a = square_operand(runtime, bound.required("a"), gufunc)?;
    let n = a.rows;
    let count = a.count();
    let result_bytes = if vectors { a.data.len() * 8 } else { 0 };
    let working = matrices(n, n, 3).saturating_add(result_bytes);
    charge(runtime, count, (n as u64).saturating_pow(2), working)?;
    let mut eigenvalues = Vec::with_capacity(count.saturating_mul(n));
    let mut eigenvectors = Vec::with_capacity(if vectors { a.data.len() } else { 0 });
    for index in 0..count {
        let matrix = if upper {
            a.matrix(index).transpose()
        } else {
            a.matrix(index)
        };
        let finite = (0..n).all(|row| (0..=row).all(|column| matrix.get(row, column).is_finite()));
        if !finite {
            eigenvalues.extend(std::iter::repeat_n(f64::NAN, n));
            if vectors {
                eigenvectors.extend(Matrix::identity(n).data);
            }
            continue;
        }
        let (values, basis) =
            dense::symmetric_eigen(&matrix, vectors, &mut |cost| runtime.charge_cpu(cost))?;
        eigenvalues.extend(values);
        if let Some(basis) = basis {
            eigenvectors.extend(basis.data);
        }
    }
    let eigenvalues = float_output(runtime, &a.batch, &[n], &eigenvalues)?;
    if !vectors {
        return Ok(eigenvalues);
    }
    let eigenvectors = float_output(runtime, &a.batch, &[n, n], &eigenvectors)?;
    runtime.new_tuple(vec![eigenvalues, eigenvectors])
}

fn svd_values(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    svd(runtime, args, &SVD, SvdVectors::None)
}

fn svd_s(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    svd(runtime, args, &SVD_S, SvdVectors::Reduced)
}

fn svd_f(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    svd(runtime, args, &SVD_F, SvdVectors::Full)
}

/// Singular values, and `(U, S, Vh)` unless `mode` is [`SvdVectors::None`]. Non-finite input
/// does not converge, as in LAPACK.
fn svd(runtime: &mut dyn PyRuntime, args: CallArgs, gufunc: &Gufunc, mode: SvdVectors) -> PyResult {
    let bound = ONE_ARGUMENT.bind(&args)?;
    let a = operand(runtime, bound.required("a"), gufunc, 0, 2)?;
    let (m, n) = (a.rows, a.columns);
    let k = m.min(n);
    let (u_columns, v_rows) = match mode {
        SvdVectors::Full => (m, n),
        SvdVectors::Reduced | SvdVectors::None => (k, k),
    };
    let count = a.count();
    let results = if mode == SvdVectors::None {
        0
    } else {
        matrices(m, u_columns, count).saturating_add(matrices(v_rows, n, count))
    };
    let square = m.max(n);
    let working = matrices(square, square, 4).saturating_add(results);
    charge(runtime, count, 1, working)?;
    let mut singular_values = Vec::with_capacity(count.saturating_mul(k));
    let (mut left, mut right) = (Vec::new(), Vec::new());
    for index in 0..count {
        if !a.values(index).iter().all(|value| value.is_finite()) {
            return Err(linalg_error("SVD did not converge"));
        }
        let result = dense::svd(&a.matrix(index), mode, &mut |cost| runtime.charge_cpu(cost))?;
        singular_values.extend(result.values);
        if let Some((u, vt)) = result.vectors {
            left.extend(u.data);
            right.extend(vt.data);
        }
    }
    let values = float_output(runtime, &a.batch, &[k], &singular_values)?;
    if mode == SvdVectors::None {
        return Ok(values);
    }
    let u = float_output(runtime, &a.batch, &[m, u_columns], &left)?;
    let vh = float_output(runtime, &a.batch, &[v_rows, n], &right)?;
    runtime.new_tuple(vec![u, values, vh])
}

/// `qr_r_raw(a)`: `dgeqrf`'s output as `(factors, tau)`. NumPy's gufunc overwrites `a` with
/// the factors in place; this one returns them, and `numpy/linalg.py` rebinds `a`.
fn qr_r_raw(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = ONE_ARGUMENT.bind(&args)?;
    let a = operand(runtime, bound.required("a"), &QR_R_RAW, 0, 2)?;
    let (m, n) = (a.rows, a.columns);
    let k = m.min(n);
    let per_matrix = (m as u64).saturating_mul((n as u64).saturating_pow(2));
    charge(runtime, a.count(), per_matrix, matrices(m, n, 1))?;
    let mut factors = Vec::with_capacity(a.data.len());
    let mut taus = Vec::with_capacity(a.count().saturating_mul(k));
    for index in 0..a.count() {
        let (factored, tau) = dense::householder_qr(a.matrix(index));
        factors.extend(factored.data);
        taus.extend(tau);
    }
    let factors = float_output(runtime, &a.batch, &[m, n], &factors)?;
    let tau = float_output(runtime, &a.batch, &[k], &taus)?;
    runtime.new_tuple(vec![factors, tau])
}

fn qr_reduced(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    qr_q(runtime, args, &QR_REDUCED, false)
}

fn qr_complete(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    qr_q(runtime, args, &QR_COMPLETE, true)
}

/// `Q` from [`qr_r_raw`]'s factors: its first `min(m, n)` columns, or all `m` if `complete`.
fn qr_q(runtime: &mut dyn PyRuntime, args: CallArgs, gufunc: &Gufunc, complete: bool) -> PyResult {
    let bound = TWO_ARGUMENTS.bind(&args)?;
    let a = operand(runtime, bound.required("a"), gufunc, 0, 2)?;
    let tau = operand(runtime, bound.required("b"), gufunc, 1, 1)?;
    let (m, n) = (a.rows, a.columns);
    let k = m.min(n);
    if tau.rows != k || tau.batch != a.batch {
        return Err(core_mismatch(gufunc, 1, 0, tau.rows, k));
    }
    let columns = if complete { m } else { k };
    let count = a.count();
    let per_matrix = (m as u64)
        .saturating_mul(columns as u64)
        .saturating_mul(k.max(1) as u64);
    charge(runtime, count, per_matrix, matrices(m, columns, count))?;
    let mut values = Vec::with_capacity(count.saturating_mul(m * columns));
    for index in 0..count {
        let q = dense::householder_q(&a.matrix(index), tau.values(index), columns);
        values.extend(q.data);
    }
    float_output(runtime, &a.batch, &[m, columns], &values)
}

/// `lstsq(a, b, rcond)` for one matrix pair, since `numpy.linalg.lstsq` requires 2-d
/// operands: `(x, residuals, rank, singular_values)` with `rank` an `int32` scalar.
fn lstsq(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    let bound = LSTSQ_ARGUMENTS.bind(&args)?;
    let a = operand(runtime, bound.required("a"), &LSTSQ, 0, 2)?;
    let b = operand(runtime, bound.required("b"), &LSTSQ, 1, 2)?;
    let rcond = args::float_arg(runtime, &bound.required("rcond"))?;
    if !a.batch.is_empty() || !b.batch.is_empty() {
        return Err(PyError::unsupported(
            "numpy.linalg.lstsq does not support stacked matrices",
        ));
    }
    if b.rows != a.rows {
        return Err(core_mismatch(&LSTSQ, 1, 0, b.rows, a.rows));
    }
    let (m, n) = (a.rows, a.columns);
    let square = m.max(n);
    let working = matrices(square, square, 4).saturating_add(matrices(n, b.columns, 1));
    let per_matrix = (m as u64)
        .saturating_mul(n as u64)
        .saturating_mul(b.columns as u64);
    charge(runtime, 1, per_matrix, working)?;
    let finite = |stack: &Stack| stack.data.iter().all(|value| value.is_finite());
    if !finite(&a) || !finite(&b) {
        return Err(linalg_error("SVD did not converge in Linear Least Squares"));
    }
    let fit = dense::least_squares(&a.matrix(0), &b.matrix(0), rcond, &mut |cost| {
        runtime.charge_cpu(cost)
    })?;
    let solution = float_output(runtime, &[], &[n, b.columns], &fit.solution.data)?;
    let residuals = float_output(runtime, &[], &[b.columns], &fit.residuals)?;
    let rank = i32::try_from(fit.rank)
        .map_err(|_| PyError::overflow_error("lstsq rank does not fit in int32"))?;
    let rank = output(runtime, DType::INT32, &[], &[], &[rank])?;
    let singular_values = float_output(runtime, &[], &[m.min(n)], &fit.singular_values)?;
    runtime.new_tuple(vec![solution, residuals, rank, singular_values])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_index_broadcasts_length_one_and_missing_axes() {
        // Output batch (2, 3) against operand batches (3,), (2, 1), and ().
        let positions = (0..6).collect::<Vec<_>>();
        let along_last = positions
            .iter()
            .map(|flat| batch_index(*flat, &[2, 3], &[3]))
            .collect::<Vec<_>>();
        assert_eq!(along_last, [0, 1, 2, 0, 1, 2]);
        let along_first = positions
            .iter()
            .map(|flat| batch_index(*flat, &[2, 3], &[2, 1]))
            .collect::<Vec<_>>();
        assert_eq!(along_first, [0, 0, 0, 1, 1, 1]);
        assert!(positions
            .iter()
            .all(|flat| batch_index(*flat, &[2, 3], &[]) == 0));
    }
}
