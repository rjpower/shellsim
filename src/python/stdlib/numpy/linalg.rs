//! `numpy.linalg`'s native primitives: `_numpy_linalg`.
//!
//! This module is the thin runtime-facing layer over [`dense`], the shared kernel module: it
//! reads operand arrays, batches over stacked leading dimensions, charges CPU for the cubic (or,
//! for the Jacobi and QR-iteration solvers, per-sweep) work before running it, calls into
//! [`dense`], and writes results back into new arrays. `source/numpy/linalg.py` composes these
//! primitives (plus ordinary NumPy array operations) into the full `numpy.linalg` surface:
//! functions such as `matrix_power`, `matrix_rank`, `pinv`, `lstsq` and `norm` are plain Python
//! built from `inv`, `solve`, `svd` and `eigh` here. `lu` and `solve_triangular` are exposed only
//! for `scipy.linalg` to build its own `lu`/`lu_factor`/`solve_triangular`/`cho_solve` on top of,
//! not part of `numpy.linalg`'s own public surface.
//!
//! Every primitive computes in `f64` regardless of the input's integer or floating dtype, and
//! casts the result to `float32` only when every real operand was `float32`, rounding once. This
//! is simpler than running parallel single- and double-precision kernels and, because it uses
//! more precision than the target rather than less, cannot make results less accurate; see
//! `docs/numpy.md` for the resulting (deliberate) difference from real NumPy's single-precision
//! LAPACK calls. `float16` and complex *input* are both rejected as unsupported (see
//! `docs/numpy.md`); `eig`'s *output* is still complex, matching `numpy.linalg.eig`.

pub(in crate::python) mod dense;

use dense::{Cplx, Mat, Trans};

use super::super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyArrayBuffer, PyError, PyResult, PyRuntime,
};
use super::super::super::Value;
use super::args::Signature;
use super::array::{self, Array};
use super::convert;
use super::dtype::{Category, DType, Kind};
use super::element::{C128, C64};

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
        module: "numpy.linalg",
        name,
        call,
    }
}

static FUNCTIONS: &[FunctionDef] = &[
    function("inv", inv),
    function("solve", solve),
    function("det", det),
    function("slogdet", slogdet),
    function("cholesky", cholesky),
    function("qr", qr),
    function("eigh", eigh),
    function("eig", eig),
    function("svd", svd),
    function("lu", lu),
    function("solve_triangular", solve_triangular),
];

/// Whether every real operand was `float32`, the one case where a result narrows from the
/// `f64` working precision.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Precision {
    Single,
    Double,
}

impl Precision {
    fn combine(self, other: Precision) -> Precision {
        if self == Precision::Single && other == Precision::Single {
            Precision::Single
        } else {
            Precision::Double
        }
    }
}

fn square_error() -> PyError {
    PyError::exception(
        "LinAlgError",
        "Last 2 dimensions of the array must be square",
    )
}

fn need_2d(ndim: usize) -> PyError {
    PyError::exception(
        "LinAlgError",
        format!("{ndim}-dimensional array given. Array must be at least two-dimensional"),
    )
}

/// Check that a real operand's dtype is usable, and report whether it is `float32`.
fn check_dtype(array: &Array, function: &str) -> PyResult<Precision> {
    if array.dtype.kind() == Kind::Float16 {
        return Err(PyError::type_error(
            "array type float16 is unsupported in linalg",
        ));
    }
    if array.dtype.category() == Category::Complex {
        return Err(PyError::unsupported(format!(
            "complex input to numpy.linalg.{function} is not supported by shellsim's NumPy"
        )));
    }
    Ok(if array.dtype == DType::FLOAT32 {
        Precision::Single
    } else {
        Precision::Double
    })
}

/// `array`'s elements as `f64`, gathered in C order (works for any strided view).
fn as_f64(runtime: &mut dyn PyRuntime, array: &Array) -> PyResult<Vec<f64>> {
    let cast = convert::cast_array(runtime, array, DType::FLOAT64, false)?;
    array::read_elements::<f64>(runtime, &cast)
}

/// `array` broadcast to `batch_shape ++ core_shape` and read as `f64` in C order. `array`'s own
/// shape must already end with `core_shape`.
fn broadcast_f64(
    runtime: &mut dyn PyRuntime,
    array: &Array,
    batch_shape: &[usize],
    core_shape: &[usize],
) -> PyResult<Vec<f64>> {
    let mut shape = batch_shape.to_vec();
    shape.extend_from_slice(core_shape);
    let buffer = array::broadcast_buffer(runtime, array, DType::FLOAT64, &shape)?;
    let PyArrayBuffer::Bytes(bytes) = buffer else {
        return Err(PyError::runtime_error("linalg operand is not numeric"));
    };
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("8-byte chunk")))
        .collect())
}

fn batch_count(batch_shape: &[usize]) -> usize {
    batch_shape.iter().product()
}

fn chunks(flat: &[f64], rows: usize, cols: usize) -> Vec<Mat> {
    flat.chunks_exact(rows * cols)
        .map(|chunk| Mat::from_row_major(rows, cols, chunk.to_vec()))
        .collect()
}

/// `array`'s shape split into leading batch dimensions and a trailing square `n x n` core,
/// requiring at least two dimensions and an equal last two.
fn square_shape(array: &Array) -> PyResult<(Vec<usize>, usize)> {
    if array.ndim() < 2 {
        return Err(need_2d(array.ndim()));
    }
    let n = array.shape()[array.ndim() - 1];
    if array.shape()[array.ndim() - 2] != n {
        return Err(square_error());
    }
    Ok((array.shape()[..array.ndim() - 2].to_vec(), n))
}

fn rect_shape(array: &Array) -> PyResult<(Vec<usize>, usize, usize)> {
    if array.ndim() < 2 {
        return Err(need_2d(array.ndim()));
    }
    let (rows, cols) = (
        array.shape()[array.ndim() - 2],
        array.shape()[array.ndim() - 1],
    );
    Ok((array.shape()[..array.ndim() - 2].to_vec(), rows, cols))
}

/// Build a new array from batched `n x n`-shaped `f64` matrices, casting to `precision`.
fn array_from_batches(
    runtime: &mut dyn PyRuntime,
    batch_shape: &[usize],
    rows: usize,
    cols: usize,
    mats: &[Mat],
    precision: Precision,
) -> PyResult<Array> {
    let mut flat = Vec::with_capacity(mats.len() * rows * cols);
    for mat in mats {
        flat.extend_from_slice(&mat.data);
    }
    let mut shape = batch_shape.to_vec();
    shape.push(rows);
    shape.push(cols);
    let array = array::array_from_elements::<f64>(runtime, DType::FLOAT64, shape, &flat)?;
    match precision {
        Precision::Double => Ok(array),
        Precision::Single => convert::cast_array(runtime, &array, DType::FLOAT32, false),
    }
}

/// Build a new array from batched length-`n` `f64` vectors, casting to `precision`.
fn array_from_vectors(
    runtime: &mut dyn PyRuntime,
    batch_shape: &[usize],
    n: usize,
    vectors: &[Vec<f64>],
    precision: Precision,
) -> PyResult<Array> {
    let mut flat = Vec::with_capacity(vectors.len() * n);
    for vector in vectors {
        flat.extend_from_slice(vector);
    }
    let mut shape = batch_shape.to_vec();
    shape.push(n);
    let array = array::array_from_elements::<f64>(runtime, DType::FLOAT64, shape, &flat)?;
    match precision {
        Precision::Double => Ok(array),
        Precision::Single => convert::cast_array(runtime, &array, DType::FLOAT32, false),
    }
}

/// A 0-d result unboxes to a NumPy scalar, as `det` and the like return; other shapes stay
/// arrays.
fn scalar_or_array(runtime: &mut dyn PyRuntime, array: &Array) -> PyResult {
    if array.ndim() == 0 {
        return convert::element_to_scalar(runtime, array, array.view.offset);
    }
    Ok(array.value())
}

fn singular_error() -> PyError {
    PyError::exception("LinAlgError", "Singular matrix")
}

// ---------------------------------------------------------------------------------------------
// inv
// ---------------------------------------------------------------------------------------------

fn inv(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("inv", &["a"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let a = convert::as_array(runtime, bound.required("a"))?;
    let precision = check_dtype(&a, "inv")?;
    let (batch_shape, n) = square_shape(&a)?;
    let count = batch_count(&batch_shape);
    runtime.charge_cpu(dense::factor_cost(n as u64, n as u64) * count as u64)?;
    let flat = as_f64(runtime, &a)?;
    let mats = chunks(&flat, n, n);
    let mut results = Vec::with_capacity(mats.len());
    for mat in &mats {
        let factorization = dense::lu_factor(mat);
        if factorization.singular_at.is_some() {
            return Err(singular_error());
        }
        results.push(dense::lu_invert(&factorization.lu, &factorization.piv));
    }
    let result = array_from_batches(runtime, &batch_shape, n, n, &results, precision)?;
    Ok(result.value())
}

// ---------------------------------------------------------------------------------------------
// solve
// ---------------------------------------------------------------------------------------------

fn solve(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("solve", &["a", "b"], 2);
    let bound = SIGNATURE.bind(&args)?;
    let a = convert::as_array(runtime, bound.required("a"))?;
    let b = convert::as_array(runtime, bound.required("b"))?;
    let precision = check_dtype(&a, "solve")?.combine(check_dtype(&b, "solve")?);
    let (a_batch, n) = square_shape(&a)?;
    if b.ndim() == 0 {
        return Err(need_2d(0));
    }
    // NumPy 2's gufunc solve: `b` is a batch of vectors when its rank is one less than `a`'s
    // (signature `(m,m),(m)->(m)`), else a batch of matrices (`(m,m),(m,k)->(m,k)`).
    let vector_rhs = b.ndim() < a.ndim();
    let (b_batch, k): (Vec<usize>, usize) = if vector_rhs {
        if b.shape().last().copied() != Some(n) {
            return Err(PyError::value_error(format!(
                "solve: Input operand 1 has a mismatch in its core dimension 0, with gufunc \
                 signature (m,m),(m)->(m) (size {} is different from {n})",
                b.shape().last().copied().unwrap_or(0)
            )));
        }
        (b.shape()[..b.ndim() - 1].to_vec(), 1)
    } else {
        let bn = b.shape()[b.ndim() - 2];
        if bn != n {
            return Err(PyError::value_error(format!(
                "solve: Input operand 1 has a mismatch in its core dimension 0, with gufunc \
                 signature (m,m),(m,n)->(m,n) (size {bn} is different from {n})"
            )));
        }
        (b.shape()[..b.ndim() - 2].to_vec(), b.shape()[b.ndim() - 1])
    };
    let batch_shape = array::broadcast_shapes(&[&a_batch, &b_batch])?;
    let count = batch_count(&batch_shape);
    runtime.charge_cpu(
        dense::factor_cost(n as u64, n as u64).saturating_add((n * n * k) as u64) * count as u64,
    )?;
    let a_flat = broadcast_f64(runtime, &a, &batch_shape, &[n, n])?;
    let b_core: Vec<usize> = if vector_rhs { vec![n] } else { vec![n, k] };
    let b_flat = broadcast_f64(runtime, &b, &batch_shape, &b_core)?;
    let a_mats = chunks(&a_flat, n, n);
    let b_mats = chunks(&b_flat, n, k);
    let mut results = Vec::with_capacity(a_mats.len());
    for (a_mat, b_mat) in a_mats.iter().zip(&b_mats) {
        let factorization = dense::lu_factor(a_mat);
        if factorization.singular_at.is_some() {
            return Err(singular_error());
        }
        results.push(dense::lu_solve(
            &factorization.lu,
            &factorization.piv,
            b_mat,
            Trans::No,
        ));
    }
    let result = if vector_rhs {
        array_from_vectors(
            runtime,
            &batch_shape,
            n,
            &results.iter().map(|m| m.data.clone()).collect::<Vec<_>>(),
            precision,
        )?
    } else {
        array_from_batches(runtime, &batch_shape, n, k, &results, precision)?
    };
    scalar_or_array(runtime, &result)
}

// ---------------------------------------------------------------------------------------------
// det / slogdet
// ---------------------------------------------------------------------------------------------

fn det(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("det", &["a"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let a = convert::as_array(runtime, bound.required("a"))?;
    let precision = check_dtype(&a, "det")?;
    let (batch_shape, n) = square_shape(&a)?;
    let count = batch_count(&batch_shape);
    runtime.charge_cpu(dense::factor_cost(n as u64, n as u64) * count as u64)?;
    let flat = as_f64(runtime, &a)?;
    let mats = chunks(&flat, n, n);
    let values: Vec<f64> = mats
        .iter()
        .map(|mat| dense::lu_det(&dense::lu_factor(mat)))
        .collect();
    let result = array_from_vectors(runtime, &batch_shape, 1, &wrap(&values), precision)?;
    let result = drop_last_axis(runtime, &result)?;
    scalar_or_array(runtime, &result)
}

fn slogdet(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("slogdet", &["a"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let a = convert::as_array(runtime, bound.required("a"))?;
    let precision = check_dtype(&a, "slogdet")?;
    let (batch_shape, n) = square_shape(&a)?;
    let count = batch_count(&batch_shape);
    runtime.charge_cpu(dense::factor_cost(n as u64, n as u64) * count as u64)?;
    let flat = as_f64(runtime, &a)?;
    let mats = chunks(&flat, n, n);
    let mut signs = Vec::with_capacity(mats.len());
    let mut logs = Vec::with_capacity(mats.len());
    for mat in &mats {
        let (sign, log_det) = dense::lu_slogdet(&dense::lu_factor(mat));
        signs.push(sign);
        logs.push(log_det);
    }
    let sign_array = array_from_vectors(runtime, &batch_shape, 1, &wrap(&signs), precision)?;
    let log_array = array_from_vectors(runtime, &batch_shape, 1, &wrap(&logs), Precision::Double)?;
    let sign_array = drop_last_axis(runtime, &sign_array)?;
    let log_array = drop_last_axis(runtime, &log_array)?;
    let sign_value = scalar_or_array(runtime, &sign_array)?;
    let log_value = scalar_or_array(runtime, &log_array)?;
    runtime.new_tuple(vec![sign_value, log_value])
}

fn wrap(values: &[f64]) -> Vec<Vec<f64>> {
    values.iter().map(|v| vec![*v]).collect()
}

fn drop_last_axis(runtime: &mut dyn PyRuntime, array: &Array) -> PyResult<Array> {
    let shape = array.shape()[..array.ndim() - 1].to_vec();
    let strides = array.strides()[..array.ndim() - 1].to_vec();
    array::new_view(
        runtime,
        array,
        array.dtype,
        shape,
        strides,
        array.view.offset,
    )
}

// ---------------------------------------------------------------------------------------------
// cholesky
// ---------------------------------------------------------------------------------------------

fn cholesky(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("cholesky", &["a"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let a = convert::as_array(runtime, bound.required("a"))?;
    let precision = check_dtype(&a, "cholesky")?;
    let (batch_shape, n) = square_shape(&a)?;
    let count = batch_count(&batch_shape);
    runtime.charge_cpu(dense::factor_cost(n as u64, n as u64) * count as u64)?;
    let flat = as_f64(runtime, &a)?;
    let mats = chunks(&flat, n, n);
    let mut results = Vec::with_capacity(mats.len());
    for mat in &mats {
        results.push(
            dense::cholesky_lower(mat).map_err(|_| {
                PyError::exception("LinAlgError", "Matrix is not positive definite")
            })?,
        );
    }
    let result = array_from_batches(runtime, &batch_shape, n, n, &results, precision)?;
    Ok(result.value())
}

// ---------------------------------------------------------------------------------------------
// qr
// ---------------------------------------------------------------------------------------------

fn qr(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("qr", &["a", "mode"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let a = convert::as_array(runtime, bound.required("a"))?;
    let mode = match bound.value("mode") {
        Some(value) => runtime.string_value(&value)?.unwrap_or_default(),
        None => "reduced".to_string(),
    };
    if !["reduced", "complete", "r", "raw"].contains(&mode.as_str()) {
        return Err(PyError::value_error(format!("Unrecognized mode '{mode}'")));
    }
    let precision = check_dtype(&a, "qr")?;
    let (batch_shape, rows, cols) = rect_shape(&a)?;
    let k = rows.min(cols);
    let count = batch_count(&batch_shape);
    runtime.charge_cpu(dense::factor_cost(rows as u64, cols as u64) * count as u64)?;
    let flat = as_f64(runtime, &a)?;
    let mats = chunks(&flat, rows, cols);
    if mode == "raw" {
        let mut h = Vec::with_capacity(mats.len());
        let mut taus = Vec::with_capacity(mats.len());
        for mat in &mats {
            let factorization = dense::householder_qr(mat);
            h.push(factorization.factored.data.clone());
            taus.push(factorization.tau.clone());
        }
        let mut shape = batch_shape.clone();
        shape.push(cols);
        shape.push(rows);
        let mut flat_h = Vec::with_capacity(h.len() * rows * cols);
        for m in &h {
            flat_h.extend_from_slice(m);
        }
        let h_array =
            array::fortran_array_from_elements::<f64>(runtime, DType::FLOAT64, shape, &flat_h)?;
        let h_array = match precision {
            Precision::Double => h_array,
            Precision::Single => convert::cast_array(runtime, &h_array, DType::FLOAT32, false)?,
        };
        let tau_values: Vec<Vec<f64>> = taus;
        let tau_array = array_from_vectors(runtime, &batch_shape, k, &tau_values, precision)?;
        return runtime.new_tuple(vec![h_array.value(), tau_array.value()]);
    }
    let full = mode == "complete";
    let mut rs = Vec::with_capacity(mats.len());
    let mut qs = Vec::with_capacity(mats.len());
    for mat in &mats {
        let factorization = dense::householder_qr(mat);
        if full {
            rs.push(dense::qr_explicit_r_full(&factorization));
        } else {
            rs.push(dense::qr_explicit_r(&factorization));
        }
        if mode != "r" {
            qs.push(dense::qr_explicit_q(&factorization, full));
        }
    }
    let r_rows = if full { rows } else { k };
    let r_array = array_from_batches(runtime, &batch_shape, r_rows, cols, &rs, precision)?;
    if mode == "r" {
        return runtime.new_tuple(vec![r_array.value()]);
    }
    let q_width = if full { rows } else { k };
    let q_array = array_from_batches(runtime, &batch_shape, rows, q_width, &qs, precision)?;
    runtime.new_tuple(vec![q_array.value(), r_array.value()])
}

// ---------------------------------------------------------------------------------------------
// eigh
// ---------------------------------------------------------------------------------------------

fn eigh(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("eigh", &["a", "UPLO", "compute_vectors"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let a = convert::as_array(runtime, bound.required("a"))?;
    let lower = match bound.value("UPLO") {
        Some(value) => runtime.string_value(&value)?.unwrap_or_default() != "U",
        None => true,
    };
    let compute_vectors = match bound.value("compute_vectors") {
        Some(value) => runtime.truth(&value)?,
        None => true,
    };
    let precision = check_dtype(&a, "eigh")?;
    let (batch_shape, n) = square_shape(&a)?;
    let flat = as_f64(runtime, &a)?;
    let mats = chunks(&flat, n, n);
    let mut values = Vec::with_capacity(mats.len());
    let mut vectors = Vec::with_capacity(mats.len());
    for mat in &mats {
        let symmetric = symmetrize(mat, lower);
        let (w, v) =
            dense::jacobi_eigh(&symmetric, compute_vectors, |cost| runtime.charge_cpu(cost))?;
        values.push(w);
        if let Some(v) = v {
            vectors.push(v);
        }
    }
    let w_array = array_from_vectors(runtime, &batch_shape, n, &values, precision)?;
    if !compute_vectors {
        return runtime.new_tuple(vec![w_array.value(), Value::None]);
    }
    let v_array = array_from_batches(runtime, &batch_shape, n, n, &vectors, precision)?;
    runtime.new_tuple(vec![w_array.value(), v_array.value()])
}

/// Mirror the trusted triangle of `mat` into the other, so the Jacobi kernel (which reads both)
/// sees a genuinely symmetric matrix regardless of what the untrusted triangle holds.
fn symmetrize(mat: &Mat, lower: bool) -> Mat {
    let n = mat.rows;
    let mut result = mat.clone();
    for i in 0..n {
        for j in (i + 1)..n {
            if lower {
                result.set(i, j, mat.get(j, i));
            } else {
                result.set(j, i, mat.get(i, j));
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------------------------
// svd
// ---------------------------------------------------------------------------------------------

fn svd(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("svd", &["a", "full_matrices", "compute_uv"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let a = convert::as_array(runtime, bound.required("a"))?;
    let full_matrices = match bound.value("full_matrices") {
        Some(value) => runtime.truth(&value)?,
        None => true,
    };
    let compute_uv = match bound.value("compute_uv") {
        Some(value) => runtime.truth(&value)?,
        None => true,
    };
    let precision = check_dtype(&a, "svd")?;
    let (batch_shape, rows, cols) = rect_shape(&a)?;
    let k = rows.min(cols);
    let flat = as_f64(runtime, &a)?;
    let mats = chunks(&flat, rows, cols);
    let mut singular = Vec::with_capacity(mats.len());
    let mut us = Vec::with_capacity(mats.len());
    let mut vts = Vec::with_capacity(mats.len());
    for mat in &mats {
        let (u, s, vt) = dense::jacobi_svd(mat, full_matrices, compute_uv, |cost| {
            runtime.charge_cpu(cost)
        })?;
        singular.push(s);
        if let Some(u) = u {
            us.push(u);
        }
        if let Some(vt) = vt {
            vts.push(vt);
        }
    }
    let s_array = array_from_vectors(runtime, &batch_shape, k, &singular, precision)?;
    if !compute_uv {
        return scalar_or_array(runtime, &s_array);
    }
    let u_width = if full_matrices { rows } else { k };
    let vt_height = if full_matrices { cols } else { k };
    let u_array = array_from_batches(runtime, &batch_shape, rows, u_width, &us, precision)?;
    let vt_array = array_from_batches(runtime, &batch_shape, vt_height, cols, &vts, precision)?;
    runtime.new_tuple(vec![u_array.value(), s_array.value(), vt_array.value()])
}

// ---------------------------------------------------------------------------------------------
// eig
// ---------------------------------------------------------------------------------------------

fn non_convergence_error() -> PyError {
    PyError::exception("LinAlgError", "Eigenvalues did not converge")
}

fn eig(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("eig", &["a", "compute_vectors"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let a = convert::as_array(runtime, bound.required("a"))?;
    let compute_vectors = match bound.value("compute_vectors") {
        Some(value) => runtime.truth(&value)?,
        None => true,
    };
    let precision = check_dtype(&a, "eig")?;
    let (batch_shape, n) = square_shape(&a)?;
    let flat = as_f64(runtime, &a)?;
    let mats = chunks(&flat, n, n);
    let mut values = Vec::with_capacity(mats.len());
    let mut vectors = Vec::with_capacity(mats.len());
    for mat in &mats {
        let Some(result) =
            dense::eig_general(mat, compute_vectors, |cost| runtime.charge_cpu(cost))?
        else {
            return Err(non_convergence_error());
        };
        values.push(result.values);
        if let Some(v) = result.vectors {
            vectors.push(v);
        }
    }
    let w_array = complex_array_from_vectors(runtime, &batch_shape, n, &values, precision)?;
    if !compute_vectors {
        return runtime.new_tuple(vec![w_array.value(), Value::None]);
    }
    let v_array = complex_array_from_columns(runtime, &batch_shape, n, &vectors, precision)?;
    runtime.new_tuple(vec![w_array.value(), v_array.value()])
}

/// Build a new array from batched length-`n` complex vectors (eigenvalues), narrowing to
/// `complex64` when `precision` is `Single`, matching how the real primitives narrow to
/// `float32`.
fn complex_array_from_vectors(
    runtime: &mut dyn PyRuntime,
    batch_shape: &[usize],
    n: usize,
    vectors: &[Vec<Cplx>],
    precision: Precision,
) -> PyResult<Array> {
    let mut shape = batch_shape.to_vec();
    shape.push(n);
    let flat: Vec<C128> = vectors
        .iter()
        .flatten()
        .map(|value| C128 {
            re: value.re,
            im: value.im,
        })
        .collect();
    match precision {
        Precision::Double => {
            array::array_from_elements::<C128>(runtime, DType::COMPLEX128, shape, &flat)
        }
        Precision::Single => {
            let narrowed: Vec<C64> = flat
                .iter()
                .map(|value| C64 {
                    re: value.re as f32,
                    im: value.im as f32,
                })
                .collect();
            array::array_from_elements::<C64>(runtime, DType::COMPLEX64, shape, &narrowed)
        }
    }
}

/// Build a new array from batched `n x n` complex matrices, each given as `n` length-`n`
/// eigenvector columns (`columns[k]` is column `k`, as [`dense::eig_general`] returns them),
/// narrowing to `complex64` when `precision` is `Single`.
fn complex_array_from_columns(
    runtime: &mut dyn PyRuntime,
    batch_shape: &[usize],
    n: usize,
    batches: &[Vec<Vec<Cplx>>],
    precision: Precision,
) -> PyResult<Array> {
    let mut shape = batch_shape.to_vec();
    shape.push(n);
    shape.push(n);
    let mut flat = Vec::with_capacity(batches.len() * n * n);
    for columns in batches {
        let mut mat = vec![C128 { re: 0.0, im: 0.0 }; n * n];
        for (col_index, column) in columns.iter().enumerate() {
            for (row_index, value) in column.iter().enumerate() {
                mat[row_index * n + col_index] = C128 {
                    re: value.re,
                    im: value.im,
                };
            }
        }
        flat.extend(mat);
    }
    match precision {
        Precision::Double => {
            array::array_from_elements::<C128>(runtime, DType::COMPLEX128, shape, &flat)
        }
        Precision::Single => {
            let narrowed: Vec<C64> = flat
                .iter()
                .map(|value| C64 {
                    re: value.re as f32,
                    im: value.im as f32,
                })
                .collect();
            array::array_from_elements::<C64>(runtime, DType::COMPLEX64, shape, &narrowed)
        }
    }
}

// ---------------------------------------------------------------------------------------------
// lu (exposed for scipy.linalg's `lu`/`lu_factor`/`lu_solve`, not part of numpy.linalg itself)
// ---------------------------------------------------------------------------------------------

/// `_numpy_linalg.lu(a)`: the packed `getrf`-style LU factorization [`dense::lu_factor`]
/// computes (`lu`, and the 0-based `piv` such that step `k` swapped row `k` with row `piv[k]`),
/// batched like this module's other primitives. `a` need not be square. `scipy.linalg.lu_factor`
/// returns this pair directly; `scipy.linalg.lu` and `.lu_solve` build on it in Python.
fn lu(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("lu", &["a"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let a = convert::as_array(runtime, bound.required("a"))?;
    let precision = check_dtype(&a, "lu")?;
    let (batch_shape, rows, cols) = rect_shape(&a)?;
    let k = rows.min(cols);
    let count = batch_count(&batch_shape);
    runtime.charge_cpu(dense::factor_cost(rows as u64, cols as u64) * count as u64)?;
    let flat = as_f64(runtime, &a)?;
    let mats = chunks(&flat, rows, cols);
    let mut lus = Vec::with_capacity(mats.len());
    let mut pivs = Vec::with_capacity(mats.len());
    for mat in &mats {
        let factorization = dense::lu_factor(mat);
        lus.push(factorization.lu);
        pivs.push(
            factorization
                .piv
                .iter()
                .map(|&value| value as i32)
                .collect::<Vec<i32>>(),
        );
    }
    let lu_array = array_from_batches(runtime, &batch_shape, rows, cols, &lus, precision)?;
    let piv_array = array_from_int_vectors(runtime, &batch_shape, k, &pivs)?;
    runtime.new_tuple(vec![lu_array.value(), piv_array.value()])
}

/// Build a new C-ordered `int32` array from batched length-`n` pivot vectors.
fn array_from_int_vectors(
    runtime: &mut dyn PyRuntime,
    batch_shape: &[usize],
    n: usize,
    vectors: &[Vec<i32>],
) -> PyResult<Array> {
    let mut flat = Vec::with_capacity(vectors.len() * n);
    for vector in vectors {
        flat.extend_from_slice(vector);
    }
    let mut shape = batch_shape.to_vec();
    shape.push(n);
    array::array_from_elements::<i32>(runtime, DType::INT32, shape, &flat)
}

// ---------------------------------------------------------------------------------------------
// solve_triangular (exposed for scipy.linalg's `solve_triangular` and `cho_solve`)
// ---------------------------------------------------------------------------------------------

/// `_numpy_linalg.solve_triangular(a, b, lower, transpose, unit_diagonal)`: the solution of the
/// triangular system `a @ x == b` (or, if `transpose`, `a.T @ x == b`), batched like `solve`.
/// Real NumPy has no public triangular solve; this exists for `scipy.linalg.solve_triangular`
/// and, applied twice, `scipy.linalg.cho_solve`.
fn solve_triangular(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new(
        "solve_triangular",
        &["a", "b", "lower", "transpose", "unit_diagonal"],
        2,
    );
    let bound = SIGNATURE.bind(&args)?;
    let a = convert::as_array(runtime, bound.required("a"))?;
    let b = convert::as_array(runtime, bound.required("b"))?;
    let lower = match bound.value("lower") {
        Some(value) => runtime.truth(&value)?,
        None => false,
    };
    let transpose = match bound.value("transpose") {
        Some(value) => runtime.truth(&value)?,
        None => false,
    };
    let unit_diagonal = match bound.value("unit_diagonal") {
        Some(value) => runtime.truth(&value)?,
        None => false,
    };
    let trans = if transpose {
        Trans::Transpose
    } else {
        Trans::No
    };
    let precision =
        check_dtype(&a, "solve_triangular")?.combine(check_dtype(&b, "solve_triangular")?);
    let (a_batch, n) = square_shape(&a)?;
    if b.ndim() == 0 {
        return Err(need_2d(0));
    }
    let vector_rhs = b.ndim() < a.ndim();
    let (b_batch, k): (Vec<usize>, usize) = if vector_rhs {
        (b.shape()[..b.ndim() - 1].to_vec(), 1)
    } else {
        (b.shape()[..b.ndim() - 2].to_vec(), b.shape()[b.ndim() - 1])
    };
    let batch_shape = array::broadcast_shapes(&[&a_batch, &b_batch])?;
    let count = batch_count(&batch_shape);
    runtime.charge_cpu(dense::factor_cost(n as u64, n as u64) * count as u64)?;
    let a_flat = broadcast_f64(runtime, &a, &batch_shape, &[n, n])?;
    let b_core: Vec<usize> = if vector_rhs { vec![n] } else { vec![n, k] };
    let b_flat = broadcast_f64(runtime, &b, &batch_shape, &b_core)?;
    let a_mats = chunks(&a_flat, n, n);
    let b_mats = chunks(&b_flat, n, k);
    let mut results = Vec::with_capacity(a_mats.len());
    for (a_mat, b_mat) in a_mats.iter().zip(&b_mats) {
        match dense::triangular_solve(a_mat, b_mat, lower, trans, unit_diagonal) {
            Ok(x) => results.push(x),
            Err(_) => return Err(singular_error()),
        }
    }
    let result = if vector_rhs {
        array_from_vectors(
            runtime,
            &batch_shape,
            n,
            &results.iter().map(|m| m.data.clone()).collect::<Vec<_>>(),
            precision,
        )?
    } else {
        array_from_batches(runtime, &batch_shape, n, k, &results, precision)?
    };
    scalar_or_array(runtime, &result)
}
