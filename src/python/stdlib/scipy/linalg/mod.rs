//! `scipy.linalg`'s native primitives: `_scipy_linalg`.
//!
//! Every routine here works on one matrix (plus one right-hand side, where relevant), matching
//! LAPACK's and BLAS's own single-call shape and the f2py wrappers `scipy.linalg.lapack` and
//! `scipy.linalg.blas` expose. `source/scipy/linalg/__init__.py` and `source/scipy/linalg/lapack.py`
//! loop over stacked leading dimensions in Python, calling these once per matrix, which charges
//! CPU for the whole stack automatically (each call charges its own cubic work) and keeps the
//! native surface small. The higher-level convenience functions (`solve`, `inv`, `lu`, `qr`,
//! `eigh`, `svd`, `lstsq`, `pinv`, the special matrices, and so on) are plain Python built from
//! these primitives and from `numpy.linalg`, as `docs/scipy.md` describes.
//!
//! Every routine reads its operand(s) as `f64` regardless of whether the caller resolved
//! `float32` or `float64` precision (`source/scipy/linalg/__init__.py` does that resolution, and
//! its `DeprecationWarning`s for `float16`/`bool` input, in Python), and rounds the result back
//! to `float32` only when asked. See `src/python/stdlib/numpy/linalg.rs` for why: it is simpler
//! than parallel single- and double-precision kernels and cannot lose accuracy relative to
//! running the target precision throughout.
//!
//! `lapack.rs` holds the LAPACK-named routines (`getrf`, `getrs`, `gecon`, `getri`, `trtrs`,
//! `trtri`, `potrf`, `potrs`, `potri`, `gtsv`, `gbsv`, `lange`); `expm` and the one BLAS routine
//! (`nrm2`) live here.

mod lapack;

use super::super::super::native::{CallArgs, FunctionDef, ModuleDef, PyError, PyResult, PyRuntime};
use super::super::super::Value;
use super::super::numpy::linalg::dense::{self, Mat};
use super::super::numpy::{
    array_from_elements, as_array, cast_array, flag, fortran_array_from_elements, read_elements,
    Array, Category, DType, Signature,
};

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
        module: "scipy.linalg",
        name,
        call,
    }
}

static FUNCTIONS: &[FunctionDef] = &[
    function("expm", expm),
    function("nrm2", nrm2),
    function("eigh_gen", eigh_gen),
    function("check_real", check_real),
    function("qr_pivoted", qr_pivoted),
    function("sgetrf", lapack::sgetrf),
    function("dgetrf", lapack::dgetrf),
    function("sgetrs", lapack::sgetrs),
    function("dgetrs", lapack::dgetrs),
    function("sgecon", lapack::sgecon),
    function("dgecon", lapack::dgecon),
    function("sgetri", lapack::sgetri),
    function("dgetri", lapack::dgetri),
    function("strtrs", lapack::strtrs),
    function("dtrtrs", lapack::dtrtrs),
    function("strtri", lapack::strtri),
    function("dtrtri", lapack::dtrtri),
    function("spotrf", lapack::spotrf),
    function("dpotrf", lapack::dpotrf),
    function("spotrs", lapack::spotrs),
    function("dpotrs", lapack::dpotrs),
    function("spotri", lapack::spotri),
    function("dpotri", lapack::dpotri),
    function("sgtsv", lapack::sgtsv),
    function("dgtsv", lapack::dgtsv),
    function("sgbsv", lapack::sgbsv),
    function("dgbsv", lapack::dgbsv),
    function("slange", lapack::slange),
    function("dlange", lapack::dlange),
];

/// Whether a real operand is `float32`, the one case where a result narrows from the `f64`
/// working precision. The Python layer has already resolved every operand to `float32` or
/// `float64` (see the module doc), so nothing else is expected here.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Precision {
    Single,
    Double,
}

pub(super) fn precision_of(array: &Array) -> Precision {
    if array.dtype == DType::FLOAT32 {
        Precision::Single
    } else {
        Precision::Double
    }
}

/// Read a 2-D array (any layout) as a row-major [`Mat`], charging for the read.
pub(super) fn to_mat(runtime: &mut dyn PyRuntime, array: &Array) -> PyResult<Mat> {
    let rows = array.shape()[0];
    let cols = array.shape().get(1).copied().unwrap_or(1);
    let cast = cast_array(runtime, array, DType::FLOAT64, false)?;
    let flat = read_elements::<f64>(runtime, &cast)?;
    Ok(Mat::from_row_major(rows, cols, flat))
}

/// Read a 1-D array as a plain `Vec<f64>`.
pub(super) fn to_vec(runtime: &mut dyn PyRuntime, array: &Array) -> PyResult<Vec<f64>> {
    let cast = cast_array(runtime, array, DType::FLOAT64, false)?;
    read_elements::<f64>(runtime, &cast)
}

/// A new Fortran-ordered array holding `mat`, rounded to `precision`, as f2py's wrappers return.
///
/// `fortran_array_from_elements` takes its `values` slice in column-major order for the given
/// shape (it lays raw bytes straight into the buffer and computes F-order strides around them,
/// rather than permuting elements itself). `mat.data` is row-major (`Mat::get`/`set` index it as
/// `i * cols + j`), so this transposes first: `mat.transpose().data`, read row-major over the
/// `cols x rows` transpose, visits `mat`'s entries column by column, which is `mat`'s own data
/// in column-major order.
pub(super) fn fortran_array(
    runtime: &mut dyn PyRuntime,
    mat: &Mat,
    precision: Precision,
) -> PyResult<Array> {
    let shape = vec![mat.rows, mat.cols];
    let column_major = mat.transpose().data;
    match precision {
        Precision::Double => {
            fortran_array_from_elements::<f64>(runtime, DType::FLOAT64, shape, &column_major)
        }
        Precision::Single => {
            let narrowed: Vec<f32> = column_major.iter().map(|&value| value as f32).collect();
            fortran_array_from_elements::<f32>(runtime, DType::FLOAT32, shape, &narrowed)
        }
    }
}

/// A right-hand-side result array shaped like the `b` f2py's routine solved for: `mat`'s single
/// column as a 1-D array if `b_was_vector` (f2py's routines return `x` in `b`'s own shape, and
/// `to_mat` always turns a 1-D `b` into a one-column [`Mat`], losing that shape), otherwise the
/// ordinary Fortran-ordered 2-D array [`fortran_array`] builds.
pub(super) fn fortran_array_like_b(
    runtime: &mut dyn PyRuntime,
    mat: &Mat,
    b_was_vector: bool,
    precision: Precision,
) -> PyResult<Array> {
    if b_was_vector {
        vector_array(runtime, &mat.data, precision)
    } else {
        fortran_array(runtime, mat, precision)
    }
}

/// A new C-ordered vector array holding `values`, rounded to `precision`.
pub(super) fn vector_array(
    runtime: &mut dyn PyRuntime,
    values: &[f64],
    precision: Precision,
) -> PyResult<Array> {
    match precision {
        Precision::Double => {
            array_from_elements::<f64>(runtime, DType::FLOAT64, vec![values.len()], values)
        }
        Precision::Single => {
            let narrowed: Vec<f32> = values.iter().map(|&value| value as f32).collect();
            array_from_elements::<f32>(runtime, DType::FLOAT32, vec![values.len()], &narrowed)
        }
    }
}

/// A new C-ordered `int32` vector array, as f2py's pivot and permutation outputs are.
pub(super) fn int32_array(runtime: &mut dyn PyRuntime, values: &[i32]) -> PyResult<Array> {
    array_from_elements::<i32>(runtime, DType::INT32, vec![values.len()], values)
}

pub(super) fn linalg_error(message: impl Into<String>) -> PyError {
    PyError::exception("LinAlgError", message.into())
}

/// `_scipy_linalg.check_real(a, function)`: reject complex input the way real SciPy's own
/// LAPACK-backed functions do (they wrap only the real LAPACK routines; the complex (`c`/`z`)
/// ones are part of the explicit unsupported frontier `docs/scipy.md` documents). `function` is
/// the `scipy.linalg` name to report, e.g. `"solve"`.
fn check_real(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("check_real", &["a", "function"], 2);
    let bound = SIGNATURE.bind(&args)?;
    let a = as_array(runtime, bound.required("a"))?;
    let function = runtime
        .string_value(&bound.required("function"))?
        .unwrap_or_default();
    if a.dtype.category() == Category::Complex {
        return Err(PyError::unsupported(format!(
            "complex input to scipy.linalg.{function} is not supported by shellsim's SciPy"
        )));
    }
    Ok(Value::None)
}

// ---------------------------------------------------------------------------------------------
// expm
// ---------------------------------------------------------------------------------------------

/// `_scipy_linalg.expm(a)`: the matrix exponential of one square matrix, already resolved to
/// `float32` or `float64` by the Python layer. Charges one matrix product's worth of cubic work
/// per scaling-and-squaring step in addition to the factorization itself, matching `docs/scipy.md`.
fn expm(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("expm", &["a"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let a = as_array(runtime, bound.required("a"))?;
    if a.ndim() != 2 || a.shape()[0] != a.shape()[1] {
        return Err(linalg_error(
            "Last 2 dimensions of the array must be square",
        ));
    }
    let n = a.shape()[0] as u64;
    let precision = precision_of(&a);
    // A handful of matrix products (Pade evaluation plus squaring) on top of the LU solve that
    // combines them, each cubic: a conservative constant multiple of `factor_cost`.
    runtime.charge_cpu(dense::factor_cost(n, n) * 6)?;
    let mat = to_mat(runtime, &a)?;
    let result = dense::expm(&mat);
    let shape = vec![result.rows, result.cols];
    let array = match precision {
        Precision::Double => {
            array_from_elements::<f64>(runtime, DType::FLOAT64, shape, &result.data)?
        }
        Precision::Single => {
            let narrowed: Vec<f32> = result.data.iter().map(|&value| value as f32).collect();
            array_from_elements::<f32>(runtime, DType::FLOAT32, shape, &narrowed)?
        }
    };
    Ok(array.value())
}

// ---------------------------------------------------------------------------------------------
// nrm2 (BLAS)
// ---------------------------------------------------------------------------------------------

/// `_scipy_linalg.nrm2(x)`: the Euclidean norm of one vector.
fn nrm2(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("nrm2", &["x"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let x = as_array(runtime, bound.required("x"))?;
    let precision = precision_of(&x);
    runtime.charge_cpu(x.size() as u64 + 1)?;
    let values = to_vec(runtime, &x)?;
    let norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
    let value = match precision {
        Precision::Double => norm,
        Precision::Single => norm as f32 as f64,
    };
    Ok(Value::Float(value))
}

// ---------------------------------------------------------------------------------------------
// eigh_gen (the generalized symmetric eigenproblem behind scipy.linalg.eigh's `b` argument)
// ---------------------------------------------------------------------------------------------

/// `_scipy_linalg.eigh_gen(a, b, lower, compute_vectors)`: the generalized symmetric
/// eigenproblem `A x = lambda B x`, via [`dense::eigh_generalized`] (a Cholesky-based reduction
/// to a standard eigenproblem). `scipy.linalg` has no LAPACK-named routine of its own for this
/// (real LAPACK's is `sygv`, which this module does not otherwise wrap), so it is not exposed
/// through `lapack.py`; `source/scipy/linalg/__init__.py`'s `eigh` calls it directly when a
/// second matrix is given.
fn eigh_gen(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature =
        Signature::new("eigh_gen", &["a", "b", "lower", "compute_vectors"], 2);
    let bound = SIGNATURE.bind(&args)?;
    let a = as_array(runtime, bound.required("a"))?;
    let b = as_array(runtime, bound.required("b"))?;
    let lower = flag(runtime, bound.value("lower"), true)?;
    let compute_vectors = flag(runtime, bound.value("compute_vectors"), true)?;
    let n = a.shape()[0];
    // The reduction does a Cholesky factorization of `b`, two triangular solves against `a`,
    // and (if vectors are wanted) one more triangular solve to map eigenvectors back: a small
    // constant multiple of one factorization's cubic work, on top of the Jacobi sweeps charged
    // as they run.
    runtime.charge_cpu(dense::factor_cost(n as u64, n as u64) * 4)?;
    let a_mat = symmetrize(&to_mat(runtime, &a)?, lower);
    let b_mat = symmetrize(&to_mat(runtime, &b)?, lower);
    match dense::eigh_generalized(&a_mat, &b_mat, compute_vectors, |cost| {
        runtime.charge_cpu(cost)
    })? {
        Err(_info) => Err(linalg_error(
            "The leading minor of order 1 of B is not positive definite. The factorization \
             of B could not be completed and no eigenvalues or eigenvectors were computed.",
        )),
        Ok((values, vectors)) => {
            let w_array = vector_array(runtime, &values, Precision::Double)?;
            if !compute_vectors {
                return Ok(w_array.value());
            }
            let v = vectors.expect("compute_vectors requested");
            let v_array =
                array_from_elements::<f64>(runtime, DType::FLOAT64, vec![v.rows, v.cols], &v.data)?;
            runtime.new_tuple(vec![w_array.value(), v_array.value()])
        }
    }
}

// ---------------------------------------------------------------------------------------------
// qr_pivoted (column-pivoted QR behind scipy.linalg.qr(pivoting=True))
// ---------------------------------------------------------------------------------------------

/// `_scipy_linalg.qr_pivoted(a)`: the full-shape column-pivoted QR factorization
/// [`dense::householder_qr_pivoted`] computes: `(q, r, jpvt)` with `a[:, jpvt] == q @ r`, `q`
/// square (`rows x rows`) and `r` shaped `rows x cols`, `jpvt` an `int32` array of the 0-based
/// column permutation, as LAPACK's `geqp3` (behind SciPy's `qr(pivoting=True)`) reports it.
/// `source/scipy/linalg/__init__.py`'s `qr` reduces `q`/`r` to the economic shape itself when
/// asked, the same way it already does for the unpivoted case.
fn qr_pivoted(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    static SIGNATURE: Signature = Signature::new("qr_pivoted", &["a"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let a = as_array(runtime, bound.required("a"))?;
    let (rows, cols) = (a.shape()[0], a.shape().get(1).copied().unwrap_or(1));
    runtime.charge_cpu(dense::factor_cost(rows as u64, cols as u64))?;
    let mat = to_mat(runtime, &a)?;
    let (factorization, jpvt) = dense::householder_qr_pivoted(&mat);
    let q = dense::qr_explicit_q(&factorization, true);
    let r = dense::qr_explicit_r_full(&factorization);
    let precision = precision_of(&a);
    let q_array = real_array(runtime, &q, precision)?;
    let r_array = real_array(runtime, &r, precision)?;
    let jpvt_i32: Vec<i32> = jpvt.iter().map(|&value| value as i32).collect();
    let jpvt_array = int32_array(runtime, &jpvt_i32)?;
    runtime.new_tuple(vec![q_array.value(), r_array.value(), jpvt_array.value()])
}

/// A new C-ordered array holding `mat`, rounded to `precision` (unlike [`fortran_array`], for
/// results `scipy.linalg`'s own Python code returns C-ordered, such as `qr`'s `q`/`r`).
fn real_array(runtime: &mut dyn PyRuntime, mat: &Mat, precision: Precision) -> PyResult<Array> {
    let shape = vec![mat.rows, mat.cols];
    match precision {
        Precision::Double => array_from_elements::<f64>(runtime, DType::FLOAT64, shape, &mat.data),
        Precision::Single => {
            let narrowed: Vec<f32> = mat.data.iter().map(|&value| value as f32).collect();
            array_from_elements::<f32>(runtime, DType::FLOAT32, shape, &narrowed)
        }
    }
}

/// Build a genuinely symmetric matrix from `mat`, trusting only the triangle `lower` names
/// (mirroring it into the other). [`dense::cholesky_lower`] already reads only its own lower
/// triangle, but [`dense::eigh_generalized`]'s reduction uses its `a` operand as a full dense
/// matrix, so `a` (and, for a `lower=false` caller, `b`) must be made symmetric first.
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
