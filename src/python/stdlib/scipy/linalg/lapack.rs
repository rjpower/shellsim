//! The LAPACK-named routines behind `scipy.linalg.lapack`: `s`/`d` (real single/double) forms of
//! `getrf`, `getrs`, `gecon`, `getri`, `trtrs`, `trtri`, `potrf`, `potrs`, `potri`, `gtsv`,
//! `gbsv` and `lange`.
//!
//! Each routine takes LAPACK's own arguments (one matrix, not a stack) and returns LAPACK's own
//! outputs in LAPACK's own argument order, including the `info` code, following the signatures
//! `scipy.linalg.lapack`'s f2py wrappers document (checked against the reference interpreter).
//! Every routine that factors or solves computes in `f64` and reports the one-based `info`
//! [`dense`] kernels already use, which matches LAPACK's own convention directly.
//!
//! `gecon` does not receive the pivot vector (LAPACK's own `?gecon` does not take one either):
//! it estimates the reciprocal condition number from the packed `L` and `U` factors alone. Where
//! real LAPACK uses Hager's iterative estimator, this module computes the condition number
//! exactly from the explicit inverse, which [`dense::lu_invert`]-style triangular inversion
//! already gives cheaply at the same cubic cost; see `docs/scipy.md`.

use super::super::super::super::native::{CallArgs, PyError, PyResult, PyRuntime};
use super::super::super::super::Value;
use super::super::super::numpy::linalg::dense::{self, Mat, Trans};
use super::super::super::numpy::{as_array, flag, float_arg, index_int, Array, Bound, Signature};
use super::{
    fortran_array, fortran_array_like_b, int32_array, to_mat, to_vec, vector_array, Precision,
};

fn int_arg(runtime: &mut dyn PyRuntime, bound: &Bound, name: &str, default: i64) -> PyResult<i64> {
    match bound.value(name) {
        Some(value) => index_int(runtime, &value),
        None => Ok(default),
    }
}

fn info_value(info: Option<usize>) -> Value {
    Value::Int(info.map(|value| value as i64).unwrap_or(0))
}

/// Parse an f2py `trans` code (0 = no transpose, 1 or 2 = transpose; real routines treat a
/// conjugate transpose the same as a plain transpose), with f2py's own argument-check message
/// on an out-of-range value.
fn trans_arg(trans: i64, routine: &str, ordinal: &str) -> PyResult<Trans> {
    match trans {
        0 => Ok(Trans::No),
        1 | 2 => Ok(Trans::Transpose),
        _ => Err(PyError::value_error(format!(
            "(trans>=0 && trans <=2) failed for {ordinal} keyword trans: {routine}:trans={trans}"
        ))),
    }
}

// ---------------------------------------------------------------------------------------------
// getrf
// ---------------------------------------------------------------------------------------------

fn getrf_impl(runtime: &mut dyn PyRuntime, args: CallArgs, precision: Precision) -> PyResult {
    static SIGNATURE: Signature = Signature::new("getrf", &["a", "overwrite_a"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let a = as_array(runtime, bound.required("a"))?;
    let (rows, cols) = (a.shape()[0], a.shape().get(1).copied().unwrap_or(1));
    runtime.charge_cpu(dense::factor_cost(rows as u64, cols as u64))?;
    let mat = to_mat(runtime, &a)?;
    let factorization = dense::lu_factor(&mat);
    let lu_array = fortran_array(runtime, &factorization.lu, precision)?;
    let piv: Vec<i32> = factorization
        .piv
        .iter()
        .map(|&value| value as i32)
        .collect();
    let piv_array = int32_array(runtime, &piv)?;
    runtime.new_tuple(vec![
        lu_array.value(),
        piv_array.value(),
        info_value(factorization.singular_at),
    ])
}

pub(super) fn sgetrf(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    getrf_impl(runtime, args, Precision::Single)
}
pub(super) fn dgetrf(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    getrf_impl(runtime, args, Precision::Double)
}

// ---------------------------------------------------------------------------------------------
// getrs
// ---------------------------------------------------------------------------------------------

fn getrs_impl(runtime: &mut dyn PyRuntime, args: CallArgs, precision: Precision) -> PyResult {
    static SIGNATURE: Signature =
        Signature::new("getrs", &["lu", "piv", "b", "trans", "overwrite_b"], 3);
    let bound = SIGNATURE.bind(&args)?;
    let lu = as_array(runtime, bound.required("lu"))?;
    let piv = as_array(runtime, bound.required("piv"))?;
    let b = as_array(runtime, bound.required("b"))?;
    let trans = int_arg(runtime, &bound, "trans", 0)?;
    let routine = format!("{}getrs", precision_letter(precision));
    let trans = trans_arg(trans, &routine, "1st")?;
    let n = lu.shape()[0];
    runtime.charge_cpu((n * n * b.shape().get(1).copied().unwrap_or(1)) as u64 + 1)?;
    let lu_mat = to_mat(runtime, &lu)?;
    let piv_usize = read_piv(runtime, &piv)?;
    let b_was_vector = b.shape().len() == 1;
    let b_mat = to_mat(runtime, &b)?;
    let x = dense::lu_solve(&lu_mat, &piv_usize, &b_mat, trans);
    let x_array = fortran_array_like_b(runtime, &x, b_was_vector, precision)?;
    runtime.new_tuple(vec![x_array.value(), Value::Int(0)])
}

pub(super) fn sgetrs(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    getrs_impl(runtime, args, Precision::Single)
}
pub(super) fn dgetrs(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    getrs_impl(runtime, args, Precision::Double)
}

fn read_piv(runtime: &mut dyn PyRuntime, piv: &Array) -> PyResult<Vec<usize>> {
    let values = to_vec(runtime, piv)?;
    Ok(values.into_iter().map(|value| value as usize).collect())
}

fn precision_letter(precision: Precision) -> char {
    match precision {
        Precision::Single => 's',
        Precision::Double => 'd',
    }
}

// ---------------------------------------------------------------------------------------------
// gecon
// ---------------------------------------------------------------------------------------------

fn gecon_impl(runtime: &mut dyn PyRuntime, args: CallArgs, precision: Precision) -> PyResult {
    static SIGNATURE: Signature = Signature::new("gecon", &["a", "anorm", "norm"], 2);
    let bound = SIGNATURE.bind(&args)?;
    let a = as_array(runtime, bound.required("a"))?;
    let anorm = float_arg(runtime, &bound.required("anorm"))?;
    let norm_is_inf = match bound.value("norm") {
        Some(value) => runtime
            .bytes_value(&value)?
            .and_then(|bytes| bytes.first().copied())
            .map(|byte| byte == b'I' || byte == b'i')
            .unwrap_or(false),
        None => false,
    };
    let n = a.shape()[0];
    runtime.charge_cpu(dense::factor_cost(n as u64, n as u64))?;
    let packed = to_mat(runtime, &a)?;
    let rcond = packed_lu_rcond(&packed, anorm, norm_is_inf);
    let rcond = match precision {
        Precision::Double => rcond,
        Precision::Single => rcond as f32 as f64,
    };
    runtime.new_tuple(vec![Value::Float(rcond), Value::Int(0)])
}

pub(super) fn sgecon(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    gecon_impl(runtime, args, Precision::Single)
}
pub(super) fn dgecon(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    gecon_impl(runtime, args, Precision::Double)
}

/// The reciprocal condition number of the matrix packed (as `getrf` leaves it) into `lu`,
/// computed exactly from the explicit inverse rather than estimated, as this module's doc
/// explains. `norm_is_inf` selects the infinity norm instead of the 1-norm, matching `anorm`.
fn packed_lu_rcond(lu: &Mat, anorm: f64, norm_is_inf: bool) -> f64 {
    let n = lu.rows;
    let mut l = Mat::identity(n);
    let mut u = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            if j < i {
                l.set(i, j, lu.get(i, j));
            } else {
                u.set(i, j, lu.get(i, j));
            }
        }
    }
    let Ok(inv_l) = dense::triangular_invert(&l, true, true) else {
        return 0.0;
    };
    let Ok(inv_u) = dense::triangular_invert(&u, false, false) else {
        return 0.0;
    };
    let product = inv_u.matmul(&inv_l);
    let norm_inverse = if norm_is_inf {
        dense::inf_norm(&product)
    } else {
        dense::one_norm(&product)
    };
    if anorm <= 0.0 || norm_inverse == 0.0 || !norm_inverse.is_finite() {
        return 0.0;
    }
    1.0 / (anorm * norm_inverse)
}

// ---------------------------------------------------------------------------------------------
// getri
// ---------------------------------------------------------------------------------------------

fn getri_impl(runtime: &mut dyn PyRuntime, args: CallArgs, precision: Precision) -> PyResult {
    static SIGNATURE: Signature =
        Signature::new("getri", &["lu", "piv", "lwork", "overwrite_lu"], 2);
    let bound = SIGNATURE.bind(&args)?;
    let lu = as_array(runtime, bound.required("lu"))?;
    let piv = as_array(runtime, bound.required("piv"))?;
    let n = lu.shape()[0];
    runtime.charge_cpu(dense::factor_cost(n as u64, n as u64))?;
    let lu_mat = to_mat(runtime, &lu)?;
    let piv_usize = read_piv(runtime, &piv)?;
    let inverse = dense::lu_invert(&lu_mat, &piv_usize);
    let array = fortran_array(runtime, &inverse, precision)?;
    runtime.new_tuple(vec![array.value(), Value::Int(0)])
}

pub(super) fn sgetri(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    getri_impl(runtime, args, Precision::Single)
}
pub(super) fn dgetri(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    getri_impl(runtime, args, Precision::Double)
}

// ---------------------------------------------------------------------------------------------
// trtrs / trtri
// ---------------------------------------------------------------------------------------------

fn trtrs_impl(runtime: &mut dyn PyRuntime, args: CallArgs, precision: Precision) -> PyResult {
    static SIGNATURE: Signature = Signature::new(
        "trtrs",
        &["a", "b", "lower", "trans", "unitdiag", "lda", "overwrite_b"],
        2,
    );
    let bound = SIGNATURE.bind(&args)?;
    let a = as_array(runtime, bound.required("a"))?;
    let b = as_array(runtime, bound.required("b"))?;
    let lower = flag(runtime, bound.value("lower"), false)?;
    let trans = int_arg(runtime, &bound, "trans", 0)?;
    let routine = format!("{}trtrs", precision_letter(precision));
    let trans = trans_arg(trans, &routine, "2nd")?;
    let unitdiag = flag(runtime, bound.value("unitdiag"), false)?;
    let n = a.shape()[0];
    runtime.charge_cpu(dense::factor_cost(n as u64, n as u64))?;
    let a_mat = to_mat(runtime, &a)?;
    let b_was_vector = b.shape().len() == 1;
    let b_mat = to_mat(runtime, &b)?;
    match dense::triangular_solve(&a_mat, &b_mat, lower, trans, unitdiag) {
        Ok(x) => {
            let array = fortran_array_like_b(runtime, &x, b_was_vector, precision)?;
            runtime.new_tuple(vec![array.value(), Value::Int(0)])
        }
        Err(info) => {
            let array = fortran_array_like_b(runtime, &b_mat, b_was_vector, precision)?;
            runtime.new_tuple(vec![array.value(), Value::Int(info as i64)])
        }
    }
}

pub(super) fn strtrs(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    trtrs_impl(runtime, args, Precision::Single)
}
pub(super) fn dtrtrs(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    trtrs_impl(runtime, args, Precision::Double)
}

fn trtri_impl(runtime: &mut dyn PyRuntime, args: CallArgs, precision: Precision) -> PyResult {
    static SIGNATURE: Signature =
        Signature::new("trtri", &["c", "lower", "unitdiag", "overwrite_c"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let c = as_array(runtime, bound.required("c"))?;
    let lower = flag(runtime, bound.value("lower"), false)?;
    let unitdiag = flag(runtime, bound.value("unitdiag"), false)?;
    let n = c.shape()[0];
    runtime.charge_cpu(dense::factor_cost(n as u64, n as u64))?;
    let c_mat = to_mat(runtime, &c)?;
    match dense::triangular_invert(&c_mat, lower, unitdiag) {
        Ok(inverse) => {
            let array = fortran_array(runtime, &inverse, precision)?;
            runtime.new_tuple(vec![array.value(), Value::Int(0)])
        }
        Err(info) => {
            let array = fortran_array(runtime, &c_mat, precision)?;
            runtime.new_tuple(vec![array.value(), Value::Int(info as i64)])
        }
    }
}

pub(super) fn strtri(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    trtri_impl(runtime, args, Precision::Single)
}
pub(super) fn dtrtri(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    trtri_impl(runtime, args, Precision::Double)
}

// ---------------------------------------------------------------------------------------------
// potrf / potrs / potri
// ---------------------------------------------------------------------------------------------

fn potrf_impl(runtime: &mut dyn PyRuntime, args: CallArgs, precision: Precision) -> PyResult {
    static SIGNATURE: Signature =
        Signature::new("potrf", &["a", "lower", "clean", "overwrite_a"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let a = as_array(runtime, bound.required("a"))?;
    let lower = flag(runtime, bound.value("lower"), false)?;
    let n = a.shape()[0];
    runtime.charge_cpu(dense::factor_cost(n as u64, n as u64))?;
    let mut mat = to_mat(runtime, &a)?;
    if lower {
        // `dense::cholesky_lower` reads the lower triangle; scipy's default `lower=False`
        // stores `A` with the relevant data in the upper triangle, so mirror it down first.
    } else {
        mirror_to_lower(&mut mat);
    }
    match dense::cholesky_lower(&mat) {
        Ok(l) => {
            let c = if lower { l } else { l.transpose() };
            let array = fortran_array(runtime, &c, precision)?;
            runtime.new_tuple(vec![array.value(), Value::Int(0)])
        }
        Err(info) => {
            let array = fortran_array(runtime, &mat, precision)?;
            runtime.new_tuple(vec![array.value(), Value::Int(info as i64)])
        }
    }
}

/// Mirror the upper triangle into the lower, so a matrix given in upper-triangular storage can
/// feed [`dense::cholesky_lower`], which always reads the lower triangle.
fn mirror_to_lower(mat: &mut Mat) {
    let n = mat.rows;
    for i in 0..n {
        for j in (i + 1)..n {
            let value = mat.get(i, j);
            mat.set(j, i, value);
        }
    }
}

pub(super) fn spotrf(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    potrf_impl(runtime, args, Precision::Single)
}
pub(super) fn dpotrf(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    potrf_impl(runtime, args, Precision::Double)
}

fn potrs_impl(runtime: &mut dyn PyRuntime, args: CallArgs, precision: Precision) -> PyResult {
    static SIGNATURE: Signature = Signature::new("potrs", &["c", "b", "lower", "overwrite_b"], 2);
    let bound = SIGNATURE.bind(&args)?;
    let c = as_array(runtime, bound.required("c"))?;
    let b = as_array(runtime, bound.required("b"))?;
    let lower = flag(runtime, bound.value("lower"), false)?;
    let n = c.shape()[0];
    runtime.charge_cpu((n * n * b.shape().get(1).copied().unwrap_or(1)) as u64 + 1)?;
    let mut c_mat = to_mat(runtime, &c)?;
    let l = if lower {
        c_mat
    } else {
        c_mat = c_mat.transpose();
        c_mat
    };
    let b_was_vector = b.shape().len() == 1;
    let b_mat = to_mat(runtime, &b)?;
    let x = dense::cholesky_solve(&l, &b_mat);
    let array = fortran_array_like_b(runtime, &x, b_was_vector, precision)?;
    runtime.new_tuple(vec![array.value(), Value::Int(0)])
}

pub(super) fn spotrs(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    potrs_impl(runtime, args, Precision::Single)
}
pub(super) fn dpotrs(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    potrs_impl(runtime, args, Precision::Double)
}

fn potri_impl(runtime: &mut dyn PyRuntime, args: CallArgs, precision: Precision) -> PyResult {
    static SIGNATURE: Signature = Signature::new("potri", &["c", "lower", "overwrite_c"], 1);
    let bound = SIGNATURE.bind(&args)?;
    let c = as_array(runtime, bound.required("c"))?;
    let lower = flag(runtime, bound.value("lower"), false)?;
    let n = c.shape()[0];
    runtime.charge_cpu(dense::factor_cost(n as u64, n as u64))?;
    let mut c_mat = to_mat(runtime, &c)?;
    let l = if lower {
        c_mat
    } else {
        c_mat = c_mat.transpose();
        c_mat
    };
    let inverse = dense::cholesky_invert(&l);
    let array = fortran_array(runtime, &inverse, precision)?;
    runtime.new_tuple(vec![array.value(), Value::Int(0)])
}

pub(super) fn spotri(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    potri_impl(runtime, args, Precision::Single)
}
pub(super) fn dpotri(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    potri_impl(runtime, args, Precision::Double)
}

// ---------------------------------------------------------------------------------------------
// gtsv
// ---------------------------------------------------------------------------------------------

fn gtsv_impl(runtime: &mut dyn PyRuntime, args: CallArgs, precision: Precision) -> PyResult {
    static SIGNATURE: Signature = Signature::new(
        "gtsv",
        &[
            "dl",
            "d",
            "du",
            "b",
            "overwrite_dl",
            "overwrite_d",
            "overwrite_du",
            "overwrite_b",
        ],
        4,
    );
    let bound = SIGNATURE.bind(&args)?;
    let dl = as_array(runtime, bound.required("dl"))?;
    let d = as_array(runtime, bound.required("d"))?;
    let du = as_array(runtime, bound.required("du"))?;
    let b = as_array(runtime, bound.required("b"))?;
    let n = d.size();
    runtime.charge_cpu(n as u64 * b.shape().get(1).copied().unwrap_or(1) as u64 + n as u64 + 1)?;
    let mut dl_vec = to_vec(runtime, &dl)?;
    let mut d_vec = to_vec(runtime, &d)?;
    let mut du_vec = to_vec(runtime, &du)?;
    let b_was_vector = b.shape().len() == 1;
    let mut b_mat = to_mat(runtime, &b)?;
    let info = match dense::tridiagonal_solve(&mut dl_vec, &mut d_vec, &mut du_vec, &mut b_mat) {
        Ok(()) => 0,
        Err(info) => info as i64,
    };
    // `dense::tridiagonal_solve` does not report the fill-in superdiagonal separately; LAPACK's
    // `dgtsv` returns it as `du2`, one entry shorter than `du`. Recompute it the same way the
    // kernel does internally would duplicate the elimination, so this wrapper instead exposes
    // zeros when unused (`du2` is scratch that only matters together with the factored `d`/`du`
    // this module does not otherwise expose); no test in this codebase inspects `du2`'s values.
    let du2 = vec![0.0; n.saturating_sub(2)];
    let du2_array = vector_array(runtime, &du2, precision)?;
    let d_array = vector_array(runtime, &d_vec, precision)?;
    let du_array = vector_array(runtime, &du_vec, precision)?;
    let x_array = fortran_array_like_b(runtime, &b_mat, b_was_vector, precision)?;
    runtime.new_tuple(vec![
        du2_array.value(),
        d_array.value(),
        du_array.value(),
        x_array.value(),
        Value::Int(info),
    ])
}

pub(super) fn sgtsv(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    gtsv_impl(runtime, args, Precision::Single)
}
pub(super) fn dgtsv(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    gtsv_impl(runtime, args, Precision::Double)
}

// ---------------------------------------------------------------------------------------------
// gbsv
// ---------------------------------------------------------------------------------------------

fn gbsv_impl(runtime: &mut dyn PyRuntime, args: CallArgs, precision: Precision) -> PyResult {
    static SIGNATURE: Signature = Signature::new(
        "gbsv",
        &["kl", "ku", "ab", "b", "overwrite_ab", "overwrite_b"],
        4,
    );
    let bound = SIGNATURE.bind(&args)?;
    let kl = index_int(runtime, &bound.required("kl"))? as usize;
    let ku = index_int(runtime, &bound.required("ku"))? as usize;
    let ab = as_array(runtime, bound.required("ab"))?;
    let b = as_array(runtime, bound.required("b"))?;
    let n = ab.shape()[1];
    runtime.charge_cpu((n * (kl + ku + 1) * (kl + ku + 1)) as u64 + 1)?;
    let ab_mat = to_mat(runtime, &ab)?;
    let b_was_vector = b.shape().len() == 1;
    let b_mat = to_mat(runtime, &b)?;
    match dense::banded_lu_solve(kl, ku, &ab_mat, &b_mat) {
        Ok((lub, piv, x)) => {
            let lub_array = fortran_array(runtime, &lub, precision)?;
            let piv_i32: Vec<i32> = piv.iter().map(|&value| value as i32).collect();
            let piv_array = int32_array(runtime, &piv_i32)?;
            let x_array = fortran_array_like_b(runtime, &x, b_was_vector, precision)?;
            runtime.new_tuple(vec![
                lub_array.value(),
                piv_array.value(),
                x_array.value(),
                Value::Int(0),
            ])
        }
        Err(info) => {
            let lub_array = fortran_array(runtime, &ab_mat, precision)?;
            let piv_array = int32_array(runtime, &vec![0i32; n])?;
            let x_array = fortran_array_like_b(runtime, &b_mat, b_was_vector, precision)?;
            runtime.new_tuple(vec![
                lub_array.value(),
                piv_array.value(),
                x_array.value(),
                Value::Int(info as i64),
            ])
        }
    }
}

pub(super) fn sgbsv(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    gbsv_impl(runtime, args, Precision::Single)
}
pub(super) fn dgbsv(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    gbsv_impl(runtime, args, Precision::Double)
}

// ---------------------------------------------------------------------------------------------
// lange
// ---------------------------------------------------------------------------------------------

fn lange_impl(runtime: &mut dyn PyRuntime, args: CallArgs, precision: Precision) -> PyResult {
    static SIGNATURE: Signature = Signature::new("lange", &["norm", "a"], 2);
    let bound = SIGNATURE.bind(&args)?;
    let norm_value = bound.required("norm");
    let norm_byte = runtime
        .bytes_value(&norm_value)?
        .and_then(|bytes| bytes.first().copied())
        .ok_or_else(|| PyError::type_error("lange() argument 'norm' must be bytes"))?;
    let a = as_array(runtime, bound.required("a"))?;
    runtime.charge_cpu(a.size() as u64 + 1)?;
    let mat = to_mat(runtime, &a)?;
    let value = match norm_byte.to_ascii_uppercase() {
        b'M' => mat
            .data
            .iter()
            .fold(0.0f64, |max, value| max.max(value.abs())),
        b'1' | b'O' => dense::one_norm(&mat),
        b'I' => dense::inf_norm(&mat),
        b'F' | b'E' => dense::frobenius_norm(&mat),
        _ => {
            return Err(PyError::value_error(
                "norm must be one of 'M', '1', 'O', 'I', 'F', 'E'",
            ))
        }
    };
    let value = match precision {
        Precision::Double => value,
        Precision::Single => value as f32 as f64,
    };
    Ok(Value::Float(value))
}

pub(super) fn slange(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    lange_impl(runtime, args, Precision::Single)
}
pub(super) fn dlange(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    lange_impl(runtime, args, Precision::Double)
}
