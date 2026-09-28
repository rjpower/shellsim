//! shellsim's `scipy.special`: the native kernels behind the ufunc table entries the NumPy core
//! dispatches to (see `numpy::ufunc::Family::Special` and `special_loop`), and the native module
//! `_scipy_special` that exposes them to the frozen `scipy.special` module.
//!
//! This module holds only the "real kernels" a composition cannot easily reproduce at full
//! accuracy: `gamma`/`gammaln`/`loggamma`/`psi`, `erf`/`erfc` and their inverses, `ndtr`/
//! `log_ndtr`/`ndtri`, the incomplete gamma and beta functions and their inverses, and the
//! Hurwitz zeta function. Everything else `scipy.special` exposes (`expit`, `logit`, `xlogy`,
//! `entr`, `beta`, `betaln`, `binom`, `comb`, `boxcox`, the distribution functions and more) is a
//! composition of these kernels (or of plain NumPy ufuncs) written directly in frozen Python;
//! see `source/scipy/special.py`.
//!
//! Every function here is a plain `f64 -> f64` (or higher arity) kernel; [`Function::eval`]
//! dispatches to one, and [`Function::eval_f32`] reruns it in `f64` and rounds, since a `float32`
//! result correctly rounded from a `float64` computation meets the crate's accuracy target and
//! needs no separate single-precision code path. [`evaluate`] wraps each element's kernel call
//! with [`meter`]'s CPU accounting, and the ufunc loop in `numpy::ufunc` calls both.
//!
//! The module is organized by shared numerical machinery rather than by SciPy's own file layout:
//! [`gamma`] is the Lanczos gamma/digamma core (plus a private `binom` helper [`ibeta`] uses for
//! its exactness fast path); [`igam`] and [`ibeta`] are the one incomplete gamma and one
//! incomplete beta implementation; [`erf`] and [`zeta`] are the remaining families with their own
//! series.

mod erf;
mod gamma;
mod ibeta;
mod igam;
mod meter;
mod zeta;

use super::super::super::native::{ModuleDef, ValueDef};
use super::super::numpy::special_ufunc_value;

pub(in crate::python) use meter::evaluate;

/// Relative CPU cost of one element of a `scipy.special` ufunc loop, charged up front (see
/// `numpy::ufunc::element_cost`) before [`evaluate`] additionally charges for the iterative work
/// a particular element's arguments need. Higher than the plain floating-point ufuncs (cost `4`)
/// because even the closed-form kernels here run several transcendental calls per element.
pub(in crate::python) const ELEMENT_COST: u64 = 8;

/// One `scipy.special` native kernel: an entry in the ufunc table ([`super::super::numpy::ufunc`])
/// closed by [`ALL`] in this order (`numpy::ufunc::special_index` checks that at compile time).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::python) enum Function {
    Erf,
    Erfc,
    Erfinv,
    Erfcinv,
    Gamma,
    Gammaln,
    Loggamma,
    Psi,
    Betainc,
    Betaincc,
    Betaincinv,
    Gammainc,
    Gammaincc,
    Gammaincinv,
    Gammainccinv,
    Ndtr,
    LogNdtr,
    Ndtri,
    Zeta,
}

impl Function {
    /// Every function, in the order they close the ufunc table (see the module doc comment and
    /// `numpy::ufunc::special_index`).
    pub(in crate::python) const ALL: &'static [Function] = &[
        Self::Erf,
        Self::Erfc,
        Self::Erfinv,
        Self::Erfcinv,
        Self::Gamma,
        Self::Gammaln,
        Self::Loggamma,
        Self::Psi,
        Self::Betainc,
        Self::Betaincc,
        Self::Betaincinv,
        Self::Gammainc,
        Self::Gammaincc,
        Self::Gammaincinv,
        Self::Gammainccinv,
        Self::Ndtr,
        Self::LogNdtr,
        Self::Ndtri,
        Self::Zeta,
    ];

    /// The ufunc's Python name. `psi`'s `digamma` alias is bound in frozen Python
    /// (`source/scipy/special.py`) so `special.psi is special.digamma`. `zeta` is the raw
    /// two-argument Hurwitz form; the frozen `zeta(x, q=None)` wrapper handles the one-argument
    /// Riemann case and the two-argument domain restriction (see `zeta.rs`).
    pub(in crate::python) const fn name(&self) -> &'static str {
        match self {
            Self::Erf => "erf",
            Self::Erfc => "erfc",
            Self::Erfinv => "erfinv",
            Self::Erfcinv => "erfcinv",
            Self::Gamma => "gamma",
            Self::Gammaln => "gammaln",
            Self::Loggamma => "loggamma",
            Self::Psi => "psi",
            Self::Betainc => "betainc",
            Self::Betaincc => "betaincc",
            Self::Betaincinv => "betaincinv",
            Self::Gammainc => "gammainc",
            Self::Gammaincc => "gammaincc",
            Self::Gammaincinv => "gammaincinv",
            Self::Gammainccinv => "gammainccinv",
            Self::Ndtr => "ndtr",
            Self::LogNdtr => "log_ndtr",
            Self::Ndtri => "ndtri",
            Self::Zeta => "zeta",
        }
    }

    /// Number of positional arguments.
    pub(in crate::python) const fn nin(&self) -> usize {
        match self {
            Self::Erf
            | Self::Erfc
            | Self::Erfinv
            | Self::Erfcinv
            | Self::Gamma
            | Self::Gammaln
            | Self::Loggamma
            | Self::Psi
            | Self::Ndtr
            | Self::LogNdtr
            | Self::Ndtri => 1,
            Self::Gammainc
            | Self::Gammaincc
            | Self::Gammaincinv
            | Self::Gammainccinv
            | Self::Zeta => 2,
            Self::Betainc | Self::Betaincc | Self::Betaincinv => 3,
        }
    }

    /// Whether SciPy registers a complex loop for this function (so complex input is an explicit
    /// "unsupported", not a missing-loop `TypeError`); see `numpy::ufunc::resolve_special`.
    pub(in crate::python) const fn supports_complex(&self) -> bool {
        matches!(
            self,
            Self::Erf
                | Self::Erfc
                | Self::Gamma
                | Self::Loggamma
                | Self::Psi
                | Self::Ndtr
                | Self::LogNdtr
        )
    }

    /// Evaluate in `f64`.
    pub(in crate::python) fn eval(&self, args: &[f64]) -> f64 {
        match self {
            Self::Erf => erf::erf(args[0]),
            Self::Erfc => erf::erfc(args[0]),
            Self::Erfinv => erf::erfinv(args[0]),
            Self::Erfcinv => erf::erfcinv(args[0]),
            Self::Gamma => gamma::gamma(args[0]),
            Self::Gammaln => gamma::gammaln(args[0]),
            Self::Loggamma => gamma::loggamma(args[0]),
            Self::Psi => gamma::digamma(args[0]),
            Self::Betainc => ibeta::betainc(args[0], args[1], args[2]),
            Self::Betaincc => ibeta::betaincc(args[0], args[1], args[2]),
            Self::Betaincinv => ibeta::betaincinv(args[0], args[1], args[2]),
            Self::Gammainc => igam::gammainc(args[0], args[1]),
            Self::Gammaincc => igam::gammaincc(args[0], args[1]),
            Self::Gammaincinv => igam::gammaincinv(args[0], args[1]),
            Self::Gammainccinv => igam::gammainccinv(args[0], args[1]),
            Self::Ndtr => erf::ndtr(args[0]),
            Self::LogNdtr => erf::log_ndtr(args[0]),
            Self::Ndtri => erf::ndtri(args[0]),
            Self::Zeta => zeta::zeta(args[0], args[1]),
        }
    }

    /// Evaluate in `f32`, by computing in `f64` and rounding (see the module doc comment).
    pub(in crate::python) fn eval_f32(&self, args: &[f32]) -> f32 {
        let mut wide = [0.0_f64; 3];
        for (slot, value) in wide.iter_mut().zip(args) {
            *slot = *value as f64;
        }
        self.eval(&wide[..args.len()]) as f32
    }
}

const fn function_value(function: Function, position: usize) -> ValueDef {
    special_ufunc_value(function.name(), position)
}

macro_rules! module_values {
    ($($function:ident),* $(,)?) => {
        &[$(function_value(Function::$function, position_of(Function::$function)),)*]
    };
}

/// The position of `function` within [`Function::ALL`], at compile time.
const fn position_of(function: Function) -> usize {
    let mut index = 0;
    while index < Function::ALL.len() {
        if same(Function::ALL[index], function) {
            return index;
        }
        index += 1;
    }
    panic!("function is listed in Function::ALL")
}

const fn same(a: Function, b: Function) -> bool {
    a as u8 == b as u8
}

static VALUES: &[ValueDef] = module_values!(
    Erf,
    Erfc,
    Erfinv,
    Erfcinv,
    Gamma,
    Gammaln,
    Loggamma,
    Psi,
    Betainc,
    Betaincc,
    Betaincinv,
    Gammainc,
    Gammaincc,
    Gammaincinv,
    Gammainccinv,
    Ndtr,
    LogNdtr,
    Ndtri,
    Zeta,
);

/// `_scipy_special` exports only ufunc values (every SciPy-written-in-Python name, such as
/// `comb`, `beta` or `logsumexp`, lives in frozen Python instead); it declares no native
/// functions.
pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_scipy_special",
    functions: &[],
    values: VALUES,
};
