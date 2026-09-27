//! shellsim's `scipy.special`: the native kernels behind the ufunc table entries the NumPy core
//! dispatches to (see `numpy::ufunc::Family::Special` and `special_loop`), and the native module
//! `_scipy_special` that exposes them to the frozen `scipy.special` package.
//!
//! Every function here is a plain `f64 -> f64` (or higher arity) kernel; [`Function::eval`]
//! dispatches to one, and [`Function::eval_f32`] reruns it in `f64` and rounds, since a `float32`
//! result correctly rounded from a `float64` computation meets the crate's accuracy target and
//! needs no separate single-precision code path. [`evaluate`] wraps each element's kernel call
//! with [`meter`]'s CPU accounting, and the ufunc loop in `numpy::ufunc` calls both.
//!
//! The module is organized by shared numerical machinery rather than by SciPy's own file layout:
//! [`gamma`] is the Lanczos gamma/digamma/beta/binomial core; [`igam`] and [`ibeta`] are the one
//! incomplete gamma and one incomplete beta implementation every distribution function in
//! [`distributions`] is built on; [`erf`] and [`zeta`] are the remaining families with their own
//! series; [`misc`] holds the closed-form functions with no iteration at all.

mod distributions;
mod erf;
mod gamma;
mod ibeta;
mod igam;
mod meter;
mod misc;
mod zeta;

use super::super::super::native::{ModuleDef, ValueDef};
use super::super::numpy::special_ufunc_value;

pub(in crate::python) use meter::evaluate;

/// Relative CPU cost of one element of a `scipy.special` ufunc loop, charged up front (see
/// `numpy::ufunc::element_cost`) before [`evaluate`] additionally charges for the iterative work
/// a particular element's arguments need. Higher than the plain floating-point ufuncs (cost `4`)
/// because even the closed-form kernels here run several transcendental calls per element.
pub(in crate::python) const ELEMENT_COST: u64 = 8;

/// How NumPy picks the `float32` vs. `float64` loop for a `scipy.special` ufunc (see
/// `numpy::ufunc::resolve_special`); the loop selection itself lives there, this just classifies
/// each function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::python) enum Loops {
    /// The ordinary case: `float32` and `float64` loops in that order, so any input that safely
    /// casts to `float32` computes in single precision. `complex` marks whether SciPy also
    /// registers complex loops (so complex input is an explicit "unsupported", not a missing
    /// loop `TypeError`).
    Real { complex: bool },
    /// `logit` registers its `float64` loop first, so only exact `float32` input selects single
    /// precision.
    DoubleFirst,
    /// `bdtr` and `bdtrc`: a third loop takes the trial count as an integer, so a
    /// floating-point trial count with a `float64`-resolving call issues SciPy's
    /// `DeprecationWarning` (see `numpy::ufunc::evaluate`).
    Binomial,
}

/// One `scipy.special` function: an entry in the ufunc table ([`super::super::numpy::ufunc`])
/// closed by [`ALL`] in this order (`numpy::ufunc::special_index` checks that at compile time).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::python) enum Function {
    Erf,
    Erfc,
    Erfinv,
    Erfcinv,
    Gamma,
    Rgamma,
    Gammaln,
    Loggamma,
    Psi,
    Beta,
    Betaln,
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
    Expit,
    Logit,
    LogExpit,
    Xlogy,
    Xlog1py,
    Entr,
    RelEntr,
    KlDiv,
    Binom,
    Poch,
    Stdtr,
    Stdtrit,
    Chdtr,
    Chdtrc,
    Chdtri,
    Fdtr,
    Fdtrc,
    Fdtri,
    Pdtr,
    Pdtrc,
    Pdtrik,
    Bdtr,
    Bdtrc,
    BinomPmf,
    BinomCdf,
    BinomSf,
    BinomPpf,
    BinomIsf,
    Boxcox,
    InvBoxcox,
    Expm1,
    Log1p,
    RiemannZeta,
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
        Self::Rgamma,
        Self::Gammaln,
        Self::Loggamma,
        Self::Psi,
        Self::Beta,
        Self::Betaln,
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
        Self::Expit,
        Self::Logit,
        Self::LogExpit,
        Self::Xlogy,
        Self::Xlog1py,
        Self::Entr,
        Self::RelEntr,
        Self::KlDiv,
        Self::Binom,
        Self::Poch,
        Self::Stdtr,
        Self::Stdtrit,
        Self::Chdtr,
        Self::Chdtrc,
        Self::Chdtri,
        Self::Fdtr,
        Self::Fdtrc,
        Self::Fdtri,
        Self::Pdtr,
        Self::Pdtrc,
        Self::Pdtrik,
        Self::Bdtr,
        Self::Bdtrc,
        Self::BinomPmf,
        Self::BinomCdf,
        Self::BinomSf,
        Self::BinomPpf,
        Self::BinomIsf,
        Self::Boxcox,
        Self::InvBoxcox,
        Self::Expm1,
        Self::Log1p,
        Self::RiemannZeta,
        Self::Zeta,
    ];

    /// The ufunc's Python name. `psi`'s `digamma` alias is bound in frozen Python
    /// (`source/scipy/special/__init__.py`) so `special.psi is special.digamma`. `Expm1` and
    /// `Log1p` share their name with a NumPy ufunc (see `numpy::ufunc::special_index`'s doc
    /// comment) but are separate `scipy.special` objects. The private binomial ufuncs
    /// `scipy.stats` calls are named with a leading underscore, as SciPy's are.
    pub(in crate::python) const fn name(&self) -> &'static str {
        match self {
            Self::Erf => "erf",
            Self::Erfc => "erfc",
            Self::Erfinv => "erfinv",
            Self::Erfcinv => "erfcinv",
            Self::Gamma => "gamma",
            Self::Rgamma => "rgamma",
            Self::Gammaln => "gammaln",
            Self::Loggamma => "loggamma",
            Self::Psi => "psi",
            Self::Beta => "beta",
            Self::Betaln => "betaln",
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
            Self::Expit => "expit",
            Self::Logit => "logit",
            Self::LogExpit => "log_expit",
            Self::Xlogy => "xlogy",
            Self::Xlog1py => "xlog1py",
            Self::Entr => "entr",
            Self::RelEntr => "rel_entr",
            Self::KlDiv => "kl_div",
            Self::Binom => "binom",
            Self::Poch => "poch",
            Self::Stdtr => "stdtr",
            Self::Stdtrit => "stdtrit",
            Self::Chdtr => "chdtr",
            Self::Chdtrc => "chdtrc",
            Self::Chdtri => "chdtri",
            Self::Fdtr => "fdtr",
            Self::Fdtrc => "fdtrc",
            Self::Fdtri => "fdtri",
            Self::Pdtr => "pdtr",
            Self::Pdtrc => "pdtrc",
            Self::Pdtrik => "pdtrik",
            Self::Bdtr => "bdtr",
            Self::Bdtrc => "bdtrc",
            Self::BinomPmf => "_binom_pmf",
            Self::BinomCdf => "_binom_cdf",
            Self::BinomSf => "_binom_sf",
            Self::BinomPpf => "_binom_ppf",
            Self::BinomIsf => "_binom_isf",
            Self::Boxcox => "boxcox",
            Self::InvBoxcox => "inv_boxcox",
            Self::Expm1 => "expm1",
            Self::Log1p => "log1p",
            Self::RiemannZeta => "_riemann_zeta",
            Self::Zeta => "_zeta",
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
            | Self::Rgamma
            | Self::Gammaln
            | Self::Loggamma
            | Self::Psi
            | Self::Ndtr
            | Self::LogNdtr
            | Self::Ndtri
            | Self::Expit
            | Self::Logit
            | Self::LogExpit
            | Self::Entr
            | Self::Expm1
            | Self::Log1p
            | Self::RiemannZeta => 1,
            Self::Beta
            | Self::Betaln
            | Self::Gammainc
            | Self::Gammaincc
            | Self::Gammaincinv
            | Self::Gammainccinv
            | Self::Xlogy
            | Self::Xlog1py
            | Self::RelEntr
            | Self::KlDiv
            | Self::Binom
            | Self::Poch
            | Self::Stdtr
            | Self::Stdtrit
            | Self::Chdtr
            | Self::Chdtrc
            | Self::Chdtri
            | Self::Pdtr
            | Self::Pdtrc
            | Self::Pdtrik
            | Self::Boxcox
            | Self::InvBoxcox
            | Self::Zeta => 2,
            Self::Betainc
            | Self::Betaincc
            | Self::Betaincinv
            | Self::Fdtr
            | Self::Fdtrc
            | Self::Fdtri
            | Self::Bdtr
            | Self::Bdtrc
            | Self::BinomPmf
            | Self::BinomCdf
            | Self::BinomSf
            | Self::BinomPpf
            | Self::BinomIsf => 3,
        }
    }

    /// How NumPy resolves this function's loop dtype (see [`Loops`]).
    pub(in crate::python) const fn loops(&self) -> Loops {
        match self {
            Self::Erf
            | Self::Erfc
            | Self::Gamma
            | Self::Rgamma
            | Self::Loggamma
            | Self::Psi
            | Self::Ndtr
            | Self::LogNdtr
            | Self::Xlogy
            | Self::Xlog1py
            | Self::Expm1
            | Self::Log1p
            | Self::RiemannZeta => Loops::Real { complex: true },
            Self::Logit => Loops::DoubleFirst,
            Self::Bdtr | Self::Bdtrc => Loops::Binomial,
            _ => Loops::Real { complex: false },
        }
    }

    /// Evaluate in `f64`.
    pub(in crate::python) fn eval(&self, args: &[f64]) -> f64 {
        match self {
            Self::Erf => erf::erf(args[0]),
            Self::Erfc => erf::erfc(args[0]),
            Self::Erfinv => erf::erfinv(args[0]),
            Self::Erfcinv => erf::erfcinv(args[0]),
            Self::Gamma => gamma::gamma(args[0]),
            Self::Rgamma => gamma::rgamma(args[0]),
            Self::Gammaln => gamma::gammaln(args[0]),
            Self::Loggamma => gamma::loggamma(args[0]),
            Self::Psi => gamma::digamma(args[0]),
            Self::Beta => gamma::beta(args[0], args[1]),
            Self::Betaln => gamma::betaln(args[0], args[1]),
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
            Self::Expit => misc::expit(args[0]),
            Self::Logit => misc::logit(args[0]),
            Self::LogExpit => misc::log_expit(args[0]),
            Self::Xlogy => misc::xlogy(args[0], args[1]),
            Self::Xlog1py => misc::xlog1py(args[0], args[1]),
            Self::Entr => misc::entr(args[0]),
            Self::RelEntr => misc::rel_entr(args[0], args[1]),
            Self::KlDiv => misc::kl_div(args[0], args[1]),
            Self::Binom => gamma::binom(args[0], args[1]),
            Self::Poch => gamma::poch(args[0], args[1]),
            Self::Stdtr => distributions::stdtr(args[0], args[1]),
            Self::Stdtrit => distributions::stdtrit(args[0], args[1]),
            Self::Chdtr => distributions::chdtr(args[0], args[1]),
            Self::Chdtrc => distributions::chdtrc(args[0], args[1]),
            Self::Chdtri => distributions::chdtri(args[0], args[1]),
            Self::Fdtr => distributions::fdtr(args[0], args[1], args[2]),
            Self::Fdtrc => distributions::fdtrc(args[0], args[1], args[2]),
            Self::Fdtri => distributions::fdtri(args[0], args[1], args[2]),
            Self::Pdtr => distributions::pdtr(args[0], args[1]),
            Self::Pdtrc => distributions::pdtrc(args[0], args[1]),
            Self::Pdtrik => distributions::pdtrik(args[0], args[1]),
            Self::Bdtr => distributions::bdtr(args[0], args[1], args[2]),
            Self::Bdtrc => distributions::bdtrc(args[0], args[1], args[2]),
            Self::BinomPmf => distributions::binom_pmf(args[0], args[1], args[2]),
            Self::BinomCdf => distributions::bdtr(args[0], args[1], args[2]),
            Self::BinomSf => distributions::bdtrc(args[0], args[1], args[2]),
            Self::BinomPpf => distributions::binom_ppf(args[0], args[1], args[2]),
            Self::BinomIsf => distributions::binom_isf(args[0], args[1], args[2]),
            Self::Boxcox => misc::boxcox(args[0], args[1]),
            Self::InvBoxcox => misc::inv_boxcox(args[0], args[1]),
            Self::Expm1 => misc::expm1(args[0]),
            Self::Log1p => misc::log1p(args[0]),
            Self::RiemannZeta => zeta::riemann_zeta(args[0]),
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
    Rgamma,
    Gammaln,
    Loggamma,
    Psi,
    Beta,
    Betaln,
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
    Expit,
    Logit,
    LogExpit,
    Xlogy,
    Xlog1py,
    Entr,
    RelEntr,
    KlDiv,
    Binom,
    Poch,
    Stdtr,
    Stdtrit,
    Chdtr,
    Chdtrc,
    Chdtri,
    Fdtr,
    Fdtrc,
    Fdtri,
    Pdtr,
    Pdtrc,
    Pdtrik,
    Bdtr,
    Bdtrc,
    BinomPmf,
    BinomCdf,
    BinomSf,
    BinomPpf,
    BinomIsf,
    Boxcox,
    InvBoxcox,
    Expm1,
    Log1p,
    RiemannZeta,
    Zeta,
);

/// `_scipy_special` exports only ufunc values (every SciPy-written-in-Python name, such as
/// `comb` or `logsumexp`, lives in frozen Python instead); it declares no native functions.
pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_scipy_special",
    functions: &[],
    values: VALUES,
};
