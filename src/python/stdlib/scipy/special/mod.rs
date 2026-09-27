//! `scipy.special` ufuncs: the numeric kernels and the `_scipy_special` native module that
//! exports them.
//!
//! Each [`Function`] is a NumPy ufunc: its `numpy.ufunc` value indexes the shared ufunc table,
//! and NumPy's ufunc machinery (`numpy::ufunc`) handles broadcasting, casting, `out=`, and
//! scalar boxing. This module supplies what differs per function: the arity, the loops SciPy
//! registers (which decide the result dtype) and the kernel, which [`evaluate`] runs under a
//! work meter.
//!
//! Kernels are ports of the implementations SciPy 1.18 uses: Cephes (through SciPy's xsf
//! library) for most functions and Boost.Math for the incomplete beta family and the t and F
//! distributions. Like SciPy's loops, kernels never raise floating-point errors or warnings;
//! domain errors return NaN and poles return infinities. Complex loops are not implemented and
//! fail explicitly.
//!
//! Some kernels iterate many times per element for extreme arguments, so every element is
//! charged a flat [`ELEMENT_COST`] and every metered iteration (see [`meter`]) a further
//! `STEP_COST`.

mod distributions;
mod elementary;
mod erf;
mod gamma;
mod ibeta;
mod ibeta_inv;
mod igam;
mod meter;
mod poly;
mod unity;

use super::super::super::native::{ModuleDef, PyResult, PyRuntime, ValueDef};
use super::super::numpy::named_ufunc_value;

/// A `scipy.special` ufunc.
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
    Bdtr,
    Bdtrc,
    Boxcox,
    InvBoxcox,
    RiemannZeta,
    Zeta,
}

/// The loops SciPy registers for a function, which decide the loop NumPy selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::python) enum Loops {
    /// `float32` then `float64` loops. With `complex`, SciPy also has complex loops, which
    /// shellsim does not implement.
    Real { complex: bool },
    /// `logit` registers its `float64` loop first, so only `float32` input, which matches the
    /// `float32` loop exactly, computes in single precision.
    DoubleFirst,
    /// `bdtr` and `bdtrc` take the trial count `n` as a C `int` in their `(float64, int64,
    /// float64)` loop. Their `float32` and all-`float64` loops accept a floating-point `n`, which
    /// SciPy deprecates with a warning on every call.
    Binomial,
}

/// CPU units per element, for the closed-form part of every kernel.
pub(in crate::python) const ELEMENT_COST: u64 = 8;
/// CPU units per metered step: one iteration of a series, continued fraction or root finder, or
/// one evaluation of the incomplete beta or gamma function.
const STEP_COST: u64 = 4;
/// Steps an element may take before it is charged and retried with a doubled allowance. Most
/// elements take well under a hundred.
const FIRST_ALLOWANCE: u64 = 1 << 12;

/// Evaluate one element with `kernel`, charging `runtime` for every step the kernel takes.
///
/// An element that exhausts its allowance is charged and evaluated again with twice the
/// allowance, so its work is bounded by the CPU budget, and the repeated work is at most the
/// work of the final evaluation.
pub(in crate::python) fn evaluate<T>(
    runtime: &mut dyn PyRuntime,
    kernel: impl Fn() -> T,
) -> PyResult<T> {
    let mut allowance = FIRST_ALLOWANCE;
    loop {
        let (value, steps) = meter::run(allowance, &kernel);
        runtime.charge_cpu(steps.saturating_mul(STEP_COST))?;
        if let Some(value) = value {
            return Ok(value);
        }
        allowance = allowance.saturating_mul(2);
    }
}

impl Function {
    /// Every function, in `numpy.ufunc` table order.
    pub(in crate::python) const ALL: [Self; 47] = [
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
        Self::Bdtr,
        Self::Bdtrc,
        Self::Boxcox,
        Self::InvBoxcox,
        Self::RiemannZeta,
        Self::Zeta,
    ];

    /// The ufunc's `__name__`.
    pub(in crate::python) const fn name(self) -> &'static str {
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
            Self::Bdtr => "bdtr",
            Self::Bdtrc => "bdtrc",
            Self::Boxcox => "boxcox",
            Self::InvBoxcox => "inv_boxcox",
            Self::RiemannZeta => "_riemann_zeta",
            Self::Zeta => "_zeta",
        }
    }

    /// The number of inputs.
    pub(in crate::python) const fn nin(self) -> usize {
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
            | Self::RiemannZeta => 1,
            Self::Betainc
            | Self::Betaincc
            | Self::Betaincinv
            | Self::Fdtr
            | Self::Fdtrc
            | Self::Fdtri
            | Self::Bdtr
            | Self::Bdtrc => 3,
            _ => 2,
        }
    }

    pub(in crate::python) const fn loops(self) -> Loops {
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
            | Self::RiemannZeta
            | Self::Zeta => Loops::Real { complex: true },
            Self::Logit => Loops::DoubleFirst,
            Self::Bdtr | Self::Bdtrc => Loops::Binomial,
            _ => Loops::Real { complex: false },
        }
    }

    /// Evaluate the `float64` loop on one element's arguments; `args` has [`Self::nin`] values.
    pub(in crate::python) fn eval(self, args: &[f64]) -> f64 {
        let arg = |index: usize| args[index];
        match self {
            Self::Erf => erf::erf(arg(0)),
            Self::Erfc => erf::erfc(arg(0)),
            Self::Erfinv => erf::erfinv(arg(0)),
            Self::Erfcinv => erf::erfcinv(arg(0)),
            Self::Gamma => gamma::gamma(arg(0)),
            Self::Rgamma => gamma::rgamma(arg(0)),
            Self::Gammaln => gamma::lgam(arg(0)),
            Self::Loggamma => gamma::loggamma(arg(0)),
            Self::Psi => gamma::digamma(arg(0)),
            Self::Beta => gamma::beta(arg(0), arg(1)),
            Self::Betaln => gamma::lbeta(arg(0), arg(1)),
            Self::Betainc => ibeta::betainc(arg(0), arg(1), arg(2)),
            Self::Betaincc => ibeta::betaincc(arg(0), arg(1), arg(2)),
            Self::Betaincinv => ibeta_inv::betaincinv(arg(0), arg(1), arg(2)),
            Self::Gammainc => igam::igam(arg(0), arg(1)),
            Self::Gammaincc => igam::igamc(arg(0), arg(1)),
            Self::Gammaincinv => igam::igami(arg(0), arg(1)),
            Self::Gammainccinv => igam::igamci(arg(0), arg(1)),
            Self::Ndtr => erf::ndtr(arg(0)),
            Self::LogNdtr => erf::log_ndtr(arg(0)),
            Self::Ndtri => erf::ndtri(arg(0)),
            Self::Expit => elementary::expit(arg(0)),
            Self::Logit => elementary::logit(arg(0)),
            Self::LogExpit => elementary::log_expit(arg(0)),
            Self::Xlogy => elementary::xlogy(arg(0), arg(1)),
            Self::Xlog1py => elementary::xlog1py(arg(0), arg(1)),
            Self::Entr => elementary::entr(arg(0)),
            Self::RelEntr => elementary::rel_entr(arg(0), arg(1)),
            Self::KlDiv => elementary::kl_div(arg(0), arg(1)),
            Self::Binom => elementary::binom(arg(0), arg(1)),
            Self::Poch => gamma::poch(arg(0), arg(1)),
            Self::Stdtr => distributions::stdtr(arg(0), arg(1)),
            Self::Stdtrit => distributions::stdtrit(arg(0), arg(1)),
            Self::Chdtr => distributions::chdtr(arg(0), arg(1)),
            Self::Chdtrc => distributions::chdtrc(arg(0), arg(1)),
            Self::Chdtri => distributions::chdtri(arg(0), arg(1)),
            Self::Fdtr => distributions::fdtr(arg(0), arg(1), arg(2)),
            Self::Fdtrc => distributions::fdtrc(arg(0), arg(1), arg(2)),
            Self::Fdtri => distributions::fdtri(arg(0), arg(1), arg(2)),
            Self::Pdtr => distributions::pdtr(arg(0), arg(1)),
            Self::Pdtrc => distributions::pdtrc(arg(0), arg(1)),
            Self::Bdtr => distributions::bdtr(arg(0), arg(1), arg(2)),
            Self::Bdtrc => distributions::bdtrc(arg(0), arg(1), arg(2)),
            Self::Boxcox => elementary::boxcox(arg(0), arg(1)),
            Self::InvBoxcox => elementary::inv_boxcox(arg(0), arg(1)),
            Self::RiemannZeta => gamma::riemann_zeta(arg(0)),
            Self::Zeta => gamma::zeta(arg(0), arg(1)),
        }
    }

    /// Evaluate the `float32` loop. SciPy's templated kernels compute in single precision;
    /// the others compute in `f64` and round the result.
    pub(in crate::python) fn eval_f32(self, args: &[f32]) -> f32 {
        let arg = |index: usize| args[index];
        match self {
            Self::Expit => elementary::expit_f32(arg(0)),
            Self::Logit => elementary::logit_f32(arg(0)),
            Self::LogExpit => elementary::log_expit_f32(arg(0)),
            Self::Xlogy => elementary::xlogy_f32(arg(0), arg(1)),
            Self::Xlog1py => elementary::xlog1py_f32(arg(0), arg(1)),
            _ => {
                let mut wide = [0.0; 3];
                for (wide, narrow) in wide.iter_mut().zip(args) {
                    *wide = f64::from(*narrow);
                }
                #[allow(clippy::cast_possible_truncation)]
                let result = self.eval(&wide[..args.len()]) as f32;
                result
            }
        }
    }
}

/// `scipy.special`'s ufuncs, star-imported by the frozen `scipy.special` package.
pub(in crate::python) static MODULE: ModuleDef = ModuleDef {
    name: "_scipy_special",
    functions: &[],
    values: &VALUES,
};

static VALUES: [ValueDef; Function::ALL.len() + 1] = values();

/// One value per function under its own name, then `digamma`, SciPy's alias of `psi`.
const fn values() -> [ValueDef; Function::ALL.len() + 1] {
    let mut values =
        [const { named_ufunc_value("digamma", Function::Psi.name()) }; Function::ALL.len() + 1];
    let mut index = 0;
    while index < Function::ALL.len() {
        let name = Function::ALL[index].name();
        values[index] = named_ufunc_value(name, name);
        index += 1;
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arity_matches_the_argument_count_each_kernel_reads() {
        for function in Function::ALL {
            let args = [0.5; 3];
            // Reading past nin would panic on the slice.
            let _ = function.eval(&args[..function.nin()]);
            let _ = function.eval_f32(&[0.5f32; 3][..function.nin()]);
        }
    }

    #[test]
    fn float32_kernels_round_like_single_precision() {
        let wide = Function::Erf.eval(&[0.5]);
        #[allow(clippy::cast_possible_truncation)]
        let narrow = wide as f32;
        assert_eq!(Function::Erf.eval_f32(&[0.5]), narrow);
        assert_eq!(Function::Expit.eval_f32(&[0.0]), 0.5);
    }
}
