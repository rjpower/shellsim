//! Ziggurat sampling for the standard normal and standard exponential distributions.
//!
//! This follows Marsaglia & Tsang's 2000 "The Ziggurat Method for Generating Random Variables"
//! with 256 layers, the layer count NumPy also documents. The layer boundaries `x[0..256]` are
//! the unique solution (found here by bisection, not copied from any table) of the standard
//! ziggurat balance equations: every layer, plus the tail beyond `x[255]`, has the same area.
//! Solving them independently reproduces the well-known `x[255] = 3.6541528853610...` constant
//! for the 256-layer normal ziggurat, which is a property of the equations and `n = 256`, not of
//! any particular implementation. Each draw either accepts within a layer's fast rectangle,
//! falls back to an acceptance test against the true density in the layer's "wedge", or (in the
//! bottom layer) samples the unbounded tail with Marsaglia's exponential-based tail algorithm.
//!
//! **Accuracy**: this reproduces the ziggurat method faithfully and its output is statistically
//! standard normal / standard exponential, but it does not reproduce NumPy 2.5.3's stream bit
//! for bit. NumPy packs the layer index, sign, and fraction bits of one 64-bit draw in a way
//! its public documentation does not specify, and this module's (reasonable, but unverified)
//! choice of that packing does not reproduce NumPy's own table lookups word for word. See
//! `docs/numpy.md`. Downstream distributions built on the normal (Generator's `standard_gamma`
//! for shape >= 1, and therefore `gamma`, `chisquare`, `f`, and `standard_t`) inherit this same
//! difference; legacy `RandomState`'s Gaussian (the polar method, see `legacy_gauss` in
//! `gamma.rs`) is unaffected and matches NumPy exactly.

use std::sync::OnceLock;

use super::bitgen::BitGen;

const LAYERS: usize = 256;
const MANTISSA_BITS: u32 = 55;

struct NormalTables {
    /// Boundaries `x[0] = r` (largest, adjoining the tail) down to `x[255]` (smallest). Kept
    /// alongside the derived `k`/`w`/`f` tables for documentation even though sampling only
    /// reads the derived tables.
    #[allow(dead_code)]
    x: [f64; LAYERS],
    f: [f64; LAYERS],
    /// Fast-accept threshold for each layer's mantissa, in mantissa units.
    k: [u64; LAYERS],
    /// Fast-accept scale for each layer's mantissa: `x = mantissa * w[idx]`.
    w: [f64; LAYERS],
    r: f64,
}

fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp()
}

/// `sqrt(pi/2) * erfc(r / sqrt(2))`: the Gaussian tail integral.
fn tail_area(r: f64) -> f64 {
    std::f64::consts::FRAC_PI_2.sqrt() * (1.0 - erf(r / std::f64::consts::SQRT_2))
}

/// `erf(x)` by its defining power series, `2/sqrt(pi) * sum (-1)^n x^(2n+1) / (n! (2n+1))`.
/// The series converges for every `x`; this sums until a term stops changing the total, which
/// is a few dozen terms for the ziggurat's `x` values (a few units). Table construction runs
/// once per process, so the series' simplicity matters more than its cost.
fn erf(x: f64) -> f64 {
    let x2 = x * x;
    let mut term = x;
    let mut total = x;
    let mut n = 0.0f64;
    loop {
        n += 1.0;
        term *= -x2 / n;
        let add = term / (2.0 * n + 1.0);
        total += add;
        if add.abs() < 1e-18 * total.abs() || n > 500.0 {
            break;
        }
    }
    total * 2.0 / std::f64::consts::PI.sqrt()
}

fn build_normal() -> NormalTables {
    let residual = |r: f64| -> Option<f64> {
        let area = r * normal_pdf(r) + tail_area(r);
        let mut x = [0.0f64; LAYERS];
        x[LAYERS - 1] = r;
        let mut prev_f = normal_pdf(r);
        for i in (0..LAYERS - 1).rev() {
            let value = prev_f + area / x[i + 1];
            if value >= 1.0 {
                return None;
            }
            x[i] = (-2.0 * value.ln()).sqrt();
            prev_f = value;
        }
        Some(x[0] * (1.0 - normal_pdf(x[0])) - area)
    };
    let (mut lo, mut hi) = (3.0f64, 4.0f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        match residual(mid) {
            None => lo = mid,
            Some(_) => hi = mid,
        }
    }
    let r = hi;
    let area = r * normal_pdf(r) + tail_area(r);
    let mut x_asc = [0.0f64; LAYERS];
    x_asc[LAYERS - 1] = r;
    let mut prev_f = normal_pdf(r);
    for i in (0..LAYERS - 1).rev() {
        let value = prev_f + area / x_asc[i + 1];
        x_asc[i] = (-2.0 * value.ln()).sqrt();
        prev_f = value;
    }
    // Reverse into NumPy's draw order: index 0 is the tail-adjoining (widest) layer.
    let mut x = [0.0f64; LAYERS];
    for i in 0..LAYERS {
        x[i] = x_asc[LAYERS - 1 - i];
    }
    let mut f = [0.0f64; LAYERS];
    for i in 0..LAYERS {
        f[i] = normal_pdf(x[i]);
    }
    let scale = (1u64 << MANTISSA_BITS) as f64;
    let mut w = [0.0f64; LAYERS];
    let mut k = [0u64; LAYERS];
    w[0] = (area / normal_pdf(r)) / scale;
    k[0] = (scale * r * normal_pdf(r) / area) as u64;
    for i in 1..LAYERS {
        w[i] = x[i] / scale;
        k[i] = (scale * x[i - 1] / x[i]) as u64;
    }
    NormalTables { x, f, k, w, r }
}

static NORMAL: OnceLock<NormalTables> = OnceLock::new();

fn normal_tables() -> &'static NormalTables {
    NORMAL.get_or_init(build_normal)
}

/// Marsaglia's tail algorithm: an accept-reject sampler for the Gaussian tail beyond `r`.
fn sample_tail(bitgen: &mut BitGen, r: f64) -> f64 {
    loop {
        let x = -bitgen.next_double().ln() / r;
        let y = -bitgen.next_double().ln();
        if 2.0 * y > x * x {
            return r + x;
        }
    }
}

/// One standard normal draw.
pub(in crate::python) fn next_gauss(bitgen: &mut BitGen) -> f64 {
    let tables = normal_tables();
    loop {
        let word = bitgen.next_u64();
        let idx = (word & 0xff) as usize;
        let rest = word >> 8;
        let sign = rest & 1;
        let mantissa = rest >> 1;
        let x = mantissa as f64 * tables.w[idx];
        if mantissa < tables.k[idx] {
            return if sign != 0 { -x } else { x };
        }
        if idx == 0 {
            let tail = sample_tail(bitgen, tables.r);
            return if sign != 0 { -tail } else { tail };
        }
        let u = bitgen.next_double();
        if u * (tables.f[idx - 1] - tables.f[idx]) + tables.f[idx] < normal_pdf(x) {
            return if sign != 0 { -x } else { x };
        }
    }
}

struct ExponentialTables {
    /// See `NormalTables::x`: kept for documentation, unused by sampling.
    #[allow(dead_code)]
    x: [f64; LAYERS],
    f: [f64; LAYERS],
    k: [u64; LAYERS],
    w: [f64; LAYERS],
    r: f64,
}

fn exp_pdf(x: f64) -> f64 {
    (-x).exp()
}

fn build_exponential() -> ExponentialTables {
    // The same balance equations, with f(x) = exp(-x) instead of the Gaussian density; the
    // inverse of f is `-ln`, and the tail integral of `exp(-x)` beyond `r` is `exp(-r)`.
    let residual = |r: f64| -> Option<f64> {
        let area = r * exp_pdf(r) + exp_pdf(r);
        let mut x = [0.0f64; LAYERS];
        x[LAYERS - 1] = r;
        let mut prev_f = exp_pdf(r);
        for i in (0..LAYERS - 1).rev() {
            let value = prev_f + area / x[i + 1];
            if value >= 1.0 {
                return None;
            }
            x[i] = -value.ln();
            prev_f = value;
        }
        Some(x[0] * (1.0 - exp_pdf(x[0])) - area)
    };
    let (mut lo, mut hi) = (5.0f64, 8.0f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        match residual(mid) {
            None => lo = mid,
            Some(_) => hi = mid,
        }
    }
    let r = hi;
    let area = r * exp_pdf(r) + exp_pdf(r);
    let mut x_asc = [0.0f64; LAYERS];
    x_asc[LAYERS - 1] = r;
    let mut prev_f = exp_pdf(r);
    for i in (0..LAYERS - 1).rev() {
        let value = prev_f + area / x_asc[i + 1];
        x_asc[i] = -value.ln();
        prev_f = value;
    }
    let mut x = [0.0f64; LAYERS];
    for i in 0..LAYERS {
        x[i] = x_asc[LAYERS - 1 - i];
    }
    let mut f = [0.0f64; LAYERS];
    for i in 0..LAYERS {
        f[i] = exp_pdf(x[i]);
    }
    let scale = (1u64 << MANTISSA_BITS) as f64;
    let mut w = [0.0f64; LAYERS];
    let mut k = [0u64; LAYERS];
    w[0] = (area / exp_pdf(r)) / scale;
    k[0] = (scale * r * exp_pdf(r) / area) as u64;
    for i in 1..LAYERS {
        w[i] = x[i] / scale;
        k[i] = (scale * x[i - 1] / x[i]) as u64;
    }
    ExponentialTables { x, f, k, w, r }
}

static EXPONENTIAL: OnceLock<ExponentialTables> = OnceLock::new();

fn exponential_tables() -> &'static ExponentialTables {
    EXPONENTIAL.get_or_init(build_exponential)
}

/// One standard exponential draw by the ziggurat method (`method="zig"`, the default).
pub(in crate::python) fn next_exponential_zig(bitgen: &mut BitGen) -> f64 {
    let tables = exponential_tables();
    loop {
        let word = bitgen.next_u64();
        let idx = (word & 0xff) as usize;
        let mantissa = word >> 8;
        let x = mantissa as f64 * tables.w[idx];
        if mantissa < tables.k[idx] {
            return x;
        }
        if idx == 0 {
            return tables.r - bitgen.next_double().ln();
        }
        let u = bitgen.next_double();
        if u * (tables.f[idx - 1] - tables.f[idx]) + tables.f[idx] < exp_pdf(x) {
            return x;
        }
    }
}

/// One standard exponential draw by inversion (`method="inv"`): `-log(1 - U)`, the same
/// formula legacy `RandomState.standard_exponential` always uses.
pub(in crate::python) fn next_exponential_inv(bitgen: &mut BitGen) -> f64 {
    -(1.0 - bitgen.next_double()).ln()
}
