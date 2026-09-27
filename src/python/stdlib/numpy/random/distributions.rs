//! Parametric samplers behind `numpy.random`: exponential, gamma, chi-square, F, Student's t,
//! binomial and Poisson.
//!
//! Each sampler ports one function of NumPy 2.5's `distributions.c`, which `Generator` uses, or
//! of `legacy-distributions.c`, which keeps `RandomState` on its NumPy 1.16 streams. The two
//! differ in how they draw normals (ziggurat or polar method) and exponentials (ziggurat or
//! inversion), and in details of the binomial sampler, so the samplers take a [`Family`].
//! Arithmetic follows the C expressions operation for operation, including integer arithmetic
//! and evaluation order, and `exp`, `log`, `log1p` and `pow` come from the platform libm, so
//! seeded streams match NumPy bit for bit.
//!
//! The Python layer validates parameters before a sampler runs, as NumPy's Cython layer does.
//! Every loop draws from the [`Stream`] or spends credit on it, so rejection loops stay metered.

use super::bitgen::Stream;
use super::ziggurat::{
    EXP_R, EXP_R_F, FE_DOUBLE, FE_FLOAT, KE_DOUBLE, KE_FLOAT, WE_DOUBLE, WE_FLOAT,
};
use super::{polar_gauss, ziggurat_normal, ziggurat_normal_f32};
use crate::python::native::PyResult;

/// Which of NumPy's two sampler families to follow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Family {
    /// `Generator`, on `distributions.c`.
    Generator,
    /// `RandomState`, on `legacy-distributions.c`.
    Legacy,
}

/// A continuous distribution drawn with per-element parameters `a` and `b`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Continuous {
    /// `exponential(scale=a)`.
    Exponential,
    /// `standard_gamma(shape=a)`.
    StandardGamma,
    /// `gamma(shape=a, scale=b)`.
    Gamma,
    /// `chisquare(df=a)`.
    Chisquare,
    /// `f(dfnum=a, dfden=b)`.
    F,
    /// `standard_t(df=a)`.
    StandardT,
}

impl Continuous {
    pub(super) fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "exponential" => Self::Exponential,
            "standard_gamma" => Self::StandardGamma,
            "gamma" => Self::Gamma,
            "chisquare" => Self::Chisquare,
            "f" => Self::F,
            "standard_t" => Self::StandardT,
            _ => return None,
        })
    }

    /// One draw, as NumPy's `random_*` or `legacy_*` function of the same name.
    pub(super) fn sample(
        self,
        stream: &mut Stream<'_>,
        family: Family,
        a: f64,
        b: f64,
    ) -> PyResult<f64> {
        Ok(match self {
            Self::Exponential => a * standard_exponential(stream, family)?,
            Self::StandardGamma => standard_gamma(stream, family, a)?,
            Self::Gamma => b * standard_gamma(stream, family, a)?,
            Self::Chisquare => chisquare(stream, family, a)?,
            Self::F => {
                let numerator = chisquare(stream, family, a)? * b;
                let denominator = chisquare(stream, family, b)? * a;
                numerator / denominator
            }
            Self::StandardT => {
                let num = normal(stream, family)?;
                let denom = standard_gamma(stream, family, a / 2.0)?;
                (a / 2.0).sqrt() * num / denom.sqrt()
            }
        })
    }
}

fn normal(stream: &mut Stream<'_>, family: Family) -> PyResult<f64> {
    match family {
        Family::Generator => ziggurat_normal(stream),
        Family::Legacy => polar_gauss(stream),
    }
}

/// `random_standard_exponential` or `legacy_standard_exponential`.
pub(super) fn standard_exponential(stream: &mut Stream<'_>, family: Family) -> PyResult<f64> {
    match family {
        Family::Generator => ziggurat_exponential(stream),
        // U is in [0, 1), so 1 - U is never 0.
        Family::Legacy => Ok(-(1.0 - stream.next_double()?).ln()),
    }
}

/// `random_standard_exponential`: 53 bits of magnitude and an 8-bit layer index from one 64-bit
/// draw. NumPy retries by tail recursion; this loops.
fn ziggurat_exponential(stream: &mut Stream<'_>) -> PyResult<f64> {
    loop {
        let mut ri = stream.next_u64()? >> 3;
        let idx = (ri & 0xff) as usize;
        ri >>= 8;
        let x = ri as f64 * WE_DOUBLE[idx];
        if ri < KE_DOUBLE[idx] {
            return Ok(x);
        }
        if idx == 0 {
            return Ok(EXP_R - (-stream.next_double()?).ln_1p());
        }
        if (FE_DOUBLE[idx - 1] - FE_DOUBLE[idx]) * stream.next_double()? + FE_DOUBLE[idx]
            < (-x).exp()
        {
            return Ok(x);
        }
    }
}

/// `random_standard_exponential_f`: the float32 ziggurat on one 32-bit draw.
pub(super) fn ziggurat_exponential_f32(stream: &mut Stream<'_>) -> PyResult<f32> {
    loop {
        let mut ri = stream.next_u32()? >> 1;
        let idx = (ri & 0xff) as usize;
        ri >>= 8;
        let x = ri as f32 * WE_FLOAT[idx];
        if ri < KE_FLOAT[idx] {
            return Ok(x);
        }
        if idx == 0 {
            return Ok(EXP_R_F - (-stream.next_float()?).ln_1p());
        }
        if (FE_FLOAT[idx - 1] - FE_FLOAT[idx]) * stream.next_float()? + FE_FLOAT[idx] < (-x).exp() {
            return Ok(x);
        }
    }
}

/// `random_standard_exponential_inv_fill`: inversion of one uniform.
pub(super) fn inverse_exponential(stream: &mut Stream<'_>) -> PyResult<f64> {
    Ok(-(-stream.next_double()?).ln_1p())
}

/// `random_standard_exponential_inv_fill_f`, which takes `log1p` in double precision.
pub(super) fn inverse_exponential_f32(stream: &mut Stream<'_>) -> PyResult<f32> {
    let uniform = f64::from(stream.next_float()?);
    #[allow(clippy::cast_possible_truncation)]
    let value = -(-uniform).ln_1p() as f32;
    Ok(value)
}

/// `random_standard_gamma` or `legacy_standard_gamma`: Marsaglia and Tsang's method for
/// `shape > 1`, and Johnk's rejection method below 1.
fn standard_gamma(stream: &mut Stream<'_>, family: Family, shape: f64) -> PyResult<f64> {
    if shape == 1.0 {
        return standard_exponential(stream, family);
    }
    if shape == 0.0 {
        return Ok(0.0);
    }
    if shape < 1.0 {
        loop {
            let u = stream.next_double()?;
            let v = standard_exponential(stream, family)?;
            if u <= 1.0 - shape {
                let x = u.powf(1.0 / shape);
                if x <= v {
                    return Ok(x);
                }
            } else {
                let y = -((1.0 - u) / shape).ln();
                let x = (1.0 - shape + shape * y).powf(1.0 / shape);
                if x <= v + y {
                    return Ok(x);
                }
            }
        }
    }
    let b = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * b).sqrt();
    loop {
        // C's `do { ... } while (V <= 0.0)` ends on a NaN, which a NaN shape produces.
        let (x, v) = loop {
            let x = normal(stream, family)?;
            let v = 1.0 + c * x;
            if v > 0.0 || v.is_nan() {
                break (x, v);
            }
        };
        let v = v * v * v;
        let u = stream.next_double()?;
        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return Ok(b * v);
        }
        if u.ln() < 0.5 * x * x + b * (1.0 - v + v.ln()) {
            return Ok(b * v);
        }
    }
}

/// `random_standard_gamma_f`: [`standard_gamma`] in single precision for `Generator`.
pub(super) fn standard_gamma_f32(stream: &mut Stream<'_>, shape: f32) -> PyResult<f32> {
    if shape == 1.0 {
        return ziggurat_exponential_f32(stream);
    }
    if shape == 0.0 {
        return Ok(0.0);
    }
    if shape < 1.0 {
        loop {
            let u = stream.next_float()?;
            let v = ziggurat_exponential_f32(stream)?;
            if u <= 1.0 - shape {
                let x = u.powf(1.0 / shape);
                if x <= v {
                    return Ok(x);
                }
            } else {
                let y = -((1.0 - u) / shape).ln();
                let x = (1.0 - shape + shape * y).powf(1.0 / shape);
                if x <= v + y {
                    return Ok(x);
                }
            }
        }
    }
    let b = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * b).sqrt();
    loop {
        let (x, v) = loop {
            let x = ziggurat_normal_f32(stream)?;
            let v = 1.0 + c * x;
            if v > 0.0 || v.is_nan() {
                break (x, v);
            }
        };
        let v = v * v * v;
        let u = stream.next_float()?;
        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return Ok(b * v);
        }
        if u.ln() < 0.5 * x * x + b * (1.0 - v + v.ln()) {
            return Ok(b * v);
        }
    }
}

fn chisquare(stream: &mut Stream<'_>, family: Family, df: f64) -> PyResult<f64> {
    Ok(2.0 * standard_gamma(stream, family, df / 2.0)?)
}

/// `random_loggam`: `log(gamma(x))` for the Poisson sampler, by Stirling's series after shifting
/// `x` to at least 7.
fn loggam(x: f64) -> f64 {
    const A: [f64; 10] = [
        8.333333333333333e-02,
        -2.777777777777778e-03,
        7.936507936507937e-04,
        -5.952380952380952e-04,
        8.417508417508418e-04,
        -1.917526917526918e-03,
        6.41025641025641e-03,
        -2.955065359477124e-02,
        1.796443723688307e-01,
        -1.39243221690590e+00,
    ];
    if x == 1.0 || x == 2.0 {
        return 0.0;
    }
    #[allow(clippy::cast_possible_truncation)]
    let n = if x < 7.0 { (7.0 - x) as i64 } else { 0 };
    let mut x0 = x + n as f64;
    let x2 = (1.0 / x0) * (1.0 / x0);
    let lg2pi = 1.8378770664093453e+00;
    let mut gl0 = A[9];
    for coefficient in A[..9].iter().rev() {
        gl0 *= x2;
        gl0 += coefficient;
    }
    let mut gl = gl0 / x0 + 0.5 * lg2pi + (x0 - 0.5) * x0.ln() - x0;
    if x < 7.0 {
        for _ in 1..=n {
            gl -= (x0 - 1.0).ln();
            x0 -= 1.0;
        }
    }
    gl
}

/// `random_poisson`, which `legacy_random_poisson` shares: multiplication of uniforms below
/// `lam = 10`, and Hormann's transformed rejection (PTRS) above.
pub(super) fn poisson(stream: &mut Stream<'_>, lam: f64) -> PyResult<i64> {
    if lam >= 10.0 {
        poisson_ptrs(stream, lam)
    } else if lam == 0.0 {
        Ok(0)
    } else {
        let enlam = (-lam).exp();
        let mut x = 0;
        let mut product = 1.0;
        loop {
            product *= stream.next_double()?;
            if product > enlam {
                x += 1;
            } else {
                return Ok(x);
            }
        }
    }
}

#[allow(clippy::cast_possible_truncation)]
fn poisson_ptrs(stream: &mut Stream<'_>, lam: f64) -> PyResult<i64> {
    let slam = lam.sqrt();
    let loglam = lam.ln();
    let b = 0.931 + 2.53 * slam;
    let a = -0.059 + 0.02483 * b;
    let invalpha = 1.1239 + 1.1328 / (b - 3.4);
    let vr = 0.9277 - 3.6224 / (b - 2.0);
    loop {
        let u = stream.next_double()? - 0.5;
        let v = stream.next_double()?;
        let us = 0.5 - u.abs();
        // Out-of-range conversions saturate, as x86-64's conversion does for C's cast.
        let k = ((2.0 * a / us + b) * u + lam + 0.43).floor() as i64;
        if us >= 0.07 && v <= vr {
            return Ok(k);
        }
        if k < 0 || (us < 0.013 && v > us) {
            continue;
        }
        if v.ln() + invalpha.ln() - (a / (us * us) + b).ln()
            <= -lam + k as f64 * loglam - loggam(k as f64 + 1.0)
        {
            return Ok(k);
        }
    }
}

/// `random_binomial` or `legacy_random_binomial`: inversion when the mean of the smaller tail
/// is at most 30, and BTPE above.
pub(super) fn binomial(stream: &mut Stream<'_>, family: Family, p: f64, n: i64) -> PyResult<i64> {
    // Only Generator skips the draw for a degenerate distribution.
    if family == Family::Generator && (n == 0 || p == 0.0) {
        return Ok(0);
    }
    if p <= 0.5 {
        if p * n as f64 <= 30.0 {
            binomial_inversion(stream, family, n, p)
        } else {
            binomial_btpe(stream, family, n, p)
        }
    } else {
        let q = 1.0 - p;
        let tail = if q * n as f64 <= 30.0 {
            binomial_inversion(stream, family, n, q)?
        } else {
            binomial_btpe(stream, family, n, q)?
        };
        Ok(n - tail)
    }
}

#[allow(clippy::cast_possible_truncation)]
fn binomial_inversion(stream: &mut Stream<'_>, family: Family, n: i64, p: f64) -> PyResult<i64> {
    let q = 1.0 - p;
    let qn = match family {
        Family::Generator => (n as f64 * (-p).ln_1p()).exp(),
        Family::Legacy => (n as f64 * q.ln()).exp(),
    };
    let np = n as f64 * p;
    // C's MIN(n, ...) compares and yields a double.
    let limit = np + 10.0 * (np * q + 1.0).sqrt();
    let bound = (if (n as f64) < limit { n as f64 } else { limit }) as i64;
    let mut x = 0;
    let mut px = qn;
    let mut u = stream.next_double()?;
    while u > px {
        x += 1;
        if x > bound {
            x = 0;
            px = qn;
            u = stream.next_double()?;
        } else {
            u -= px;
            px = ((n - x + 1) as f64 * p * px) / (x as f64 * q);
        }
    }
    Ok(x)
}

/// Kachitvichyanukul and Schmeiser's BTPE. `Generator` subtracts the last two Stirling error
/// terms, correcting the 1988 paper; `RandomState` keeps the paper's additions and its 13680.
#[allow(clippy::cast_possible_truncation, clippy::many_single_char_names)]
fn binomial_btpe(stream: &mut Stream<'_>, family: Family, n: i64, p: f64) -> PyResult<i64> {
    let r = if p < 1.0 - p { p } else { 1.0 - p };
    let q = 1.0 - r;
    let fm = n as f64 * r + r;
    let m = fm.floor() as i64;
    let p1 = (2.195 * (n as f64 * r * q).sqrt() - 4.6 * q).floor() + 0.5;
    let xm = m as f64 + 0.5;
    let xl = xm - p1;
    let xr = xm + p1;
    let c = 0.134 + 20.5 / (15.3 + m as f64);
    let a = (fm - xl) / (fm - xl * r);
    let laml = a * (1.0 + a / 2.0);
    let a = (xr - fm) / (xr * q);
    let lamr = a * (1.0 + a / 2.0);
    let p2 = p1 * (1.0 + 2.0 * c);
    let p3 = p2 + c / laml;
    let p4 = p3 + c / lamr;
    let y = loop {
        // Step 10.
        let nrq = n as f64 * r * q;
        let u = stream.next_double()? * p4;
        let mut v = stream.next_double()?;
        if u <= p1 {
            break (xm - p1 * v + u).floor() as i64;
        }
        let y = if u <= p2 {
            // Step 20.
            let x = xl + (u - p1) / c;
            v = v * c + 1.0 - (m as f64 - x + 0.5).abs() / p1;
            if v > 1.0 {
                continue;
            }
            x.floor() as i64
        } else if u <= p3 {
            // Step 30.
            let y = (xl + v.ln() / laml).floor() as i64;
            if y < 0 || v == 0.0 {
                continue;
            }
            v = v * (u - p2) * laml;
            y
        } else {
            // Step 40.
            let y = (xr - v.ln() / lamr).floor() as i64;
            if y > n || v == 0.0 {
                continue;
            }
            v = v * (u - p3) * lamr;
            y
        };
        // Step 50.
        let k = (y - m).abs();
        if !(k > 20 && (k as f64) < nrq / 2.0 - 1.0) {
            let s = r / q;
            let a = s * (n + 1) as f64;
            let mut f = 1.0;
            if m < y {
                for i in m + 1..=y {
                    f *= a / i as f64 - s;
                }
            } else if m > y {
                for i in y + 1..=m {
                    f /= a / i as f64 - s;
                }
            }
            stream.spend(k.unsigned_abs())?;
            if v > f {
                continue;
            }
            break y;
        }
        // Step 52.
        let kf = k as f64;
        let rho = (kf / nrq) * ((kf * (kf / 3.0 + 0.625) + 0.16666666666666666) / nrq + 0.5);
        // C computes -k * k in 64-bit integers.
        let t = (-k).wrapping_mul(k) as f64 / (2.0 * nrq);
        let big_a = v.ln();
        if big_a < t - rho {
            break y;
        }
        if big_a > t + rho {
            continue;
        }
        let x1 = y as f64 + 1.0;
        let f1 = m as f64 + 1.0;
        let z = n as f64 + 1.0 - m as f64;
        let w = n as f64 - y as f64 + 1.0;
        let (x2, f2, z2, w2) = (x1 * x1, f1 * f1, z * z, w * w);
        let stirling = |c0: f64, square: f64, base: f64| {
            (c0 - (462.0 - (132.0 - (99.0 - 140.0 / square) / square) / square) / square)
                / base
                / 166320.0
        };
        let head = xm * (f1 / x1).ln()
            + ((n - m) as f64 + 0.5) * (z / w).ln()
            + (y - m) as f64 * (w * r / (x1 * q)).ln();
        let bound = match family {
            Family::Generator => {
                head + stirling(13860.0, f2, f1) + stirling(13860.0, z2, z)
                    - stirling(13860.0, x2, x1)
                    - stirling(13860.0, w2, w)
            }
            Family::Legacy => {
                head + stirling(13680.0, f2, f1)
                    + stirling(13680.0, z2, z)
                    + stirling(13680.0, x2, x1)
                    + stirling(13680.0, w2, w)
            }
        };
        if big_a > bound {
            continue;
        }
        break y;
    };
    // Step 60. Callers pass p <= 0.5, so this never flips, as in NumPy.
    Ok(if p > 0.5 { n - y } else { y })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loggam_matches_known_values() {
        assert_eq!(loggam(1.0), 0.0);
        assert!((loggam(5.0) - 24f64.ln()).abs() < 1e-14);
        assert!((loggam(30.5) - 72.9534711841694).abs() < 1e-12);
    }
}
