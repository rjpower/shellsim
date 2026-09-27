//! The gamma function and its relatives: log-gamma, reciprocal gamma, beta, digamma, the
//! Pochhammer symbol, and the Riemann and Hurwitz zeta functions.
//!
//! These are ports of the Cephes routines as SciPy's xsf library carries them
//! (`xsf/cephes/{gamma,rgamma,beta,psi,poch,zeta,zetac,lanczos}.h`, `xsf/digamma.h`), so results
//! match SciPy's to the last bit or two. Errors follow SciPy's defaults: domain errors and poles
//! return NaN or infinity without warning.
//!
//! Numeric constants are copied digit for digit from the C sources, so some carry more
//! precision than an `f64` holds.
#![allow(clippy::excessive_precision, clippy::unreadable_literal)]

use std::f64::consts::{E, PI};

use super::meter;
use super::poly::{chbevl, p1evl, polevl, ratevl};

pub(super) const MACHEP: f64 = 1.1102230246251565404236316680908203125E-16;
pub(super) const MAXLOG: f64 = 7.097827128933839730962063185871E2;
pub(super) const MAXGAM: f64 = 171.624376956302725;
const SQRT2PI: f64 = 2.506628274631000502415765284811045253007;
const SQRT2OPI: f64 = 7.978845608028653558798921198687637369517E-1;
const LOGPI: f64 = 1.144729885849400174143427351353058711647;
const EULER: f64 = 0.577215664901532860606512090082402431;

/// `sin(pi x)`, exact at integers.
pub(super) fn sinpi(x: f64) -> f64 {
    let (x, sign) = if x.is_sign_negative() {
        (-x, -1.0)
    } else {
        (x, 1.0)
    };
    let r = x % 2.0;
    if r < 0.5 {
        sign * (PI * r).sin()
    } else if r > 1.5 {
        sign * (PI * (r - 2.0)).sin()
    } else {
        -sign * (PI * (r - 1.0)).sin()
    }
}

const GAMMA_P: [f64; 7] = [
    1.60119522476751861407E-4,
    1.19135147006586384913E-3,
    1.04213797561761569935E-2,
    4.76367800457137231464E-2,
    2.07448227648435975150E-1,
    4.94214826801497100753E-1,
    9.99999999999999996796E-1,
];
const GAMMA_Q: [f64; 8] = [
    -2.31581873324120129819E-5,
    5.39605580493303397842E-4,
    -4.45641913851797240494E-3,
    1.18139785222060435552E-2,
    3.58236398605498653373E-2,
    -2.34591795718243348568E-1,
    7.14304917030273074085E-2,
    1.00000000000000000320E0,
];
const GAMMA_STIR: [f64; 5] = [
    7.87311395793093628397E-4,
    -2.29549961613378126380E-4,
    -2.68132617805781232825E-3,
    3.47222221605458667310E-3,
    8.33333333333482257126E-2,
];
const MAXSTIR: f64 = 143.01608;

/// Gamma by Stirling's formula, valid for `33 <= x <= 172`.
fn stirf(x: f64) -> f64 {
    if x >= MAXGAM {
        return f64::INFINITY;
    }
    let w = 1.0 / x;
    let w = 1.0 + w * polevl(w, &GAMMA_STIR);
    let mut y = x.exp();
    if x > MAXSTIR {
        // Avoid overflow in pow().
        let v = x.powf(0.5 * x - 0.25);
        y = v * (v / y);
    } else {
        y = x.powf(x - 0.5) / y;
    }
    SQRT2PI * y * w
}

/// The gamma function. Poles at nonpositive integers give NaN, except that zero gives an
/// infinity with the sign of the zero.
pub(super) fn gamma(x: f64) -> f64 {
    if !x.is_finite() {
        return if x > 0.0 { x } else { f64::NAN };
    }
    if x == 0.0 {
        return f64::INFINITY.copysign(x);
    }
    let q = x.abs();
    if q > 33.0 {
        if x >= 0.0 {
            return stirf(x);
        }
        let mut p = q.floor();
        if p == q {
            return f64::NAN;
        }
        let sign = if (p as i64) & 1 == 0 { -1.0 } else { 1.0 };
        let mut z = q - p;
        if z > 0.5 {
            p += 1.0;
            z = q - p;
        }
        z = q * sinpi(z);
        if z == 0.0 {
            return sign * f64::INFINITY;
        }
        return sign * (PI / (z.abs() * stirf(q)));
    }
    let mut x = x;
    let mut z = 1.0;
    while x >= 3.0 {
        x -= 1.0;
        z *= x;
    }
    while x < 0.0 {
        if x > -1e-9 {
            return small_gamma(x, z);
        }
        z /= x;
        x += 1.0;
    }
    while x < 2.0 {
        if x < 1e-9 {
            return small_gamma(x, z);
        }
        z /= x;
        x += 1.0;
    }
    if x == 2.0 {
        return z;
    }
    x -= 2.0;
    z * polevl(x, &GAMMA_P) / polevl(x, &GAMMA_Q)
}

fn small_gamma(x: f64, z: f64) -> f64 {
    if x == 0.0 {
        // x started as a negative integer.
        f64::NAN
    } else {
        z / ((1.0 + 0.5772156649015329 * x) * x)
    }
}

const GAMMA_A: [f64; 5] = [
    8.11614167470508450300E-4,
    -5.95061904284301438324E-4,
    7.93650340457716943945E-4,
    -2.77777777730099687205E-3,
    8.33333333333331927722E-2,
];
const GAMMA_B: [f64; 6] = [
    -1.37825152569120859100E3,
    -3.88016315134637840924E4,
    -3.31612992738871184744E5,
    -1.16237097492762307383E6,
    -1.72173700820839662146E6,
    -8.53555664245765465627E5,
];
const GAMMA_C: [f64; 6] = [
    -3.51815701436523470549E2,
    -1.70642106651881159223E4,
    -2.20528590553854454839E5,
    -1.13933444367982507207E6,
    -2.53252307177582951285E6,
    -2.01889141433532773231E6,
];
const LS2PI: f64 = 0.91893853320467274178;
const MAXLGM: f64 = 2.556_348e305;

fn lgam_large_x(x: f64) -> f64 {
    let q = (x - 0.5) * x.ln() - x + LS2PI;
    if x > 1.0e8 {
        return q;
    }
    let p = 1.0 / (x * x);
    let p = ((7.9365079365079365079365e-4 * p - 2.7777777777777777777778e-3) * p
        + 0.0833333333333333333333)
        / x;
    q + p
}

/// `log|gamma(x)|` and the sign of `gamma(x)`. Poles give `+inf`.
pub(super) fn lgam_sgn(x: f64) -> (f64, f64) {
    if !x.is_finite() {
        return (x, 1.0);
    }
    if x < -34.0 {
        let q = -x;
        let (w, _) = lgam_sgn(q);
        let mut p = q.floor();
        if p == q {
            return (f64::INFINITY, 1.0);
        }
        let sign = if (p as i64) & 1 == 0 { -1.0 } else { 1.0 };
        let mut z = q - p;
        if z > 0.5 {
            p += 1.0;
            z = p - q;
        }
        z = q * sinpi(z);
        if z == 0.0 {
            return (f64::INFINITY, sign);
        }
        return (LOGPI - z.ln() - w, sign);
    }
    if x < 13.0 {
        let mut z = 1.0;
        let mut p = 0.0;
        let mut u = x;
        while u >= 3.0 {
            p -= 1.0;
            u = x + p;
            z *= u;
        }
        while u < 2.0 {
            if u == 0.0 {
                return (f64::INFINITY, 1.0);
            }
            z /= u;
            p += 1.0;
            u = x + p;
        }
        let sign = if z < 0.0 {
            z = -z;
            -1.0
        } else {
            1.0
        };
        if u == 2.0 {
            return (z.ln(), sign);
        }
        p -= 2.0;
        let x = x + p;
        let p = x * polevl(x, &GAMMA_B) / p1evl(x, &GAMMA_C);
        return (z.ln() + p, sign);
    }
    if x > MAXLGM {
        return (f64::INFINITY, 1.0);
    }
    if x >= 1000.0 {
        return (lgam_large_x(x), 1.0);
    }
    let q = (x - 0.5) * x.ln() - x + LS2PI;
    let p = 1.0 / (x * x);
    (q + polevl(p, &GAMMA_A) / x, 1.0)
}

/// `log|gamma(x)|`, SciPy's `gammaln`.
pub(super) fn lgam(x: f64) -> f64 {
    lgam_sgn(x).0
}

/// The sign of `gamma(x)`: NaN at poles and `-inf`.
pub(super) fn gammasgn(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x > 0.0 {
        return 1.0;
    }
    if x == 0.0 {
        return 1.0_f64.copysign(x);
    }
    if x.is_infinite() {
        return f64::NAN;
    }
    let fx = x.floor();
    if x - fx == 0.0 {
        return f64::NAN;
    }
    if (fx as i64) % 2 != 0 {
        -1.0
    } else {
        1.0
    }
}

/// SciPy's real `loggamma`: `log(gamma(x))`, NaN where gamma is negative.
pub(super) fn loggamma(x: f64) -> f64 {
    if x < 0.0 {
        return f64::NAN;
    }
    lgam(x)
}

const RGAMMA_R: [f64; 16] = [
    3.13173458231230000000E-17,
    -6.70718606477908000000E-16,
    2.20039078172259550000E-15,
    2.47691630348254132600E-13,
    -6.60074100411295197440E-12,
    5.13850186324226978840E-11,
    1.08965386454418662084E-9,
    -3.33964630686836942556E-8,
    2.68975996440595483619E-7,
    2.96001177518801696639E-6,
    -8.04814124978471142852E-5,
    4.16609138709688864714E-4,
    5.06579864028608725080E-3,
    -6.41925436109158228810E-2,
    -4.98558728684003594785E-3,
    1.27546015610523951063E-1,
];

/// `1 / gamma(x)`, zero at the poles.
pub(super) fn rgamma(x: f64) -> f64 {
    if x == 0.0 {
        return x;
    }
    if x < 0.0 && x == x.floor() {
        return 0.0;
    }
    if x.abs() > 4.0 {
        return 1.0 / gamma(x);
    }
    let mut z = 1.0;
    let mut w = x;
    while w > 1.0 {
        w -= 1.0;
        z *= w;
    }
    while w < 0.0 {
        z /= w;
        w += 1.0;
    }
    if w == 0.0 {
        return 0.0;
    }
    if w == 1.0 {
        return 1.0 / z;
    }
    w * (1.0 + chbevl(4.0 * w - 2.0, &RGAMMA_R)) / z
}

const LANCZOS_SUM_EXPG_SCALED_NUM: [f64; 13] = [
    0.006061842346248906525783753964555936883222,
    0.5098416655656676188125178644804694509993,
    19.51992788247617482847860966235652136208,
    449.9445569063168119446858607650988409623,
    6955.999602515376140356310115515198987526,
    75999.29304014542649875303443598909137092,
    601859.6171681098786670226533699352302507,
    3481712.15498064590882071018964774556468,
    14605578.08768506808414169982791359218571,
    43338889.32467613834773723740590533316085,
    86363131.28813859145546927288977868422342,
    103794043.1163445451906271053616070238554,
    56906521.91347156388090791033559122686859,
];
const LANCZOS_SUM_EXPG_SCALED_DENOM: [f64; 13] = [
    1.0,
    66.0,
    1925.0,
    32670.0,
    357423.0,
    2637558.0,
    13339535.0,
    45995730.0,
    105258076.0,
    150917976.0,
    120543840.0,
    39916800.0,
    0.0,
];
pub(super) const LANCZOS_G: f64 = 6.024680040776729583740234375;

pub(super) fn lanczos_sum_expg_scaled(x: f64) -> f64 {
    ratevl(
        x,
        &LANCZOS_SUM_EXPG_SCALED_NUM,
        &LANCZOS_SUM_EXPG_SCALED_DENOM,
    )
}

const BETA_ASYMP_FACTOR: f64 = 1e6;

/// `ln|B(a, b)|` for `a > ASYMP_FACTOR * max(|b|, 1)`, with the sign of `B`.
fn lbeta_asymp(a: f64, b: f64) -> (f64, f64) {
    let (mut r, sign) = lgam_sgn(b);
    r -= b * a.ln();
    r += b * (1.0 - b) / (2.0 * a);
    r += b * (1.0 - b) * (1.0 - 2.0 * b) / (12.0 * a * a);
    r += -b * b * (1.0 - b) * (1.0 - b) / (12.0 * a * a * a);
    (r, sign)
}

/// A negative integer `a` as C's `int` conversion sees it, or `None` when it does not fit.
fn as_int(a: f64) -> Option<i64> {
    (a == a.trunc() && a.abs() < i32::MAX as f64).then_some(a as i64)
}

fn beta_negint(a: i64, b: f64) -> f64 {
    if b == b.trunc() && b.abs() < i32::MAX as f64 && 1.0 - a as f64 - b > 0.0 {
        let sign = if (b as i64) % 2 == 0 { 1.0 } else { -1.0 };
        sign * beta(1.0 - a as f64 - b, b)
    } else {
        f64::INFINITY
    }
}

fn lbeta_negint(a: i64, b: f64) -> f64 {
    if b == b.trunc() && b.abs() < i32::MAX as f64 && 1.0 - a as f64 - b > 0.0 {
        lbeta(1.0 - a as f64 - b, b)
    } else {
        f64::INFINITY
    }
}

/// Where `beta` and `lbeta` handle a nonpositive integer argument specially.
enum Negint {
    Value(f64),
    Overflow,
}

fn negint_case(a: f64, b: f64, log: bool) -> Option<Negint> {
    for (first, second) in [(a, b), (b, a)] {
        if first <= 0.0 && first == first.floor() {
            return Some(match as_int(first) {
                Some(n) if log => Negint::Value(lbeta_negint(n, second)),
                Some(n) => Negint::Value(beta_negint(n, second)),
                None => Negint::Overflow,
            });
        }
    }
    None
}

/// The beta function `B(a, b) = gamma(a) gamma(b) / gamma(a + b)`.
pub(super) fn beta(a: f64, b: f64) -> f64 {
    match negint_case(a, b, false) {
        Some(Negint::Value(value)) => return value,
        Some(Negint::Overflow) => return f64::INFINITY,
        None => {}
    }
    let (a, b) = if a.abs() < b.abs() { (b, a) } else { (a, b) };
    if a.abs() > BETA_ASYMP_FACTOR * b.abs() && a > BETA_ASYMP_FACTOR {
        // Avoid loss of precision in lgam(a + b) - lgam(a).
        let (y, sign) = lbeta_asymp(a, b);
        return sign * y.exp();
    }
    let y = a + b;
    if y.abs() > MAXGAM || a.abs() > MAXGAM || b.abs() > MAXGAM {
        let (y, s1) = lgam_sgn(y);
        let (lb, s2) = lgam_sgn(b);
        let y = lb - y;
        let (la, s3) = lgam_sgn(a);
        let y = la + y;
        let sign = s1 * s2 * s3;
        if y > MAXLOG {
            return sign * f64::INFINITY;
        }
        return sign * y.exp();
    }
    let y = rgamma(y);
    let a = gamma(a);
    let b = gamma(b);
    if y.is_infinite() {
        return f64::INFINITY;
    }
    if ((a * y).abs() - 1.0).abs() > ((b * y).abs() - 1.0).abs() {
        b * y * a
    } else {
        a * y * b
    }
}

/// `ln|B(a, b)|`, SciPy's `betaln`.
pub(super) fn lbeta(a: f64, b: f64) -> f64 {
    match negint_case(a, b, true) {
        Some(Negint::Value(value)) => return value,
        Some(Negint::Overflow) => return f64::INFINITY,
        None => {}
    }
    let (a, b) = if a.abs() < b.abs() { (b, a) } else { (a, b) };
    if a.abs() > BETA_ASYMP_FACTOR * b.abs() && a > BETA_ASYMP_FACTOR {
        return lbeta_asymp(a, b).0;
    }
    let y = a + b;
    if y.abs() > MAXGAM || a.abs() > MAXGAM || b.abs() > MAXGAM {
        let y = lgam_sgn(y).0;
        let y = lgam_sgn(b).0 - y;
        return lgam_sgn(a).0 + y;
    }
    let y = rgamma(y);
    let a = gamma(a);
    let b = gamma(b);
    if y.is_infinite() {
        return f64::INFINITY;
    }
    let y = if ((a * y).abs() - 1.0).abs() > ((b * y).abs() - 1.0).abs() {
        b * y * a
    } else {
        a * y * b
    };
    y.abs().ln()
}

const PSI_A: [f64; 7] = [
    8.33333333333333333333E-2,
    -2.10927960927960927961E-2,
    7.57575757575757575758E-3,
    -4.16666666666666666667E-3,
    3.96825396825396825397E-3,
    -8.33333333333333333333E-3,
    8.33333333333333333333E-2,
];
const PSI_Y: f32 = 0.99558162689208984;
const PSI_ROOT1: f64 = 1569415565.0 / 1073741824.0;
const PSI_ROOT2: f64 = (381566830.0 / 1073741824.0) / 1073741824.0;
const PSI_ROOT3: f64 = 0.9016312093258695918615325266959189453125e-19;
const PSI_P: [f64; 6] = [
    -0.0020713321167745952,
    -0.045251321448739056,
    -0.28919126444774784,
    -0.65031853770896507,
    -0.32555031186804491,
    0.25479851061131551,
];
const PSI_Q: [f64; 7] = [
    -0.55789841321675513e-6,
    0.0021284987017821144,
    0.054151797245674225,
    0.43593529692665969,
    1.4606242909763515,
    2.0767117023730469,
    1.0,
];

/// Digamma on `[1, 2]`: a rational approximation from Boost around the positive root.
fn digamma_imp_1_2(x: f64) -> f64 {
    let mut g = x - PSI_ROOT1;
    g -= PSI_ROOT2;
    g -= PSI_ROOT3;
    let r = polevl(x - 1.0, &PSI_P) / polevl(x - 1.0, &PSI_Q);
    g * f64::from(PSI_Y) + g * r
}

fn psi_asy(x: f64) -> f64 {
    let y = if x < 1.0e17 {
        let z = 1.0 / (x * x);
        z * polevl(z, &PSI_A)
    } else {
        0.0
    };
    x.ln() - (0.5 / x) - y
}

/// Cephes' `psi`: the digamma function.
fn psi(x: f64) -> f64 {
    let mut y = 0.0;
    let mut x = x;
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == f64::NEG_INFINITY {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY.copysign(-x);
    }
    if x < 0.0 {
        // Argument reduction before evaluating tan(pi x).
        let r = x - x.trunc();
        if r == 0.0 {
            return f64::NAN;
        }
        y = -PI / (PI * r).tan();
        x = 1.0 - x;
    }
    // Positive integers up to 10.
    if x <= 10.0 && x == x.floor() {
        let n = x as i64;
        for i in 1..n {
            y += 1.0 / i as f64;
        }
        return y - EULER;
    }
    // Use the recurrence to move x into [1, 2].
    if x < 1.0 {
        y -= 1.0 / x;
        x += 1.0;
    } else if x < 10.0 {
        while x > 2.0 {
            x -= 1.0;
            y += 1.0 / x;
        }
    }
    if (1.0..=2.0).contains(&x) {
        return y + digamma_imp_1_2(x);
    }
    y + psi_asy(x)
}

const DIGAMMA_NEGROOT: f64 = -0.504083008264455409;
const DIGAMMA_NEGROOTVAL: f64 = 7.2897639029768949e-17;

/// SciPy's digamma: Cephes' `psi`, with a Taylor series around the negative root nearest zero.
pub(super) fn digamma(z: f64) -> f64 {
    if (z - DIGAMMA_NEGROOT).abs() < 0.3 {
        let mut result = DIGAMMA_NEGROOTVAL;
        let mut coefficient = -1.0;
        let shifted = z - DIGAMMA_NEGROOT;
        for n in 1..100 {
            coefficient *= -shifted;
            let term = coefficient * zeta(f64::from(n + 1), DIGAMMA_NEGROOT);
            result += term;
            if term.abs() < f64::EPSILON * result.abs() {
                break;
            }
        }
        return result;
    }
    psi(z)
}

fn is_nonpos_int(x: f64) -> bool {
    x <= 0.0 && x == x.ceil() && x.abs() < 1e13
}

/// The Pochhammer symbol `(a)_m = gamma(a + m) / gamma(a)`.
pub(super) fn poch(a: f64, m: f64) -> f64 {
    let mut r = 1.0;
    let mut m = m;
    // Reduce |m| below 1 with the recurrences.
    while m >= 1.0 && meter::step() {
        if a + m == 1.0 {
            break;
        }
        m -= 1.0;
        r *= a + m;
        if !r.is_finite() || r == 0.0 {
            break;
        }
    }
    while m <= -1.0 && meter::step() {
        if a + m == 0.0 {
            break;
        }
        r /= a + m;
        m += 1.0;
        if !r.is_finite() || r == 0.0 {
            break;
        }
    }
    if m == 0.0 {
        return r;
    }
    if a > 1e4 && m.abs() <= 1.0 {
        // Avoid loss of precision.
        return r
            * a.powf(m)
            * (1.0
                + m * (m - 1.0) / (2.0 * a)
                + m * (m - 1.0) * (m - 2.0) * (3.0 * m - 1.0) / (24.0 * a * a)
                + m * m * (m - 1.0) * (m - 1.0) * (m - 2.0) * (m - 3.0) / (48.0 * a * a * a));
    }
    if is_nonpos_int(a + m) && !is_nonpos_int(a) && a + m != m {
        return f64::INFINITY;
    }
    if !is_nonpos_int(a + m) && is_nonpos_int(a) {
        return 0.0;
    }
    r * (lgam(a + m) - lgam(a)).exp() * gammasgn(a + m) * gammasgn(a)
}

/// `(2k)! / B_2k` for the Euler-Maclaurin summation.
const ZETA_A: [f64; 12] = [
    12.0,
    -720.0,
    30240.0,
    -1209600.0,
    47900160.0,
    -1.8924375803183791606e9,
    7.47242496e10,
    -2.950130727918164224e12,
    1.1646782814350067249e14,
    -4.5979787224074726105e15,
    1.8152105401943546773e17,
    -7.1661652561756670113e18,
];

/// The Hurwitz zeta function `sum((k + q)^-x, k >= 0)` for `x > 1`.
pub(super) fn zeta(x: f64, q: f64) -> f64 {
    if x == 1.0 {
        return f64::INFINITY;
    }
    if x < 1.0 {
        return f64::NAN;
    }
    if q <= 0.0 {
        if q == q.floor() {
            return f64::INFINITY;
        }
        if x != x.floor() {
            // q^-x is not defined.
            return f64::NAN;
        }
    }
    if q > 1e8 {
        // Asymptotic expansion, https://dlmf.nist.gov/25.11#E43.
        return (1.0 / (x - 1.0) + 1.0 / (2.0 * q)) * q.powf(1.0 - x);
    }
    // Euler-Maclaurin summation.
    let mut s = q.powf(-x);
    let mut a = q;
    let mut i = 0;
    let mut b = 0.0;
    // For negative q the sum runs about |q| terms, as in Cephes.
    while (i < 9 || a <= 9.0) && meter::step() {
        i += 1;
        a += 1.0;
        b = a.powf(-x);
        s += b;
        if (b / s).abs() < MACHEP {
            return s;
        }
    }
    let w = a;
    s += b * w / (x - 1.0);
    s -= 0.5 * b;
    let mut a = 1.0;
    let mut k = 0.0;
    for coefficient in ZETA_A {
        a *= x + k;
        b /= w;
        let t = a * b / coefficient;
        s += t;
        if (t / s).abs() < MACHEP {
            return s;
        }
        k += 1.0;
        a *= x + k;
        b /= w;
        k += 1.0;
    }
    s
}

/// `zeta(n) - 1` for integers `0 <= n <= 30`; index 1 is unused.
const AZETAC: [f64; 31] = [
    -1.50000000000000000000E0,
    0.0,
    6.44934066848226436472E-1,
    2.02056903159594285400E-1,
    8.23232337111381915160E-2,
    3.69277551433699263314E-2,
    1.73430619844491397145E-2,
    8.34927738192282683980E-3,
    4.07735619794433937869E-3,
    2.00839282608221441785E-3,
    9.94575127818085337146E-4,
    4.94188604119464558702E-4,
    2.46086553308048298638E-4,
    1.22713347578489146752E-4,
    6.12481350587048292585E-5,
    3.05882363070204935517E-5,
    1.52822594086518717326E-5,
    7.63719763789976227360E-6,
    3.81729326499983985646E-6,
    1.90821271655393892566E-6,
    9.53962033872796113152E-7,
    4.76932986787806463117E-7,
    2.38450502727732990004E-7,
    1.19219925965311073068E-7,
    5.96081890512594796124E-8,
    2.98035035146522801861E-8,
    1.49015548283650412347E-8,
    7.45071178983542949198E-9,
    3.72533402478845705482E-9,
    1.86265972351304900640E-9,
    9.31327432419668182872E-10,
];
const ZETAC_P: [f64; 9] = [
    5.85746514569725319540E11,
    2.57534127756102572888E11,
    4.87781159567948256438E10,
    5.15399538023885770696E9,
    3.41646073514754094281E8,
    1.60837006880656492731E7,
    5.92785467342109522998E5,
    1.51129169964938823117E4,
    2.01822444485997955865E2,
];
const ZETAC_Q: [f64; 8] = [
    3.90497676373371157516E11,
    5.22858235368272161797E10,
    5.64451517271280543351E9,
    3.39006746015350418834E8,
    1.79410371500126453702E7,
    5.66666825131384797029E5,
    1.60382976810944131506E4,
    1.96436237223387314144E2,
];
const ZETAC_A: [f64; 11] = [
    8.70728567484590192539E6,
    1.76506865670346462757E8,
    2.60889506707483264896E10,
    5.29806374009894791647E11,
    2.26888156119238241487E13,
    3.31884402932705083599E14,
    5.13778997975868230192E15,
    -1.98123688133907171455E15,
    -9.92763810039983572356E16,
    7.82905376180870586444E16,
    9.26786275768927717187E16,
];
const ZETAC_B: [f64; 10] = [
    -7.92625410563741062861E6,
    -1.60529969932920229676E8,
    -2.37669260975543221788E10,
    -4.80319584350455169857E11,
    -2.07820961754173320170E13,
    -2.96075404507272223680E14,
    -4.86299103694609136686E15,
    5.34589509675789930199E15,
    5.71464111092297631292E16,
    -1.79915597658676556828E16,
];
const ZETAC_R: [f64; 6] = [
    -3.28717474506562731748E-1,
    1.55162528742623950834E1,
    -2.48762831680821954401E2,
    1.01050368053237678329E3,
    1.26726061410235149405E4,
    -1.11578094770515181334E5,
];
const ZETAC_S: [f64; 5] = [
    1.95107674914060531512E1,
    3.17710311750646984099E2,
    3.03835500874445748734E3,
    2.03665876435770579345E4,
    7.43853965136767874343E4,
];
const ZETAC_TAYLOR0: [f64; 10] = [
    -1.0000000009110164892,
    -1.0000000057646759799,
    -9.9999983138417361078e-1,
    -1.0000013011460139596,
    -1.000001940896320456,
    -9.9987929950057116496e-1,
    -1.000785194477042408,
    -1.0031782279542924256,
    -9.1893853320467274178e-1,
    -1.5,
];
const ZETAC_MAXL2: f64 = 127.0;

fn zetac_positive(x: f64) -> f64 {
    if x == 1.0 {
        return f64::INFINITY;
    }
    if x >= ZETAC_MAXL2 {
        // The first term is 2^-x.
        return 0.0;
    }
    let w = x.floor();
    if w == x && x < 31.0 {
        return AZETAC[x as usize];
    }
    if x < 1.0 {
        let w = 1.0 - x;
        return polevl(x, &ZETAC_R) / (w * p1evl(x, &ZETAC_S));
    }
    if x <= 10.0 {
        let b = 2.0_f64.powf(x) * (x - 1.0);
        let w = 1.0 / x;
        return (x * polevl(w, &ZETAC_P)) / (b * p1evl(w, &ZETAC_Q));
    }
    if x <= 50.0 {
        let b = 2.0_f64.powf(-x);
        let w = polevl(x, &ZETAC_A) / p1evl(x, &ZETAC_B);
        return w.exp() + b;
    }
    // Sum of inverse powers.
    let mut s = 0.0;
    let mut a: f64 = 1.0;
    loop {
        a += 2.0;
        let b = a.powf(-x);
        s += b;
        if b / s <= MACHEP {
            break;
        }
    }
    let b = 2.0_f64.powf(-x);
    (s + b) / (1.0 - b)
}

/// `zeta(-x)` for positive `x` through the reflection formula (DLMF 25.4.2), with the Lanczos
/// approximation for gamma to avoid overflow.
fn zeta_reflection(x: f64) -> f64 {
    let hx = x / 2.0;
    if hx == hx.floor() {
        // A zero of the sine factor.
        return 0.0;
    }
    let x_shift = x % 4.0;
    let mut small_term = -SQRT2OPI * (0.5 * PI * x_shift).sin();
    small_term *= lanczos_sum_expg_scaled(x + 1.0) * zeta(x + 1.0, 1.0);
    let base = (x + LANCZOS_G + 0.5) / (2.0 * PI * E);
    let large_term = base.powf(x + 0.5);
    if large_term.is_finite() {
        return large_term * small_term;
    }
    let large_term = base.powf(0.5 * x + 0.25);
    (large_term * small_term) * large_term
}

/// The Riemann zeta function.
pub(super) fn riemann_zeta(x: f64) -> f64 {
    if x.is_nan() {
        x
    } else if x == f64::NEG_INFINITY {
        f64::NAN
    } else if x < 0.0 && x > -0.01 {
        1.0 + polevl(x, &ZETAC_TAYLOR0)
    } else if x < 0.0 {
        zeta_reflection(-x)
    } else {
        1.0 + zetac_positive(x)
    }
}
