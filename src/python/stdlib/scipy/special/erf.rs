//! The error function family and the standard normal distribution: `erf`, `erfc`, `erfcinv`,
//! `ndtr`, `ndtri`, and `log_ndtr`.
//!
//! `erf`, `erfc`, `ndtr`, `ndtri` and `erfcinv` are ports of Cephes
//! (`xsf/cephes/{ndtr,ndtri,erfinv}.h`), as SciPy uses. SciPy computes `erfinv` with Boost, in
//! `erf_inv.rs`, and `log_ndtr` with Faddeeva's `erfcx`; here `log_ndtr` uses the logarithm of
//! `ndtr` with an asymptotic series in the far tail, which agrees with SciPy to within a few
//! units in the last place.
#![allow(clippy::excessive_precision, clippy::unreadable_literal)]

use std::f64::consts::{FRAC_1_SQRT_2, PI};

use super::gamma::MAXLOG;
use super::poly::{p1evl, polevl};

const NDTR_P: [f64; 9] = [
    2.46196981473530512524E-10,
    5.64189564831068821977E-1,
    7.46321056442269912687E0,
    4.86371970985681366614E1,
    1.96520832956077098242E2,
    5.26445194995477358631E2,
    9.34528527171957607540E2,
    1.02755188689515710272E3,
    5.57535335369399327526E2,
];
const NDTR_Q: [f64; 8] = [
    1.32281951154744992508E1,
    8.67072140885989742329E1,
    3.54937778887819891062E2,
    9.75708501743205489753E2,
    1.82390916687909736289E3,
    2.24633760818710981792E3,
    1.65666309194161350182E3,
    5.57535340817727675546E2,
];
const NDTR_R: [f64; 6] = [
    5.64189583547755073984E-1,
    1.27536670759978104416E0,
    5.01905042251180477414E0,
    6.16021097993053585195E0,
    7.40974269950448939160E0,
    2.97886665372100240670E0,
];
const NDTR_S: [f64; 6] = [
    2.26052863220117276590E0,
    9.39603524938001434673E0,
    1.20489539808096656605E1,
    1.70814450747565897222E1,
    9.60896809063285878198E0,
    3.36907645100081516050E0,
];
const NDTR_T: [f64; 5] = [
    9.60497373987051638749E0,
    9.00260197203842689217E1,
    2.23200534594684319226E3,
    7.00332514112805075473E3,
    5.55923013010394962768E4,
];
const NDTR_U: [f64; 5] = [
    3.35617141647503099647E1,
    5.21357949780152679795E2,
    4.59432382970980127987E3,
    2.26290000613890934246E4,
    4.92673942608635921086E4,
];

/// The complementary error function.
pub(super) fn erfc(a: f64) -> f64 {
    if a.is_nan() {
        return f64::NAN;
    }
    let x = a.abs();
    if x < 1.0 {
        return 1.0 - erf(a);
    }
    let z = -a * a;
    if z >= -MAXLOG {
        let z = z.exp();
        let (p, q) = if x < 8.0 {
            (polevl(x, &NDTR_P), p1evl(x, &NDTR_Q))
        } else {
            (polevl(x, &NDTR_R), p1evl(x, &NDTR_S))
        };
        let mut y = (z * p) / q;
        if a < 0.0 {
            y = 2.0 - y;
        }
        if y != 0.0 {
            return y;
        }
    }
    // Underflow.
    if a < 0.0 {
        2.0
    } else {
        0.0
    }
}

/// The error function.
pub(super) fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return -erf(-x);
    }
    if x.abs() > 1.0 {
        return 1.0 - erfc(x);
    }
    let z = x * x;
    x * polevl(z, &NDTR_T) / p1evl(z, &NDTR_U)
}

/// The standard normal cumulative distribution function.
pub(super) fn ndtr(a: f64) -> f64 {
    if a.is_nan() {
        return f64::NAN;
    }
    let x = a * FRAC_1_SQRT_2;
    let z = x.abs();
    if z < 1.0 {
        return 0.5 + 0.5 * erf(x);
    }
    let y = 0.5 * erfc(z);
    if x > 0.0 {
        1.0 - y
    } else {
        y
    }
}

/// `log(ndtr(x))`, accurate in both tails.
pub(super) fn log_ndtr(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let t = x * FRAC_1_SQRT_2;
    if x >= -1.0 {
        return (-erfc(t) / 2.0).ln_1p();
    }
    if x > -20.0 {
        return (0.5 * erfc(-t)).ln();
    }
    if x == f64::NEG_INFINITY {
        return x;
    }
    // Mills' ratio: ndtr(x) = phi(x) / |x| * sum((-1)^k (2k - 1)!! / x^(2k)).
    let inverse_square = 1.0 / (x * x);
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..20 {
        term *= -f64::from(2 * k - 1) * inverse_square;
        sum += term;
        if term.abs() < f64::EPSILON * sum.abs() {
            break;
        }
    }
    -0.5 * x * x - (-x).ln() - 0.5 * (2.0 * PI).ln() + sum.ln()
}

const NDTRI_P0: [f64; 5] = [
    -5.99633501014107895267E1,
    9.80010754185999661536E1,
    -5.66762857469070293439E1,
    1.39312609387279679503E1,
    -1.23916583867381258016E0,
];
const NDTRI_Q0: [f64; 8] = [
    1.95448858338141759834E0,
    4.67627912898881538453E0,
    8.63602421390890590575E1,
    -2.25462687854119370527E2,
    2.00260212380060660359E2,
    -8.20372256168333339912E1,
    1.59056225126211695515E1,
    -1.18331621121330003142E0,
];
const NDTRI_P1: [f64; 9] = [
    4.05544892305962419923E0,
    3.15251094599893866154E1,
    5.71628192246421288162E1,
    4.40805073893200834700E1,
    1.46849561928858024014E1,
    2.18663306850790267539E0,
    -1.40256079171354495875E-1,
    -3.50424626827848203418E-2,
    -8.57456785154685413611E-4,
];
const NDTRI_Q1: [f64; 8] = [
    1.57799883256466749731E1,
    4.53907635128879210584E1,
    4.13172038254672030440E1,
    1.50425385692907503408E1,
    2.50464946208309415979E0,
    -1.42182922854787788574E-1,
    -3.80806407691578277194E-2,
    -9.33259480895457427372E-4,
];
const NDTRI_P2: [f64; 9] = [
    3.23774891776946035970E0,
    6.91522889068984211695E0,
    3.93881025292474443415E0,
    1.33303460815807542389E0,
    2.01485389549179081538E-1,
    1.23716634817820021358E-2,
    3.01581553508235416007E-4,
    2.65806974686737550832E-6,
    6.23974539184983293730E-9,
];
const NDTRI_Q2: [f64; 8] = [
    6.02427039364742014255E0,
    3.67983563856160859403E0,
    1.37702099489081330271E0,
    2.16236993594496635890E-1,
    1.34204006088543189037E-2,
    3.28014464682127739104E-4,
    2.89247864745380683936E-6,
    6.79019408009981274425E-9,
];
const SQRT2PI: f64 = 2.506628274631000502415765284811045253007;
const EXP_MINUS_2: f64 = 0.13533528323661269189;

/// The inverse of `ndtr`.
pub(super) fn ndtri(y0: f64) -> f64 {
    if y0 == 0.0 {
        return f64::NEG_INFINITY;
    }
    if y0 == 1.0 {
        return f64::INFINITY;
    }
    if !(0.0..=1.0).contains(&y0) {
        return f64::NAN;
    }
    let mut negate = true;
    let mut y = y0;
    if y > 1.0 - EXP_MINUS_2 {
        y = 1.0 - y;
        negate = false;
    }
    if y > EXP_MINUS_2 {
        let y = y - 0.5;
        let y2 = y * y;
        let x = y + y * (y2 * polevl(y2, &NDTRI_P0) / p1evl(y2, &NDTRI_Q0));
        return x * SQRT2PI;
    }
    let x = (-2.0 * y.ln()).sqrt();
    let x0 = x - x.ln() / x;
    let z = 1.0 / x;
    let x1 = if x < 8.0 {
        z * polevl(z, &NDTRI_P1) / p1evl(z, &NDTRI_Q1)
    } else {
        z * polevl(z, &NDTRI_P2) / p1evl(z, &NDTRI_Q2)
    };
    let x = x0 - x1;
    if negate {
        -x
    } else {
        x
    }
}

/// The inverse complementary error function on `[0, 2]`, as SciPy's `erfcinv` computes it from
/// Cephes' `ndtri` (`xsf/cephes/erfinv.h`).
pub(super) fn erfcinv(y: f64) -> f64 {
    if y > 0.0 && y < 2.0 {
        return -ndtri(0.5 * y) * FRAC_1_SQRT_2;
    }
    if y == 0.0 {
        return f64::INFINITY;
    }
    if y == 2.0 {
        return f64::NEG_INFINITY;
    }
    f64::NAN
}
