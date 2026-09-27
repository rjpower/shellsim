//! The regularized incomplete beta function `I_x(a, b)`, its complement, derivative, and
//! inverse.
//!
//! SciPy computes `betainc` and `betaincc` with Boost's `ibeta`, whose Lanczos-based power terms
//! keep full precision when `a` and `b` are large. This is a port of Boost's double-precision
//! path (`boost/math/special_functions/beta.hpp`): the power terms, the series, the continued
//! fraction, DiDonato and Morris' BGRAT series, the binomial sum for integer parameters, and the
//! erf asymptotic for very large parameters. Where Boost calls its own incomplete gamma or error
//! function, this port calls the Cephes versions, which agree to a few units in the last place.
//!
//! The inverse is in [`super::ibeta_inv`].
#![allow(clippy::excessive_precision, clippy::unreadable_literal)]

use std::f64::consts::{E, FRAC_PI_2, PI};

use super::erf::{erf, erfc};
use super::gamma::{beta, gamma, lanczos_sum_expg_scaled, lgam, LANCZOS_G};
use super::igam::igamc;
use super::meter;
use super::poly::ratevl;

const EPSILON: f64 = f64::EPSILON;
const LOG_MAX: f64 = 709.782712893384;
const LOG_MIN: f64 = -708.3964185322641;
const MAX_ITERATIONS: usize = 1_000_000;

const LANCZOS_NUM: [f64; 13] = [
    2.506628274631000270164908177133837338626,
    210.8242777515793458725097339207133627117,
    8071.672002365816210638002902272250613822,
    186056.2653952234950402949897160456992822,
    2876370.628935372441225409051620849613599,
    31426415.58540019438061423162831820536287,
    248874557.8620541565114603864132294232163,
    1439720407.311721673663223072794912393972,
    6039542586.35202800506429164430729792107,
    17921034426.03720969991975575445893111267,
    35711959237.35566804944018545154716670596,
    42919803642.64909876895789904700198885093,
    23531376880.41075968857200767445163675473,
];
const LANCZOS_DENOM: [f64; 13] = [
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

/// Boost's `lanczos13m53::lanczos_sum`.
fn lanczos_sum(z: f64) -> f64 {
    ratevl(z, &LANCZOS_NUM, &LANCZOS_DENOM)
}

/// Correctly rounded `n!` for `n <= 170`, as Boost's `unchecked_factorial` tabulates them.
const FACTORIALS: [f64; 171] = [
    1.0,
    1.0,
    2.0,
    6.0,
    24.0,
    120.0,
    720.0,
    5040.0,
    40320.0,
    362880.0,
    3628800.0,
    39916800.0,
    479001600.0,
    6227020800.0,
    87178291200.0,
    1307674368000.0,
    20922789888000.0,
    355687428096000.0,
    6402373705728000.0,
    1.21645100408832e+17,
    2.43290200817664e+18,
    5.109094217170944e+19,
    1.1240007277776077e+21,
    2.585201673888498e+22,
    6.204484017332394e+23,
    1.5511210043330986e+25,
    4.0329146112660565e+26,
    1.0888869450418352e+28,
    3.0488834461171387e+29,
    8.841761993739702e+30,
    2.6525285981219107e+32,
    8.222838654177922e+33,
    2.631308369336935e+35,
    8.683317618811886e+36,
    2.9523279903960416e+38,
    1.0333147966386145e+40,
    3.7199332678990125e+41,
    1.3763753091226346e+43,
    5.230226174666011e+44,
    2.0397882081197444e+46,
    8.159152832478977e+47,
    3.345252661316381e+49,
    1.40500611775288e+51,
    6.041526306337383e+52,
    2.658271574788449e+54,
    1.1962222086548019e+56,
    5.502622159812089e+57,
    2.5862324151116818e+59,
    1.2413915592536073e+61,
    6.082818640342675e+62,
    3.0414093201713376e+64,
    1.5511187532873822e+66,
    8.065817517094388e+67,
    4.2748832840600255e+69,
    2.308436973392414e+71,
    1.2696403353658276e+73,
    7.109985878048635e+74,
    4.0526919504877214e+76,
    2.3505613312828785e+78,
    1.3868311854568984e+80,
    8.32098711274139e+81,
    5.075802138772248e+83,
    3.146997326038794e+85,
    1.98260831540444e+87,
    1.2688693218588417e+89,
    8.247650592082472e+90,
    5.443449390774431e+92,
    3.647111091818868e+94,
    2.4800355424368305e+96,
    1.711224524281413e+98,
    1.1978571669969892e+100,
    8.504785885678623e+101,
    6.1234458376886085e+103,
    4.4701154615126844e+105,
    3.307885441519386e+107,
    2.48091408113954e+109,
    1.8854947016660504e+111,
    1.4518309202828587e+113,
    1.1324281178206297e+115,
    8.946182130782976e+116,
    7.156945704626381e+118,
    5.797126020747368e+120,
    4.753643337012842e+122,
    3.945523969720659e+124,
    3.314240134565353e+126,
    2.81710411438055e+128,
    2.4227095383672734e+130,
    2.107757298379528e+132,
    1.8548264225739844e+134,
    1.650795516090846e+136,
    1.4857159644817615e+138,
    1.352001527678403e+140,
    1.2438414054641308e+142,
    1.1567725070816416e+144,
    1.087366156656743e+146,
    1.032997848823906e+148,
    9.916779348709496e+149,
    9.619275968248212e+151,
    9.426890448883248e+153,
    9.332621544394415e+155,
    9.332621544394415e+157,
    9.42594775983836e+159,
    9.614466715035127e+161,
    9.90290071648618e+163,
    1.0299016745145628e+166,
    1.081396758240291e+168,
    1.1462805637347084e+170,
    1.226520203196138e+172,
    1.324641819451829e+174,
    1.4438595832024937e+176,
    1.588245541522743e+178,
    1.7629525510902446e+180,
    1.974506857221074e+182,
    2.2311927486598138e+184,
    2.5435597334721877e+186,
    2.925093693493016e+188,
    3.393108684451898e+190,
    3.969937160808721e+192,
    4.684525849754291e+194,
    5.574585761207606e+196,
    6.689502913449127e+198,
    8.094298525273444e+200,
    9.875044200833601e+202,
    1.214630436702533e+205,
    1.506141741511141e+207,
    1.882677176888926e+209,
    2.372173242880047e+211,
    3.0126600184576594e+213,
    3.856204823625804e+215,
    4.974504222477287e+217,
    6.466855489220474e+219,
    8.47158069087882e+221,
    1.1182486511960043e+224,
    1.4872707060906857e+226,
    1.9929427461615188e+228,
    2.6904727073180504e+230,
    3.659042881952549e+232,
    5.012888748274992e+234,
    6.917786472619489e+236,
    9.615723196941089e+238,
    1.3462012475717526e+241,
    1.898143759076171e+243,
    2.695364137888163e+245,
    3.854370717180073e+247,
    5.5502938327393044e+249,
    8.047926057471992e+251,
    1.1749972043909107e+254,
    1.727245890454639e+256,
    2.5563239178728654e+258,
    3.80892263763057e+260,
    5.713383956445855e+262,
    8.62720977423324e+264,
    1.3113358856834524e+267,
    2.0063439050956823e+269,
    3.0897696138473508e+271,
    4.789142901463394e+273,
    7.471062926282894e+275,
    1.1729568794264145e+278,
    1.853271869493735e+280,
    2.9467022724950384e+282,
    4.7147236359920616e+284,
    7.590705053947219e+286,
    1.2296942187394494e+289,
    2.0044015765453026e+291,
    3.287218585534296e+293,
    5.423910666131589e+295,
    9.003691705778438e+297,
    1.503616514864999e+300,
    2.5260757449731984e+302,
    4.269068009004705e+304,
    7.257415615307999e+306,
];

/// `(x^a)(y^b)/B(a, b)` times `prefix`, computed without the cancellation of logarithms.
fn ibeta_power_terms(a: f64, b: f64, x: f64, y: f64, prefix: f64) -> f64 {
    let c = a + b;
    let gh = LANCZOS_G - 0.5;
    let agh = a + gh;
    let bgh = b + gh;
    let cgh = c + gh;
    let mut result = if a < f64::MIN_POSITIVE || b < f64::MIN_POSITIVE {
        // The denominator overflows.
        0.0
    } else {
        lanczos_sum_expg_scaled(c) / (lanczos_sum_expg_scaled(a) * lanczos_sum_expg_scaled(b))
    };
    result *= prefix;
    result *= (bgh / E).sqrt();
    result *= (agh / cgh).sqrt();
    // The bases of the exponents minus one.
    let l1 = ((x * b - y * a) - y * gh) / agh;
    let l2 = ((y * a - x * b) - x * gh) / bgh;
    if l1.abs().min(l2.abs()) < 0.2 {
        // A base near 1 needs care.
        if l1 * l2 > 0.0 || a.min(b) < 1.0 {
            // Both powers move the same way, or one exponent is small.
            if l1.abs() < 0.1 {
                result *= (a * l1.ln_1p()).exp();
            } else {
                result *= ((x * cgh) / agh).powf(a);
            }
            if l2.abs() < 0.1 {
                result *= (b * l2.ln_1p()).exp();
            } else {
                result *= ((y * cgh) / bgh).powf(b);
            }
        } else if l1.abs().max(l2.abs()) < 0.5 {
            // Both bases are near 1 and the powers pull in opposite directions: move the larger
            // power inside the other while l3 stays small.
            let small_a = a < b;
            let ratio = b / a;
            if (small_a && ratio * l2 < 0.1) || (!small_a && l1 / ratio > 0.1) {
                let l3 = (ratio * l2.ln_1p()).exp_m1();
                let l3 = l1 + l3 + l3 * l1;
                result *= (a * l3.ln_1p()).exp();
            } else {
                let l3 = (l1.ln_1p() / ratio).exp_m1();
                let l3 = l2 + l3 + l3 * l2;
                result *= (b * l3.ln_1p()).exp();
            }
        } else if l1.abs() < l2.abs() {
            // Only the first base is near 1.
            let mut l = a * l1.ln_1p() + b * ((y * cgh) / bgh).ln();
            if l <= LOG_MIN || l >= LOG_MAX {
                l += result.ln();
                if l >= LOG_MAX {
                    return f64::INFINITY;
                }
                result = l.exp();
            } else {
                result *= l.exp();
            }
        } else {
            // Only the second base is near 1.
            let mut l = b * l2.ln_1p() + a * ((x * cgh) / agh).ln();
            if l <= LOG_MIN || l >= LOG_MAX {
                l += result.ln();
                if l >= LOG_MAX {
                    return f64::INFINITY;
                }
                result = l.exp();
            } else {
                result *= l.exp();
            }
        }
    } else {
        let b1 = (x * cgh) / agh;
        let b2 = (y * cgh) / bgh;
        let l1 = a * b1.ln();
        let mut l2 = b * b2.ln();
        if l1 >= LOG_MAX || l1 <= LOG_MIN || l2 >= LOG_MAX || l2 <= LOG_MIN {
            // Sidestep overflow and underflow where possible.
            if a < b {
                let p1 = b2.powf(b / a);
                let l3 = if b1 != 0.0 && p1 != 0.0 {
                    a * (b1.ln() + p1.ln())
                } else {
                    f64::MAX
                };
                if l3 < LOG_MAX && l3 > LOG_MIN {
                    result *= (p1 * b1).powf(a);
                } else {
                    l2 += l1 + result.ln();
                    if l2 >= LOG_MAX {
                        return f64::INFINITY;
                    }
                    result = l2.exp();
                }
            } else {
                // This protects against spurious overflow in a/b.
                let p1 = if b1 < 1.0 && b < 1.0 && f64::MAX * b < a {
                    0.0
                } else {
                    b1.powf(a / b)
                };
                let l3 = if p1 != 0.0 && b2 != 0.0 {
                    (p1.ln() + b2.ln()) * b
                } else {
                    f64::MAX
                };
                if l3 < LOG_MAX && l3 > LOG_MIN {
                    result *= (p1 * b2).powf(b);
                } else if result != 0.0 {
                    l2 += l1 + result.ln();
                    if l2 >= LOG_MAX {
                        return f64::INFINITY;
                    }
                    result = l2.exp();
                }
            }
        } else {
            result *= b1.powf(a) * b2.powf(b);
        }
    }
    result
}

/// The power series for `I_x(a, b)`, added to `s0`.
fn ibeta_series(a: f64, b: f64, x: f64, s0: f64) -> f64 {
    let c = a + b;
    let agh = a + LANCZOS_G - 0.5;
    let bgh = b + LANCZOS_G - 0.5;
    let cgh = c + LANCZOS_G - 0.5;
    let mut result = if a < f64::MIN_POSITIVE || b < f64::MIN_POSITIVE {
        0.0
    } else {
        let l1 = lanczos_sum_expg_scaled(c);
        let l2 = lanczos_sum_expg_scaled(a);
        let l3 = lanczos_sum_expg_scaled(b);
        if l2 > 1.0 && l3 > 1.0 && f64::MAX / l2 < l3 {
            (l1 / l2) / l3
        } else {
            l1 / (l2 * l3)
        }
    };
    if !result.is_finite() {
        result = 0.0;
    }
    let l1 = (cgh / bgh).ln() * (b - 0.5);
    let l2 = (x * cgh / agh).ln() * a;
    if l1 > LOG_MIN && l1 < LOG_MAX && l2 > LOG_MIN && l2 < LOG_MAX {
        if a * b < bgh * 10.0 {
            result *= ((b - 0.5) * (a / bgh).ln_1p()).exp();
        } else {
            result *= (cgh / bgh).powf(b - 0.5);
        }
        result *= (x * cgh / agh).powf(a);
        result *= (agh / E).sqrt();
    } else if result != 0.0 {
        // Logarithms, which will cancel.
        result = (result.ln() + l1 + l2 + (agh.ln() - 1.0) / 2.0).exp();
    }
    if result < f64::MIN_POSITIVE {
        // The series cannot cope with denormals.
        return s0;
    }
    let mut term_scale = result;
    let mut apn = a;
    let mut poch = 1.0 - b;
    let mut n = 1.0;
    let mut sum = s0;
    for _ in 0..MAX_ITERATIONS {
        if !meter::step() {
            break;
        }
        let term = term_scale / apn;
        apn += 1.0;
        term_scale *= poch * x / n;
        n += 1.0;
        poch += 1.0;
        sum += term;
        if (EPSILON * sum).abs() >= term.abs() {
            break;
        }
    }
    sum
}

/// `I_x(a, b)` by the continued fraction.
fn ibeta_fraction2(a: f64, b: f64, x: f64, y: f64) -> f64 {
    let power = ibeta_power_terms(a, b, x, y, 1.0);
    if power == 0.0 {
        return 0.0;
    }
    let term = |m: f64| {
        let denom = a + 2.0 * m - 1.0;
        let a_n = (m * (a + m - 1.0) / denom) * ((a + b + m - 1.0) / denom) * (b - m) * x * x;
        let mut b_n = m;
        b_n += (m * (b - m) * x) / (a + 2.0 * m - 1.0);
        b_n += ((a + m) * (a * y - b * x + 1.0 + m * (2.0 - x))) / (a + 2.0 * m + 1.0);
        (a_n, b_n)
    };
    // Modified Lentz, as Boost's continued_fraction_b.
    let tiny = 16.0 * f64::MIN_POSITIVE;
    let (_, b0) = term(0.0);
    let mut f = if b0 == 0.0 { tiny } else { b0 };
    let mut c = f;
    let mut d = 0.0;
    let mut m = 1.0;
    for _ in 0..MAX_ITERATIONS {
        if !meter::step() {
            break;
        }
        let (a_n, b_n) = term(m);
        m += 1.0;
        d = b_n + a_n * d;
        if d == 0.0 {
            d = tiny;
        }
        c = b_n + a_n / c;
        if c == 0.0 {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() <= EPSILON {
            break;
        }
    }
    power / f
}

/// `I_x(a, b) - I_x(a + k, b)`.
fn ibeta_a_step(a: f64, b: f64, x: f64, y: f64, k: i32) -> f64 {
    let prefix = ibeta_power_terms(a, b, x, y, 1.0) / a;
    if prefix == 0.0 {
        return prefix;
    }
    let mut sum = 1.0;
    let mut term = 1.0;
    for i in 0..k - 1 {
        let i = f64::from(i);
        term *= (a + b + i) * x / (a + i + 1.0);
        sum += term;
    }
    prefix * sum
}

/// Boost's `regularised_gamma_prefix`: `z^a exp(-z) / gamma(a)`.
fn regularised_gamma_prefix(a: f64, z: f64) -> f64 {
    if z >= f64::MAX || (a > 0.0 && z == 0.0) {
        return 0.0;
    }
    let agh = a + LANCZOS_G - 0.5;
    let d = ((z - a) - LANCZOS_G + 0.5) / agh;
    let prefix = if a < 1.0 {
        // The Lanczos approximation is tuned for a > 1.
        if z <= LOG_MIN || a < 1.0 / f64::MAX {
            return (a * z.ln() - z - lgam(a)).exp();
        }
        return z.powf(a) * (-z).exp() / gamma(a);
    } else if (d * d * a).abs() <= 100.0 && a > 150.0 {
        // Large a with a near z.
        (a * super::unity::log1pmx(d) + z * (0.5 - LANCZOS_G) / agh).exp()
    } else {
        let alz = a * (z / agh).ln();
        let amz = a - z;
        let low = alz.min(amz);
        let high = alz.max(amz);
        if low <= LOG_MIN || high >= LOG_MAX {
            let amza = amz / a;
            if low / 2.0 > LOG_MIN && high / 2.0 < LOG_MAX {
                // The square root of the result, squared.
                let sq = (z / agh).powf(a / 2.0) * (amz / 2.0).exp();
                sq * sq
            } else if low / 4.0 > LOG_MIN && high / 4.0 < LOG_MAX && z > a {
                // The fourth root, squared twice.
                let sq = (z / agh).powf(a / 4.0) * (amz / 4.0).exp();
                let p = sq * sq;
                p * p
            } else if amza > LOG_MIN && amza < LOG_MAX {
                ((z * amza.exp()) / agh).powf(a)
            } else {
                (alz + amz).exp()
            }
        } else {
            (z / agh).powf(a) * amz.exp()
        }
    };
    prefix * (agh / E).sqrt() / lanczos_sum_expg_scaled(a)
}

fn tgamma_delta_ratio_final(z: f64, delta: f64) -> f64 {
    let zgh = z + LANCZOS_G - 0.5;
    let mut result = if z + delta == z {
        (-delta).exp()
    } else {
        let power = if delta.abs() < 10.0 {
            ((0.5 - z) * (delta / zgh).ln_1p()).exp()
        } else {
            (zgh / (zgh + delta)).powf(z - 0.5)
        };
        power * (lanczos_sum(z) / lanczos_sum(z + delta))
    };
    result *= (E / (zgh + delta)).powf(delta);
    result
}

/// `gamma(z) / gamma(z + delta)`, as Boost's `tgamma_delta_ratio`.
pub(super) fn tgamma_delta_ratio(z: f64, delta: f64) -> f64 {
    if z <= 0.0 || z + delta <= 0.0 {
        return gamma(z) / gamma(z + delta);
    }
    if delta.floor() == delta {
        if z.floor() == z && z <= 170.0 && z + delta <= 170.0 {
            return FACTORIALS[z as usize - 1] / FACTORIALS[(z + delta) as usize - 1];
        }
        if delta.abs() < 20.0 {
            if delta == 0.0 {
                return 1.0;
            }
            let mut z = z;
            let mut delta = delta;
            if delta < 0.0 {
                z -= 1.0;
                let mut result = z;
                loop {
                    delta += 1.0;
                    if delta == 0.0 {
                        return result;
                    }
                    z -= 1.0;
                    result *= z;
                }
            }
            let mut result = 1.0 / z;
            loop {
                delta -= 1.0;
                if delta == 0.0 {
                    return result;
                }
                z += 1.0;
                result /= z;
            }
        }
    }
    if z < EPSILON {
        if 170.0 < delta {
            let mut ratio = tgamma_delta_ratio_final(delta, 170.0 - delta);
            ratio *= z;
            ratio *= FACTORIALS[169];
            return 1.0 / ratio;
        }
        return 1.0 / (z * gamma(z + delta));
    }
    tgamma_delta_ratio_final(z, delta)
}

/// DiDonato and Morris' BGRAT series for large `a` and small `b` (their equations 9 to 9.6),
/// added to `s0`.
fn beta_small_b_large_a_series(a: f64, b: f64, x: f64, y: f64, s0: f64) -> f64 {
    let bm1 = b - 1.0;
    let t = a + bm1 / 2.0;
    let lx = if y < 0.35 { (-y).ln_1p() } else { x.ln() };
    let u = -t * lx;
    let h = regularised_gamma_prefix(b, u);
    if h <= f64::MIN_POSITIVE {
        return s0;
    }
    let prefix = h / tgamma_delta_ratio(a, b) / t.powf(b);
    let mut p = [0.0; 30];
    p[0] = 1.0;
    let mut j = igamc(b, u) / h;
    let mut sum = s0 + prefix * j;
    let mut tnp1 = 1usize;
    let mut lx2 = lx / 2.0;
    lx2 *= lx2;
    let mut lxp = 1.0;
    let t4 = 4.0 * t * t;
    let mut b2n = b;
    for n in 1..p.len() {
        tnp1 += 2;
        p[n] = 0.0;
        let mut tmp1 = 3usize;
        for m in 1..n {
            let mbn = m as f64 * b - n as f64;
            p[n] += mbn * p[n - m] / FACTORIALS[tmp1];
            tmp1 += 2;
        }
        p[n] /= n as f64;
        p[n] += bm1 / FACTORIALS[tnp1];
        j = (b2n * (b2n + 1.0) * j + (u + b2n + 1.0) * lxp) / t4;
        lxp *= lx2;
        b2n += 2.0;
        let r = prefix * p[n] * j;
        sum += r;
        if (r / EPSILON).abs() < sum.abs() {
            break;
        }
    }
    sum
}

/// Boost's `binomial_coefficient` for `n <= 170`, or through the beta function beyond.
fn binomial_coefficient(n: u32, k: u32) -> f64 {
    if k == 0 || k == n {
        return 1.0;
    }
    if k == 1 || k == n - 1 {
        return f64::from(n);
    }
    let result = if n <= 170 {
        FACTORIALS[n as usize] / FACTORIALS[(n - k) as usize] / FACTORIALS[k as usize]
    } else {
        let value = if k < n - k {
            f64::from(k) * beta(f64::from(k), f64::from(n - k + 1))
        } else {
            f64::from(n - k) * beta(f64::from(k + 1), f64::from(n - k))
        };
        if value == 0.0 {
            return f64::INFINITY;
        }
        1.0 / value
    };
    (result - 0.5).ceil()
}

/// The upper tail of the binomial distribution, a finite sum for integer parameters.
fn binomial_ccdf(n: f64, k: f64, x: f64, y: f64) -> f64 {
    let mut result = x.powf(n);
    if result > f64::MIN_POSITIVE {
        let mut term = result;
        let mut i = (n - 1.0).trunc() as u32;
        while f64::from(i) > k {
            term *= (f64::from(i + 1) * y) / ((n - f64::from(i)) * x);
            result += term;
            i -= 1;
        }
        return result;
    }
    let mut start = (n * x).trunc() as i64;
    if start as f64 <= k + 1.0 {
        start = (k + 2.0).trunc() as i64;
    }
    let n_int = n.trunc() as u32;
    let start_f = start as f64;
    result = x.powf(start_f) * y.powf(n - start_f) * binomial_coefficient(n_int, start as u32);
    if result == 0.0 {
        let mut i = start - 1;
        while i as f64 > k {
            let i_f = i as f64;
            result += x.powf(i_f) * y.powf(n - i_f) * binomial_coefficient(n_int, i as u32);
            i -= 1;
        }
    } else {
        let mut term = result;
        let start_term = result;
        let mut i = start - 1;
        while i as f64 > k {
            let i_f = i as f64;
            term *= ((i_f + 1.0) * y) / ((n - i_f) * x);
            result += term;
            i -= 1;
        }
        term = start_term;
        let mut i = start + 1;
        while i as f64 <= n {
            let i_f = i as f64;
            term *= (n - i_f + 1.0) * x / (i_f * y);
            result += term;
            i += 1;
        }
    }
    result
}

/// The erf asymptotic for very large `a` and `b` near the saddle point.
fn ibeta_large_ab(a: f64, b: f64, x: f64, y: f64, invert: bool) -> f64 {
    let x0 = a / (a + b);
    let y0 = b / (a + b);
    let mut nu = x0 * (x / x0).ln() + y0 * (y / y0).ln();
    if nu > 0.0 || x == x0 || y == y0 {
        nu = 0.0;
    }
    nu = (-2.0 * nu).sqrt();
    if nu != 0.0 && nu / (x - x0) < 0.0 {
        nu = -nu;
    }
    let argument = -nu * ((a + b) / 2.0).sqrt();
    if invert {
        (1.0 + erf(argument)) / 2.0
    } else {
        erfc(argument) / 2.0
    }
}

/// Boost's `ibeta_imp` for the regularized function with `0 < x < 1` and positive finite
/// parameters: `I_x(a, b)`, or `1 - I_x(a, b)` when `invert`.
pub(super) fn ibeta_imp(a: f64, b: f64, x: f64, invert: bool) -> f64 {
    if !meter::step() {
        return f64::NAN;
    }
    let (mut a, mut b, mut x) = (a, b, x);
    let mut y = 1.0 - x;
    let mut invert = invert;
    if x == 0.0 {
        return if invert { 1.0 } else { 0.0 };
    }
    if x == 1.0 {
        return if invert { 0.0 } else { 1.0 };
    }
    if a == 0.5 && b == 0.5 {
        return if invert {
            y.sqrt().asin()
        } else {
            x.sqrt().asin()
        } / FRAC_PI_2;
    }
    if a == 1.0 {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut x, &mut y);
        invert = !invert;
    }
    if b == 1.0 {
        if a == 1.0 {
            return if invert { y } else { x };
        }
        return if y < 0.5 {
            let log = a * (-y).ln_1p();
            if invert {
                -log.exp_m1()
            } else {
                log.exp()
            }
        } else if invert {
            -(x.powf(a) - 1.0)
        } else {
            x.powf(a)
        };
    }
    // Each branch leaves `fract`, to be subtracted from 1 when `invert` is still set.
    let fract;
    if a.min(b) <= 1.0 {
        if x > 0.5 {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut x, &mut y);
            invert = !invert;
        }
        if a.max(b) <= 1.0 {
            if a >= b.min(0.2) || x.powf(a) <= 0.9 {
                fract = series(a, b, x, &mut invert);
            } else {
                std::mem::swap(&mut a, &mut b);
                std::mem::swap(&mut x, &mut y);
                invert = !invert;
                fract = if y >= 0.3 {
                    series(a, b, x, &mut invert)
                } else {
                    a_step_then_bgrat(a, b, x, y, &mut invert)
                };
            }
        } else if b <= 1.0 || (x < 0.1 && (b * x).powf(a) <= 0.7) {
            fract = series(a, b, x, &mut invert);
        } else {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut x, &mut y);
            invert = !invert;
            fract = if y >= 0.3 {
                series(a, b, x, &mut invert)
            } else if a >= 15.0 {
                if std::mem::take(&mut invert) {
                    -beta_small_b_large_a_series(a, b, x, y, -1.0)
                } else {
                    beta_small_b_large_a_series(a, b, x, y, 0.0)
                }
            } else {
                a_step_then_bgrat(a, b, x, y, &mut invert)
            };
        }
    } else {
        let lambda = if a < b {
            a - (a + b) * x
        } else {
            (a + b) * y - b
        };
        if lambda < 0.0 {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut x, &mut y);
            invert = !invert;
        }
        if b < 40.0 {
            if a.floor() == a && b.floor() == b && a < f64::from(i32::MAX - 100) && y != 1.0 {
                // The binomial distribution's finite sum.
                let k = a - 1.0;
                let n = b + k;
                fract = binomial_ccdf(n, k, x, y);
            } else if b * x <= 0.7 {
                fract = series(a, b, x, &mut invert);
            } else if a > 15.0 {
                // Sidestep to the series representation.
                let mut n = b.floor() as i32;
                if f64::from(n) == b {
                    n -= 1;
                }
                let bbar = b - f64::from(n);
                let step = ibeta_a_step(bbar, a, y, x, n);
                fract = beta_small_b_large_a_series(a, bbar, x, y, step);
            } else {
                let mut n = b.floor() as i32;
                let mut bbar = b - f64::from(n);
                if bbar <= 0.0 {
                    n -= 1;
                    bbar += 1.0;
                }
                let mut total = ibeta_a_step(bbar, a, y, x, n) + ibeta_a_step(a, bbar, x, y, 20);
                if invert {
                    total -= 1.0;
                }
                total = beta_small_b_large_a_series(a + 20.0, bbar, x, y, total);
                if std::mem::take(&mut invert) {
                    total = -total;
                }
                fract = total;
            }
        } else {
            // a and b both large.
            let ma = a.max(b);
            let xa = if ma == a { x } else { y };
            let saddle = ma / (a + b);
            let mut powers = 0.0;
            let mut use_asym = false;
            let limit = if xa < saddle { 2.0 } else { 15.0 };
            if ma > 1e-5 / EPSILON && ma / a.min(b) < limit {
                if a == b {
                    use_asym = true;
                } else {
                    powers = ((x / (a / (a + b))).ln() * a + (y / (b / (a + b))).ln() * b).exp();
                    if powers < EPSILON {
                        use_asym = true;
                    }
                }
            }
            let asymptotic = use_asym
                .then(|| ibeta_large_ab(a, b, x, y, invert))
                // A correction term too large for the erf approximation falls back.
                .filter(|estimate| estimate * EPSILON >= powers);
            fract = match asymptotic {
                Some(estimate) => {
                    invert = false;
                    estimate
                }
                None => ibeta_fraction2(a, b, x, y),
            };
        }
    }
    if invert {
        1.0 - fract
    } else {
        fract
    }
}

/// The series, or `-(series added to -1)` when inverting, as Boost's branches write it; clears
/// `invert` because the result is already complemented.
fn series(a: f64, b: f64, x: f64, invert: &mut bool) -> f64 {
    if std::mem::take(invert) {
        -ibeta_series(a, b, x, -1.0)
    } else {
        ibeta_series(a, b, x, 0.0)
    }
}

/// `ibeta_a_step` by 20, then BGRAT from `a + 20`; clears `invert` like [`series`].
fn a_step_then_bgrat(a: f64, b: f64, x: f64, y: f64, invert: &mut bool) -> f64 {
    let fract = ibeta_a_step(a, b, x, y, 20);
    if std::mem::take(invert) {
        -beta_small_b_large_a_series(a + 20.0, b, x, y, fract - 1.0)
    } else {
        beta_small_b_large_a_series(a + 20.0, b, x, y, fract)
    }
}

/// The derivative Boost's `ibeta_imp` reports with `I_x(a, b)`, which its inverse iterates on:
/// the beta density, with Boost's values at the endpoints and a finite stand-in for overflow.
pub(super) fn ibeta_imp_derivative(a: f64, b: f64, x: f64) -> f64 {
    let edge = |parameter: f64| {
        if parameter == 1.0 {
            1.0
        } else if parameter < 1.0 {
            f64::MAX / 2.0
        } else {
            f64::MIN_POSITIVE * 2.0
        }
    };
    let y = 1.0 - x;
    if x == 0.0 {
        return edge(a);
    }
    if x == 1.0 {
        return edge(b);
    }
    if a == 0.5 && b == 0.5 {
        return 1.0 / (PI * (y * x).sqrt());
    }
    let (a1, b1, x1) = if a == 1.0 { (b, a, y) } else { (a, b, x) };
    if b1 == 1.0 {
        return if a1 == 1.0 {
            1.0
        } else {
            a1 * x1.powf(a1 - 1.0)
        };
    }
    let terms = ibeta_power_terms(a, b, x, y, 1.0);
    let div = y * x;
    if terms == 0.0 {
        0.0
    } else if f64::MAX * div < terms {
        f64::MAX / 2.0
    } else {
        terms / div
    }
}

/// SciPy's limits shared by `betainc` and `betaincc`: `Some(I_x(a, b))` for the degenerate
/// parameters and domain errors.
fn betainc_limit(a: f64, b: f64, x: f64) -> Option<f64> {
    if a.is_nan() || b.is_nan() || x.is_nan() || a < 0.0 || b < 0.0 || !(0.0..=1.0).contains(&x) {
        return Some(f64::NAN);
    }
    if (a == 0.0 && b == 0.0) || (a.is_infinite() && b.is_infinite()) {
        return Some(f64::NAN);
    }
    if a == 0.0 || b.is_infinite() {
        // A point distribution at 0.
        return Some(if x > 0.0 { 1.0 } else { 0.0 });
    }
    if b == 0.0 || a.is_infinite() {
        // A point distribution at 1.
        return Some(if x < 1.0 { 0.0 } else { 1.0 });
    }
    None
}

/// The regularized incomplete beta function `I_x(a, b)`, SciPy's `betainc`.
pub(super) fn betainc(a: f64, b: f64, x: f64) -> f64 {
    betainc_limit(a, b, x).unwrap_or_else(|| ibeta_imp(a, b, x, false))
}

/// `1 - I_x(a, b)`, SciPy's `betaincc`.
pub(super) fn betaincc(a: f64, b: f64, x: f64) -> f64 {
    match betainc_limit(a, b, x) {
        Some(value) if value.is_nan() => value,
        Some(value) => 1.0 - value,
        None => ibeta_imp(a, b, x, true),
    }
}
