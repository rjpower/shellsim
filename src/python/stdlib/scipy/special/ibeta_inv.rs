//! The inverse of the regularized incomplete beta function, and Student's t quantile.
//!
//! SciPy computes `betaincinv`, `stdtrit` and `fdtri` with Boost.Math, so this is a port of
//! Boost's double-precision code: `ibeta_inv_imp` (`special_functions/detail/ibeta_inverse.hpp`),
//! which starts from one of Temme's asymptotic inversions, Hill's Student's t approximation, or
//! a power-term estimate and refines it by Halley iteration on the incomplete beta function; the
//! root finders of `tools/roots.hpp`; and the Student's t quantile of
//! `special_functions/detail/t_distribution_inv.hpp`.
//!
//! Boost's evaluation errors (a root finder that finds no bracket, or that exhausts SciPy's
//! limit of 400 iterations) surface as [`EvaluationError`], which SciPy's wrappers turn into NaN.
//! Where Boost calls its own incomplete gamma inverses and beta function for starting values,
//! this port calls the Cephes versions; the iteration removes the difference. Polynomials use
//! plain Horner evaluation, where Boost unrolls some fixed-size ones.
#![allow(clippy::excessive_precision, clippy::unreadable_literal)]

use std::cmp::Ordering;
use std::f64::consts::{FRAC_PI_2, PI, SQRT_2};
use std::mem::swap;

use super::erf::erfcinv;
use super::gamma::{beta, lgam};
use super::ibeta::{ibeta_imp, ibeta_imp_derivative, tgamma_delta_ratio};
use super::igam::{igamci, igami};
use super::meter;

const MIN_VALUE: f64 = f64::MIN_POSITIVE;
const MAX_VALUE: f64 = f64::MAX;
const EPSILON: f64 = f64::EPSILON;
/// Bits of precision Boost's policies report for `double`.
const DIGITS: i32 = 53;
/// SciPy's `max_root_iterations` policy for its Boost functions.
const MAX_ROOT_ITERATIONS: u32 = 400;
/// Boost runs Newton's method on Temme's equations without a limit; it converges quadratically
/// within a few steps, and this cap only bounds pathological input.
const MAX_NEWTON_ITERATIONS: u32 = 200;

/// A Boost evaluation error: no root was found within the iteration limit.
#[derive(Debug)]
pub(super) struct EvaluationError;

/// A single-precision literal from Boost's source, as C++ promotes it to double.
fn single(value: f32) -> f64 {
    f64::from(value)
}

fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Horner evaluation with coefficients in increasing degree, Boost's `evaluate_polynomial`.
fn polynomial(coefficients: &[f64], z: f64) -> f64 {
    let (last, rest) = coefficients
        .split_last()
        .expect("polynomials have coefficients");
    rest.iter().rev().fold(*last, |sum, c| sum * z + c)
}

fn even_polynomial(coefficients: &[f64], z: f64) -> f64 {
    polynomial(coefficients, z * z)
}

/// The binary exponent `frexp` reports: `x = m * 2^e` with `0.5 <= |m| < 1`.
fn frexp_exponent(x: f64) -> i32 {
    if x == 0.0 || !x.is_finite() {
        return 0;
    }
    #[allow(clippy::cast_possible_truncation)]
    let biased = ((x.to_bits() >> 52) & 0x7ff) as i32;
    if biased == 0 {
        // Subnormal: scale into the normal range first.
        return frexp_exponent(x * 2f64.powi(64)) - 64;
    }
    biased - 1022
}

/// The number of representable doubles from `a` to `b`, Boost's `float_distance`.
fn float_distance(a: f64, b: f64) -> f64 {
    fn ordered(x: f64) -> i64 {
        #[allow(clippy::cast_possible_wrap)]
        let bits = x.to_bits() as i64;
        if bits < 0 {
            i64::MIN - bits
        } else {
            bits
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let distance = (ordered(b) - ordered(a)) as f64;
    distance
}

/// Boost's `handle_zero_derivative`: pick a bisection step when the derivative vanishes.
#[allow(clippy::too_many_arguments)]
fn handle_zero_derivative(
    value: &dyn Fn(f64) -> f64,
    last_f0: &mut f64,
    f0: f64,
    delta: &mut f64,
    result: f64,
    guess: &mut f64,
    min: f64,
    max: f64,
) {
    if *last_f0 == 0.0 {
        // The first iteration: pretend the previous one was at min or max.
        *guess = if result == min { max } else { min };
        *last_f0 = value(*guess);
        *delta = *guess - result;
    }
    let crossed = sign(*last_f0) * sign(f0) < 0.0;
    *delta = if crossed == (*delta < 0.0) {
        (result - min) / 2.0
    } else {
        (result - max) / 2.0
    };
}

/// Boost's `newton_raphson_iterate`, for Temme's equations relating eta and x.
fn newton_raphson(
    f: impl Fn(f64) -> (f64, f64),
    guess: f64,
    mut min: f64,
    mut max: f64,
    digits: i32,
) -> Result<f64, EvaluationError> {
    if min > max {
        return Err(EvaluationError);
    }
    let value = |x: f64| f(x).0;
    let mut f0 = 0.0;
    let mut result = guess;
    let mut guess = guess;
    let factor = 2f64.powi(1 - digits);
    let mut delta = MAX_VALUE;
    let mut delta1 = MAX_VALUE;
    let mut max_range_f = 0.0;
    let mut min_range_f = 0.0;
    let mut count = MAX_NEWTON_ITERATIONS;
    loop {
        if !meter::step() {
            return Err(EvaluationError);
        }
        let mut last_f0 = f0;
        let delta2 = delta1;
        delta1 = delta;
        let (value0, f1) = f(result);
        f0 = value0;
        count -= 1;
        if f0 == 0.0 {
            break;
        }
        if f1 == 0.0 {
            handle_zero_derivative(
                &value,
                &mut last_f0,
                f0,
                &mut delta,
                result,
                &mut guess,
                min,
                max,
            );
        } else {
            delta = f0 / f1;
        }
        if (delta * 2.0).abs() > delta2.abs() {
            // The last two steps have not converged.
            let shift = if delta > 0.0 {
                (result - min) / 2.0
            } else {
                (result - max) / 2.0
            };
            delta = if result != 0.0 && shift.abs() > result.abs() {
                sign(delta) * result.abs()
            } else {
                shift
            };
            delta1 = 3.0 * delta;
        }
        guess = result;
        result -= delta;
        if result <= min {
            delta = 0.5 * (guess - min);
            result = guess - delta;
            if result == min || result == max {
                break;
            }
        } else if result >= max {
            delta = 0.5 * (guess - max);
            result = guess - delta;
            if result == min || result == max {
                break;
            }
        }
        if delta > 0.0 {
            max = guess;
            max_range_f = f0;
        } else {
            min = guess;
            min_range_f = f0;
        }
        if max_range_f * min_range_f > 0.0 {
            return Err(EvaluationError);
        }
        // A NaN step ends the iteration, as Boost's `<` test does.
        let shrinking = (result * factor).abs().partial_cmp(&delta.abs()) == Some(Ordering::Less);
        if count == 0 || !shrinking {
            break;
        }
    }
    Ok(result)
}

/// Boost's Halley step, falling back to Newton's when the Halley denominator would overflow.
fn halley_step(f0: f64, f1: f64, f2: f64) -> f64 {
    let denom = 2.0 * f0;
    let num = 2.0 * f1 - f0 * (f2 / f1);
    if num.abs() < 1.0 && denom.abs() >= num.abs() * MAX_VALUE {
        f0 / f1
    } else {
        denom / num
    }
}

/// Boost's `bracket_root_towards_max`: step `guess` towards `max` until the sign changes.
fn bracket_towards_max(
    value: &dyn Fn(f64) -> f64,
    mut guess: f64,
    f0: f64,
    min: &mut f64,
    max: &mut f64,
    count: &mut u32,
) -> f64 {
    if *count < 2 {
        return guess - (*max + *min) / 2.0;
    }
    let e = frexp_exponent(*max / guess).abs();
    let guess0 = guess;
    let mut multiplier = if e < 64 { 2.0 } else { 2f64.powi(e / 32) };
    let mut f_current = f0;
    let growth = if e > 1024 { 8.0 } else { 2.0 };
    // Boost multiplies towards max unless the range is negative, deciding once.
    let multiply = min.abs() < max.abs();
    loop {
        *count -= 1;
        if *count == 0 || (f_current < 0.0) != (f0 < 0.0) {
            break;
        }
        *min = guess;
        if multiply {
            guess *= multiplier;
        } else {
            guess /= multiplier;
        }
        if guess > *max {
            guess = *max;
            // There must be a change of sign.
            f_current = -f_current;
            break;
        }
        multiplier *= growth;
        f_current = value(guess);
    }
    if *count != 0 {
        *max = guess;
        if multiplier > 16.0 {
            return (guess0 - guess)
                + bracket_towards_min(value, guess, f_current, min, max, count);
        }
    }
    guess0 - (*max + *min) / 2.0
}

/// Boost's `bracket_root_towards_min`.
fn bracket_towards_min(
    value: &dyn Fn(f64) -> f64,
    mut guess: f64,
    f0: f64,
    min: &mut f64,
    max: &mut f64,
    count: &mut u32,
) -> f64 {
    if *count < 2 {
        return guess - (*max + *min) / 2.0;
    }
    let e = frexp_exponent(guess / *min).abs();
    let guess0 = guess;
    let mut multiplier = if e < 64 { 2.0 } else { 2f64.powi(e / 32) };
    let mut f_current = f0;
    let growth = if e > 1024 { 8.0 } else { 2.0 };
    // Boost divides towards min unless the range is negative, deciding once.
    let divide = min.abs() < max.abs();
    loop {
        *count -= 1;
        if *count == 0 || (f_current < 0.0) != (f0 < 0.0) {
            break;
        }
        *max = guess;
        if divide {
            guess /= multiplier;
        } else {
            guess *= multiplier;
        }
        if guess < *min {
            guess = *min;
            f_current = -f_current;
            break;
        }
        multiplier *= growth;
        f_current = value(guess);
    }
    if *count != 0 {
        *min = guess;
        if multiplier > 16.0 {
            return (guess0 - guess)
                + bracket_towards_max(value, guess, f_current, min, max, count);
        }
    }
    guess0 - (*max + *min) / 2.0
}

/// Boost's `halley_iterate` (`second_order_root_finder` with a Halley step). Returns the root
/// and the number of iterations used.
#[allow(clippy::too_many_lines)]
fn halley(
    f: impl Fn(f64) -> (f64, f64, f64),
    guess: f64,
    mut min: f64,
    mut max: f64,
    digits: i32,
    max_iter: u32,
) -> Result<(f64, u32), EvaluationError> {
    if min >= max {
        return Err(EvaluationError);
    }
    let value = |x: f64| f(x).0;
    let mut f0 = 0.0;
    let mut result = guess;
    let mut guess = guess;
    let factor = 2f64.powi(1 - digits);
    let mut delta = (10_000_000.0 * guess).max(10_000_000.0);
    let mut delta1 = delta;
    let mut out_of_bounds_sentry = false;
    let mut max_range_f = 0.0;
    let mut min_range_f = 0.0;
    let mut count = max_iter;
    'iterate: loop {
        if !meter::step() {
            return Err(EvaluationError);
        }
        'step: {
            let mut last_f0 = f0;
            let delta2 = delta1;
            delta1 = delta;
            let (value0, f1, f2) = f(result);
            f0 = value0;
            count -= 1;
            if f0 == 0.0 {
                break 'iterate;
            }
            if f1 == 0.0 {
                handle_zero_derivative(
                    &value,
                    &mut last_f0,
                    f0,
                    &mut delta,
                    result,
                    &mut guess,
                    min,
                    max,
                );
            } else if f2 != 0.0 {
                delta = halley_step(f0, f1, f2);
                if delta * f1 / f0 < 0.0 {
                    // Newton and Halley disagree on the direction; take a bounded Newton step.
                    delta = f0 / f1;
                    if delta.abs() > 2.0 * result.abs() {
                        delta = sign(delta) * 2.0 * result.abs();
                    }
                }
            } else {
                delta = f0 / f1;
            }
            let convergence = if delta2.abs() > 1.0 || (MAX_VALUE * delta2).abs() > delta.abs() {
                (delta / delta2).abs()
            } else {
                MAX_VALUE
            };
            if convergence > 0.8 && convergence < 2.0 {
                // The last two steps have not converged.
                let wide = if min.abs() < 1.0 {
                    (1000.0 * min).abs() < max.abs()
                } else {
                    (max / min).abs() > 1000.0
                };
                if wide {
                    delta = if delta > 0.0 {
                        bracket_towards_min(&value, result, f0, &mut min, &mut max, &mut count)
                    } else {
                        bracket_towards_max(&value, result, f0, &mut min, &mut max, &mut count)
                    };
                } else {
                    delta = if delta > 0.0 {
                        (result - min) / 2.0
                    } else {
                        (result - max) / 2.0
                    };
                    if result != 0.0 && delta.abs() > result {
                        delta = sign(delta) * result.abs() * single(0.9);
                    }
                }
                delta1 = delta * 3.0;
            }
            guess = result;
            result -= delta;
            if result < min {
                let mut diff = if min.abs() < 1.0
                    && result.abs() > 1.0
                    && MAX_VALUE / result.abs() < min.abs()
                {
                    1000.0
                } else if min.abs() < 1.0 && (MAX_VALUE * min).abs() < result.abs() {
                    if (min < 0.0) != (result < 0.0) {
                        -MAX_VALUE
                    } else {
                        MAX_VALUE
                    }
                } else {
                    result / min
                };
                if diff.abs() < 1.0 {
                    diff = 1.0 / diff;
                }
                if !out_of_bounds_sentry && diff > 0.0 && diff < 3.0 {
                    // A small step out of bounds: assume the root is near min.
                    delta = single(0.99) * (guess - min);
                    result = guess - delta;
                    out_of_bounds_sentry = true;
                } else {
                    if float_distance(min, max).abs() < 2.0 {
                        result = (min + max) / 2.0;
                        break 'iterate;
                    }
                    delta = bracket_towards_min(&value, guess, f0, &mut min, &mut max, &mut count);
                    result = guess - delta;
                    if result <= min {
                        result = min.next_up();
                    }
                    if result >= max {
                        result = max.next_down();
                    }
                    guess = min;
                    break 'step;
                }
            } else if result > max {
                let mut diff = if max.abs() < 1.0
                    && result.abs() > 1.0
                    && MAX_VALUE / result.abs() < max.abs()
                {
                    1000.0
                } else {
                    result / max
                };
                if diff.abs() < 1.0 {
                    diff = 1.0 / diff;
                }
                if !out_of_bounds_sentry && diff > 0.0 && diff < 3.0 {
                    delta = single(0.99) * (guess - max);
                    result = guess - delta;
                    out_of_bounds_sentry = true;
                } else {
                    if float_distance(min, max).abs() < 2.0 {
                        result = (min + max) / 2.0;
                        break 'iterate;
                    }
                    delta = bracket_towards_max(&value, guess, f0, &mut min, &mut max, &mut count);
                    result = guess - delta;
                    if result >= max {
                        result = max.next_down();
                    }
                    if result <= min {
                        result = min.next_up();
                    }
                    guess = min;
                    break 'step;
                }
            }
            if delta > 0.0 {
                max = guess;
                max_range_f = f0;
            } else {
                min = guess;
                min_range_f = f0;
            }
            if max_range_f * min_range_f > 0.0 {
                return Err(EvaluationError);
            }
        }
        // A NaN step ends the iteration, as Boost's `<` test does.
        let shrinking = (result * factor).abs().partial_cmp(&delta.abs()) == Some(Ordering::Less);
        if count == 0 || !shrinking {
            break;
        }
    }
    Ok((result, max_iter - count))
}

/// Temme's equation relating eta and x, with its derivative in x.
fn temme_root(t: f64, a: f64) -> impl Fn(f64) -> (f64, f64) {
    move |x| {
        let y = 1.0 - x;
        (x.ln() + a * y.ln() + t, 1.0 / x - a / y)
    }
}

/// Temme's first inversion (section 2), for `a` and `b` of similar size.
fn temme_method_1(a: f64, b: f64, z: f64) -> f64 {
    let r2 = SQRT_2;
    let eta0 = erfcinv(2.0 * z) / -(a / 2.0).sqrt();
    let big_b = b - a;
    let b_2 = big_b * big_b;
    let b_3 = b_2 * big_b;
    let terms = [
        eta0,
        polynomial(
            &[
                -big_b * r2 / 2.0,
                (1.0 - 2.0 * big_b) / 8.0,
                -(big_b * r2 / 48.0),
                -1.0 / 192.0,
                -big_b * r2 / 3840.0,
            ],
            eta0,
        ),
        polynomial(
            &[
                big_b * r2 * (3.0 * big_b - 2.0) / 12.0,
                (20.0 * b_2 - 12.0 * big_b + 1.0) / 128.0,
                big_b * r2 * (20.0 * big_b - 1.0) / 960.0,
                (16.0 * b_2 + 30.0 * big_b - 15.0) / 4608.0,
                big_b * r2 * (21.0 * big_b + 32.0) / 53760.0,
                (-32.0 * b_2 + 63.0) / 368640.0,
                -big_b * r2 * (120.0 * big_b + 17.0) / 25804480.0,
            ],
            eta0,
        ),
        polynomial(
            &[
                big_b * r2 * (-75.0 * b_2 + 80.0 * big_b - 16.0) / 480.0,
                (-1080.0 * b_3 + 868.0 * b_2 - 90.0 * big_b - 45.0) / 9216.0,
                big_b * r2 * (-1190.0 * b_2 + 84.0 * big_b + 373.0) / 53760.0,
                (-2240.0 * b_3 - 2508.0 * b_2 + 2100.0 * big_b - 165.0) / 368640.0,
            ],
            eta0,
        ),
    ];
    let eta = polynomial(&terms, 1.0 / a);
    let eta_2 = eta * eta;
    let c = -(-eta_2 / 2.0).exp();
    let x = if eta_2 == 0.0 {
        0.5
    } else {
        (1.0 + eta * ((1.0 + c) / eta_2).sqrt()) / 2.0
    };
    x.clamp(0.0, 1.0)
}

/// Temme's second inversion (section 3), for `a / (a + b)` between 0.2 and 0.8.
fn temme_method_2(z: f64, r: f64, theta: f64) -> f64 {
    let eta0 = erfcinv(2.0 * z) / -(r / 2.0).sqrt();
    let s = theta.sin();
    let c = theta.cos();
    let sc = s * c;
    let sc_2 = sc * sc;
    let sc_3 = sc_2 * sc;
    let sc_4 = sc_2 * sc_2;
    let sc_5 = sc_2 * sc_3;
    let sc_6 = sc_3 * sc_3;
    let sc_7 = sc_4 * sc_3;
    let e1 = polynomial(
        &[
            (2.0 * s * s - 1.0) / (3.0 * s * c),
            -even_polynomial(&[-1.0, -5.0, 5.0], s) / (36.0 * sc_2),
            even_polynomial(&[1.0, 21.0, -69.0, 46.0], s) / (1620.0 * sc_3),
            -even_polynomial(&[7.0, -2.0, 33.0, -62.0, 31.0], s) / (6480.0 * sc_4),
            even_polynomial(&[25.0, -52.0, -17.0, 88.0, -115.0, 46.0], s) / (90720.0 * sc_5),
        ],
        eta0,
    );
    let e2 = polynomial(
        &[
            -even_polynomial(&[7.0, 12.0, -78.0, 52.0], s) / (405.0 * sc_3),
            even_polynomial(&[-7.0, 2.0, 183.0, -370.0, 185.0], s) / (2592.0 * sc_4),
            -even_polynomial(&[-533.0, 776.0, -1835.0, 10240.0, -13525.0, 5410.0], s)
                / (204120.0 * sc_5),
            -even_polynomial(
                &[
                    -1579.0, 3747.0, -3372.0, -15821.0, 45588.0, -45213.0, 15071.0,
                ],
                s,
            ) / (2099520.0 * sc_6),
        ],
        eta0,
    );
    let e3 = polynomial(
        &[
            even_polynomial(&[449.0, -1259.0, -769.0, 6686.0, -9260.0, 3704.0], s)
                / (102060.0 * sc_5),
            -even_polynomial(
                &[
                    63149.0, -151557.0, 140052.0, -727469.0, 2239932.0, -2251437.0, 750479.0,
                ],
                s,
            ) / (20995200.0 * sc_6),
            even_polynomial(
                &[
                    29233.0, -78755.0, 105222.0, 146879.0, -1602610.0, 3195183.0, -2554139.0,
                    729754.0,
                ],
                s,
            ) / (36741600.0 * sc_7),
        ],
        eta0,
    );
    let eta = polynomial(&[eta0, e1, e2, e3], 1.0 / r);
    let s_2 = s * s;
    let c_2 = c * c;
    let alpha = (c / s) * (c / s);
    let lu = -(eta * eta) / (2.0 * s_2) + s_2.ln() + c_2 * c_2.ln() / s_2;
    let mut x = if eta.abs() < 0.7 {
        // Small eta: Temme's expansion in eta (section 5).
        let w0 = s * s;
        polynomial(
            &[
                w0,
                s * c,
                (1.0 - 2.0 * w0) / 3.0,
                polynomial(&[1.0, -13.0, 13.0], w0) / (36.0 * s * c),
                polynomial(&[1.0, 21.0, -69.0, 46.0], w0) / (270.0 * w0 * c * c),
            ],
            eta,
        )
    } else {
        // Large eta: a polynomial in u, then pick the root on eta's side of sin^2(theta).
        let u = lu.exp();
        let x = polynomial(
            &[
                u,
                alpha,
                0.0,
                3.0 * alpha * (3.0 * alpha + 1.0) / 6.0,
                4.0 * alpha * (4.0 * alpha + 1.0) * (4.0 * alpha + 2.0) / 24.0,
                5.0 * alpha * (5.0 * alpha + 1.0) * (5.0 * alpha + 2.0) * (5.0 * alpha + 3.0)
                    / 120.0,
            ],
            u,
        );
        if (x - s_2) * eta < 0.0 {
            1.0 - x
        } else {
            x
        }
    };
    let (lower, upper) = if eta < 0.0 { (0.0, s_2) } else { (s_2, 1.0) };
    if x < lower || x > upper {
        x = (lower + upper) / 2.0;
    }
    // Boost keeps its estimate when this polish finds no root.
    newton_raphson(temme_root(-lu, alpha), x, lower, upper, DIGITS / 2).unwrap_or(x)
}

/// Temme's third inversion (section 4), through the incomplete gamma inverse, for very
/// different `a` and `b`.
#[allow(clippy::many_single_char_names)]
fn temme_method_3(a: f64, b: f64, p: f64, q: f64) -> Result<f64, EvaluationError> {
    let eta0 = if p < q { igamci(b, p) } else { igami(b, q) } / a;
    let mu = b / a;
    let w = (1.0 + mu).sqrt();
    let w_2 = w * w;
    let w_3 = w_2 * w;
    let w_4 = w_2 * w_2;
    let w_5 = w_3 * w_2;
    let w_6 = w_3 * w_3;
    let w_7 = w_4 * w_3;
    let w_8 = w_4 * w_4;
    let w_9 = w_5 * w_4;
    let w_10 = w_5 * w_5;
    let d = eta0 - mu;
    let d_2 = d * d;
    let d_3 = d_2 * d;
    let d_4 = d_2 * d_2;
    let w1 = w + 1.0;
    let w1_2 = w1 * w1;
    let w1_3 = w1 * w1_2;
    let w1_4 = w1_2 * w1_2;

    let mut e1 = (w + 2.0) * (w - 1.0) / (3.0 * w);
    e1 += (w_3 + 9.0 * w_2 + 21.0 * w + 5.0) * d / (36.0 * w_2 * w1);
    e1 -= (w_4 - 13.0 * w_3 + 69.0 * w_2 + 167.0 * w + 46.0) * d_2 / (1620.0 * w1_2 * w_3);
    e1 -= (7.0 * w_5 + 21.0 * w_4 + 70.0 * w_3 + 26.0 * w_2 - 93.0 * w - 31.0) * d_3
        / (6480.0 * w1_3 * w_4);
    e1 -= (75.0 * w_6 + 202.0 * w_5 + 188.0 * w_4 - 888.0 * w_3 - 1345.0 * w_2 + 118.0 * w + 138.0)
        * d_4
        / (272160.0 * w1_4 * w_5);

    let mut e2 = (28.0 * w_4 + 131.0 * w_3 + 402.0 * w_2 + 581.0 * w + 208.0) * (w - 1.0)
        / (1620.0 * w1 * w_3);
    e2 -=
        (35.0 * w_6 - 154.0 * w_5 - 623.0 * w_4 - 1636.0 * w_3 - 3983.0 * w_2 - 3514.0 * w - 925.0)
            * d
            / (12960.0 * w1_2 * w_4);
    e2 -= (2132.0 * w_7
        + 7915.0 * w_6
        + 16821.0 * w_5
        + 35066.0 * w_4
        + 87490.0 * w_3
        + 141183.0 * w_2
        + 95993.0 * w
        + 21640.0)
        * d_2
        / (816480.0 * w_5 * w1_3);
    e2 -= (11053.0 * w_8 + 53308.0 * w_7 + 117010.0 * w_6 + 163924.0 * w_5 + 116188.0 * w_4
        - 258428.0 * w_3
        - 677042.0 * w_2
        - 481940.0 * w
        - 105497.0)
        * d_3
        / (14696640.0 * w1_4 * w_6);

    let mut e3 = -((3592.0 * w_7 + 8375.0 * w_6
        - 1323.0 * w_5
        - 29198.0 * w_4
        - 89578.0 * w_3
        - 154413.0 * w_2
        - 116063.0 * w
        - 29632.0)
        * (w - 1.0))
        / (816480.0 * w_5 * w1_2);
    e3 -= (442043.0 * w_9 + 2054169.0 * w_8 + 3803094.0 * w_7 + 3470754.0 * w_6 + 2141568.0 * w_5
        - 2393568.0 * w_4
        - 19904934.0 * w_3
        - 34714674.0 * w_2
        - 23128299.0 * w
        - 5253353.0)
        * d
        / (146966400.0 * w_6 * w1_3);
    e3 -= (116932.0 * w_10
        + 819281.0 * w_9
        + 2378172.0 * w_8
        + 4341330.0 * w_7
        + 6806004.0 * w_6
        + 10622748.0 * w_5
        + 18739500.0 * w_4
        + 30651894.0 * w_3
        + 30869976.0 * w_2
        + 15431867.0 * w
        + 2919016.0)
        * d_2
        / (146966400.0 * w1_4 * w_7);

    let mut eta = eta0 + e1 / a + e2 / (a * a) + e3 / (a * a * a);
    // Equation 4.2 has one root on each side of cross = 1 / (1 + mu); eta decides which.
    if eta <= 0.0 {
        eta = MIN_VALUE;
    }
    let u = eta - mu * eta.ln() + (1.0 + mu) * (1.0 + mu).ln() - mu;
    let cross = 1.0 / (1.0 + mu);
    let (lower, upper) = if eta < mu { (cross, 1.0) } else { (0.0, cross) };
    let x = (lower + upper) / 2.0;
    if cross == 0.0 || cross == 1.0 {
        return Ok(cross);
    }
    newton_raphson(temme_root(u, mu), x, lower, upper, DIGITS / 2)
}

/// The function, derivative and second derivative Halley iteration uses to solve
/// `I_x(a, b) = target`, or `1 - I_x(a, b) = target` when `invert` is set.
fn ibeta_roots(a: f64, b: f64, target: f64, invert: bool, x: f64) -> (f64, f64, f64) {
    let f = ibeta_imp(a, b, x, invert) - target;
    let mut f1 = ibeta_imp_derivative(a, b, x);
    if invert {
        f1 = -f1;
    }
    let mut y = 1.0 - x;
    if y == 0.0 {
        y = MIN_VALUE * 64.0;
    }
    let x = if x == 0.0 { MIN_VALUE * 64.0 } else { x };
    let mut f2 = f1 * (-y * a + (b - 2.0) * x + 1.0);
    if f2.abs() < y * x * MAX_VALUE {
        f2 /= y * x;
    }
    if invert {
        f2 = -f2;
    }
    if f1 == 0.0 {
        f1 = if invert { -1.0 } else { 1.0 } * MIN_VALUE * 64.0;
    }
    (f, f1, f2)
}

/// Boost's `ibeta_inv_imp`: `(x, 1 - x)` with `I_x(a, b) = p`, given `q = 1 - p`. `want_y`
/// records whether Boost's caller asked for `1 - x`, which moves the search's lower bound.
#[allow(clippy::too_many_lines, clippy::many_single_char_names)]
fn ibeta_inv_imp(
    mut a: f64,
    mut b: f64,
    mut p: f64,
    mut q: f64,
    want_y: bool,
) -> Result<(f64, f64), EvaluationError> {
    let mut invert = false;
    if q == 0.0 {
        return Ok((1.0, 0.0));
    }
    if p == 0.0 {
        return Ok((0.0, 1.0));
    }
    if a == 1.0 {
        if b == 1.0 {
            return Ok((p, 1.0 - p));
        }
        // Handle as the b == 1 case below.
        swap(&mut a, &mut b);
        swap(&mut p, &mut q);
        invert = true;
    }
    let mut x;
    let mut y;
    let mut lower = 0.0;
    let mut upper = 1.0;
    if a == 0.5 {
        if b == 0.5 {
            let x = (p * FRAC_PI_2).sin();
            let y = (q * FRAC_PI_2).sin();
            return Ok((x * x, y * y));
        } else if b > 0.5 {
            swap(&mut a, &mut b);
            swap(&mut p, &mut q);
            invert = !invert;
        }
    }
    if b == 0.5 && a >= 0.5 && p != 1.0 {
        // Student's t distribution.
        (x, y) = find_ibeta_inv_from_t_dist(a, p);
    } else if b == 1.0 {
        if p < q {
            x = p.powf(1.0 / a);
            y = if a > 1.0 {
                -(p.ln() / a).exp_m1()
            } else {
                1.0 - x
            };
        } else {
            x = ((-q).ln_1p() / a).exp();
            y = -((-q).ln_1p() / a).exp_m1();
        }
        if invert {
            swap(&mut x, &mut y);
        }
        return Ok((x, y));
    } else if a + b > 5.0 {
        // Temme's asymptotic expansions, with p < 0.5 to avoid cancellation.
        if p > 0.5 {
            swap(&mut a, &mut b);
            swap(&mut p, &mut q);
            invert = !invert;
        }
        let minv = a.min(b);
        let maxv = a.max(b);
        if minv.sqrt() > maxv - minv && minv > 5.0 {
            x = temme_method_1(a, b, p);
            y = 1.0 - x;
        } else {
            let r = a + b;
            let theta = (a / r).sqrt().asin();
            let lambda = minv / r;
            if (0.2..=0.8).contains(&lambda) && r >= 10.0 {
                let ppa = p.powf(1.0 / a);
                x = if ppa < 0.0025 && a + b < 200.0 {
                    ppa * (a * beta(a, b)).powf(1.0 / a)
                } else {
                    temme_method_2(p, r, theta)
                };
                y = 1.0 - x;
            } else {
                // Very different a and b: Temme's third method needs a > b.
                if a < b {
                    swap(&mut a, &mut b);
                    swap(&mut p, &mut q);
                    invert = !invert;
                }
                let mut bet = 0.0;
                if b < 2.0 {
                    bet = beta(a, b);
                    if bet > MAX_VALUE {
                        bet = MAX_VALUE;
                    }
                }
                if bet == 0.0 {
                    x = 0.0;
                    y = 1.0;
                } else {
                    y = (b * q * bet).powf(1.0 / b);
                    x = 1.0 - y;
                }
                if y > 1e-5 && a.min(b) < 1000.0 {
                    x = temme_method_3(a, b, p, q)?;
                    y = 1.0 - x;
                } else if y > 1e-5 && a.min(b) > 1000.0 {
                    // Start from the saddle point.
                    x = a.max(b) / (a + b);
                    y = a.min(b) / (a + b);
                }
            }
        }
    } else if a < 1.0 && b < 1.0 {
        // Start on the correct side of the inflection point xs.
        let mut xs = (1.0 - a) / (2.0 - a - b);
        let fs = ibeta_imp(a, b, xs, false) - p;
        if fs.abs() / p < EPSILON * 3.0 {
            return Ok(if invert {
                (1.0 - xs, xs)
            } else {
                (xs, 1.0 - xs)
            });
        }
        if fs < 0.0 {
            swap(&mut a, &mut b);
            swap(&mut p, &mut q);
            invert = !invert;
            xs = 1.0 - xs;
        }
        if a < MIN_VALUE && b > MIN_VALUE {
            return Ok(if invert { (1.0, 0.0) } else { (0.0, 1.0) });
        }
        let bet = beta(a, b);
        let xg = if bet.is_finite() {
            (a * p * bet).powf(1.0 / a)
        } else {
            let xg = ((lgam(a + 1.0) + lgam(b) - lgam(a + b) + p.ln()) / a).exp();
            if xg > 2.0 / EPSILON {
                2.0 / EPSILON
            } else {
                xg
            }
        };
        x = xg / (1.0 + xg);
        y = 1.0 / (1.0 + xg);
        if x > xs {
            x = xs;
        }
        upper = xs;
    } else if a > 1.0 && b > 1.0 {
        let mut xs = (a - 1.0) / (a + b - 2.0);
        let mut xs2 = (b - 1.0) / (a + b - 2.0);
        let ps = ibeta_imp(a, b, xs, false) - p;
        if ps < 0.0 {
            swap(&mut a, &mut b);
            swap(&mut p, &mut q);
            swap(&mut xs, &mut xs2);
            invert = !invert;
        }
        let lx = (p * a * beta(a, b)).ln() / a;
        x = lx.exp();
        y = if x < 0.9 { 1.0 - x } else { -lx.exp_m1() };
        if b < a && x < 0.2 {
            let mut ap1 = a - 1.0;
            let bm1 = b - 1.0;
            let a_2 = a * a;
            let a_3 = a * a_2;
            let b_2 = b * b;
            let mut terms = [0.0, 1.0, bm1 / ap1, 0.0, 0.0];
            ap1 *= ap1;
            terms[3] = bm1 * (3.0 * a * b + 5.0 * b + a_2 - a - 4.0) / (2.0 * (a + 2.0) * ap1);
            ap1 *= a + 1.0;
            terms[4] = bm1
                * (33.0 * a * b_2 + 31.0 * b_2 + 8.0 * a_2 * b_2 - 30.0 * a * b - 47.0 * b
                    + 11.0 * a_2 * b
                    + 6.0 * a_3 * b
                    + 18.0
                    + 4.0 * a
                    - a_3
                    + a_2 * a_2
                    - 10.0 * a_2)
                / (3.0 * (a + 3.0) * (a + 2.0) * ap1);
            x = polynomial(&terms, x);
        }
        if x > xs {
            x = xs;
        }
        upper = xs;
    } else {
        // One of a and b is above 1 and a + b is small: arrange b > a, which gives a concave
        // curve without inflection points.
        if b < a {
            swap(&mut a, &mut b);
            swap(&mut p, &mut q);
            invert = !invert;
        }
        if a < MIN_VALUE {
            // Avoid spurious overflow for subnormal a.
            (x, y) = if p < 1.0 { (1.0, 0.0) } else { (0.0, 1.0) };
        } else if p.powf(1.0 / a) < 0.5 {
            x = (p * a * beta(a, b)).powf(1.0 / a);
            if x > 1.0 || !x.is_finite() {
                x = 1.0;
            }
            if x == 0.0 {
                x = MIN_VALUE;
            }
            y = 1.0 - x;
        } else {
            // Model the curve as a distorted quarter circle.
            y = (1.0 - p.powf(b * beta(a, b))).powf(1.0 / b);
            if y > 1.0 || !y.is_finite() {
                y = 1.0;
            }
            if y == 0.0 {
                y = MIN_VALUE;
            }
            x = 1.0 - y;
        }
    }
    // Iterate on the smaller of x and y.
    if x > 0.5 {
        swap(&mut a, &mut b);
        swap(&mut p, &mut q);
        swap(&mut x, &mut y);
        invert = !invert;
        (lower, upper) = (1.0 - upper, 1.0 - lower);
    }
    if lower == 0.0 {
        // Subnormal answers take many iterations and have unreliable derivatives.
        lower = if invert && !want_y {
            EPSILON
        } else {
            MIN_VALUE
        };
        if x < lower {
            x = lower;
        }
    }
    let mut digits = DIGITS / 2;
    if x < 1e-50 && (a < 1.0 || b < 1.0) {
        // Where the derivative is very large, keep the root finder from stopping early.
        digits *= 3;
        digits /= 2;
    }
    let (target, root_invert) = if p < q { (p, false) } else { (q, true) };
    let (mut x, used) = halley(
        |x| ibeta_roots(a, b, target, root_invert, x),
        x,
        lower,
        upper,
        digits,
        MAX_ROOT_ITERATIONS,
    )?;
    if used >= MAX_ROOT_ITERATIONS {
        return Err(EvaluationError);
    }
    if x == lower {
        x = 0.0;
    }
    Ok(if invert { (1.0 - x, x) } else { (x, 1.0 - x) })
}

/// SciPy's `betaincinv`: `x` with `I_x(a, b) = p`.
pub(super) fn betaincinv(a: f64, b: f64, p: f64) -> f64 {
    ibeta_inv(a, b, p, false).map_or(f64::NAN, |(x, _)| x)
}

/// Boost's `ibeta_inv(a, b, p, &y)`: `(x, 1 - x)` with `I_x(a, b) = p`, or `None` for a
/// domain or evaluation error. `want_y` records whether the caller uses `1 - x`.
pub(super) fn ibeta_inv(a: f64, b: f64, p: f64, want_y: bool) -> Option<(f64, f64)> {
    if !(a > 0.0 && b > 0.0 && (0.0..=1.0).contains(&p)) {
        return None;
    }
    ibeta_inv_imp(a, b, p, 1.0 - p, want_y).ok()
}

/// The starting value for `b = 1/2`, where the incomplete beta function is Student's t.
fn find_ibeta_inv_from_t_dist(a: f64, p: f64) -> (f64, f64) {
    let u = p / 2.0;
    let v = 1.0 - u;
    let df = a * 2.0;
    let t = inverse_students_t(df, u, v).0;
    (df / (df + t * t), t * t / (df + t * t))
}

/// Hill's approximation to the Student's t quantile (Algorithm 396), for `u <= 0.5`.
fn inverse_students_t_hill(ndf: f64, u: f64) -> f64 {
    if ndf > single(1e20) {
        return -erfcinv(2.0 * u) * SQRT_2;
    }
    let a = 1.0 / (ndf - 0.5);
    let b = 48.0 / (a * a);
    let mut c = ((20700.0 * a / b - 98.0) * a - 16.0) * a + single(96.36);
    let d = ((94.5 / (b + c) - 3.0) / b + 1.0) * (a * PI / 2.0).sqrt() * ndf;
    let mut y = (d * 2.0 * u).powf(2.0 / ndf);
    if y > single(0.05) + a {
        // Asymptotic inverse expansion about the normal.
        let x = -erfcinv(2.0 * u) * SQRT_2;
        y = x * x;
        if ndf < 5.0 {
            c += single(0.3) * (ndf - 4.5) * (x + single(0.6));
        }
        c += (((single(0.05) * d * x - 5.0) * x - 7.0) * x - 2.0) * x + b;
        y = (((((single(0.4) * y + single(6.3)) * y + 36.0) * y + 94.5) / c - y - 3.0) / b + 1.0)
            * x;
        y = (a * y * y).exp_m1();
    } else {
        y = ((1.0
            / (((ndf + 6.0) / (ndf * y) - single(0.089) * d - single(0.822)) * (ndf + 2.0) * 3.0)
            + 0.5 / (ndf + 4.0))
            * y
            - 1.0)
            * (ndf + 1.0)
            / (ndf + 2.0)
            + 1.0 / y;
    }
    -(ndf * y).sqrt()
}

/// Shaw's tail series for the Student's t quantile (section 6 of his paper).
fn inverse_students_t_tail_series(df: f64, v: f64) -> f64 {
    let w = tgamma_delta_ratio(df / 2.0, 0.5) * (df * PI).sqrt() * v;
    let mut np2 = df + 2.0;
    let mut np4 = df + 4.0;
    let mut np6 = df + 6.0;
    let mut d = [1.0; 7];
    d[1] = -(df + 1.0) / (2.0 * np2);
    np2 *= df + 2.0;
    d[2] = -df * (df + 1.0) * (df + 3.0) / (8.0 * np2 * np4);
    np2 *= df + 2.0;
    d[3] =
        -df * (df + 1.0) * (df + 5.0) * (((3.0 * df) + 7.0) * df - 2.0) / (48.0 * np2 * np4 * np6);
    np2 *= df + 2.0;
    np4 *= df + 4.0;
    d[4] = -df
        * (df + 1.0)
        * (df + 7.0)
        * (((((15.0 * df + 154.0) * df + 465.0) * df + 286.0) * df - 336.0) * df + 64.0)
        / (384.0 * np2 * np4 * np6 * (df + 8.0));
    np2 *= df + 2.0;
    d[5] = -df
        * (df + 1.0)
        * (df + 3.0)
        * (df + 9.0)
        * (((((((35.0 * df + 452.0) * df + 1573.0) * df + 600.0) * df - 2020.0) * df) + 928.0)
            * df
            - 128.0)
        / (1280.0 * np2 * np4 * np6 * (df + 8.0) * (df + 10.0));
    np2 *= df + 2.0;
    np4 *= df + 4.0;
    np6 *= df + 6.0;
    d[6] = -df
        * (df + 1.0)
        * (df + 11.0)
        * ((((((((((((945.0 * df) + 31506.0) * df + 425858.0) * df + 2980236.0) * df
            + 11266745.0)
            * df
            + 20675018.0)
            * df
            + 7747124.0)
            * df
            - 22574632.0)
            * df
            - 8565600.0)
            * df
            + 18108416.0)
            * df
            - 7099392.0)
            * df
            + 884736.0)
        / (46080.0 * np2 * np4 * np6 * (df + 8.0) * (df + 10.0) * (df + 12.0));
    let rn = df.sqrt();
    let div = (rn * w).powf(1.0 / df);
    let power = div * div;
    -(polynomial(&d, power) * rn / div)
}

/// Shaw's body series for the Student's t quantile with small degrees of freedom.
fn inverse_students_t_body_series(df: f64, u: f64) -> f64 {
    let v = tgamma_delta_ratio(df / 2.0, 0.5) * (df * PI).sqrt() * (u - 0.5);
    let r#in = 1.0 / df;
    let c = [
        1.0,
        0.16666666666666666667 + 0.16666666666666666667 * r#in,
        (0.0083333333333333333333 * r#in + 0.066666666666666666667) * r#in
            + 0.058333333333333333333,
        ((0.00019841269841269841270 * r#in + 0.0017857142857142857143) * r#in
            + 0.026785714285714285714)
            * r#in
            + 0.025198412698412698413,
        (((2.7557319223985890653e-6 * r#in + 0.00037477954144620811287) * r#in
            - 0.0011078042328042328042)
            * r#in
            + 0.010559964726631393298)
            * r#in
            + 0.012039792768959435626,
        ((((2.5052108385441718775e-8 * r#in - 0.000062705427288760622094) * r#in
            + 0.00059458674042007375341)
            * r#in
            - 0.0016095979637646304313)
            * r#in
            + 0.0061039211560044893378)
            * r#in
            + 0.0038370059724226390893,
        (((((1.6059043836821614599e-10 * r#in + 0.000015401265401265401265) * r#in
            - 0.00016376804137220803887)
            * r#in
            + 0.00069084207973096861986)
            * r#in
            - 0.0012579159844784844785)
            * r#in
            + 0.0010898206731540064873)
            * r#in
            + 0.0032177478835464946576,
        ((((((7.6471637318198164759e-13 * r#in - 3.9851014346715404916e-6) * r#in
            + 0.000049255746366361445727)
            * r#in
            - 0.00024947258047043099953)
            * r#in
            + 0.00064513046951456342991)
            * r#in
            - 0.00076245135440323932387)
            * r#in
            + 0.000033530976880017885309)
            * r#in
            + 0.0017438262298340009980,
        (((((((2.8114572543455207632e-15 * r#in + 1.0914179173496789432e-6) * r#in
            - 0.000015303004486655377567)
            * r#in
            + 0.000090867107935219902229)
            * r#in
            - 0.00029133414466938067350)
            * r#in
            + 0.00051406605788341121363)
            * r#in
            - 0.00036307660358786885787)
            * r#in
            - 0.00031101086326318780412)
            * r#in
            + 0.00096472747321388644237,
        ((((((((8.2206352466243297170e-18 * r#in - 3.1239569599829868045e-7) * r#in
            + 4.8903045291975346210e-6)
            * r#in
            - 0.000033202652391372058698)
            * r#in
            + 0.00012645437628698076975)
            * r#in
            - 0.00028690924218514613987)
            * r#in
            + 0.00035764655430568632777)
            * r#in
            - 0.00010230378073700412687)
            * r#in
            - 0.00036942667800009661203)
            * r#in
            + 0.00054229262813129686486,
    ];
    // An odd polynomial in v (Shaw, equation 56).
    v * polynomial(&c, v * v)
}

/// Boost's `inverse_students_t`: an approximation to the Student's t quantile at `u`, with
/// `v = 1 - u`, and whether it is exact.
fn inverse_students_t(df: f64, mut u: f64, mut v: f64) -> (f64, bool) {
    let mut invert = false;
    if u > v {
        swap(&mut u, &mut v);
        invert = true;
    }
    let signed = |result: f64| if invert { -result } else { result };
    #[allow(clippy::cast_possible_truncation)]
    if df.floor() == df && df < 20.0 {
        // Integer degrees of freedom have closed forms (Shaw, equations 35-45).
        let tolerance = 2f64.powi((2 * DIGITS) / 3);
        match df as i32 {
            1 => {
                let result = if u == 0.5 {
                    0.0
                } else {
                    -(PI * u).cos() / (PI * u).sin()
                };
                return (signed(result), true);
            }
            2 => return (signed((2.0 * u - 1.0) / (2.0 * u * v).sqrt()), true),
            4 => {
                let alpha = 4.0 * u * v;
                let root_alpha = alpha.sqrt();
                let r = 4.0 * (root_alpha.acos() / 3.0).cos() / root_alpha;
                let x = (r - 4.0).sqrt();
                return (signed(if u - 0.5 < 0.0 { -x } else { x }), true);
            }
            6 => {
                if u < 1e-150 {
                    return (signed(inverse_students_t_hill(df, u)), false);
                }
                // Newton's method on Shaw's polynomial, from his seed value.
                let a = 4.0 * (u - u * u);
                let b = a.cbrt();
                let c = 0.85498797333834849467655443627193;
                let mut p = 6.0 * (1.0 + c * (1.0 / b - 1.0));
                for _ in 0..MAX_NEWTON_ITERATIONS {
                    let p2 = p * p;
                    let p4 = p2 * p2;
                    let p5 = p * p4;
                    let p0 = p;
                    p = 2.0 * (8.0 * a * p5 - 270.0 * p2 + 2187.0)
                        / (5.0 * (4.0 * a * p4 - 216.0 * p - 243.0));
                    let change = ((p - p0) / p).abs();
                    if change.partial_cmp(&tolerance) != Some(Ordering::Greater) {
                        break;
                    }
                }
                let p = (p - df).sqrt();
                return (signed(if u - 0.5 < 0.0 { -p } else { p }), false);
            }
            _ => {}
        }
    }
    let result = if df > 268435456.0 {
        return (signed(-erfcinv(2.0 * u) * SQRT_2), df >= 1e20);
    } else if df < 3.0 {
        // A roughly linear crossover between Shaw's tail and body series.
        let crossover = single(0.2742) - df * single(0.0242143);
        if u > crossover {
            inverse_students_t_body_series(df, u)
        } else {
            inverse_students_t_tail_series(df, u)
        }
    } else {
        // Hill's method except in the extreme tails, where the crossover is roughly
        // exponential in -df.
        let u_exp = frexp_exponent(u);
        if u > 0.0 && f64::from(u_exp) < df / single(0.654) {
            inverse_students_t_hill(df, u)
        } else {
            inverse_students_t_tail_series(df, u)
        }
    };
    (signed(result), false)
}

/// Boost's Student's t quantile for `double`, as SciPy's `stdtrit` computes it: an approximate
/// inverse polished by one Halley step. `None` stands for Boost's overflow error, which SciPy
/// reports as `+inf`.
pub(super) fn students_t_quantile(df: f64, p: f64) -> Option<f64> {
    if p == 0.0 || p == 1.0 {
        return None;
    }
    if p == 0.5 {
        return Some(0.0);
    }
    if df < 2.0 && df.floor() != df {
        // Non-integer df below 2: the incomplete beta inverse.
        let probability = if p > 0.5 { 1.0 - p } else { p };
        let Some((x, y)) = ibeta_inv(df / 2.0, 0.5, 2.0 * probability, true) else {
            return Some(f64::NAN);
        };
        if df * y > MAX_VALUE * x {
            return None;
        }
        let t = (df * y / x).sqrt();
        return Some(if p < 0.5 { -t } else { t });
    }
    let (p, invert) = if p > 0.5 { (1.0 - p, true) } else { (p, false) };
    let (t, exact) = inverse_students_t(df, p, 1.0 - p);
    if t == 0.0 || exact {
        return Some(if invert { -t } else { t });
    }
    // One Halley step in the incomplete beta formulation.
    let t2 = t * t;
    let xb = df / (df + t2);
    let y = t2 / (df + t2);
    let a = df / 2.0;
    if xb == 0.0 {
        return Some(t);
    }
    let (f0, f1) = if xb < y {
        (
            ibeta_imp(a, 0.5, xb, false),
            ibeta_imp_derivative(a, 0.5, xb),
        )
    } else {
        (ibeta_imp(0.5, a, y, true), ibeta_imp_derivative(0.5, a, y))
    };
    let p0 = f0 / 2.0 - p;
    let p1 = f1 * (y * xb * xb * xb / df).sqrt();
    let p2 = t * (df + 1.0) / (t * t + df);
    let t = t.abs() + p0 / (p1 + p0 * p2 / 2.0);
    Some(if invert { t } else { -t })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frexp_exponent_matches_c() {
        assert_eq!(frexp_exponent(1.0), 1);
        assert_eq!(frexp_exponent(0.5), 0);
        assert_eq!(frexp_exponent(0.75), 0);
        assert_eq!(frexp_exponent(1e-310), -1029);
    }

    #[test]
    fn float_distance_counts_representable_values() {
        assert_eq!(float_distance(1.0, 1.0f64.next_up()), 1.0);
        assert_eq!(float_distance(-0.0, 0.0), 0.0);
        assert_eq!(float_distance(0.0, f64::from_bits(3)), 3.0);
    }

    #[test]
    fn inverse_returns_both_complements() {
        let (x, y) = ibeta_inv(2.0, 3.0, 0.3, true).expect("in domain");
        assert!((x + y - 1.0).abs() < 1e-15);
        assert!(ibeta_inv(0.0, 3.0, 0.3, true).is_none());
    }
}
