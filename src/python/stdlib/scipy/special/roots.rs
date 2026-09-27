//! Boost's root finders (`boost/math/tools/toms748_solve.hpp`), as the Boost ports use them:
//! Alefeld, Potra and Shi's TOMS 748 algorithm, and the search that brackets a root from a guess
//! before handing it to TOMS 748.
//!
//! The ports repeat Boost's arithmetic step by step, so they evaluate the function at the same
//! points and return the same bracket, from which callers take the midpoint as SciPy does.
//! Boost reports a failure (ends that do not bracket a root, or no bracket within the iteration
//! limit) by raising an error, which SciPy's wrappers turn into NaN; here it is `None`. Every
//! function evaluation is a metered step, and an exhausted meter also gives `None`.

use super::meter;

/// Boost's `sign`: -1, 0 or 1, with 0 for both zeros.
pub(super) fn sign(x: f64) -> i32 {
    if x == 0.0 {
        0
    } else if x.is_sign_negative() {
        -1
    } else {
        1
    }
}

/// Boost's `eps_tolerance` for `bits` bits: the ends agree to a relative
/// `max(2^(1 - bits), 4 EPSILON)` of the smaller.
pub(super) fn eps_tolerance(bits: i32) -> impl Fn(f64, f64) -> bool {
    let eps = 2f64.powi(1 - bits).max(4.0 * f64::EPSILON);
    move |a: f64, b: f64| (a - b).abs() <= eps * a.abs().min(b.abs())
}

/// Boost's `equal_ceil`: the ends have the same ceiling, or agree to `2 EPSILON` relative to `b`.
pub(super) fn equal_ceil(a: f64, b: f64) -> bool {
    a.ceil() == b.ceil() || ((b - a) / b).abs() < f64::EPSILON * 2.0
}

/// Evaluate `f` at `x` as one metered step.
fn metered(f: &mut impl FnMut(f64) -> f64, x: f64) -> Option<f64> {
    meter::step().then(|| f(x))
}

/// The state TOMS 748 carries between steps: the bracket `[a, b]`, the end `d` most recently
/// discarded, and the function's values there.
struct Bracket {
    a: f64,
    b: f64,
    d: f64,
    fa: f64,
    fb: f64,
    fd: f64,
}

impl Bracket {
    /// Boost's `detail::bracket`: evaluate at `c`, nudged inside `[a, b]`, and keep the half
    /// where the sign changes. A zero at `c` collapses the bracket onto it.
    fn split(&mut self, f: &mut impl FnMut(f64) -> f64, c: f64) -> Option<()> {
        let tol = f64::EPSILON * 2.0;
        let c = if (self.b - self.a) < 2.0 * tol * self.a {
            self.a + (self.b - self.a) / 2.0
        } else if c <= self.a + self.a.abs() * tol {
            self.a + self.a.abs() * tol
        } else if c >= self.b - self.b.abs() * tol {
            self.b - self.b.abs() * tol
        } else {
            c
        };
        let fc = metered(f, c)?;
        if fc == 0.0 {
            self.a = c;
            self.fa = 0.0;
            self.d = 0.0;
            self.fd = 0.0;
        } else if sign(self.fa) * sign(fc) < 0 {
            self.d = self.b;
            self.fd = self.fb;
            self.b = c;
            self.fb = fc;
        } else {
            self.d = self.a;
            self.fd = self.fa;
            self.a = c;
            self.fa = fc;
        }
        Some(())
    }

    /// Whether any two of the four known values are within `32 MIN_POSITIVE` of each other,
    /// where Boost falls back from cubic to quadratic interpolation.
    fn values_too_close(&self, fe: f64) -> bool {
        let min_diff = f64::MIN_POSITIVE * 32.0;
        let (fa, fb, fd) = (self.fa, self.fb, self.fd);
        (fa - fb).abs() < min_diff
            || (fa - fd).abs() < min_diff
            || (fa - fe).abs() < min_diff
            || (fb - fd).abs() < min_diff
            || (fb - fe).abs() < min_diff
            || (fd - fe).abs() < min_diff
    }
}

/// Boost's `safe_div`: `num / denom`, or `r` where the quotient would overflow.
fn safe_div(num: f64, denom: f64, r: f64) -> f64 {
    if denom.abs() < 1.0 && (denom * f64::MAX).abs() <= num.abs() {
        return r;
    }
    num / denom
}

fn secant_interpolate(a: f64, b: f64, fa: f64, fb: f64) -> f64 {
    let tol = f64::EPSILON * 5.0;
    let c = a - (fa / (fb - fa)) * (b - a);
    if c <= a + a.abs() * tol || c >= b - b.abs() * tol {
        return (a + b) / 2.0;
    }
    c
}

/// Newton steps on the quadratic through `a`, `b` and `d`, falling back to the secant.
fn quadratic_interpolate(s: &Bracket, count: u32) -> f64 {
    let (a, b, d, fa, fb, fd) = (s.a, s.b, s.d, s.fa, s.fb, s.fd);
    let big_b = safe_div(fb - fa, b - a, f64::MAX);
    let big_a = safe_div(fd - fb, d - b, f64::MAX);
    let big_a = safe_div(big_a - big_b, d - a, 0.0);
    if big_a == 0.0 {
        return secant_interpolate(a, b, fa, fb);
    }
    let mut c = if sign(big_a) * sign(fa) > 0 { a } else { b };
    for _ in 0..count {
        c -= safe_div(
            fa + (big_b + big_a * (c - b)) * (c - a),
            big_b + big_a * (2.0 * c - a - b),
            1.0 + c - a,
        );
    }
    if c <= a || c >= b {
        c = secant_interpolate(a, b, fa, fb);
    }
    c
}

/// Inverse cubic interpolation through `a`, `b`, `d` and `e`, falling back to the quadratic.
fn cubic_interpolate(s: &Bracket, e: f64, fe: f64) -> f64 {
    let (a, b, d, fa, fb, fd) = (s.a, s.b, s.d, s.fa, s.fb, s.fd);
    let q11 = (d - e) * fd / (fe - fd);
    let q21 = (b - d) * fb / (fd - fb);
    let q31 = (a - b) * fa / (fb - fa);
    let d21 = (b - d) * fd / (fd - fb);
    let d31 = (a - b) * fb / (fb - fa);
    let q22 = (d21 - q11) * fb / (fe - fb);
    let q32 = (d31 - q21) * fa / (fd - fa);
    let d32 = (d31 - q21) * fd / (fd - fa);
    let q33 = (d32 - q22) * fa / (fe - fa);
    let c = q31 + q32 + q33 + a;
    if c <= a || c >= b {
        return quadratic_interpolate(s, 3);
    }
    c
}

/// Boost's `toms748_solve`: narrow `[ax, bx]`, whose ends have values `fax` and `fbx` of opposite
/// signs, until `tol` accepts it. `max_iter` bounds the evaluations and receives the number
/// used.
pub(super) fn toms748_solve(
    f: &mut impl FnMut(f64) -> f64,
    (ax, bx): (f64, f64),
    (fax, fbx): (f64, f64),
    tol: &impl Fn(f64, f64) -> bool,
    max_iter: &mut u64,
) -> Option<(f64, f64)> {
    if *max_iter == 0 {
        return Some((ax, bx));
    }
    let mut count = *max_iter;
    if ax >= bx {
        return None;
    }
    let mut s = Bracket {
        a: ax,
        b: bx,
        d: 0.0,
        fa: fax,
        fb: fbx,
        fd: 1e5,
    };
    if tol(s.a, s.b) || s.fa == 0.0 || s.fb == 0.0 {
        *max_iter = 0;
        if s.fa == 0.0 {
            s.b = s.a;
        } else if s.fb == 0.0 {
            s.a = s.b;
        }
        return Some((s.a, s.b));
    }
    if sign(s.fa) * sign(s.fb) > 0 {
        return None;
    }
    let (mut e, mut fe) = (1e5, 1e5);
    // Two steps of secant and quadratic interpolation start the iteration.
    let c = secant_interpolate(s.a, s.b, s.fa, s.fb);
    s.split(f, c)?;
    count -= 1;
    if count != 0 && s.fa != 0.0 && !tol(s.a, s.b) {
        let c = quadratic_interpolate(&s, 2);
        e = s.d;
        fe = s.fd;
        s.split(f, c)?;
        count -= 1;
    }
    while count != 0 && s.fa != 0.0 && !tol(s.a, s.b) {
        let (a0, b0) = (s.a, s.b);
        // Two interpolation steps, a double-length secant step, and a bisection if the bracket
        // did not at least halve.
        let c = if s.values_too_close(fe) {
            quadratic_interpolate(&s, 2)
        } else {
            cubic_interpolate(&s, e, fe)
        };
        e = s.d;
        fe = s.fd;
        s.split(f, c)?;
        count -= 1;
        if count == 0 || s.fa == 0.0 || tol(s.a, s.b) {
            break;
        }
        let c = if s.values_too_close(fe) {
            quadratic_interpolate(&s, 3)
        } else {
            cubic_interpolate(&s, e, fe)
        };
        s.split(f, c)?;
        count -= 1;
        if count == 0 || s.fa == 0.0 || tol(s.a, s.b) {
            break;
        }
        let (u, fu) = if s.fa.abs() < s.fb.abs() {
            (s.a, s.fa)
        } else {
            (s.b, s.fb)
        };
        let mut c = u - 2.0 * (fu / (s.fb - s.fa)) * (s.b - s.a);
        if (c - u).abs() > (s.b - s.a) / 2.0 {
            c = s.a + (s.b - s.a) / 2.0;
        }
        e = s.d;
        fe = s.fd;
        s.split(f, c)?;
        count -= 1;
        if count == 0 || s.fa == 0.0 || tol(s.a, s.b) {
            break;
        }
        if (s.b - s.a) < 0.5 * (b0 - a0) {
            continue;
        }
        e = s.d;
        fe = s.fd;
        s.split(f, s.a + (s.b - s.a) / 2.0)?;
        count -= 1;
    }
    *max_iter -= count;
    if s.fa == 0.0 {
        s.b = s.a;
    } else if s.fb == 0.0 {
        s.a = s.b;
    }
    Some((s.a, s.b))
}

/// Boost's `bracket_and_solve_root`: step from `guess` by `factor`, doubling it every so often,
/// until the function changes sign, then solve with TOMS 748. `rising` says whether the
/// function increases with its argument. `max_iter` bounds the evaluations and receives the
/// number used.
pub(super) fn bracket_and_solve_root(
    f: &mut impl FnMut(f64) -> f64,
    guess: f64,
    mut factor: f64,
    rising: bool,
    tol: &impl Fn(f64, f64) -> bool,
    max_iter: &mut u64,
) -> Option<(f64, f64)> {
    let (mut a, mut b) = (guess, guess);
    let mut fa = metered(f, a)?;
    let mut fb = fa;
    let mut count = *max_iter - 1;
    let mut step = 32;
    if (fa < 0.0) == (if guess < 0.0 { !rising } else { rising }) {
        // The root lies above the guess.
        while sign(fb) == sign(fa) {
            if count == 0 {
                return None;
            }
            if (*max_iter - count).is_multiple_of(step) {
                factor *= 2.0;
                if step > 1 {
                    step /= 2;
                }
            }
            a = b;
            fa = fb;
            b *= factor;
            fb = metered(f, b)?;
            count -= 1;
        }
    } else {
        while sign(fb) == sign(fa) {
            if a.abs() < f64::MIN_POSITIVE {
                // The root is at zero, or too close to it to resolve.
                *max_iter = *max_iter - count + 1;
                return Some(if a > 0.0 { (0.0, a) } else { (a, 0.0) });
            }
            if count == 0 {
                return None;
            }
            if (*max_iter - count).is_multiple_of(step) {
                factor *= 2.0;
                if step > 1 {
                    step /= 2;
                }
            }
            b = a;
            fb = fa;
            a /= factor;
            fa = metered(f, a)?;
            count -= 1;
        }
    }
    *max_iter = *max_iter - count + 1;
    let ends = if a < 0.0 { (b, a) } else { (a, b) };
    let values = if a < 0.0 { (fb, fa) } else { (fa, fb) };
    let solution = toms748_solve(f, ends, values, tol, &mut count)?;
    *max_iter += count;
    Some(solution)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toms748_converges_to_the_tolerance() {
        let tol = eps_tolerance(53);
        let mut iterations = 100;
        let mut f = |x: f64| x * x * x - 2.0;
        let (a, b) = toms748_solve(&mut f, (0.0, 2.0), (-2.0, 6.0), &tol, &mut iterations)
            .expect("the ends bracket the root");
        let root = 2f64.cbrt();
        assert!(a <= root && root <= b && tol(a, b), "{a} {b}");
        assert!(iterations < 20, "{iterations}");
    }

    #[test]
    fn toms748_rejects_ends_without_a_sign_change() {
        let mut f = |x: f64| x + 10.0;
        let tol = eps_tolerance(53);
        assert!(toms748_solve(&mut f, (0.0, 1.0), (10.0, 11.0), &tol, &mut 50).is_none());
        assert!(toms748_solve(&mut f, (1.0, 0.0), (11.0, 10.0), &tol, &mut 50).is_none());
    }

    #[test]
    fn bracketing_searches_both_directions() {
        let tol = eps_tolerance(53);
        for (guess, rising) in [(1.0, true), (1e6, true), (1.0, false), (1e-6, false)] {
            let mut iterations = 400;
            let root = 1000.0;
            let mut f = |x: f64| if rising { x - root } else { root - x };
            let (a, b) = bracket_and_solve_root(&mut f, guess, 2.0, rising, &tol, &mut iterations)
                .expect("the root can be bracketed");
            assert!(
                ((a + b) / 2.0 - root).abs() <= 1e-12 * root,
                "{guess} {rising}"
            );
        }
        // A root at zero ends the downward search.
        let mut f = |x: f64| x;
        let mut iterations = 2000;
        let (a, b) = bracket_and_solve_root(&mut f, 1.0, 2.0, true, &tol, &mut iterations)
            .expect("zero is reached");
        assert_eq!(a, 0.0);
        assert!(b > 0.0 && b < f64::MIN_POSITIVE);
    }
}
