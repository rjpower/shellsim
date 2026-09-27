//! The numeric core of `numpy.fft`: an O(n log n) discrete Fourier transform for every length.
//!
//! Powers of two run through an iterative radix-2 Cooley–Tukey transform (Cooley & Tukey,
//! "An Algorithm for the Machine Calculation of Complex Fourier Series", 1965). Every other
//! length runs through Bluestein's chirp-z transform (Bluestein, 1970), which rewrites the
//! length-`n` DFT as a length-`m` convolution (`m` a power of two, `m >= 2n-1`) computed with
//! two more radix-2 transforms, so the whole algorithm stays O(n log n) regardless of length.
//!
//! Twiddle factors and chirp factors both come from [`cos_sin_turn`], which folds the turn
//! fraction into the first quadrant before calling `sin`/`cos`, so the trigonometric argument
//! never exceeds pi/2. That keeps rounding close to the last bit no matter how large `k` or the
//! transform length are, and it gives exact values (not just close ones) at the quarter-turn
//! points every transform relies on for cancellation.
//!
//! Every value here is `f64`; [`super`] rounds to `f32` only when the caller's dtype calls for
//! single precision. Callers charge CPU and reserve memory ([`transform_cost`],
//! [`working_size`]) before calling [`dft`], as the crate's resource-metering rules require.

use crate::python::stdlib::numpy::element::C128;

/// cos(2*pi*numerator/denominator) and sin(2*pi*numerator/denominator) for any numerator and
/// any positive denominator, folded by the turn's own symmetry into the first quadrant so the
/// underlying `sin`/`cos` call never sees an argument beyond pi/2.
///
/// The fold is two reflections done on the *fraction* `k/d` (not the angle), each an exact or
/// near-exact subtraction of two similarly scaled values, so it adds negligible error beyond
/// the single division that turns the integer ratio into a fraction:
/// 1. reflect `k` across the half turn (`k` vs `d-k`) so `k <= d/2`, flipping the sign of sin;
/// 2. reflect the resulting fraction across the quarter turn (`f` vs `0.5-f`) so `f <= 0.25`,
///    flipping the sign of cos.
pub(super) fn cos_sin_turn(numerator: u64, denominator: u64) -> (f64, f64) {
    debug_assert!(denominator > 0);
    let mut k = numerator % denominator;
    let mut sin_sign = 1.0f64;
    if 2 * u128::from(k) > u128::from(denominator) {
        k = denominator - k;
        sin_sign = -1.0;
    }
    let mut fraction = k as f64 / denominator as f64;
    let mut cos_sign = 1.0f64;
    if fraction > 0.25 {
        fraction = 0.5 - fraction;
        cos_sign = -1.0;
    }
    let (sin, cos) = (2.0 * std::f64::consts::PI * fraction).sin_cos();
    (cos_sign * cos, sin_sign * sin)
}

fn add(a: C128, b: C128) -> C128 {
    C128 {
        re: a.re + b.re,
        im: a.im + b.im,
    }
}

fn sub(a: C128, b: C128) -> C128 {
    C128 {
        re: a.re - b.re,
        im: a.im - b.im,
    }
}

fn mul(a: C128, b: C128) -> C128 {
    C128 {
        re: a.re * b.re - a.im * b.im,
        im: a.re * b.im + a.im * b.re,
    }
}

fn conj(a: C128) -> C128 {
    C128 {
        re: a.re,
        im: -a.im,
    }
}

/// In-place iterative radix-2 Cooley–Tukey transform. `data.len()` must be a power of two.
/// Both directions are unnormalized (neither divides by `n`); [`super`] applies the `norm=`
/// scale once, after the transform.
fn radix2(data: &mut [C128], inverse: bool) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two());
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if j > i {
            data.swap(i, j);
        }
    }
    let mut size = 2usize;
    while size <= n {
        let half = size / 2;
        let twiddles: Vec<C128> = (0..half)
            .map(|k| {
                let (cos, sin) = cos_sin_turn(k as u64, size as u64);
                C128 {
                    re: cos,
                    im: if inverse { sin } else { -sin },
                }
            })
            .collect();
        let mut start = 0;
        while start < n {
            for k in 0..half {
                let a = data[start + k];
                let b = mul(data[start + k + half], twiddles[k]);
                data[start + k] = add(a, b);
                data[start + k + half] = sub(a, b);
            }
            start += size;
        }
        size *= 2;
    }
}

/// `k*k mod 2n`, computed in `u128` so squaring never overflows regardless of `n`.
fn chirp_numerator(k: usize, n: usize) -> u64 {
    let modulus = 2u128 * n as u128;
    let k = k as u128 % modulus;
    ((k * k) % modulus) as u64
}

/// Bluestein's chirp-z transform: `X[k] = chirp[k] * conv(x*chirp, conj(chirp))[k]`, where
/// `chirp[k] = exp(sign*i*pi*k^2/n)` and `conv` is a length-`n` linear convolution computed as
/// a circular convolution of the next power of two at least `2n-1`, via two radix-2 transforms.
/// See Bluestein (1970) or Rabiner & Gold, "Theory and Application of Digital Signal
/// Processing" (1975), section 6.10, for the derivation.
fn bluestein(data: &mut [C128], inverse: bool) {
    let n = data.len();
    let sign = if inverse { 1.0 } else { -1.0 };
    let chirp: Vec<C128> = (0..n)
        .map(|k| {
            let (cos, sin) = cos_sin_turn(chirp_numerator(k, n), 2 * n as u64);
            C128 {
                re: cos,
                im: sign * sin,
            }
        })
        .collect();
    let m = (2 * n - 1).next_power_of_two();
    let mut a = vec![C128::default(); m];
    let mut b = vec![C128::default(); m];
    b[0] = conj(chirp[0]);
    for k in 0..n {
        a[k] = mul(data[k], chirp[k]);
        if k > 0 {
            let c = conj(chirp[k]);
            b[k] = c;
            b[m - k] = c;
        }
    }
    radix2(&mut a, false);
    radix2(&mut b, false);
    for i in 0..m {
        a[i] = mul(a[i], b[i]);
    }
    radix2(&mut a, true);
    let scale = 1.0 / m as f64;
    for (k, slot) in data.iter_mut().enumerate() {
        let scaled = C128 {
            re: a[k].re * scale,
            im: a[k].im * scale,
        };
        *slot = mul(scaled, chirp[k]);
    }
}

/// The unnormalized O(n log n) DFT of `data`, in place: forward (`inverse=false`) computes
/// `X[k] = sum_j data[j] * exp(-2*pi*i*j*k/n)`; `inverse=true` computes the same sum with a
/// `+` sign and, like `radix2`/`bluestein`, does not divide by `n`.
pub(super) fn dft(data: &mut [C128], inverse: bool) {
    if data.len().is_power_of_two() {
        radix2(data, inverse);
    } else {
        bluestein(data, inverse);
    }
}

/// The size of the radix-2 transform(s) [`dft`] actually runs for a length-`n` input: `n`
/// itself when it is already a power of two, otherwise Bluestein's padded convolution size.
pub(super) fn working_size(n: usize) -> usize {
    if n.is_power_of_two() {
        n
    } else {
        (2 * n - 1).next_power_of_two()
    }
}

/// CPU units for one length-`n` transform, proportional to `n*log2(n)` of the actual radix-2
/// work `dft` performs (which for Bluestein is `working_size(n)`, not `n`). The constant is a
/// conservative, order-of-magnitude estimate of the butterfly and trigonometric work per
/// element, not a precise flop count; see `AGENTS.md` on resource costs.
pub(super) fn transform_cost(n: usize) -> u64 {
    let working = working_size(n) as u64;
    let bits = working.max(2).ilog2() as u64;
    working.saturating_mul(bits).saturating_mul(4) + working
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn direct_dft(input: &[C128], inverse: bool) -> Vec<C128> {
        let n = input.len();
        let sign = if inverse { 1.0 } else { -1.0 };
        (0..n)
            .map(|k| {
                input
                    .iter()
                    .enumerate()
                    .fold(C128::default(), |acc, (j, x)| {
                        let angle = sign * 2.0 * PI * (j * k) as f64 / n as f64;
                        add(
                            acc,
                            mul(
                                *x,
                                C128 {
                                    re: angle.cos(),
                                    im: angle.sin(),
                                },
                            ),
                        )
                    })
            })
            .collect()
    }

    fn close(a: &[C128], b: &[C128], atol: f64) {
        for (x, y) in a.iter().zip(b) {
            assert!(
                (x.re - y.re).abs() < atol && (x.im - y.im).abs() < atol,
                "{x:?} != {y:?}"
            );
        }
    }

    #[test]
    fn cos_sin_turn_matches_cardinal_and_diagonal_angles() {
        assert_eq!(cos_sin_turn(0, 4), (1.0, 0.0));
        let (c, s) = cos_sin_turn(1, 4);
        assert!(c.abs() < 1e-15 && (s - 1.0).abs() < 1e-15);
        assert_eq!(cos_sin_turn(2, 4), (-1.0, 0.0));
        let (c, s) = cos_sin_turn(3, 4);
        assert!(c.abs() < 1e-15 && (s + 1.0).abs() < 1e-15);
        let (c, s) = cos_sin_turn(1, 8);
        let expected = std::f64::consts::FRAC_1_SQRT_2;
        assert!((c - expected).abs() < 1e-15 && (s - expected).abs() < 1e-15);
    }

    #[test]
    fn radix2_matches_direct_dft_for_powers_of_two() {
        for n in [1usize, 2, 4, 8, 16, 32] {
            let input: Vec<C128> = (0..n)
                .map(|k| C128 {
                    re: (k as f64).sin(),
                    im: (k as f64 * 0.3).cos(),
                })
                .collect();
            for inverse in [false, true] {
                let mut got = input.clone();
                dft(&mut got, inverse);
                close(&got, &direct_dft(&input, inverse), 1e-9);
            }
        }
    }

    #[test]
    fn bluestein_matches_direct_dft_for_non_power_of_two_lengths() {
        for n in [3usize, 5, 6, 7, 11, 13, 17, 100] {
            let input: Vec<C128> = (0..n)
                .map(|k| C128 {
                    re: (k as f64 * 0.7).cos(),
                    im: (k as f64 * 0.4).sin(),
                })
                .collect();
            for inverse in [false, true] {
                let mut got = input.clone();
                dft(&mut got, inverse);
                close(&got, &direct_dft(&input, inverse), 1e-7);
            }
        }
    }

    #[test]
    fn forward_then_inverse_recovers_the_input_scaled_by_n() {
        for n in [1usize, 5, 8, 13] {
            let input: Vec<C128> = (0..n)
                .map(|k| C128 {
                    re: k as f64,
                    im: -(k as f64),
                })
                .collect();
            let mut data = input.clone();
            dft(&mut data, false);
            dft(&mut data, true);
            let scaled: Vec<C128> = input
                .iter()
                .map(|v| C128 {
                    re: v.re * n as f64,
                    im: v.im * n as f64,
                })
                .collect();
            close(&data, &scaled, 1e-6);
        }
    }
}
