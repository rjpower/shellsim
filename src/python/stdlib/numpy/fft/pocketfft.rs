//! A port of pocketfft's one-dimensional transforms, the kernels behind NumPy's `numpy.fft`.
//!
//! The source is `pocketfft_hdronly.h` at commit 33ae5dc, the copy NumPy 2.5 vendors. The port
//! keeps pocketfft's arithmetic operation for operation so that results match NumPy bit for
//! bit on the same `libm`: every butterfly adds and multiplies in the same order, the radix
//! constants are the same doubles, and a plan picks the same factorization and the same choice
//! between FFTPACK passes and Bluestein's algorithm. Plans are generic over pocketfft's `T0`,
//! `f64` or `f32`; as in pocketfft, `f32` plans still compute twiddle factors in double
//! precision and round them.
//!
//! Two details carry the exactness. Twiddle factors come from `sincos_2pibyn`, which computes
//! its base angle `π/(4n)` in x87 extended precision before rounding it to double;
//! [`quarter_pi_over`] reproduces that double rounding with integer arithmetic. And the complex
//! passes for radices 2, 3, 4, 5, 7, 8 and 11 share one loop, [`pass`], around per-radix
//! butterflies; pocketfft writes each pass out separately, but the operations per element are
//! the same.
//!
//! Callers bound the length: plans allocate `O(n)` twiddles and scratch, and every index below
//! stays under `8 * n`, which cannot overflow for an allocatable length.

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub};

/// A plan's precision, pocketfft's `T0`.
pub(super) trait Real:
    Copy
    + Debug
    + Default
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + MulAssign
{
    /// Round a double to this precision, as C's conversion does.
    fn from_f64(value: f64) -> Self;

    fn to_f64(self) -> f64;
}

impl Real for f64 {
    fn from_f64(value: f64) -> Self {
        value
    }

    fn to_f64(self) -> f64 {
        self
    }
}

impl Real for f32 {
    fn from_f64(value: f64) -> Self {
        value as f32
    }

    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

/// A complex value with pocketfft's `cmplx` arithmetic.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Cmplx<T> {
    pub r: T,
    pub i: T,
}

const fn cmplx<T>(r: T, i: T) -> Cmplx<T> {
    Cmplx { r, i }
}

impl<T: Real> Add for Cmplx<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        cmplx(self.r + other.r, self.i + other.i)
    }
}

impl<T: Real> Sub for Cmplx<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        cmplx(self.r - other.r, self.i - other.i)
    }
}

impl<T: Real> Mul<T> for Cmplx<T> {
    type Output = Self;

    fn mul(self, factor: T) -> Self {
        cmplx(self.r * factor, self.i * factor)
    }
}

/// A constant rounded to the plan's precision.
fn constant<T: Real>(value: f64) -> T {
    T::from_f64(value)
}

fn one<T: Real>() -> Cmplx<T> {
    cmplx(constant(1.0), T::default())
}

/// `PM(a, b, c, d)`: `(c + d, c - d)`.
fn pm<T: Add<Output = T> + Sub<Output = T> + Copy>(c: T, d: T) -> (T, T) {
    (c + d, c - d)
}

/// `special_mul<fwd>`: multiply by `w`, or by its conjugate in the forward direction.
fn special_mul<T: Real>(v: Cmplx<T>, w: Cmplx<T>, forward: bool) -> Cmplx<T> {
    if forward {
        cmplx(v.r * w.r + v.i * w.i, v.i * w.r - v.r * w.i)
    } else {
        cmplx(v.r * w.r - v.i * w.i, v.r * w.i + v.i * w.r)
    }
}

/// `ROTX90<fwd>`: multiply by `-i` forward and by `i` backward.
fn rotx90<T: Real>(a: Cmplx<T>, forward: bool) -> Cmplx<T> {
    if forward {
        cmplx(a.i, -a.r)
    } else {
        cmplx(-a.i, a.r)
    }
}

const HSQT2: f64 = std::f64::consts::FRAC_1_SQRT_2;
const SQRT2: f64 = std::f64::consts::SQRT_2;
const TAUR: f64 = -0.5;
const TAUI: f64 = 0.8660254037844386;

fn rotx45<T: Real>(a: Cmplx<T>, forward: bool) -> Cmplx<T> {
    let hsqt2 = constant::<T>(HSQT2);
    if forward {
        cmplx(hsqt2 * (a.r + a.i), hsqt2 * (a.i - a.r))
    } else {
        cmplx(hsqt2 * (a.r - a.i), hsqt2 * (a.i + a.r))
    }
}

fn rotx135<T: Real>(a: Cmplx<T>, forward: bool) -> Cmplx<T> {
    let hsqt2 = constant::<T>(HSQT2);
    if forward {
        cmplx(hsqt2 * (a.i - a.r), hsqt2 * (-a.r - a.i))
    } else {
        cmplx(hsqt2 * (-a.r - a.i), hsqt2 * (a.r - a.i))
    }
}

/// `double(0.25L * pi / n)` as x87 extended precision evaluates it: `π` rounded to a 64-bit
/// significand, divided by `n` and rounded to 64 bits, then rounded again to 53 bits. Rounding
/// the exact quotient once gives a different double for about a quarter of all lengths.
///
/// ```text
/// quarter_pi_over(3) == f64::from_bits(0x3fd0c152382d7366)  // one ulp above π/4 / 3.0
/// ```
fn quarter_pi_over(n: usize) -> f64 {
    // π as an x87 long double is 0xC90FDAA22168C235 × 2^-62, so π/4 is that significand ×
    // 2^-64, and π/(4n) is `numerator / n × 2^-128`.
    const PI_SIGNIFICAND: u128 = 0xC90F_DAA2_2168_C235;
    let numerator = PI_SIGNIFICAND << 64;
    let divisor = n as u128;
    let (extended, first) =
        round_to_bits(numerator / divisor, !numerator.is_multiple_of(divisor), 64);
    let (double, second) = round_to_bits(extended, false, 53);
    let exponent = (first + second) as i32 - 128;
    double as f64 * f64::from_bits(((exponent + 1023) as u64) << 52)
}

/// Round `value`, followed by further nonzero bits when `sticky`, to `bits` significant bits
/// with ties to even. Returns the rounded significand and how many bits were dropped.
fn round_to_bits(value: u128, sticky: bool, bits: u32) -> (u128, u32) {
    let length = 128 - value.leading_zeros();
    if length <= bits {
        return (value, 0);
    }
    let dropped = length - bits;
    let kept = value >> dropped;
    let rest = value & ((1u128 << dropped) - 1);
    let half = 1u128 << (dropped - 1);
    let round_up = rest > half || (rest == half && (sticky || kept & 1 == 1));
    let kept = kept + u128::from(round_up);
    if kept >> bits != 0 {
        (kept >> 1, dropped + 1)
    } else {
        (kept, dropped)
    }
}

/// `sincos_2pibyn`: `e^(2πik/n)` for every `k`, from two short tables whose product gives each
/// entry.
struct Twiddles {
    n: usize,
    mask: usize,
    shift: u32,
    v1: Vec<Cmplx<f64>>,
    v2: Vec<Cmplx<f64>>,
}

impl Twiddles {
    fn new(n: usize) -> Self {
        let angle = quarter_pi_over(n);
        let nval = (n + 2) / 2;
        let mut shift = 1;
        while (1usize << shift) * (1usize << shift) < nval {
            shift += 1;
        }
        let mask = (1usize << shift) - 1;
        let mut v1 = vec![one(); mask + 1];
        for (index, value) in v1.iter_mut().enumerate().skip(1) {
            *value = Self::calc(index, n, angle);
        }
        let mut v2 = vec![one(); (nval + mask) / (mask + 1)];
        for (index, value) in v2.iter_mut().enumerate().skip(1) {
            *value = Self::calc(index * (mask + 1), n, angle);
        }
        Self {
            n,
            mask,
            shift,
            v1,
            v2,
        }
    }

    /// `e^(2πix/n)` from the octant that contains it, so each sine and cosine argument stays
    /// within `[0, π/4]`.
    fn calc(x: usize, n: usize, angle: f64) -> Cmplx<f64> {
        let at = |x: usize| x as f64 * angle;
        let mut x = x << 3;
        if x < 4 * n {
            if x < 2 * n {
                if x < n {
                    return cmplx(at(x).cos(), at(x).sin());
                }
                return cmplx(at(2 * n - x).sin(), at(2 * n - x).cos());
            }
            x -= 2 * n;
            if x < n {
                return cmplx(-at(x).sin(), at(x).cos());
            }
            return cmplx(-at(2 * n - x).cos(), at(2 * n - x).sin());
        }
        x = 8 * n - x;
        if x < 2 * n {
            if x < n {
                return cmplx(at(x).cos(), -at(x).sin());
            }
            return cmplx(at(2 * n - x).sin(), -at(2 * n - x).cos());
        }
        x -= 2 * n;
        if x < n {
            return cmplx(-at(x).sin(), -at(x).cos());
        }
        cmplx(-at(2 * n - x).cos(), -at(2 * n - x).sin())
    }

    /// `e^(2πi·index/n)`, computed in double precision and rounded to `T`.
    fn get<T: Real>(&self, index: usize) -> Cmplx<T> {
        let (index, conjugate) = if 2 * index <= self.n {
            (index, false)
        } else {
            (self.n - index, true)
        };
        let x1 = self.v1[index & self.mask];
        let x2 = self.v2[index >> self.shift];
        let imaginary = T::from_f64(x1.r * x2.i + x1.i * x2.r);
        cmplx(
            T::from_f64(x1.r * x2.r - x1.i * x2.i),
            if conjugate { -imaginary } else { imaginary },
        )
    }
}

fn largest_prime_factor(mut n: usize) -> usize {
    let mut result = 1;
    while n & 1 == 0 {
        result = 2;
        n >>= 1;
    }
    let mut x = 3;
    while x * x <= n {
        while n.is_multiple_of(x) {
            result = x;
            n /= x;
        }
        x += 2;
    }
    if n > 1 {
        result = n;
    }
    result
}

/// pocketfft's operation-count estimate, which penalizes factors without a dedicated pass.
fn cost_guess(n: usize) -> f64 {
    const PENALTY: f64 = 1.1;
    let original = n;
    let mut n = n;
    let mut result = 0.0;
    while n & 1 == 0 {
        result += 2.0;
        n >>= 1;
    }
    let mut x = 3;
    while x * x <= n {
        while n.is_multiple_of(x) {
            result += if x <= 5 { x as f64 } else { PENALTY * x as f64 };
            n /= x;
        }
        x += 2;
    }
    if n > 1 {
        result += if n <= 5 { n as f64 } else { PENALTY * n as f64 };
    }
    result * original as f64
}

/// The smallest product of 2, 3, 5, 7 and 11 that is at least `n`.
fn good_size_cmplx(n: usize) -> usize {
    if n <= 12 {
        return n;
    }
    let mut best = 2 * n;
    let mut f11 = 1;
    while f11 < best {
        let mut f117 = f11;
        while f117 < best {
            let mut f1175 = f117;
            while f1175 < best {
                let mut x = f1175;
                while x < n {
                    x *= 2;
                }
                loop {
                    if x < n {
                        x *= 3;
                    } else if x > n {
                        best = best.min(x);
                        if x & 1 == 1 {
                            break;
                        }
                        x >>= 1;
                    } else {
                        return n;
                    }
                }
                f1175 *= 5;
            }
            f117 *= 7;
        }
        f11 *= 11;
    }
    best
}

/// The padded length Bluestein's algorithm would use for `length`, when pocketfft prefers it to
/// the FFTPACK passes. A real transform's FFTPACK cost counts half.
fn bluestein_length(length: usize, real: bool) -> Option<usize> {
    let largest = if length < 50 {
        0
    } else {
        largest_prime_factor(length)
    };
    if largest * largest <= length {
        return None;
    }
    let direct = cost_guess(length) * if real { 0.5 } else { 1.0 };
    let padded = good_size_cmplx(2 * length - 1);
    let bluestein = 2.0 * cost_guess(padded) * 1.5;
    (bluestein < direct).then_some(padded)
}

/// The length of the complex transforms a plan for `length` runs: `length` itself, or the
/// padded length of Bluestein's algorithm.
pub(super) fn plan_length(length: usize, real: bool) -> usize {
    bluestein_length(length, real).unwrap_or(length)
}

/// Estimated operations for one transform of `length`, from pocketfft's cost model.
pub(super) fn transform_work(length: usize, real: bool) -> u64 {
    match bluestein_length(length, real) {
        Some(padded) => (2.0 * cost_guess(padded)) as u64 + 4 * padded as u64,
        None => cost_guess(length) as u64,
    }
}

/// One factor of a complex plan and its twiddle factors.
struct ComplexFactor<T> {
    radix: usize,
    twiddles: Vec<Cmplx<T>>,
    /// The `radix` roots of unity the generic pass needs, for radices above 11.
    roots: Vec<Cmplx<T>>,
}

/// `cfftp`: FFTPACK's mixed-radix complex transform.
struct Cfftp<T> {
    length: usize,
    factors: Vec<ComplexFactor<T>>,
}

impl<T: Real> Cfftp<T> {
    fn new(length: usize) -> Self {
        if length == 1 {
            return Self {
                length,
                factors: Vec::new(),
            };
        }
        let mut radices = Vec::new();
        let mut rest = length;
        while rest & 7 == 0 {
            radices.push(8);
            rest >>= 3;
        }
        while rest & 3 == 0 {
            radices.push(4);
            rest >>= 2;
        }
        if rest & 1 == 0 {
            rest >>= 1;
            // A factor of 2 goes first.
            radices.push(2);
            let last = radices.len() - 1;
            radices.swap(0, last);
        }
        push_odd_factors(&mut radices, rest);

        let twiddle = Twiddles::new(length);
        let mut l1 = 1;
        let factors = radices
            .into_iter()
            .map(|radix| {
                let ido = length / (l1 * radix);
                let mut twiddles = Vec::with_capacity((radix - 1) * (ido - 1));
                for j in 1..radix {
                    for i in 1..ido {
                        twiddles.push(twiddle.get(j * l1 * i));
                    }
                }
                let roots = if radix > 11 {
                    (0..radix).map(|j| twiddle.get(j * l1 * ido)).collect()
                } else {
                    Vec::new()
                };
                l1 *= radix;
                ComplexFactor {
                    radix,
                    twiddles,
                    roots,
                }
            })
            .collect();
        Self { length, factors }
    }

    /// `pass_all`: transform `c` in place and scale it by `fct`.
    fn exec(&self, c: &mut [Cmplx<T>], fct: T, forward: bool) {
        if self.length == 1 {
            c[0] = c[0] * fct;
            return;
        }
        let mut ch = vec![Cmplx::default(); self.length];
        let mut in_c = true;
        let mut l1 = 1;
        for factor in &self.factors {
            let radix = factor.radix;
            let ido = self.length / (l1 * radix);
            let (p1, p2): (&mut [Cmplx<T>], &mut [Cmplx<T>]) = if in_c {
                (&mut *c, &mut ch)
            } else {
                (&mut ch, &mut *c)
            };
            if radix <= 11 {
                pass(radix, ido, l1, p1, p2, &factor.twiddles, forward);
            } else {
                // The generic pass leaves its result in its input buffer.
                passg(
                    ido,
                    radix,
                    l1,
                    p1,
                    p2,
                    &factor.twiddles,
                    &factor.roots,
                    forward,
                );
                in_c = !in_c;
            }
            in_c = !in_c;
            l1 *= radix;
        }
        let unscaled = fct == constant(1.0);
        if !in_c {
            if unscaled {
                c.copy_from_slice(&ch);
            } else {
                for (target, value) in c.iter_mut().zip(&ch) {
                    *target = *value * fct;
                }
            }
        } else if !unscaled {
            for value in c.iter_mut() {
                *value = *value * fct;
            }
        }
    }
}

/// Append the odd prime factors of `rest`, smallest first.
fn push_odd_factors(radices: &mut Vec<usize>, mut rest: usize) {
    let mut divisor = 3;
    while divisor * divisor <= rest {
        while rest.is_multiple_of(divisor) {
            radices.push(divisor);
            rest /= divisor;
        }
        divisor += 2;
    }
    if rest > 1 {
        radices.push(rest);
    }
}

/// `cos(2πm/p)` and `sin(2πm/p)` for `m = 1..=(p-1)/2`: pocketfft's constants for the odd
/// radices with dedicated passes.
const RADIX3: [(f64, f64); 1] = [(TAUR, TAUI)];
const RADIX5: [(f64, f64); 2] = [
    (0.30901699437494745, 0.9510565162951535),
    (-0.8090169943749475, 0.5877852522924731),
];
const RADIX7: [(f64, f64); 3] = [
    (0.6234898018587335, 0.7818314824680298),
    (-0.2225209339563144, 0.9749279121818236),
    (-0.9009688679024191, 0.4338837391175581),
];
const RADIX11: [(f64, f64); 5] = [
    (0.8412535328311812, 0.5406408174555976),
    (0.41541501300188644, 0.9096319953545183),
    (-0.14231483827328514, 0.9898214418809327),
    (-0.6548607339452851, 0.7557495743542583),
    (-0.9594929736144974, 0.28173255684142967),
];

/// The complex passes `pass2` through `pass11`: a radix-`radix` butterfly over each group of
/// inputs, with the outputs of every group but the first multiplied by twiddle factors.
fn pass<T: Real>(
    radix: usize,
    ido: usize,
    l1: usize,
    cc: &[Cmplx<T>],
    ch: &mut [Cmplx<T>],
    wa: &[Cmplx<T>],
    forward: bool,
) {
    let odd = match radix {
        3 => Some(OddButterfly::new(&RADIX3, forward)),
        5 => Some(OddButterfly::new(&RADIX5, forward)),
        7 => Some(OddButterfly::new(&RADIX7, forward)),
        11 => Some(OddButterfly::new(&RADIX11, forward)),
        _ => None,
    };
    let mut input = [Cmplx::default(); 11];
    let mut output = [Cmplx::default(); 11];
    for k in 0..l1 {
        for i in 0..ido {
            for (j, value) in input[..radix].iter_mut().enumerate() {
                *value = cc[i + ido * (j + radix * k)];
            }
            let input = &input[..radix];
            let output = &mut output[..radix];
            match (&odd, radix) {
                (Some(butterfly), _) => butterfly.apply(input, output),
                (None, 2) => (output[0], output[1]) = pm(input[0], input[1]),
                (None, 4) => radix4(input, output, forward),
                (None, 8) => radix8(input, output, forward),
                _ => unreachable!("radix {radix} has no dedicated pass"),
            }
            ch[i + ido * k] = output[0];
            for (u, value) in output.iter().enumerate().skip(1) {
                ch[i + ido * (k + l1 * u)] = if i == 0 {
                    *value
                } else {
                    special_mul(*value, wa[i - 1 + (u - 1) * (ido - 1)], forward)
                };
            }
        }
    }
}

fn radix4<T: Real>(input: &[Cmplx<T>], output: &mut [Cmplx<T>], forward: bool) {
    let (t2, t1) = pm(input[0], input[2]);
    let (t3, t4) = pm(input[1], input[3]);
    let t4 = rotx90(t4, forward);
    (output[0], output[2]) = pm(t2, t3);
    (output[1], output[3]) = pm(t1, t4);
}

fn radix8<T: Real>(input: &[Cmplx<T>], output: &mut [Cmplx<T>], forward: bool) {
    let (a1, a5) = pm(input[1], input[5]);
    let (a3, a7) = pm(input[3], input[7]);
    let a7 = rotx90(a7, forward);
    let (a1, a3) = pm(a1, a3);
    let a3 = rotx90(a3, forward);
    let (a5, a7) = pm(a5, a7);
    let a5 = rotx45(a5, forward);
    let a7 = rotx135(a7, forward);
    let (a0, a4) = pm(input[0], input[4]);
    let (a2, a6) = pm(input[2], input[6]);
    let (a0, a2) = pm(a0, a2);
    (output[0], output[4]) = pm(a0, a1);
    (output[2], output[6]) = pm(a2, a3);
    let a6 = rotx90(a6, forward);
    let (a4, a6) = pm(a4, a6);
    (output[1], output[5]) = pm(a4, a5);
    (output[3], output[7]) = pm(a6, a7);
}

/// The butterfly of `pass3`, `pass5`, `pass7` and `pass11`. Output `u` and `p - u` combine the
/// sums and differences of opposite inputs with `cos(2πuj/p)` and `±sin(2πuj/p)`, accumulated
/// left to right in `j` as pocketfft's macros spell them out.
struct OddButterfly<T> {
    /// `(cos, sin)` for each `(u, j)`, with the direction's sign applied to `sin`.
    coefficients: [[(T, T); 5]; 5],
    half: usize,
}

impl<T: Real> OddButterfly<T> {
    fn new(constants: &[(f64, f64)], forward: bool) -> Self {
        let half = constants.len();
        let radix = 2 * half + 1;
        let mut coefficients = [[(T::default(), T::default()); 5]; 5];
        for u in 1..=half {
            for j in 1..=half {
                let m = u * j % radix;
                let (index, negate) = if m <= half {
                    (m - 1, forward)
                } else {
                    (radix - m - 1, !forward)
                };
                let (cos, sin) = constants[index];
                let sin = constant::<T>(sin);
                coefficients[u - 1][j - 1] = (constant(cos), if negate { -sin } else { sin });
            }
        }
        Self { coefficients, half }
    }

    fn apply(&self, input: &[Cmplx<T>], output: &mut [Cmplx<T>]) {
        let radix = input.len();
        let t0 = input[0];
        let mut sums = [Cmplx::default(); 5];
        let mut differences = [Cmplx::default(); 5];
        for j in 1..=self.half {
            (sums[j - 1], differences[j - 1]) = pm(input[j], input[radix - j]);
        }
        let sums = &sums[..self.half];
        let differences = &differences[..self.half];
        let mut total = t0;
        for sum in sums {
            total = total + *sum;
        }
        output[0] = total;
        for u in 1..=self.half {
            let coefficients = &self.coefficients[u - 1][..self.half];
            let (mut real, mut imaginary) = (t0.r, t0.i);
            for (&(cos, _), sum) in coefficients.iter().zip(sums) {
                real += cos * sum.r;
                imaginary += cos * sum.i;
            }
            let (_, sin) = coefficients[0];
            let (mut rotated_imaginary, mut rotated_real) =
                (sin * differences[0].r, sin * differences[0].i);
            for (&(_, sin), difference) in coefficients.iter().zip(differences).skip(1) {
                rotated_imaginary += sin * difference.r;
                rotated_real += sin * difference.i;
            }
            let ca = cmplx(real, imaginary);
            let cb = cmplx(-rotated_real, rotated_imaginary);
            (output[u], output[radix - u]) = pm(ca, cb);
        }
    }
}

/// `passg`: the generic pass for a prime radix above 11. The result is left in `cc`.
#[allow(clippy::too_many_arguments)]
fn passg<T: Real>(
    ido: usize,
    ip: usize,
    l1: usize,
    cc: &mut [Cmplx<T>],
    ch: &mut [Cmplx<T>],
    wa: &[Cmplx<T>],
    csarr: &[Cmplx<T>],
    forward: bool,
) {
    let ipph = ip.div_ceil(2);
    let idl1 = ido * l1;
    let ch_at = |a: usize, b: usize, c: usize| a + ido * (b + l1 * c);
    let cc_at = |a: usize, b: usize, c: usize| a + ido * (b + ip * c);
    let x2 = |a: usize, b: usize| a + idl1 * b;

    let mut wal = vec![one(); ip];
    for (target, root) in wal.iter_mut().zip(csarr).skip(1) {
        *target = cmplx(root.r, if forward { -root.i } else { root.i });
    }

    for k in 0..l1 {
        for i in 0..ido {
            ch[ch_at(i, k, 0)] = cc[cc_at(i, 0, k)];
        }
    }
    for j in 1..ipph {
        let jc = ip - j;
        for k in 0..l1 {
            for i in 0..ido {
                (ch[ch_at(i, k, j)], ch[ch_at(i, k, jc)]) =
                    pm(cc[cc_at(i, j, k)], cc[cc_at(i, jc, k)]);
            }
        }
    }
    for k in 0..l1 {
        for i in 0..ido {
            let mut total = ch[ch_at(i, k, 0)];
            for j in 1..ipph {
                total = total + ch[ch_at(i, k, j)];
            }
            cc[ch_at(i, k, 0)] = total;
        }
    }
    for l in 1..ipph {
        let lc = ip - l;
        for ik in 0..idl1 {
            let (h0, h1, h2) = (ch[x2(ik, 0)], ch[x2(ik, 1)], ch[x2(ik, 2)]);
            let (last, before) = (ch[x2(ik, ip - 1)], ch[x2(ik, ip - 2)]);
            cc[x2(ik, l)] = cmplx(
                h0.r + wal[l].r * h1.r + wal[2 * l].r * h2.r,
                h0.i + wal[l].r * h1.i + wal[2 * l].r * h2.i,
            );
            cc[x2(ik, lc)] = cmplx(
                -wal[l].i * last.i - wal[2 * l].i * before.i,
                wal[l].i * last.r + wal[2 * l].i * before.r,
            );
        }
        let mut iwal = 2 * l;
        let mut j = 3;
        let mut jc = ip - 3;
        let next_root = |iwal: &mut usize| {
            *iwal += l;
            if *iwal > ip {
                *iwal -= ip;
            }
            wal[*iwal]
        };
        while j < ipph - 1 {
            let xwal = next_root(&mut iwal);
            let xwal2 = next_root(&mut iwal);
            for ik in 0..idl1 {
                let (a, b) = (ch[x2(ik, j)], ch[x2(ik, j + 1)]);
                let (c, d) = (ch[x2(ik, jc)], ch[x2(ik, jc - 1)]);
                let target = &mut cc[x2(ik, l)];
                target.r += a.r * xwal.r + b.r * xwal2.r;
                target.i += a.i * xwal.r + b.i * xwal2.r;
                let target = &mut cc[x2(ik, lc)];
                target.r = target.r - (c.i * xwal.i + d.i * xwal2.i);
                target.i += c.r * xwal.i + d.r * xwal2.i;
            }
            j += 2;
            jc -= 2;
        }
        while j < ipph {
            let xwal = next_root(&mut iwal);
            for ik in 0..idl1 {
                let (a, c) = (ch[x2(ik, j)], ch[x2(ik, jc)]);
                let target = &mut cc[x2(ik, l)];
                target.r += a.r * xwal.r;
                target.i += a.i * xwal.r;
                let target = &mut cc[x2(ik, lc)];
                target.r = target.r - c.i * xwal.i;
                target.i += c.r * xwal.i;
            }
            j += 1;
            jc -= 1;
        }
    }

    // Shuffling and twiddling.
    for j in 1..ipph {
        let jc = ip - j;
        if ido == 1 {
            for ik in 0..idl1 {
                (cc[x2(ik, j)], cc[x2(ik, jc)]) = pm(cc[x2(ik, j)], cc[x2(ik, jc)]);
            }
            continue;
        }
        for k in 0..l1 {
            (cc[ch_at(0, k, j)], cc[ch_at(0, k, jc)]) = pm(cc[ch_at(0, k, j)], cc[ch_at(0, k, jc)]);
            for i in 1..ido {
                let (x1, x2) = pm(cc[ch_at(i, k, j)], cc[ch_at(i, k, jc)]);
                cc[ch_at(i, k, j)] = special_mul(x1, wa[(j - 1) * (ido - 1) + i - 1], forward);
                cc[ch_at(i, k, jc)] = special_mul(x2, wa[(jc - 1) * (ido - 1) + i - 1], forward);
            }
        }
    }
}

/// One factor of a real plan and its twiddle factors, stored as interleaved real and
/// imaginary parts.
struct RealFactor<T> {
    radix: usize,
    twiddles: Vec<T>,
    /// `2 * radix` values for the generic passes, for radices above 5.
    roots: Vec<T>,
}

/// `rfftp`: FFTPACK's real transform, in FFTPACK's halfcomplex order `r0, r1, i1, r2, i2, …`.
struct Rfftp<T> {
    length: usize,
    factors: Vec<RealFactor<T>>,
}

/// `(a, b) = conj(c + id) * (e + if)`: pocketfft's `MULPM`.
fn mulpm<T: Real>(c: T, d: T, e: T, f: T) -> (T, T) {
    (c * e + d * f, c * f - d * e)
}

impl<T: Real> Rfftp<T> {
    fn new(length: usize) -> Self {
        if length == 1 {
            return Self {
                length,
                factors: Vec::new(),
            };
        }
        let mut radices = Vec::new();
        let mut rest = length;
        while rest.is_multiple_of(4) {
            radices.push(4);
            rest >>= 2;
        }
        if rest.is_multiple_of(2) {
            rest >>= 1;
            radices.push(2);
            let last = radices.len() - 1;
            radices.swap(0, last);
        }
        push_odd_factors(&mut radices, rest);

        let twiddle = Twiddles::new(length);
        let count = radices.len();
        let mut l1 = 1;
        let factors = radices
            .into_iter()
            .enumerate()
            .map(|(k, radix)| {
                let ido = length / (l1 * radix);
                let mut twiddles = Vec::new();
                // The last factor has `ido == 1` and needs no twiddles.
                if k + 1 < count {
                    twiddles = vec![T::default(); (radix - 1) * (ido - 1)];
                    for j in 1..radix {
                        for i in 1..=(ido - 1) / 2 {
                            let value = twiddle.get::<T>(j * l1 * i);
                            twiddles[(j - 1) * (ido - 1) + 2 * i - 2] = value.r;
                            twiddles[(j - 1) * (ido - 1) + 2 * i - 1] = value.i;
                        }
                    }
                }
                let mut roots = Vec::new();
                if radix > 5 {
                    roots = vec![T::default(); 2 * radix];
                    roots[0] = constant(1.0);
                    let (mut i, mut ic) = (2, 2 * radix - 2);
                    while i <= ic {
                        let value = twiddle.get::<T>(i / 2 * (length / radix));
                        roots[i] = value.r;
                        roots[i + 1] = value.i;
                        roots[ic] = value.r;
                        roots[ic + 1] = -value.i;
                        i += 2;
                        ic -= 2;
                    }
                }
                l1 *= radix;
                RealFactor {
                    radix,
                    twiddles,
                    roots,
                }
            })
            .collect();
        Self { length, factors }
    }

    /// Transform `c` in place, real to halfcomplex when `r2hc` and back otherwise, and scale
    /// it by `fct`.
    fn exec(&self, c: &mut [T], fct: T, r2hc: bool) {
        if self.length == 1 {
            c[0] *= fct;
            return;
        }
        let length = self.length;
        let mut ch = vec![T::default(); length];
        let mut in_c = true;
        if r2hc {
            let mut l1 = length;
            for factor in self.factors.iter().rev() {
                let radix = factor.radix;
                let ido = length / l1;
                l1 /= radix;
                let (p1, p2): (&mut [T], &mut [T]) = if in_c {
                    (&mut *c, &mut ch)
                } else {
                    (&mut ch, &mut *c)
                };
                let wa = &factor.twiddles;
                match radix {
                    4 => radf4(ido, l1, p1, p2, wa),
                    2 => radf2(ido, l1, p1, p2, wa),
                    3 => radf3(ido, l1, p1, p2, wa),
                    5 => radf5(ido, l1, p1, p2, wa),
                    _ => {
                        radfg(ido, radix, l1, p1, p2, wa, &factor.roots);
                        in_c = !in_c;
                    }
                }
                in_c = !in_c;
            }
        } else {
            let mut l1 = 1;
            for factor in &self.factors {
                let radix = factor.radix;
                let ido = length / (radix * l1);
                let (p1, p2): (&mut [T], &mut [T]) = if in_c {
                    (&mut *c, &mut ch)
                } else {
                    (&mut ch, &mut *c)
                };
                let wa = &factor.twiddles;
                match radix {
                    4 => radb4(ido, l1, p1, p2, wa),
                    2 => radb2(ido, l1, p1, p2, wa),
                    3 => radb3(ido, l1, p1, p2, wa),
                    5 => radb5(ido, l1, p1, p2, wa),
                    _ => radbg(ido, radix, l1, p1, p2, wa, &factor.roots),
                }
                in_c = !in_c;
                l1 *= radix;
            }
        }
        let unscaled = fct == constant(1.0);
        if !in_c {
            if unscaled {
                c.copy_from_slice(&ch);
            } else {
                for (target, value) in c.iter_mut().zip(&ch) {
                    *target = fct * *value;
                }
            }
        } else if !unscaled {
            for value in c.iter_mut() {
                *value *= fct;
            }
        }
    }
}

/// Index helpers for the real passes: `cc` is read as `(ido, l1, radix)` and `ch` written as
/// `(ido, radix, l1)` going forward, and the other way round going backward.
fn at3(ido: usize, middle: usize) -> impl Fn(usize, usize, usize) -> usize {
    move |a, b, c| a + ido * (b + middle * c)
}

fn radf2<T: Real>(ido: usize, l1: usize, cc: &[T], ch: &mut [T], wa: &[T]) {
    let cc_at = at3(ido, l1);
    let ch_at = at3(ido, 2);
    let w = |x: usize, i: usize| wa[i + x * (ido - 1)];
    for k in 0..l1 {
        (ch[ch_at(0, 0, k)], ch[ch_at(ido - 1, 1, k)]) = pm(cc[cc_at(0, k, 0)], cc[cc_at(0, k, 1)]);
    }
    if ido & 1 == 0 {
        for k in 0..l1 {
            ch[ch_at(0, 1, k)] = -cc[cc_at(ido - 1, k, 1)];
            ch[ch_at(ido - 1, 0, k)] = cc[cc_at(ido - 1, k, 0)];
        }
    }
    if ido <= 2 {
        return;
    }
    for k in 0..l1 {
        for i in (2..ido).step_by(2) {
            let ic = ido - i;
            let (tr2, ti2) = mulpm(
                w(0, i - 2),
                w(0, i - 1),
                cc[cc_at(i - 1, k, 1)],
                cc[cc_at(i, k, 1)],
            );
            (ch[ch_at(i - 1, 0, k)], ch[ch_at(ic - 1, 1, k)]) = pm(cc[cc_at(i - 1, k, 0)], tr2);
            (ch[ch_at(i, 0, k)], ch[ch_at(ic, 1, k)]) = pm(ti2, cc[cc_at(i, k, 0)]);
        }
    }
}

/// `POCKETFFT_REARRANGE`: `(a, b) = (a + b, i(b - a))` on split real and imaginary parts.
fn rearrange<T: Real>(rx: T, ix: T, ry: T, iy: T) -> (T, T, T, T) {
    (rx + ry, ix + iy, ix - iy, ry - rx)
}

fn radf3<T: Real>(ido: usize, l1: usize, cc: &[T], ch: &mut [T], wa: &[T]) {
    let taur = constant::<T>(TAUR);
    let taui = constant::<T>(TAUI);
    let cc_at = at3(ido, l1);
    let ch_at = at3(ido, 3);
    let w = |x: usize, i: usize| wa[i + x * (ido - 1)];
    for k in 0..l1 {
        let cr2 = cc[cc_at(0, k, 1)] + cc[cc_at(0, k, 2)];
        ch[ch_at(0, 0, k)] = cc[cc_at(0, k, 0)] + cr2;
        ch[ch_at(0, 2, k)] = taui * (cc[cc_at(0, k, 2)] - cc[cc_at(0, k, 1)]);
        ch[ch_at(ido - 1, 1, k)] = cc[cc_at(0, k, 0)] + taur * cr2;
    }
    if ido == 1 {
        return;
    }
    for k in 0..l1 {
        for i in (2..ido).step_by(2) {
            let ic = ido - i;
            let (dr2, di2) = mulpm(
                w(0, i - 2),
                w(0, i - 1),
                cc[cc_at(i - 1, k, 1)],
                cc[cc_at(i, k, 1)],
            );
            let (dr3, di3) = mulpm(
                w(1, i - 2),
                w(1, i - 1),
                cc[cc_at(i - 1, k, 2)],
                cc[cc_at(i, k, 2)],
            );
            let (dr2, di2, dr3, di3) = rearrange(dr2, di2, dr3, di3);
            ch[ch_at(i - 1, 0, k)] = cc[cc_at(i - 1, k, 0)] + dr2;
            ch[ch_at(i, 0, k)] = cc[cc_at(i, k, 0)] + di2;
            let tr2 = cc[cc_at(i - 1, k, 0)] + taur * dr2;
            let ti2 = cc[cc_at(i, k, 0)] + taur * di2;
            let tr3 = taui * dr3;
            let ti3 = taui * di3;
            (ch[ch_at(i - 1, 2, k)], ch[ch_at(ic - 1, 1, k)]) = pm(tr2, tr3);
            (ch[ch_at(i, 2, k)], ch[ch_at(ic, 1, k)]) = pm(ti3, ti2);
        }
    }
}

fn radf4<T: Real>(ido: usize, l1: usize, cc: &[T], ch: &mut [T], wa: &[T]) {
    let hsqt2 = constant::<T>(HSQT2);
    let cc_at = at3(ido, l1);
    let ch_at = at3(ido, 4);
    let w = |x: usize, i: usize| wa[i + x * (ido - 1)];
    for k in 0..l1 {
        let tr1;
        let tr2;
        (tr1, ch[ch_at(0, 2, k)]) = pm(cc[cc_at(0, k, 3)], cc[cc_at(0, k, 1)]);
        (tr2, ch[ch_at(ido - 1, 1, k)]) = pm(cc[cc_at(0, k, 0)], cc[cc_at(0, k, 2)]);
        (ch[ch_at(0, 0, k)], ch[ch_at(ido - 1, 3, k)]) = pm(tr2, tr1);
    }
    if ido & 1 == 0 {
        for k in 0..l1 {
            let ti1 = -hsqt2 * (cc[cc_at(ido - 1, k, 1)] + cc[cc_at(ido - 1, k, 3)]);
            let tr1 = hsqt2 * (cc[cc_at(ido - 1, k, 1)] - cc[cc_at(ido - 1, k, 3)]);
            (ch[ch_at(ido - 1, 0, k)], ch[ch_at(ido - 1, 2, k)]) =
                pm(cc[cc_at(ido - 1, k, 0)], tr1);
            (ch[ch_at(0, 3, k)], ch[ch_at(0, 1, k)]) = pm(ti1, cc[cc_at(ido - 1, k, 2)]);
        }
    }
    if ido <= 2 {
        return;
    }
    for k in 0..l1 {
        for i in (2..ido).step_by(2) {
            let ic = ido - i;
            let twiddled = |x: usize| {
                mulpm(
                    w(x, i - 2),
                    w(x, i - 1),
                    cc[cc_at(i - 1, k, x + 1)],
                    cc[cc_at(i, k, x + 1)],
                )
            };
            let (cr2, ci2) = twiddled(0);
            let (cr3, ci3) = twiddled(1);
            let (cr4, ci4) = twiddled(2);
            let (tr1, tr4) = pm(cr4, cr2);
            let (ti1, ti4) = pm(ci2, ci4);
            let (tr2, tr3) = pm(cc[cc_at(i - 1, k, 0)], cr3);
            let (ti2, ti3) = pm(cc[cc_at(i, k, 0)], ci3);
            (ch[ch_at(i - 1, 0, k)], ch[ch_at(ic - 1, 3, k)]) = pm(tr2, tr1);
            (ch[ch_at(i, 0, k)], ch[ch_at(ic, 3, k)]) = pm(ti1, ti2);
            (ch[ch_at(i - 1, 2, k)], ch[ch_at(ic - 1, 1, k)]) = pm(tr3, ti4);
            (ch[ch_at(i, 2, k)], ch[ch_at(ic, 1, k)]) = pm(tr4, ti3);
        }
    }
}

const TR11: f64 = RADIX5[0].0;
const TI11: f64 = RADIX5[0].1;
const TR12: f64 = RADIX5[1].0;
const TI12: f64 = RADIX5[1].1;

fn radf5<T: Real>(ido: usize, l1: usize, cc: &[T], ch: &mut [T], wa: &[T]) {
    let tr11 = constant::<T>(TR11);
    let ti11 = constant::<T>(TI11);
    let tr12 = constant::<T>(TR12);
    let ti12 = constant::<T>(TI12);
    let cc_at = at3(ido, l1);
    let ch_at = at3(ido, 5);
    let w = |x: usize, i: usize| wa[i + x * (ido - 1)];
    for k in 0..l1 {
        let (cr2, ci5) = pm(cc[cc_at(0, k, 4)], cc[cc_at(0, k, 1)]);
        let (cr3, ci4) = pm(cc[cc_at(0, k, 3)], cc[cc_at(0, k, 2)]);
        let c0 = cc[cc_at(0, k, 0)];
        ch[ch_at(0, 0, k)] = c0 + cr2 + cr3;
        ch[ch_at(ido - 1, 1, k)] = c0 + tr11 * cr2 + tr12 * cr3;
        ch[ch_at(0, 2, k)] = ti11 * ci5 + ti12 * ci4;
        ch[ch_at(ido - 1, 3, k)] = c0 + tr12 * cr2 + tr11 * cr3;
        ch[ch_at(0, 4, k)] = ti12 * ci5 - ti11 * ci4;
    }
    if ido == 1 {
        return;
    }
    for k in 0..l1 {
        for i in (2..ido).step_by(2) {
            let ic = ido - i;
            let twiddled = |x: usize| {
                mulpm(
                    w(x, i - 2),
                    w(x, i - 1),
                    cc[cc_at(i - 1, k, x + 1)],
                    cc[cc_at(i, k, x + 1)],
                )
            };
            let (dr2, di2) = twiddled(0);
            let (dr3, di3) = twiddled(1);
            let (dr4, di4) = twiddled(2);
            let (dr5, di5) = twiddled(3);
            let (dr2, di2, dr5, di5) = rearrange(dr2, di2, dr5, di5);
            let (dr3, di3, dr4, di4) = rearrange(dr3, di3, dr4, di4);
            let (cr, ci) = (cc[cc_at(i - 1, k, 0)], cc[cc_at(i, k, 0)]);
            ch[ch_at(i - 1, 0, k)] = cr + dr2 + dr3;
            ch[ch_at(i, 0, k)] = ci + di2 + di3;
            let tr2 = cr + tr11 * dr2 + tr12 * dr3;
            let ti2 = ci + tr11 * di2 + tr12 * di3;
            let tr3 = cr + tr12 * dr2 + tr11 * dr3;
            let ti3 = ci + tr12 * di2 + tr11 * di3;
            let tr5 = ti11 * dr5 + ti12 * dr4;
            let ti5 = ti11 * di5 + ti12 * di4;
            let tr4 = ti12 * dr5 - ti11 * dr4;
            let ti4 = ti12 * di5 - ti11 * di4;
            (ch[ch_at(i - 1, 2, k)], ch[ch_at(ic - 1, 1, k)]) = pm(tr2, tr5);
            (ch[ch_at(i, 2, k)], ch[ch_at(ic, 1, k)]) = pm(ti5, ti2);
            (ch[ch_at(i - 1, 4, k)], ch[ch_at(ic - 1, 3, k)]) = pm(tr3, tr4);
            (ch[ch_at(i, 4, k)], ch[ch_at(ic, 3, k)]) = pm(ti4, ti3);
        }
    }
}

/// `radfg`: the generic forward real pass for a prime radix above 5. The result is left in
/// `cc`.
fn radfg<T: Real>(
    ido: usize,
    ip: usize,
    l1: usize,
    cc: &mut [T],
    ch: &mut [T],
    wa: &[T],
    csarr: &[T],
) {
    let ipph = ip.div_ceil(2);
    let idl1 = ido * l1;
    let cc_at = at3(ido, ip);
    let c1 = at3(ido, l1);
    let c2 = |a: usize, b: usize| a + idl1 * b;

    if ido > 1 {
        for j in 1..ipph {
            let jc = ip - j;
            let is = (j - 1) * (ido - 1);
            let is2 = (jc - 1) * (ido - 1);
            for k in 0..l1 {
                let mut idij = is;
                let mut idij2 = is2;
                for i in (1..ido - 1).step_by(2) {
                    let t1 = cc[c1(i, k, j)];
                    let t2 = cc[c1(i + 1, k, j)];
                    let t3 = cc[c1(i, k, jc)];
                    let t4 = cc[c1(i + 1, k, jc)];
                    let x1 = wa[idij] * t1 + wa[idij + 1] * t2;
                    let x2 = wa[idij] * t2 - wa[idij + 1] * t1;
                    let x3 = wa[idij2] * t3 + wa[idij2 + 1] * t4;
                    let x4 = wa[idij2] * t4 - wa[idij2 + 1] * t3;
                    (cc[c1(i, k, j)], cc[c1(i + 1, k, jc)]) = pm(x3, x1);
                    (cc[c1(i + 1, k, j)], cc[c1(i, k, jc)]) = pm(x2, x4);
                    idij += 2;
                    idij2 += 2;
                }
            }
        }
    }

    for j in 1..ipph {
        let jc = ip - j;
        for k in 0..l1 {
            // MPINPLACE(C1(0,k,jc), C1(0,k,j))
            let (a, b) = (cc[c1(0, k, jc)], cc[c1(0, k, j)]);
            cc[c1(0, k, jc)] = a - b;
            cc[c1(0, k, j)] = a + b;
        }
    }

    for l in 1..ipph {
        let lc = ip - l;
        for ik in 0..idl1 {
            ch[c2(ik, l)] =
                cc[c2(ik, 0)] + csarr[2 * l] * cc[c2(ik, 1)] + csarr[4 * l] * cc[c2(ik, 2)];
            ch[c2(ik, lc)] =
                csarr[2 * l + 1] * cc[c2(ik, ip - 1)] + csarr[4 * l + 1] * cc[c2(ik, ip - 2)];
        }
        let mut iang = 2 * l;
        let mut next_angle = || {
            iang += l;
            if iang >= ip {
                iang -= ip;
            }
            (csarr[2 * iang], csarr[2 * iang + 1])
        };
        let mut j = 3;
        let mut jc = ip - 3;
        while j + 3 < ipph {
            let (ar1, ai1) = next_angle();
            let (ar2, ai2) = next_angle();
            let (ar3, ai3) = next_angle();
            let (ar4, ai4) = next_angle();
            for ik in 0..idl1 {
                ch[c2(ik, l)] += ar1 * cc[c2(ik, j)]
                    + ar2 * cc[c2(ik, j + 1)]
                    + ar3 * cc[c2(ik, j + 2)]
                    + ar4 * cc[c2(ik, j + 3)];
                ch[c2(ik, lc)] += ai1 * cc[c2(ik, jc)]
                    + ai2 * cc[c2(ik, jc - 1)]
                    + ai3 * cc[c2(ik, jc - 2)]
                    + ai4 * cc[c2(ik, jc - 3)];
            }
            j += 4;
            jc -= 4;
        }
        while j + 1 < ipph {
            let (ar1, ai1) = next_angle();
            let (ar2, ai2) = next_angle();
            for ik in 0..idl1 {
                ch[c2(ik, l)] += ar1 * cc[c2(ik, j)] + ar2 * cc[c2(ik, j + 1)];
                ch[c2(ik, lc)] += ai1 * cc[c2(ik, jc)] + ai2 * cc[c2(ik, jc - 1)];
            }
            j += 2;
            jc -= 2;
        }
        while j < ipph {
            let (ar, ai) = next_angle();
            for ik in 0..idl1 {
                ch[c2(ik, l)] += ar * cc[c2(ik, j)];
                ch[c2(ik, lc)] += ai * cc[c2(ik, jc)];
            }
            j += 1;
            jc -= 1;
        }
    }
    for ik in 0..idl1 {
        ch[c2(ik, 0)] = cc[c2(ik, 0)];
    }
    for j in 1..ipph {
        for ik in 0..idl1 {
            ch[c2(ik, 0)] += cc[c2(ik, j)];
        }
    }

    for k in 0..l1 {
        for i in 0..ido {
            cc[cc_at(i, 0, k)] = ch[c1(i, k, 0)];
        }
    }
    for j in 1..ipph {
        let jc = ip - j;
        let j2 = 2 * j - 1;
        for k in 0..l1 {
            cc[cc_at(ido - 1, j2, k)] = ch[c1(0, k, j)];
            cc[cc_at(0, j2 + 1, k)] = ch[c1(0, k, jc)];
        }
    }
    if ido == 1 {
        return;
    }
    for j in 1..ipph {
        let jc = ip - j;
        let j2 = 2 * j - 1;
        for k in 0..l1 {
            for i in (1..ido - 1).step_by(2) {
                let ic = ido - i - 2;
                cc[cc_at(i, j2 + 1, k)] = ch[c1(i, k, j)] + ch[c1(i, k, jc)];
                cc[cc_at(ic, j2, k)] = ch[c1(i, k, j)] - ch[c1(i, k, jc)];
                cc[cc_at(i + 1, j2 + 1, k)] = ch[c1(i + 1, k, j)] + ch[c1(i + 1, k, jc)];
                cc[cc_at(ic + 1, j2, k)] = ch[c1(i + 1, k, jc)] - ch[c1(i + 1, k, j)];
            }
        }
    }
}

fn radb2<T: Real>(ido: usize, l1: usize, cc: &[T], ch: &mut [T], wa: &[T]) {
    let two = constant::<T>(2.0);
    let cc_at = at3(ido, 2);
    let ch_at = at3(ido, l1);
    let w = |x: usize, i: usize| wa[i + x * (ido - 1)];
    for k in 0..l1 {
        (ch[ch_at(0, k, 0)], ch[ch_at(0, k, 1)]) = pm(cc[cc_at(0, 0, k)], cc[cc_at(ido - 1, 1, k)]);
    }
    if ido & 1 == 0 {
        for k in 0..l1 {
            ch[ch_at(ido - 1, k, 0)] = two * cc[cc_at(ido - 1, 0, k)];
            ch[ch_at(ido - 1, k, 1)] = -two * cc[cc_at(0, 1, k)];
        }
    }
    if ido <= 2 {
        return;
    }
    for k in 0..l1 {
        for i in (2..ido).step_by(2) {
            let ic = ido - i;
            let tr2;
            let ti2;
            (ch[ch_at(i - 1, k, 0)], tr2) = pm(cc[cc_at(i - 1, 0, k)], cc[cc_at(ic - 1, 1, k)]);
            (ti2, ch[ch_at(i, k, 0)]) = pm(cc[cc_at(i, 0, k)], cc[cc_at(ic, 1, k)]);
            (ch[ch_at(i, k, 1)], ch[ch_at(i - 1, k, 1)]) =
                mulpm(w(0, i - 2), w(0, i - 1), ti2, tr2);
        }
    }
}

fn radb3<T: Real>(ido: usize, l1: usize, cc: &[T], ch: &mut [T], wa: &[T]) {
    let taur = constant::<T>(TAUR);
    let taui = constant::<T>(TAUI);
    let two = constant::<T>(2.0);
    let cc_at = at3(ido, 3);
    let ch_at = at3(ido, l1);
    let w = |x: usize, i: usize| wa[i + x * (ido - 1)];
    for k in 0..l1 {
        let tr2 = two * cc[cc_at(ido - 1, 1, k)];
        let cr2 = cc[cc_at(0, 0, k)] + taur * tr2;
        ch[ch_at(0, k, 0)] = cc[cc_at(0, 0, k)] + tr2;
        let ci3 = two * taui * cc[cc_at(0, 2, k)];
        (ch[ch_at(0, k, 2)], ch[ch_at(0, k, 1)]) = pm(cr2, ci3);
    }
    if ido == 1 {
        return;
    }
    for k in 0..l1 {
        for i in (2..ido).step_by(2) {
            let ic = ido - i;
            let tr2 = cc[cc_at(i - 1, 2, k)] + cc[cc_at(ic - 1, 1, k)];
            let ti2 = cc[cc_at(i, 2, k)] - cc[cc_at(ic, 1, k)];
            let cr2 = cc[cc_at(i - 1, 0, k)] + taur * tr2;
            let ci2 = cc[cc_at(i, 0, k)] + taur * ti2;
            ch[ch_at(i - 1, k, 0)] = cc[cc_at(i - 1, 0, k)] + tr2;
            ch[ch_at(i, k, 0)] = cc[cc_at(i, 0, k)] + ti2;
            let cr3 = taui * (cc[cc_at(i - 1, 2, k)] - cc[cc_at(ic - 1, 1, k)]);
            let ci3 = taui * (cc[cc_at(i, 2, k)] + cc[cc_at(ic, 1, k)]);
            let (dr3, dr2) = pm(cr2, ci3);
            let (di2, di3) = pm(ci2, cr3);
            (ch[ch_at(i, k, 1)], ch[ch_at(i - 1, k, 1)]) =
                mulpm(w(0, i - 2), w(0, i - 1), di2, dr2);
            (ch[ch_at(i, k, 2)], ch[ch_at(i - 1, k, 2)]) =
                mulpm(w(1, i - 2), w(1, i - 1), di3, dr3);
        }
    }
}

fn radb4<T: Real>(ido: usize, l1: usize, cc: &[T], ch: &mut [T], wa: &[T]) {
    let sqrt2 = constant::<T>(SQRT2);
    let two = constant::<T>(2.0);
    let cc_at = at3(ido, 4);
    let ch_at = at3(ido, l1);
    let w = |x: usize, i: usize| wa[i + x * (ido - 1)];
    for k in 0..l1 {
        let (tr2, tr1) = pm(cc[cc_at(0, 0, k)], cc[cc_at(ido - 1, 3, k)]);
        let tr3 = two * cc[cc_at(ido - 1, 1, k)];
        let tr4 = two * cc[cc_at(0, 2, k)];
        (ch[ch_at(0, k, 0)], ch[ch_at(0, k, 2)]) = pm(tr2, tr3);
        (ch[ch_at(0, k, 3)], ch[ch_at(0, k, 1)]) = pm(tr1, tr4);
    }
    if ido & 1 == 0 {
        for k in 0..l1 {
            let (ti1, ti2) = pm(cc[cc_at(0, 3, k)], cc[cc_at(0, 1, k)]);
            let (tr2, tr1) = pm(cc[cc_at(ido - 1, 0, k)], cc[cc_at(ido - 1, 2, k)]);
            ch[ch_at(ido - 1, k, 0)] = tr2 + tr2;
            ch[ch_at(ido - 1, k, 1)] = sqrt2 * (tr1 - ti1);
            ch[ch_at(ido - 1, k, 2)] = ti2 + ti2;
            ch[ch_at(ido - 1, k, 3)] = -sqrt2 * (tr1 + ti1);
        }
    }
    if ido <= 2 {
        return;
    }
    for k in 0..l1 {
        for i in (2..ido).step_by(2) {
            let ic = ido - i;
            let (tr2, tr1) = pm(cc[cc_at(i - 1, 0, k)], cc[cc_at(ic - 1, 3, k)]);
            let (ti1, ti2) = pm(cc[cc_at(i, 0, k)], cc[cc_at(ic, 3, k)]);
            let (tr4, ti3) = pm(cc[cc_at(i, 2, k)], cc[cc_at(ic, 1, k)]);
            let (tr3, ti4) = pm(cc[cc_at(i - 1, 2, k)], cc[cc_at(ic - 1, 1, k)]);
            let cr3;
            let ci3;
            (ch[ch_at(i - 1, k, 0)], cr3) = pm(tr2, tr3);
            (ch[ch_at(i, k, 0)], ci3) = pm(ti2, ti3);
            let (cr4, cr2) = pm(tr1, tr4);
            let (ci2, ci4) = pm(ti1, ti4);
            (ch[ch_at(i, k, 1)], ch[ch_at(i - 1, k, 1)]) =
                mulpm(w(0, i - 2), w(0, i - 1), ci2, cr2);
            (ch[ch_at(i, k, 2)], ch[ch_at(i - 1, k, 2)]) =
                mulpm(w(1, i - 2), w(1, i - 1), ci3, cr3);
            (ch[ch_at(i, k, 3)], ch[ch_at(i - 1, k, 3)]) =
                mulpm(w(2, i - 2), w(2, i - 1), ci4, cr4);
        }
    }
}

fn radb5<T: Real>(ido: usize, l1: usize, cc: &[T], ch: &mut [T], wa: &[T]) {
    let tr11 = constant::<T>(TR11);
    let ti11 = constant::<T>(TI11);
    let tr12 = constant::<T>(TR12);
    let ti12 = constant::<T>(TI12);
    let cc_at = at3(ido, 5);
    let ch_at = at3(ido, l1);
    let w = |x: usize, i: usize| wa[i + x * (ido - 1)];
    for k in 0..l1 {
        let ti5 = cc[cc_at(0, 2, k)] + cc[cc_at(0, 2, k)];
        let ti4 = cc[cc_at(0, 4, k)] + cc[cc_at(0, 4, k)];
        let tr2 = cc[cc_at(ido - 1, 1, k)] + cc[cc_at(ido - 1, 1, k)];
        let tr3 = cc[cc_at(ido - 1, 3, k)] + cc[cc_at(ido - 1, 3, k)];
        let c0 = cc[cc_at(0, 0, k)];
        ch[ch_at(0, k, 0)] = c0 + tr2 + tr3;
        let cr2 = c0 + tr11 * tr2 + tr12 * tr3;
        let cr3 = c0 + tr12 * tr2 + tr11 * tr3;
        let (ci5, ci4) = mulpm(ti5, ti4, ti11, ti12);
        (ch[ch_at(0, k, 4)], ch[ch_at(0, k, 1)]) = pm(cr2, ci5);
        (ch[ch_at(0, k, 3)], ch[ch_at(0, k, 2)]) = pm(cr3, ci4);
    }
    if ido == 1 {
        return;
    }
    for k in 0..l1 {
        for i in (2..ido).step_by(2) {
            let ic = ido - i;
            let (tr2, tr5) = pm(cc[cc_at(i - 1, 2, k)], cc[cc_at(ic - 1, 1, k)]);
            let (ti5, ti2) = pm(cc[cc_at(i, 2, k)], cc[cc_at(ic, 1, k)]);
            let (tr3, tr4) = pm(cc[cc_at(i - 1, 4, k)], cc[cc_at(ic - 1, 3, k)]);
            let (ti4, ti3) = pm(cc[cc_at(i, 4, k)], cc[cc_at(ic, 3, k)]);
            let (cr, ci) = (cc[cc_at(i - 1, 0, k)], cc[cc_at(i, 0, k)]);
            ch[ch_at(i - 1, k, 0)] = cr + tr2 + tr3;
            ch[ch_at(i, k, 0)] = ci + ti2 + ti3;
            let cr2 = cr + tr11 * tr2 + tr12 * tr3;
            let ci2 = ci + tr11 * ti2 + tr12 * ti3;
            let cr3 = cr + tr12 * tr2 + tr11 * tr3;
            let ci3 = ci + tr12 * ti2 + tr11 * ti3;
            let (cr5, cr4) = mulpm(tr5, tr4, ti11, ti12);
            let (ci5, ci4) = mulpm(ti5, ti4, ti11, ti12);
            let (dr4, dr3) = pm(cr3, ci4);
            let (di3, di4) = pm(ci3, cr4);
            let (dr5, dr2) = pm(cr2, ci5);
            let (di2, di5) = pm(ci2, cr5);
            for (x, (di, dr)) in [(di2, dr2), (di3, dr3), (di4, dr4), (di5, dr5)]
                .into_iter()
                .enumerate()
            {
                (ch[ch_at(i, k, x + 1)], ch[ch_at(i - 1, k, x + 1)]) =
                    mulpm(w(x, i - 2), w(x, i - 1), di, dr);
            }
        }
    }
}

/// `radbg`: the generic backward real pass for a prime radix above 5. The result is left in
/// `ch`.
fn radbg<T: Real>(
    ido: usize,
    ip: usize,
    l1: usize,
    cc: &mut [T],
    ch: &mut [T],
    wa: &[T],
    csarr: &[T],
) {
    let two = constant::<T>(2.0);
    let ipph = ip.div_ceil(2);
    let idl1 = ido * l1;
    let cc_at = at3(ido, ip);
    let c1 = at3(ido, l1);
    let c2 = |a: usize, b: usize| a + idl1 * b;

    for k in 0..l1 {
        for i in 0..ido {
            ch[c1(i, k, 0)] = cc[cc_at(i, 0, k)];
        }
    }
    for j in 1..ipph {
        let jc = ip - j;
        let j2 = 2 * j - 1;
        for k in 0..l1 {
            ch[c1(0, k, j)] = two * cc[cc_at(ido - 1, j2, k)];
            ch[c1(0, k, jc)] = two * cc[cc_at(0, j2 + 1, k)];
        }
    }
    if ido != 1 {
        for j in 1..ipph {
            let jc = ip - j;
            let j2 = 2 * j - 1;
            for k in 0..l1 {
                for i in (1..ido - 1).step_by(2) {
                    let ic = ido - i - 2;
                    ch[c1(i, k, j)] = cc[cc_at(i, j2 + 1, k)] + cc[cc_at(ic, j2, k)];
                    ch[c1(i, k, jc)] = cc[cc_at(i, j2 + 1, k)] - cc[cc_at(ic, j2, k)];
                    ch[c1(i + 1, k, j)] = cc[cc_at(i + 1, j2 + 1, k)] - cc[cc_at(ic + 1, j2, k)];
                    ch[c1(i + 1, k, jc)] = cc[cc_at(i + 1, j2 + 1, k)] + cc[cc_at(ic + 1, j2, k)];
                }
            }
        }
    }
    for l in 1..ipph {
        let lc = ip - l;
        for ik in 0..idl1 {
            cc[c2(ik, l)] =
                ch[c2(ik, 0)] + csarr[2 * l] * ch[c2(ik, 1)] + csarr[4 * l] * ch[c2(ik, 2)];
            cc[c2(ik, lc)] =
                csarr[2 * l + 1] * ch[c2(ik, ip - 1)] + csarr[4 * l + 1] * ch[c2(ik, ip - 2)];
        }
        let mut iang = 2 * l;
        let mut next_angle = || {
            iang += l;
            if iang > ip {
                iang -= ip;
            }
            (csarr[2 * iang], csarr[2 * iang + 1])
        };
        let mut j = 3;
        let mut jc = ip - 3;
        while j + 3 < ipph {
            let (ar1, ai1) = next_angle();
            let (ar2, ai2) = next_angle();
            let (ar3, ai3) = next_angle();
            let (ar4, ai4) = next_angle();
            for ik in 0..idl1 {
                cc[c2(ik, l)] += ar1 * ch[c2(ik, j)]
                    + ar2 * ch[c2(ik, j + 1)]
                    + ar3 * ch[c2(ik, j + 2)]
                    + ar4 * ch[c2(ik, j + 3)];
                cc[c2(ik, lc)] += ai1 * ch[c2(ik, jc)]
                    + ai2 * ch[c2(ik, jc - 1)]
                    + ai3 * ch[c2(ik, jc - 2)]
                    + ai4 * ch[c2(ik, jc - 3)];
            }
            j += 4;
            jc -= 4;
        }
        while j + 1 < ipph {
            let (ar1, ai1) = next_angle();
            let (ar2, ai2) = next_angle();
            for ik in 0..idl1 {
                cc[c2(ik, l)] += ar1 * ch[c2(ik, j)] + ar2 * ch[c2(ik, j + 1)];
                cc[c2(ik, lc)] += ai1 * ch[c2(ik, jc)] + ai2 * ch[c2(ik, jc - 1)];
            }
            j += 2;
            jc -= 2;
        }
        while j < ipph {
            let (war, wai) = next_angle();
            for ik in 0..idl1 {
                cc[c2(ik, l)] += war * ch[c2(ik, j)];
                cc[c2(ik, lc)] += wai * ch[c2(ik, jc)];
            }
            j += 1;
            jc -= 1;
        }
    }
    for j in 1..ipph {
        for ik in 0..idl1 {
            ch[c2(ik, 0)] += ch[c2(ik, j)];
        }
    }
    for j in 1..ipph {
        let jc = ip - j;
        for k in 0..l1 {
            (ch[c1(0, k, jc)], ch[c1(0, k, j)]) = pm(cc[c1(0, k, j)], cc[c1(0, k, jc)]);
        }
    }
    if ido == 1 {
        return;
    }
    for j in 1..ipph {
        let jc = ip - j;
        for k in 0..l1 {
            for i in (1..ido - 1).step_by(2) {
                ch[c1(i, k, j)] = cc[c1(i, k, j)] - cc[c1(i + 1, k, jc)];
                ch[c1(i, k, jc)] = cc[c1(i, k, j)] + cc[c1(i + 1, k, jc)];
                ch[c1(i + 1, k, j)] = cc[c1(i + 1, k, j)] + cc[c1(i, k, jc)];
                ch[c1(i + 1, k, jc)] = cc[c1(i + 1, k, j)] - cc[c1(i, k, jc)];
            }
        }
    }
    for j in 1..ip {
        let is = (j - 1) * (ido - 1);
        for k in 0..l1 {
            let mut idij = is;
            for i in (1..ido - 1).step_by(2) {
                let t1 = ch[c1(i, k, j)];
                let t2 = ch[c1(i + 1, k, j)];
                ch[c1(i, k, j)] = wa[idij] * t1 - wa[idij + 1] * t2;
                ch[c1(i + 1, k, j)] = wa[idij] * t2 + wa[idij + 1] * t1;
                idij += 2;
            }
        }
    }
}

/// `fftblue`: Bluestein's algorithm, which computes a length-`n` transform as a convolution
/// of padded length `n2`, for lengths whose large prime factors make FFTPACK slow.
struct Bluestein<T> {
    n: usize,
    n2: usize,
    plan: Cfftp<T>,
    bk: Vec<Cmplx<T>>,
    bkf: Vec<Cmplx<T>>,
}

impl<T: Real> Bluestein<T> {
    fn new(n: usize, n2: usize) -> Self {
        let plan = Cfftp::new(n2);
        let roots = Twiddles::new(2 * n);
        let mut bk = vec![one(); n];
        let mut coefficient = 0;
        for (m, value) in bk.iter_mut().enumerate().skip(1) {
            coefficient += 2 * m - 1;
            if coefficient >= 2 * n {
                coefficient -= 2 * n;
            }
            *value = roots.get(coefficient);
        }
        // The zero-padded, transformed `bk`, with the normalization folded in.
        let scale = constant::<T>(1.0) / constant(n2 as f64);
        let mut transformed = vec![Cmplx::default(); n2];
        transformed[0] = bk[0] * scale;
        for m in 1..n {
            transformed[m] = bk[m] * scale;
            transformed[n2 - m] = transformed[m];
        }
        plan.exec(&mut transformed, constant(1.0), true);
        transformed.truncate(n2 / 2 + 1);
        Self {
            n,
            n2,
            plan,
            bk,
            bkf: transformed,
        }
    }

    fn fft(&self, c: &mut [Cmplx<T>], fct: T, forward: bool) {
        let (n, n2) = (self.n, self.n2);
        let mut akf = vec![Cmplx::default(); n2];
        for m in 0..n {
            akf[m] = special_mul(c[m], self.bk[m], forward);
        }
        let zero = akf[0] * T::default();
        for value in &mut akf[n..] {
            *value = zero;
        }
        self.plan.exec(&mut akf, constant(1.0), true);
        akf[0] = special_mul(akf[0], self.bkf[0], !forward);
        for m in 1..n2.div_ceil(2) {
            akf[m] = special_mul(akf[m], self.bkf[m], !forward);
            akf[n2 - m] = special_mul(akf[n2 - m], self.bkf[m], !forward);
        }
        if n2 & 1 == 0 {
            akf[n2 / 2] = special_mul(akf[n2 / 2], self.bkf[n2 / 2], !forward);
        }
        self.plan.exec(&mut akf, constant(1.0), false);
        for m in 0..n {
            c[m] = special_mul(akf[m], self.bk[m], forward) * fct;
        }
    }

    /// `exec_r`: a real transform in FFTPACK's halfcomplex order through the complex one.
    /// Zero parts are `0 * c[0]`, so they take the sign (or NaN) of the first value.
    fn exec_real(&self, c: &mut [T], fct: T, forward: bool) {
        let n = self.n;
        let mut tmp = vec![Cmplx::default(); n];
        let zero = T::default() * c[0];
        if forward {
            for (target, value) in tmp.iter_mut().zip(c.iter()) {
                *target = cmplx(*value, zero);
            }
            self.fft(&mut tmp, fct, true);
            c[0] = tmp[0].r;
            for (index, target) in c[1..].iter_mut().enumerate() {
                *target = *interleaved(&mut tmp, index);
            }
        } else {
            tmp[0] = cmplx(c[0], zero);
            for (index, value) in c[1..].iter().enumerate() {
                *interleaved(&mut tmp, index) = *value;
            }
            if n & 1 == 0 {
                tmp[n / 2].i = zero;
            }
            for m in 1..n.div_ceil(2) {
                tmp[n - m] = cmplx(tmp[m].r, -tmp[m].i);
            }
            self.fft(&mut tmp, fct, false);
            for (target, value) in c.iter_mut().zip(&tmp) {
                *target = value.r;
            }
        }
    }
}

/// The real or imaginary part of `tmp[1 + index / 2]`: `exec_r`'s `copy_n` between a
/// halfcomplex array and the interleaved parts of a complex one.
fn interleaved<T>(tmp: &mut [Cmplx<T>], index: usize) -> &mut T {
    let value = &mut tmp[1 + index / 2];
    if index.is_multiple_of(2) {
        &mut value.r
    } else {
        &mut value.i
    }
}

/// `pocketfft_c`: a complex transform of one length.
pub(super) struct ComplexPlan<T>(ComplexMethod<T>);

enum ComplexMethod<T> {
    Direct(Cfftp<T>),
    Bluestein(Bluestein<T>),
}

impl<T: Real> ComplexPlan<T> {
    pub(super) fn new(length: usize) -> Self {
        Self(match bluestein_length(length, false) {
            Some(padded) => ComplexMethod::Bluestein(Bluestein::new(length, padded)),
            None => ComplexMethod::Direct(Cfftp::new(length)),
        })
    }

    /// Transform `c` in place, `c[k] = Σ c[j] e^(∓2πijk/n)`, and scale it by `fct`.
    pub(super) fn exec(&self, c: &mut [Cmplx<T>], fct: T, forward: bool) {
        match &self.0 {
            ComplexMethod::Direct(plan) => plan.exec(c, fct, forward),
            ComplexMethod::Bluestein(plan) => plan.fft(c, fct, forward),
        }
    }
}

/// `pocketfft_r`: a real transform of one length, to or from halfcomplex order.
pub(super) struct RealPlan<T>(RealMethod<T>);

enum RealMethod<T> {
    Direct(Rfftp<T>),
    Bluestein(Bluestein<T>),
}

impl<T: Real> RealPlan<T> {
    pub(super) fn new(length: usize) -> Self {
        Self(match bluestein_length(length, true) {
            Some(padded) => RealMethod::Bluestein(Bluestein::new(length, padded)),
            None => RealMethod::Direct(Rfftp::new(length)),
        })
    }

    /// Transform `c` in place, real to halfcomplex when `forward`, and scale it by `fct`.
    pub(super) fn exec(&self, c: &mut [T], fct: T, forward: bool) {
        match &self.0 {
            RealMethod::Direct(plan) => plan.exec(c, fct, forward),
            RealMethod::Bluestein(plan) => plan.exec_real(c, fct, forward),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angle_rounds_twice_like_x87() {
        // Reference bits from `(double)(0.25L*pi/n)` compiled by GCC on x86-64.
        for (n, bits) in [
            (1, 0x3fe9_21fb_5444_2d18_u64),
            (3, 0x3fd0_c152_382d_7366),
            (13, 0x3fae_eebf_2ca2_ada8),
            (55, 0x3f8d_3ed0_ac87_2fd2),
        ] {
            assert_eq!(quarter_pi_over(n).to_bits(), bits, "n = {n}");
        }
    }

    fn direct_dft(input: &[Cmplx<f64>], forward: bool) -> Vec<Cmplx<f64>> {
        let n = input.len();
        let sign = if forward { -1.0 } else { 1.0 };
        (0..n)
            .map(|k| {
                input
                    .iter()
                    .enumerate()
                    .fold(Cmplx::default(), |sum, (j, x)| {
                        let angle =
                            sign * 2.0 * std::f64::consts::PI * ((j * k) % n) as f64 / n as f64;
                        let (sin, cos) = angle.sin_cos();
                        sum + cmplx(x.r * cos - x.i * sin, x.r * sin + x.i * cos)
                    })
            })
            .collect()
    }

    fn signal(n: usize) -> Vec<Cmplx<f64>> {
        (0..n)
            .map(|j| {
                cmplx(
                    (j as f64 * 0.7).cos() + 0.1 * j as f64,
                    (j as f64 * 0.3).sin(),
                )
            })
            .collect()
    }

    #[test]
    fn complex_plans_match_the_direct_transform() {
        // Radices 2, 3, 4, 5, 7, 8, 11, the generic pass (13, 17) and Bluestein (101, 202).
        for n in [
            1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 16, 17, 30, 64, 77, 101, 202, 243,
        ] {
            let input = signal(n);
            for forward in [true, false] {
                let mut output = input.clone();
                ComplexPlan::new(n).exec(&mut output, 1.0, forward);
                for (got, want) in output.iter().zip(direct_dft(&input, forward)) {
                    let error = (got.r - want.r).abs().max((got.i - want.i).abs());
                    assert!(error < 1e-9 * n as f64, "n = {n}: {got:?} vs {want:?}");
                }
            }
        }
    }

    #[test]
    fn real_plans_round_trip_through_halfcomplex_order() {
        for n in [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 16, 30, 49, 101, 202] {
            let input = signal(n).iter().map(|value| value.r).collect::<Vec<_>>();
            let plan = RealPlan::new(n);
            let mut spectrum = input.clone();
            plan.exec(&mut spectrum, 1.0, true);
            let complex = input
                .iter()
                .map(|value| cmplx(*value, 0.0))
                .collect::<Vec<_>>();
            let expected = direct_dft(&complex, true);
            assert!((spectrum[0] - expected[0].r).abs() < 1e-9 * n as f64);
            for k in 1..n.div_ceil(2) {
                assert!((spectrum[2 * k - 1] - expected[k].r).abs() < 1e-9 * n as f64);
                assert!((spectrum[2 * k] - expected[k].i).abs() < 1e-9 * n as f64);
            }
            plan.exec(&mut spectrum, 1.0 / n as f64, false);
            for (got, want) in spectrum.iter().zip(&input) {
                assert!((got - want).abs() < 1e-12 * n as f64, "n = {n}");
            }
        }
    }

    #[test]
    fn plan_choice_follows_pocketfft_cost_model() {
        assert_eq!(bluestein_length(49, false), None);
        assert_eq!(bluestein_length(101, false), Some(210));
        assert_eq!(good_size_cmplx(201), 210);
        assert_eq!(largest_prime_factor(2 * 3 * 101), 101);
    }
}
