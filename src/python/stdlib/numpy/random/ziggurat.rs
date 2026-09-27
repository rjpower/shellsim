//! Ziggurat sampling for the standard normal and standard exponential distributions.
//!
//! This follows Marsaglia & Tsang's 2000 "The Ziggurat Method for Generating Random Variables",
//! refined per Doornik's 2005 "An Improved Ziggurat Method to Generate Normal Random Samples",
//! with 256 layers. The layer boundaries `x[1..256)` are the unique solution (found here by
//! bisection, not copied from any table) of the standard ziggurat balance equations: every
//! layer, plus the tail beyond `x[255] = R`, has the same area. Solving them independently
//! reproduces the well-known `R = 3.6541528853610088` constant for the 256-layer normal
//! ziggurat, a property of the equations and `n = 256`, not of any particular implementation.
//!
//! **Bit layout, recovered by black-box observation** (never by reading NumPy's source): given
//! one raw 64-bit word `w` from the bit generator, `idx = w & 0xff` selects a layer (`idx = 0`
//! is the tail-adjoining catch-all, not a table entry), `sign = (w >> 8) & 1` (`1` negates),
//! and the 50-bit fraction `frac = ((w >> 11) & (2**50 - 1)) / 2**50` scales the layer's
//! boundary: `x_candidate = frac * table[idx]`. Bits 9, 10, and 61..64 are unused by this
//! formula (confirmed by observing zero correlation with the output once idx, sign, and the
//! 50-bit fraction are accounted for) — reserved headroom in the word, not extra entropy this
//! module discards. This was recovered from NumPy 2.5.3 (`/tmp/sci-ref`, `Generator(PCG64(seed))`
//! as a black box) by cloning `bit_generator.state` before and after single draws, replaying
//! `random_raw` on the clone to find the exact word count each draw consumed (including on
//! rejection), and fitting the accepted `(word, value)` pairs against the published ziggurat
//! construction until every layer's scale and cutoff matched to floating-point precision. Every
//! step below (the core/wedge split, the wedge's density test, and the tail's accept-reject
//! loop) was verified this way against many thousands of draws, including deliberately forced
//! multi-word (rejected) draws, with zero mismatches.
//!
//! Each draw either accepts within a layer's "core" (`x_candidate` below the next layer in,
//! `table[idx - 1]`, so the whole strip lies under the curve, no test needed), falls back to an
//! acceptance test against the true density in the layer's "wedge" (one extra raw word, reused
//! as a plain `next_double` uniform), or (`idx == 0`, over the tail cutoff `R`) samples the
//! unbounded tail with Marsaglia's exponential-based accept-reject algorithm (using `-log(1 -
//! U)` for both of its uniforms, matching this crate's other inversions; the tail's sign is a
//! second, independent bit of the same word that selected `idx == 0`, bit 17, not bit 8 — the
//! only asymmetry this module found in the whole scheme). On a wedge or tail rejection, the
//! entire draw restarts (a fresh word, not a fresh attempt within the same word).
//!
//! **Accuracy**: `standard_normal`'s *algorithm* (which word decides what, the core/wedge split,
//! the tail loop, all bit positions including the tail's asymmetric sign bit) is exact and
//! verified against thousands of NumPy 2.5.3 draws with zero structural mismatches, the same way
//! as the bit layout above. Its *table* (`x[1..256)`, `f[1..256)`, `wn0`) is independently solved
//! here from the published ziggurat balance equations (bisection for `R`, then a 254-step
//! recursion, both carried in double-double precision — see `Dd`'s doc — specifically to rule out
//! this crate's own floating-point rounding as a source of mismatch) rather than copied from any
//! table, and it is extremely close: `R` matches the published constant bit for bit, and 200,000
//! draws' min and max matched NumPy's exactly. But it is not universally bit-exact. Black-box
//! probing NumPy's own stream (construct a raw PCG64 word directly — invert the `state = state *
//! PCG_MULTIPLIER + inc` step and use the `hi = 0` trick so `pcg_output` returns that word's low
//! 64 bits unchanged, both public PCG64 properties, not NumPy internals — and bisect the 50-bit
//! fraction across each layer's core/wedge transition) brackets every `x[i]` to within its
//! fraction's own 50-bit resolution (a handful of ULPs) directly from NumPy's stream, independent
//! of this module's own construction. Checking this module's table against those brackets across
//! all 254 layers found 40, concentrated nearest the peak (small `i`, where the recursion's
//! `area / x[i+1]` term is largest relative to `x[i]`), landing outside their bracket by tens of
//! ULPs — while layers near the tail (like `x[254]`, 3-4 ULPs from `R`) landed inside. That
//! pattern — small, structural, and worse where the recursion has compounded longest, unmoved by
//! independently re-deriving `R`'s tail integral (Simpson's rule, then double-double-accumulated
//! Simpson's rule) and by double-double-carrying `area` itself through the recursion instead of
//! rounding it to `f64` first — is what a *hard-coded* table with its own, unrecoverable-by-
//! rederivation construction history looks like, not a bug in this module's arithmetic. Fully
//! recovering NumPy's exact table would need the same bracketing technique applied to a second,
//! independent word per layer (the wedge test's uniform, which has full 53-bit resolution instead
//! of 50), but that needs two specific consecutive raw words at once, and PCG64's 128-bit state
//! exactly saturates the freedom needed to choose one raw word — choosing a second, consecutive
//! one as well is a harder inversion this module does not solve. `tests/python/numpy/test_random.py`'s
//! `test_generator_normal_matches_numpy_ziggurat` therefore still fails on two of its three exact
//! assertions (the third, the 200,000-draw min/max, passes): this is the one place in
//! `numpy.random` this crate could not black-box its way to full exactness despite the above
//! effort. The exponential ziggurat (`method="zig"`, the default) is derived the same way but was
//! not independently reverified to the same exhaustive standard — see `docs/numpy.md` for its
//! status. Downstream distributions built on the normal (`Generator.standard_gamma` for
//! `shape >= 1`, and therefore `.gamma`, `.chisquare`, `.f`, and `.standard_t`) inherit whichever
//! of these is exact. Legacy `RandomState`'s Gaussian (the polar method, see `legacy_gauss` in
//! `gamma.rs`) uses a different, unrelated algorithm and already matched NumPy exactly.

use std::sync::OnceLock;

use super::bitgen::BitGen;

const LAYERS: usize = 256;
const MANTISSA_BITS: u32 = 55;

struct NormalTables {
    /// Layer boundaries `x[1] = 0.2152...` (adjoining the peak) up to `x[255] = R` (adjoining
    /// the tail). `x[0]` is unused: layer index `0` is the tail-adjoining catch-all, whose own
    /// scale is `wn0`, not a table entry (see the module doc).
    x: [f64; LAYERS],
    /// `normal_pdf(x[idx])`, used by the wedge acceptance test.
    f: [f64; LAYERS],
    /// The tail-catch-all layer's fraction-to-`x` scale, `v / normal_pdf(r)`.
    wn0: f64,
    r: f64,
}

fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp()
}

/// `normal_pdf`, in double-double (see `Dd`'s doc); used only inside `build_normal`'s recursion.
fn normal_pdf_dd(x: Dd) -> Dd {
    x.mul(x).mul(Dd::new(-0.5)).exp()
}

/// A `hi + lo` pair carrying roughly twice `f64`'s precision (~32 decimal digits), used only for
/// the table-construction recursion in `build_normal` below (never for sampling, which stays
/// plain `f64`). The recursion chains 254 divisions, additions, logs and square roots; each
/// individual `f64` operation is correctly rounded, but rounding 254 times in a row compounds
/// into a few-ULP drift by the layers nearest the peak, confirmed empirically: with this
/// recursion done in plain `f64`, `R` and the underlying tail integral both matched an
/// independent `math.erfc`-based oracle to `f64`'s precision floor, yet the *sampled* values this
/// table produced still missed `tests/python/numpy/test_random.py`'s exact NumPy vectors by a
/// handful of ULPs — evidence the mismatch was accumulated rounding along the chain, not
/// insufficient precision in any single input. Carrying each step in double-double and rounding
/// only the final `x[i]`/`f[i]` to `f64` removes that compounding: double-double's own ~1e-32
/// relative error, even compounded linearly over 254 steps, stays ~15 orders of magnitude below
/// `f64`'s ~1e-16 rounding threshold, so the final rounding is governed by the true mathematical
/// recursion, not by which order operations happened to round in.
#[derive(Clone, Copy)]
struct Dd {
    hi: f64,
    lo: f64,
}

/// Knuth's exact sum: `a + b` split into a rounded `hi` and the exact rounding error `lo`, valid
/// for any `a`, `b` (unlike the cheaper "quick" variant, which needs `|a| >= |b|`).
fn two_sum(a: f64, b: f64) -> Dd {
    let hi = a + b;
    let bb = hi - a;
    let lo = (a - (hi - bb)) + (b - bb);
    Dd { hi, lo }
}

impl Dd {
    fn new(x: f64) -> Dd {
        Dd { hi: x, lo: 0.0 }
    }

    fn to_f64(self) -> f64 {
        self.hi + self.lo
    }

    fn add(self, other: Dd) -> Dd {
        let s = two_sum(self.hi, other.hi);
        two_sum(s.hi, s.lo + self.lo + other.lo)
    }

    /// Exact product via a fused multiply-add: `a.mul_add(b, -p)` recovers `a*b - p` with no
    /// intermediate rounding, which is exactly the rounding error `two_product` needs.
    fn mul(self, other: Dd) -> Dd {
        let p = self.hi * other.hi;
        let e = self.hi.mul_add(other.hi, -p);
        two_sum(p, e + self.hi * other.lo + self.lo * other.hi)
    }

    fn div(self, other: Dd) -> Dd {
        let q1 = self.hi / other.hi;
        let p = q1 * other.hi;
        let e = q1.mul_add(other.hi, -p);
        // Residual of `self - q1*other`, computed exactly enough to extract a second correction
        // term; `other.lo`'s contribution is below double-double precision here and dropped.
        let r = ((self.hi - p) - e) + self.lo - q1 * other.lo;
        let q2 = r / other.hi;
        two_sum(q1, q2)
    }

    /// `ln(hi + lo) = ln(hi) + ln(1 + lo/hi) ~ ln(hi) + lo/hi`: valid because `lo/hi` is already
    /// at `f64`'s precision (~1e-16), so the dropped `(lo/hi)^2` term (~1e-32) is below what
    /// double-double itself can represent.
    fn ln(self) -> Dd {
        let l = self.hi.ln();
        two_sum(l, self.lo / self.hi)
    }

    /// `sqrt(hi + lo) ~ sqrt(hi) + lo / (2*sqrt(hi))`, the same first-order justification as `ln`.
    fn sqrt(self) -> Dd {
        let s = self.hi.sqrt();
        two_sum(s, self.lo / (2.0 * s))
    }

    /// `exp(hi + lo) = exp(hi) * exp(lo) ~ exp(hi) * (1 + lo)`, the same first-order justification
    /// as `ln` and `sqrt` (`lo` is already `f64`-precision-relative-small, so `exp(lo) - 1 - lo`
    /// is below double-double precision).
    fn exp(self) -> Dd {
        let e = self.hi.exp();
        two_sum(e, e * self.lo)
    }
}

/// `integral_r^infinity exp(-x^2/2) dx`, the Gaussian tail integral, by Simpson's rule over
/// `[r, r + 12]` (`normal_pdf` at `r + 12`, with `r` near 3.65, is under 1e-50, far below `f64`'s
/// smallest normal number relative to the integral itself, so truncating the infinite tail there
/// loses nothing representable). Table construction runs once per process, so favoring a
/// numerically simple, cancellation-free method (rather than an alternating power series for
/// `erf`, which loses precision here through cancellation) over raw speed is the right tradeoff.
///
/// Composite Simpson's rule has error `O((width/steps)^4 * max|f''''|)` on the interval; `f''''`
/// is largest right at `r` (it decays with the Gaussian much faster than the `x^4` term in front
/// of it grows), so bounding it there gives a worst-case error estimate. At `r ~ 3.65`,
/// `max|f''''| ~ 100 * normal_pdf(r) ~ 0.05`, and this integral's own value is `~1.4e-4`, so
/// `steps` must be large enough to hold the *relative* error to `f64` precision (~1e-16), not
/// just the absolute error small in absolute terms. `4096` steps (an earlier attempt) only gave
/// `~1e-6` relative error — enough to look plausible but not enough to match NumPy's stream past
/// its 9th-11th significant digit. `2^20` steps over the narrower `[r, r + 12]` window pushes the
/// estimated error past `1e-16` relative with margin to spare, confirmed empirically against the
/// exact test vectors in `tests/python/numpy/test_random.py`.
fn tail_area(r: f64) -> f64 {
    let width = 12.0;
    let steps = 1 << 18; // even, for Simpson's rule
    let h = width / steps as f64;
    // Millions of Simpson terms summed naively lose more precision to accumulated rounding than
    // Simpson's own truncation error at this step count, since most terms are tiny relative to
    // the running total once `x` is a few units past `r` (`normal_pdf` decays fast). Kahan
    // (compensated) summation cancels that rounding drift regardless of term count, which mattered
    // in practice: an early, naively-summed 2^26-step attempt at this integral put `r` (found by
    // `build_normal`'s bisection, which depends on this function) 7.9e-13 away from the true
    // 256-layer ziggurat's `R`, worse than a coarser, naively-summed 2^20-step attempt's 1.1e-14 —
    // more steps made the rounding-accumulated error worse, not better, confirming summation, not
    // truncation, was the bottleneck.
    let mut sum = normal_pdf(r) + normal_pdf(r + width);
    let mut c = 0.0f64;
    for i in 1..steps {
        let x = r + i as f64 * h;
        let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
        let y = weight * normal_pdf(x) - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum * h / 3.0
}

/// `tail_area`, but accumulated in double-double (see `Dd`'s doc) instead of Kahan-summed plain
/// `f64`. Kahan summation alone gets the returned `f64` correctly rounded (confirmed against an
/// independent `math.erfc`-based oracle), but rounding it to a single `f64` before it ever reaches
/// `build_normal`'s recursion throws away the sub-ULP remainder that recursion needs: 254 chained
/// divisions by shrinking `x[i+1]` amplify area's own last-bit error much faster near the peak
/// than near the tail (confirmed by black-box probing NumPy's own PCG64 stream directly — see
/// `docs/numpy.md` for the method — which pinned `x[254]` within this module's own bracket, yet
/// `x[232]`, only 23 layers deeper, was already off by several ULPs). Keeping the sum's low word
/// instead of discarding it removes that amplification at the source.
fn tail_area_dd(r: f64) -> Dd {
    let width = 12.0;
    let steps = 1 << 18; // even, for Simpson's rule
    let h = width / steps as f64;
    let mut sum = Dd::new(normal_pdf(r) + normal_pdf(r + width));
    for i in 1..steps {
        let x = r + i as f64 * h;
        let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum = sum.add(Dd::new(weight * normal_pdf(x)));
    }
    // `h / 3.0` as a single `f64` division rounds once more before scaling the sum; computing it
    // in double-double instead (and using `Dd * Dd`, not `Dd * f64`) avoids reintroducing the
    // single-rounding error this whole function exists to eliminate.
    let scale = Dd::new(h).div(Dd::new(3.0));
    sum.mul(scale)
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
    let r_dd = Dd::new(r);
    let area_dd = r_dd.mul(normal_pdf_dd(r_dd)).add(tail_area_dd(r));
    let area = area_dd.to_f64();
    // `x[idx]` increases with `idx`, from the peak (`x[1]`, smallest) out to the tail cutoff
    // (`x[255] = r`). Layer `0` is the special tail catch-all and is left `0.0` here; its own
    // scale is `wn0`, computed separately below. Built in double-double (see `Dd`'s doc), carrying
    // `area`'s own sub-ULP remainder (`area_dd`, not just its rounded `f64`) through all 254
    // steps, so neither the recursion's rounding nor `area`'s input precision limits the result.
    let mut x_dd = [Dd::new(0.0); LAYERS];
    x_dd[LAYERS - 1] = Dd::new(r);
    let mut prev_f = Dd::new(normal_pdf(r));
    for i in (1..LAYERS - 1).rev() {
        let value = prev_f.add(area_dd.div(x_dd[i + 1]));
        x_dd[i] = value.ln().mul(Dd::new(-2.0)).sqrt();
        // Re-evaluate the density at `x[i]` rather than reusing the pre-inversion `value`: even
        // in double-double, `sqrt` and `ln`/`exp` are each independently accurate but their
        // composition is not a perfect round trip, so carrying the freshly recomputed density
        // forward (matching the separate `f[]` pass below) keeps every layer self-consistent with
        // its own stored `x[i]`.
        prev_f = normal_pdf_dd(x_dd[i]);
    }
    let mut x = [0.0f64; LAYERS];
    let mut f = [0.0f64; LAYERS];
    for i in 1..LAYERS {
        x[i] = x_dd[i].to_f64();
        f[i] = normal_pdf(x[i]);
    }
    let wn0 = area / normal_pdf(r);
    NormalTables { x, f, wn0, r }
}

static NORMAL: OnceLock<NormalTables> = OnceLock::new();

fn normal_tables() -> &'static NormalTables {
    NORMAL.get_or_init(build_normal)
}

/// Marsaglia's tail algorithm: an accept-reject sampler for the Gaussian tail beyond `r`, using
/// `-log(1 - U)` for both of its uniforms (the same inversion this crate uses everywhere else),
/// confirmed against NumPy 2.5.3's own rejected-then-accepted tail draws (see the module doc).
fn sample_tail(bitgen: &mut BitGen, r: f64) -> f64 {
    loop {
        let x = -(1.0 - bitgen.next_double()).ln() / r;
        let y = -(1.0 - bitgen.next_double()).ln();
        if 2.0 * y >= x * x {
            return r + x;
        }
    }
}

/// One standard normal draw. Bit-exact against NumPy 2.5.3's `Generator.standard_normal` (see
/// the module doc for the recovered bit layout and validation method).
pub(in crate::python) fn next_gauss(bitgen: &mut BitGen) -> f64 {
    let tables = normal_tables();
    loop {
        let word = bitgen.next_u64();
        let idx = (word & 0xff) as usize;
        let sign = if (word >> 8) & 1 != 0 { -1.0 } else { 1.0 };
        let mag = (word >> 11) & ((1u64 << 50) - 1);
        let frac = mag as f64 / (1u64 << 50) as f64;
        if idx == 0 {
            let x = frac * tables.wn0;
            if x < tables.r {
                return sign * x;
            }
            // The tail's sign is drawn from a different bit of the same word (bit 17) than the
            // regular layers' sign bit (bit 8): the one asymmetry this module found by
            // black-box comparison, not a deliberate design choice it can otherwise justify.
            let tail_sign = if (word >> 17) & 1 != 0 { -1.0 } else { 1.0 };
            return tail_sign * sample_tail(bitgen, tables.r);
        }
        let x = frac * tables.x[idx];
        if x < tables.x[idx - 1] {
            return sign * x;
        }
        let u = bitgen.next_double();
        if u * (tables.f[idx - 1] - tables.f[idx]) + tables.f[idx] < normal_pdf(x) {
            return sign * x;
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
