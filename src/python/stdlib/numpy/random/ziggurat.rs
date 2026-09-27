//! Ziggurat samplers for `Generator`'s standard normal and standard exponential draws.
//!
//! Both follow Marsaglia and Tsang's "The Ziggurat Method for Generating Random Variables"
//! (2000): 256 layers of equal area `v` under the density, the base layer extended by the tail
//! beyond the cutoff `r`. The layer tables are computed once from the paper's recursion. `r` and
//! `v` were solved to double precision so that the layers close exactly at the mode.
//!
//! A draw takes one raw word. Its low bits choose a layer. The remaining bits give a sign (normal
//! only) and a mantissa `m`, and the candidate is `m * w[layer]`. A mantissa below `k[layer]`
//! lies under the next layer up and is accepted at once. Otherwise the base layer samples the
//! tail, and any other layer tests the candidate against the density with one uniform. A
//! rejected candidate restarts with a fresh word. The bit positions of layer, sign and mantissa
//! match NumPy's, found by observing NumPy's output for chosen raw words.
//!
//! NumPy uses the same method, but its tables differ from the computed ones in the last few
//! bits. Seeded draws therefore agree with NumPy to about `1e-14` relative, and the streams stay
//! aligned unless a mantissa falls within a few units of a layer cutoff.

use std::sync::OnceLock;

use super::bitgen::BitGen;

const LAYERS: usize = 256;
const F32_MANTISSA_BITS: u32 = 23;

/// Tail cutoff and layer area of the normal ziggurat, for the density `exp(-x*x/2)`.
const NORMAL_R: f64 = 3.654152885361009;
const NORMAL_V: f64 = 0.004928673233974655;
/// The normal mantissa sits above the layer (bits 0-7) and sign (bit 8).
const NORMAL_MANTISSA_BITS: u32 = 52;

/// Tail cutoff and layer area of the exponential ziggurat, for the density `exp(-x)`.
const EXPONENTIAL_R: f64 = 7.69711747013105;
const EXPONENTIAL_V: f64 = 0.003949659822581557;
/// The exponential mantissa takes bits 11-63, above the layer (bits 3-10).
const EXPONENTIAL_MANTISSA_BITS: u32 = 53;

/// One ziggurat's tables. Layer `i >= 1` spans `[0, x[i]]` with `x[255] = r`; the base layer 0
/// is `x[0] = v / f(r)` wide, a rectangle with the area of the region under `f(r)` plus the tail.
struct Ziggurat {
    /// `x[i] / 2**bits`, so an accepted mantissa `m` in layer `i` returns `m * w[i]`.
    w: [f64; LAYERS],
    /// Mantissas below `k[i]` are under layer `i - 1`'s width and accept without a density test.
    /// The top layer has no such core, so `k[1]` is zero.
    k: [u64; LAYERS],
    /// The density at `x[i]`, with `f[0]` standing for the mode (density 1) above the top layer.
    f: [f64; LAYERS],
    /// `w` and `k` for the 23-bit mantissas of `float32` draws.
    w32: [f32; LAYERS],
    k32: [u32; LAYERS],
    r: f64,
}

impl Ziggurat {
    /// Builds the layer tables from Marsaglia and Tsang's recursion
    /// `x[i] = f⁻¹(v / x[i + 1] + f(x[i + 1]))`.
    fn new(density: fn(f64) -> f64, inverse: fn(f64) -> f64, r: f64, v: f64, bits: u32) -> Self {
        let scale = f64::from(bits).exp2();
        let mut x = [0.0; LAYERS];
        x[LAYERS - 1] = r;
        x[0] = v / density(r);
        for i in (1..LAYERS - 1).rev() {
            x[i] = inverse(v / x[i + 1] + density(x[i + 1]));
        }
        let mut k = [0; LAYERS];
        k[0] = (r / x[0] * scale) as u64;
        for i in 2..LAYERS {
            k[i] = (x[i - 1] / x[i] * scale) as u64;
        }
        let mut f = x.map(density);
        f[0] = 1.0;
        let shift = bits - F32_MANTISSA_BITS;
        Self {
            w: x.map(|width| width / scale),
            k,
            f,
            w32: x.map(|width| (width / f64::from(F32_MANTISSA_BITS).exp2()) as f32),
            k32: k.map(|cutoff| cutoff.div_ceil(1 << shift) as u32),
            r,
        }
    }

    /// The wedge test for a candidate `x` in layer `idx >= 1`: a uniform height between the
    /// layer's bottom and top must fall under the density.
    fn wedge_accepts(
        &self,
        bitgen: &mut BitGen,
        idx: usize,
        x: f64,
        density: fn(f64) -> f64,
    ) -> bool {
        let u = bitgen.next_double();
        u * (self.f[idx - 1] - self.f[idx]) + self.f[idx] < density(x)
    }
}

fn normal_density(x: f64) -> f64 {
    (-0.5 * x * x).exp()
}

fn normal_inverse(y: f64) -> f64 {
    (-2.0 * y.ln()).sqrt()
}

fn exponential_density(x: f64) -> f64 {
    (-x).exp()
}

fn exponential_inverse(y: f64) -> f64 {
    -y.ln()
}

fn normal() -> &'static Ziggurat {
    static TABLES: OnceLock<Ziggurat> = OnceLock::new();
    TABLES.get_or_init(|| {
        Ziggurat::new(
            normal_density,
            normal_inverse,
            NORMAL_R,
            NORMAL_V,
            NORMAL_MANTISSA_BITS,
        )
    })
}

fn exponential() -> &'static Ziggurat {
    static TABLES: OnceLock<Ziggurat> = OnceLock::new();
    TABLES.get_or_init(|| {
        Ziggurat::new(
            exponential_density,
            exponential_inverse,
            EXPONENTIAL_R,
            EXPONENTIAL_V,
            EXPONENTIAL_MANTISSA_BITS,
        )
    })
}

/// Marsaglia's normal tail beyond `r`: propose `r + x` with `x` exponential of rate `r`, and
/// accept with probability `exp(-x*x/2)`.
fn normal_tail(bitgen: &mut BitGen, r: f64) -> f64 {
    loop {
        let x = -(1.0 - bitgen.next_double()).ln() / r;
        let y = -(1.0 - bitgen.next_double()).ln();
        if 2.0 * y >= x * x {
            return r + x;
        }
    }
}

/// The exponential tail beyond `r` is `r` plus a fresh exponential draw, since the
/// distribution is memoryless.
fn exponential_tail(bitgen: &mut BitGen, r: f64) -> f64 {
    r - (1.0 - bitgen.next_double()).ln()
}

/// One standard normal draw from a 64-bit word.
pub(in crate::python) fn next_gauss(bitgen: &mut BitGen) -> f64 {
    let tables = normal();
    loop {
        let word = bitgen.next_u64();
        let idx = (word & 0xff) as usize;
        let sign = if (word >> 8) & 1 != 0 { -1.0 } else { 1.0 };
        let mantissa = (word >> 9) & ((1 << NORMAL_MANTISSA_BITS) - 1);
        let x = mantissa as f64 * tables.w[idx];
        if mantissa < tables.k[idx] {
            return sign * x;
        }
        if idx == 0 {
            // NumPy reads the tail's sign from bit 17 of the same word.
            let sign = if (word >> 17) & 1 != 0 { -1.0 } else { 1.0 };
            return sign * normal_tail(bitgen, tables.r);
        }
        if tables.wedge_accepts(bitgen, idx, x, normal_density) {
            return sign * x;
        }
    }
}

/// One standard normal draw for `dtype=np.float32`, from a 32-bit word with the same layout
/// and a 23-bit mantissa. The accepted value is computed in `f32` arithmetic, as NumPy does.
pub(in crate::python) fn next_gauss_f32(bitgen: &mut BitGen) -> f64 {
    let tables = normal();
    loop {
        let word = bitgen.next_u32();
        let idx = (word & 0xff) as usize;
        let sign = if (word >> 8) & 1 != 0 { -1.0 } else { 1.0 };
        let mantissa = (word >> 9) & ((1 << F32_MANTISSA_BITS) - 1);
        let x = mantissa as f32 * tables.w32[idx];
        if mantissa < tables.k32[idx] {
            return f64::from(sign * x);
        }
        if idx == 0 {
            let sign = if (word >> 17) & 1 != 0 { -1.0 } else { 1.0 };
            return sign * normal_tail(bitgen, tables.r);
        }
        if tables.wedge_accepts(bitgen, idx, f64::from(x), normal_density) {
            return f64::from(sign * x);
        }
    }
}

/// One standard exponential draw by the ziggurat method (`method="zig"`, the default).
pub(in crate::python) fn next_exponential_zig(bitgen: &mut BitGen) -> f64 {
    let tables = exponential();
    loop {
        let word = bitgen.next_u64();
        let idx = ((word >> 3) & 0xff) as usize;
        let mantissa = word >> 11;
        let x = mantissa as f64 * tables.w[idx];
        if mantissa < tables.k[idx] {
            return x;
        }
        if idx == 0 {
            return exponential_tail(bitgen, tables.r);
        }
        if tables.wedge_accepts(bitgen, idx, x, exponential_density) {
            return x;
        }
    }
}

/// One standard exponential draw for `dtype=np.float32`: a 32-bit word with the layer in bits
/// 1-8 and a 23-bit mantissa above it, computed in `f32` arithmetic.
pub(in crate::python) fn next_exponential_zig_f32(bitgen: &mut BitGen) -> f64 {
    let tables = exponential();
    loop {
        let word = bitgen.next_u32();
        let idx = ((word >> 1) & 0xff) as usize;
        let mantissa = word >> 9;
        let x = mantissa as f32 * tables.w32[idx];
        if mantissa < tables.k32[idx] {
            return f64::from(x);
        }
        if idx == 0 {
            return exponential_tail(bitgen, tables.r);
        }
        if tables.wedge_accepts(bitgen, idx, f64::from(x), exponential_density) {
            return f64::from(x);
        }
    }
}

/// One standard exponential draw by inversion (`method="inv"`), `-log(1 - U)`, which legacy
/// `RandomState.standard_exponential` always uses.
pub(in crate::python) fn next_exponential_inv(bitgen: &mut BitGen) -> f64 {
    -(1.0 - bitgen.next_double()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layers_close_at_the_mode_and_share_one_area() {
        assert_layers_close(normal(), normal_density, NORMAL_MANTISSA_BITS);
        assert_layers_close(
            exponential(),
            exponential_density,
            EXPONENTIAL_MANTISSA_BITS,
        );
    }

    fn assert_layers_close(tables: &Ziggurat, density: fn(f64) -> f64, bits: u32) {
        let scale = f64::from(bits).exp2();
        let x1 = tables.w[1] * scale;
        let top_area = x1 * (1.0 - density(x1));
        let base_area = tables.w[0] * scale * density(tables.r);
        assert!(
            (top_area / base_area - 1.0).abs() < 1e-12,
            "{top_area} {base_area}"
        );
        assert_eq!(tables.k[1], 0);
        assert_eq!(tables.f[0], 1.0);
    }
}
