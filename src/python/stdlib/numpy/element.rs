//! Rust element types for byte-backed dtypes and the numeric value used to move between them.
//!
//! Each numeric [`Kind`](super::dtype::Kind) maps to one [`Element`] type that reads and writes
//! its little-endian bytes. Kernels are generic over `Element` and instantiated once per kind
//! through [`dispatch_numeric!`], so arithmetic happens at the dtype's own width: `int8` wraps
//! at 8 bits and `float32` rounds to single precision after every operation. `float16` is stored
//! as IEEE binary16 and computed in `f32`, rounding to nearest-even on every store, which is how
//! NumPy's half-precision loops behave.
//!
//! [`Number`] is the dtype-independent value used by casts, Python conversion, and formatting.

use std::fmt;

/// IEEE 754 binary16, stored as raw bits.
#[derive(Clone, Copy, Default, PartialEq)]
pub(in crate::python) struct F16(pub u16);

impl F16 {
    /// Round an `f32` to the nearest binary16 value, ties to even. Widening to `f64` is exact,
    /// so this rounds once.
    pub(in crate::python) fn from_f32(value: f32) -> Self {
        Self::from_f64(f64::from(value))
    }

    /// Round an `f64` to the nearest binary16 value, ties to even.
    pub(in crate::python) fn from_f64(value: f64) -> Self {
        let bits = value.to_bits();
        let sign = ((bits >> 48) & 0x8000) as u16;
        let exponent = ((bits >> 52) & 0x7ff) as i32;
        let mantissa = bits & ((1 << 52) - 1);
        if exponent == 0x7ff {
            // Infinity keeps a zero mantissa; NaN becomes a quiet NaN.
            let nan = if mantissa != 0 { 0x0200 } else { 0 };
            return Self(sign | 0x7c00 | nan);
        }
        let half_exponent = exponent - 1023 + 15;
        if half_exponent >= 0x1f {
            return Self(sign | 0x7c00);
        }
        // Normal results keep 10 fraction bits; subnormal results shift the implicit bit in.
        let (half, shift) = if half_exponent > 0 {
            (((half_exponent as u64) << 10) | (mantissa >> 42), 42)
        } else if half_exponent >= -10 {
            let shift = (43 - half_exponent) as u32;
            ((mantissa | (1 << 52)) >> shift, shift)
        } else {
            return Self(sign);
        };
        let full = if half_exponent > 0 { mantissa } else { mantissa | (1 << 52) };
        let remainder = full & ((1 << shift) - 1);
        let halfway = 1u64 << (shift - 1);
        let round_up = remainder > halfway || (remainder == halfway && half & 1 == 1);
        // A carry out of the fraction bumps the exponent, up to infinity.
        Self(sign | (half + u64::from(round_up)) as u16)
    }

    pub(in crate::python) fn to_f32(self) -> f32 {
        let sign = u32::from(self.0 & 0x8000) << 16;
        let exponent = u32::from((self.0 >> 10) & 0x1f);
        let mantissa = u32::from(self.0 & 0x03ff);
        let bits = match (exponent, mantissa) {
            (0, 0) => sign,
            (0, _) => {
                // Subnormal: normalize into f32.
                let mut exponent = 127 - 15 + 1;
                let mut mantissa = mantissa;
                while mantissa & 0x0400 == 0 {
                    mantissa <<= 1;
                    exponent -= 1;
                }
                sign | ((exponent as u32) << 23) | ((mantissa & 0x03ff) << 13)
            }
            (0x1f, 0) => sign | 0x7f80_0000,
            (0x1f, _) => sign | 0x7fc0_0000 | (mantissa << 13),
            _ => sign | ((exponent + 127 - 15) << 23) | (mantissa << 13),
        };
        f32::from_bits(bits)
    }
}

impl fmt::Debug for F16 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.to_f32())
    }
}

/// A complex number whose parts have precision `T`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(in crate::python) struct Complex<T> {
    pub re: T,
    pub im: T,
}

pub(in crate::python) type C64 = Complex<f32>;
pub(in crate::python) type C128 = Complex<f64>;

/// A dtype-independent number used for casts, Python conversion, and scalar boxing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::python) enum Number {
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Complex(f64, f64),
}

impl Number {
    pub(in crate::python) fn as_f64(self) -> f64 {
        match self {
            Self::Bool(value) => f64::from(u8::from(value)),
            Self::Int(value) => value as f64,
            Self::UInt(value) => value as f64,
            Self::Float(value) => value,
            Self::Complex(real, _) => real,
        }
    }

    pub(in crate::python) fn as_complex(self) -> (f64, f64) {
        match self {
            Self::Complex(real, imag) => (real, imag),
            other => (other.as_f64(), 0.0),
        }
    }

    pub(in crate::python) fn is_true(self) -> bool {
        match self {
            Self::Bool(value) => value,
            Self::Int(value) => value != 0,
            Self::UInt(value) => value != 0,
            Self::Float(value) => value != 0.0,
            Self::Complex(real, imag) => real != 0.0 || imag != 0.0,
        }
    }

    /// Truncate toward zero and wrap to 64 bits, as NumPy's unsafe float → int casts do for
    /// in-range values. Out-of-range and NaN floats give `i64::MIN`, matching x86-64 NumPy.
    pub(in crate::python) fn wrapping_i64(self) -> i64 {
        match self {
            Self::Bool(value) => i64::from(value),
            Self::Int(value) => value,
            Self::UInt(value) => value as i64,
            Self::Float(value) | Self::Complex(value, _) => float_to_i64(value),
        }
    }

    pub(in crate::python) fn wrapping_u64(self) -> u64 {
        match self {
            Self::UInt(value) => value,
            Self::Float(value) | Self::Complex(value, _) if value >= 9.223_372_036_854_776e18 => {
                if value < 1.844_674_407_370_955_2e19 {
                    value as u64
                } else {
                    0x8000_0000_0000_0000
                }
            }
            other => other.wrapping_i64() as u64,
        }
    }
}

fn float_to_i64(value: f64) -> i64 {
    if value.is_nan() || value >= 9.223_372_036_854_776e18 || value < -9.223_372_036_854_776e18 {
        i64::MIN
    } else {
        value as i64
    }
}

/// One byte-backed element type.
pub(in crate::python) trait Element: Copy + PartialEq + fmt::Debug + 'static {
    /// Bytes per element.
    const SIZE: usize;

    /// Read one element from exactly `SIZE` little-endian bytes.
    fn read(bytes: &[u8]) -> Self;

    /// Write this element as `SIZE` little-endian bytes.
    fn write(self, bytes: &mut [u8]);

    /// Convert with NumPy's unsafe casting: wrap integers, truncate floats toward zero, and
    /// drop imaginary parts.
    fn from_number(value: Number) -> Self;

    fn to_number(self) -> Number;

    /// The additive identity.
    fn zero() -> Self {
        Self::from_number(Number::Int(0))
    }

    /// The multiplicative identity.
    fn one() -> Self {
        Self::from_number(Number::Int(1))
    }
}

impl Element for bool {
    const SIZE: usize = 1;

    fn read(bytes: &[u8]) -> Self {
        bytes[0] != 0
    }

    fn write(self, bytes: &mut [u8]) {
        bytes[0] = u8::from(self);
    }

    fn from_number(value: Number) -> Self {
        value.is_true()
    }

    fn to_number(self) -> Number {
        Number::Bool(self)
    }
}

macro_rules! integer_element {
    ($type:ty, $variant:ident, $wrap:ident) => {
        impl Element for $type {
            const SIZE: usize = std::mem::size_of::<$type>();

            fn read(bytes: &[u8]) -> Self {
                <$type>::from_le_bytes(bytes[..Self::SIZE].try_into().expect("element width"))
            }

            fn write(self, bytes: &mut [u8]) {
                bytes[..Self::SIZE].copy_from_slice(&self.to_le_bytes());
            }

            fn from_number(value: Number) -> Self {
                value.$wrap() as $type
            }

            fn to_number(self) -> Number {
                Number::$variant(self as _)
            }
        }
    };
}

integer_element!(i8, Int, wrapping_i64);
integer_element!(i16, Int, wrapping_i64);
integer_element!(i32, Int, wrapping_i64);
integer_element!(i64, Int, wrapping_i64);
integer_element!(u8, UInt, wrapping_u64);
integer_element!(u16, UInt, wrapping_u64);
integer_element!(u32, UInt, wrapping_u64);
integer_element!(u64, UInt, wrapping_u64);

impl Element for F16 {
    const SIZE: usize = 2;

    fn read(bytes: &[u8]) -> Self {
        Self(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn write(self, bytes: &mut [u8]) {
        bytes[..2].copy_from_slice(&self.0.to_le_bytes());
    }

    fn from_number(value: Number) -> Self {
        Self::from_f64(value.as_f64())
    }

    fn to_number(self) -> Number {
        Number::Float(f64::from(self.to_f32()))
    }
}

impl Element for f32 {
    const SIZE: usize = 4;

    fn read(bytes: &[u8]) -> Self {
        f32::from_le_bytes(bytes[..4].try_into().expect("element width"))
    }

    fn write(self, bytes: &mut [u8]) {
        bytes[..4].copy_from_slice(&self.to_le_bytes());
    }

    fn from_number(value: Number) -> Self {
        value.as_f64() as f32
    }

    fn to_number(self) -> Number {
        Number::Float(f64::from(self))
    }
}

impl Element for f64 {
    const SIZE: usize = 8;

    fn read(bytes: &[u8]) -> Self {
        f64::from_le_bytes(bytes[..8].try_into().expect("element width"))
    }

    fn write(self, bytes: &mut [u8]) {
        bytes[..8].copy_from_slice(&self.to_le_bytes());
    }

    fn from_number(value: Number) -> Self {
        value.as_f64()
    }

    fn to_number(self) -> Number {
        Number::Float(self)
    }
}

impl Element for C64 {
    const SIZE: usize = 8;

    fn read(bytes: &[u8]) -> Self {
        Self {
            re: f32::read(&bytes[..4]),
            im: f32::read(&bytes[4..8]),
        }
    }

    fn write(self, bytes: &mut [u8]) {
        self.re.write(&mut bytes[..4]);
        self.im.write(&mut bytes[4..8]);
    }

    fn from_number(value: Number) -> Self {
        let (re, im) = value.as_complex();
        Self {
            re: re as f32,
            im: im as f32,
        }
    }

    fn to_number(self) -> Number {
        Number::Complex(f64::from(self.re), f64::from(self.im))
    }
}

impl Element for C128 {
    const SIZE: usize = 16;

    fn read(bytes: &[u8]) -> Self {
        Self {
            re: f64::read(&bytes[..8]),
            im: f64::read(&bytes[8..16]),
        }
    }

    fn write(self, bytes: &mut [u8]) {
        self.re.write(&mut bytes[..8]);
        self.im.write(&mut bytes[8..16]);
    }

    fn from_number(value: Number) -> Self {
        let (re, im) = value.as_complex();
        Self { re, im }
    }

    fn to_number(self) -> Number {
        Number::Complex(self.re, self.im)
    }
}

/// Run `$body` with `$T` bound to the element type of a numeric [`Kind`](super::dtype::Kind).
/// `Str` and `Object` run `$other`.
///
/// ```ignore
/// dispatch_numeric!(dtype.kind(), T => read_all::<T>(bytes), _ => unsupported())
/// ```
macro_rules! dispatch_numeric {
    ($kind:expr, $T:ident => $body:expr, _ => $other:expr) => {{
        use $crate::python::stdlib::numpy::dtype::Kind as DispatchKind;
        #[allow(unused_imports)]
        use $crate::python::stdlib::numpy::element::{C128, C64, F16};
        match $kind {
            DispatchKind::Bool => {
                type $T = bool;
                $body
            }
            DispatchKind::Int8 => {
                type $T = i8;
                $body
            }
            DispatchKind::Int16 => {
                type $T = i16;
                $body
            }
            DispatchKind::Int32 => {
                type $T = i32;
                $body
            }
            DispatchKind::Int64 => {
                type $T = i64;
                $body
            }
            DispatchKind::UInt8 => {
                type $T = u8;
                $body
            }
            DispatchKind::UInt16 => {
                type $T = u16;
                $body
            }
            DispatchKind::UInt32 => {
                type $T = u32;
                $body
            }
            DispatchKind::UInt64 => {
                type $T = u64;
                $body
            }
            DispatchKind::Float16 => {
                type $T = F16;
                $body
            }
            DispatchKind::Float32 => {
                type $T = f32;
                $body
            }
            DispatchKind::Float64 => {
                type $T = f64;
                $body
            }
            DispatchKind::Complex64 => {
                type $T = C64;
                $body
            }
            DispatchKind::Complex128 => {
                type $T = C128;
                $body
            }
            DispatchKind::Str | DispatchKind::Object => $other,
        }
    }};
}

pub(in crate::python) use dispatch_numeric;

/// Like [`dispatch_numeric!`] for bool and integer kinds only; other kinds run `$other`.
macro_rules! dispatch_integer {
    ($kind:expr, $T:ident => $body:expr, _ => $other:expr) => {{
        use $crate::python::stdlib::numpy::dtype::Kind as DispatchKind;
        match $kind {
            DispatchKind::Bool => {
                type $T = bool;
                $body
            }
            DispatchKind::Int8 => {
                type $T = i8;
                $body
            }
            DispatchKind::Int16 => {
                type $T = i16;
                $body
            }
            DispatchKind::Int32 => {
                type $T = i32;
                $body
            }
            DispatchKind::Int64 => {
                type $T = i64;
                $body
            }
            DispatchKind::UInt8 => {
                type $T = u8;
                $body
            }
            DispatchKind::UInt16 => {
                type $T = u16;
                $body
            }
            DispatchKind::UInt32 => {
                type $T = u32;
                $body
            }
            DispatchKind::UInt64 => {
                type $T = u64;
                $body
            }
            _ => $other,
        }
    }};
}

pub(in crate::python) use dispatch_integer;

/// Like [`dispatch_numeric!`] for real floating kinds only; other kinds run `$other`.
macro_rules! dispatch_real {
    ($kind:expr, $T:ident => $body:expr, _ => $other:expr) => {{
        use $crate::python::stdlib::numpy::dtype::Kind as DispatchKind;
        match $kind {
            DispatchKind::Float16 => {
                type $T = $crate::python::stdlib::numpy::element::F16;
                $body
            }
            DispatchKind::Float32 => {
                type $T = f32;
                $body
            }
            DispatchKind::Float64 => {
                type $T = f64;
                $body
            }
            _ => $other,
        }
    }};
}

pub(in crate::python) use dispatch_real;

/// Like [`dispatch_numeric!`] for complex kinds only; other kinds run `$other`.
macro_rules! dispatch_complex {
    ($kind:expr, $T:ident => $body:expr, _ => $other:expr) => {{
        use $crate::python::stdlib::numpy::dtype::Kind as DispatchKind;
        match $kind {
            DispatchKind::Complex64 => {
                type $T = $crate::python::stdlib::numpy::element::C64;
                $body
            }
            DispatchKind::Complex128 => {
                type $T = $crate::python::stdlib::numpy::element::C128;
                $body
            }
            _ => $other,
        }
    }};
}

pub(in crate::python) use dispatch_complex;

/// Read the numeric value at `bytes` for a numeric kind.
pub(in crate::python) fn read_number(kind: super::dtype::Kind, bytes: &[u8]) -> Number {
    dispatch_numeric!(kind, T => T::read(bytes).to_number(), _ => unreachable!("numeric kind"))
}

/// Write `value` at `bytes` with unsafe-cast semantics for a numeric kind.
pub(in crate::python) fn write_number(kind: super::dtype::Kind, value: Number, bytes: &mut [u8]) {
    dispatch_numeric!(kind, T => T::from_number(value).write(bytes), _ => unreachable!("numeric kind"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float16_rounds_to_nearest_even() {
        let cases = [
            (0.1, 0.099_975_585_937_5),
            (1.0 / 3.0, 0.333_251_953_125),
            (65504.0, 65504.0),
            (2049.0, 2048.0),
            (2051.0, 2052.0),
            (0.5, 0.5),
            (1e-8, 0.0),
            (6e-8, 5.960_464_477_539_063e-8),
            (70000.0, f64::INFINITY),
        ];
        for (input, expected) in cases {
            let half = F16::from_f64(input);
            assert_eq!(f64::from(half.to_f32()), expected, "{input}");
        }
        assert!(F16::from_f64(f64::NAN).to_f32().is_nan());
        assert_eq!(F16::from_f32(-0.0).0, 0x8000);
    }

    #[test]
    fn float16_round_trips_every_finite_value() {
        for bits in 0..0x7c00u16 {
            let half = F16(bits);
            assert_eq!(F16::from_f32(half.to_f32()).0, bits);
            assert_eq!(F16::from_f64(f64::from(half.to_f32())).0, bits);
        }
    }

    #[test]
    fn integer_casts_wrap_and_floats_truncate() {
        assert_eq!(u8::from_number(Number::Int(300)), 44);
        assert_eq!(i8::from_number(Number::Int(200)), -56);
        assert_eq!(u64::from_number(Number::Int(-1)), u64::MAX);
        assert_eq!(i64::from_number(Number::Float(-1.9)), -1);
        assert_eq!(i8::from_number(Number::Float(2.5)), 2);
        assert_eq!(u64::from_number(Number::Float(1e19)), 10_000_000_000_000_000_000);
    }

    #[test]
    fn elements_round_trip_through_little_endian_bytes() {
        let mut bytes = [0u8; 16];
        C128 { re: 1.5, im: -2.0 }.write(&mut bytes);
        assert_eq!(C128::read(&bytes), C128 { re: 1.5, im: -2.0 });
        (-2i16).write(&mut bytes);
        assert_eq!(&bytes[..2], &[0xfe, 0xff]);
    }
}
