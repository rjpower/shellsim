//! NumPy data types: a closed table of element kinds, name parsing, and NEP 50 promotion.
//!
//! Every dtype is a [`DType`]: a [`Kind`] discriminant plus a character width for `str` arrays.
//! Metadata comes from a `const` table indexed by the discriminant, so lookups never scan.
//!
//! Storage is always little-endian. A multi-byte numeric dtype may still be big-endian (`>i4`):
//! the byte order is an attribute of the descriptor that applies only where NumPy exposes raw
//! bytes, such as `tobytes`, `frombuffer`, and `.npy` files. Computation reads the native
//! storage, and every computed dtype (promotion, ufunc and reduction results, scalars) is
//! native, as NumPy's `ensure_dtype_nbo` makes it. Big-endian strings, bytes, structured, and
//! datetime types are explicit frontiers.
//!
//! Promotion follows NumPy 2 (NEP 50): arrays and NumPy scalars are *strong* and promote by
//! [`promote`]; Python `bool`, `int`, `float`, and `complex` operands are *weak* and adopt the
//! strong dtype when their category allows it ([`promote_weak`]).

use super::super::super::native::{PyArrayDtype, PyError, PyResult};

/// Element kinds in NumPy's type-number order.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(in crate::python) enum Kind {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float16,
    Float32,
    Float64,
    Complex64,
    Complex128,
    Str,
    Object,
}

/// Every kind, indexed by its discriminant.
pub(in crate::python) const KINDS: [Kind; 16] = [
    Kind::Bool,
    Kind::Int8,
    Kind::Int16,
    Kind::Int32,
    Kind::Int64,
    Kind::UInt8,
    Kind::UInt16,
    Kind::UInt32,
    Kind::UInt64,
    Kind::Float16,
    Kind::Float32,
    Kind::Float64,
    Kind::Complex64,
    Kind::Complex128,
    Kind::Str,
    Kind::Object,
];

/// NumPy's dtype `kind` character classes, in same-kind casting order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::python) enum Category {
    Bool,
    Unsigned,
    Signed,
    Float,
    Complex,
    Str,
    Object,
}

struct KindInfo {
    name: &'static str,
    char: char,
    category: Category,
    /// Bytes per element; for `Str` this is the size of one character.
    itemsize: usize,
}

const INFO: [KindInfo; 16] = [
    info("bool", '?', Category::Bool, 1),
    info("int8", 'b', Category::Signed, 1),
    info("int16", 'h', Category::Signed, 2),
    info("int32", 'i', Category::Signed, 4),
    info("int64", 'l', Category::Signed, 8),
    info("uint8", 'B', Category::Unsigned, 1),
    info("uint16", 'H', Category::Unsigned, 2),
    info("uint32", 'I', Category::Unsigned, 4),
    info("uint64", 'L', Category::Unsigned, 8),
    info("float16", 'e', Category::Float, 2),
    info("float32", 'f', Category::Float, 4),
    info("float64", 'd', Category::Float, 8),
    info("complex64", 'F', Category::Complex, 8),
    info("complex128", 'D', Category::Complex, 16),
    info("str", 'U', Category::Str, 4),
    info("object", 'O', Category::Object, 8),
];

const fn info(name: &'static str, char: char, category: Category, itemsize: usize) -> KindInfo {
    KindInfo {
        name,
        char,
        category,
        itemsize,
    }
}

impl Kind {
    fn info(self) -> &'static KindInfo {
        &INFO[self as usize]
    }

    pub(in crate::python) fn category(self) -> Category {
        self.info().category
    }

    pub(in crate::python) fn name(self) -> &'static str {
        self.info().name
    }

    /// Bit width of a numeric kind; 0 for `Str` and `Object`.
    pub(in crate::python) fn bits(self) -> u32 {
        match self.category() {
            Category::Str | Category::Object => 0,
            _ => self.info().itemsize as u32 * 8,
        }
    }

    fn from_index(index: u8) -> Option<Self> {
        KINDS.get(index as usize).copied()
    }
}

/// One NumPy dtype. `chars` is meaningful only for [`Kind::Str`], and `big_endian` only for
/// numeric kinds wider than one byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::python) struct DType {
    kind: Kind,
    chars: u32,
    big_endian: bool,
}

/// Bit of a packed dtype that marks big-endian byte order; kinds use the bits below it.
const BIG_ENDIAN_BIT: u32 = 0x80;

/// Largest `str` width accepted, which keeps `itemsize` and the storage tag in range.
pub(in crate::python) const MAX_STR_CHARS: u32 = (1 << 24) - 1;

impl DType {
    pub(in crate::python) const BOOL: Self = Self::of(Kind::Bool);
    pub(in crate::python) const INT8: Self = Self::of(Kind::Int8);
    pub(in crate::python) const INT16: Self = Self::of(Kind::Int16);
    pub(in crate::python) const INT32: Self = Self::of(Kind::Int32);
    pub(in crate::python) const INT64: Self = Self::of(Kind::Int64);
    pub(in crate::python) const UINT8: Self = Self::of(Kind::UInt8);
    pub(in crate::python) const UINT16: Self = Self::of(Kind::UInt16);
    pub(in crate::python) const UINT32: Self = Self::of(Kind::UInt32);
    pub(in crate::python) const UINT64: Self = Self::of(Kind::UInt64);
    pub(in crate::python) const FLOAT16: Self = Self::of(Kind::Float16);
    pub(in crate::python) const FLOAT32: Self = Self::of(Kind::Float32);
    pub(in crate::python) const FLOAT64: Self = Self::of(Kind::Float64);
    pub(in crate::python) const COMPLEX64: Self = Self::of(Kind::Complex64);
    pub(in crate::python) const COMPLEX128: Self = Self::of(Kind::Complex128);
    pub(in crate::python) const OBJECT: Self = Self::of(Kind::Object);

    /// The dtype of a fixed-size kind. Use [`DType::str`] for strings.
    pub(in crate::python) const fn of(kind: Kind) -> Self {
        Self {
            kind,
            chars: 0,
            big_endian: false,
        }
    }

    /// A `<U{chars}` string dtype.
    pub(in crate::python) fn str(chars: usize) -> PyResult<Self> {
        let chars = u32::try_from(chars)
            .ok()
            .filter(|chars| *chars <= MAX_STR_CHARS)
            .ok_or_else(|| PyError::value_error("string dtype is too wide"))?;
        Ok(Self {
            kind: Kind::Str,
            chars,
            big_endian: false,
        })
    }

    /// This dtype in big-endian byte order. Kinds without a byte order (one-byte numbers,
    /// strings, and objects) are unchanged.
    pub(in crate::python) fn big_endian(self) -> Self {
        Self {
            big_endian: self.has_byte_order(),
            ..self
        }
    }

    /// This dtype in the host's (little-endian) byte order.
    pub(in crate::python) fn native(self) -> Self {
        Self {
            big_endian: false,
            ..self
        }
    }

    /// Whether elements are in the host's byte order, NumPy's `dtype.isnative`.
    pub(in crate::python) fn is_native(self) -> bool {
        !self.big_endian
    }

    /// Whether byte order means anything for this dtype. NumPy also gives `str` a byte order,
    /// which is not modeled.
    fn has_byte_order(self) -> bool {
        self.is_numeric() && self.itemsize() > 1
    }

    /// NumPy's `dtype.byteorder`: `|` when byte order does not apply, `>` for big-endian, and
    /// `=` for native.
    pub(in crate::python) fn byte_order(self) -> char {
        if self.big_endian {
            '>'
        } else if self.itemsize() == 1 || self.kind == Kind::Object {
            '|'
        } else {
            '='
        }
    }

    pub(in crate::python) fn kind(self) -> Kind {
        self.kind
    }

    pub(in crate::python) fn category(self) -> Category {
        self.kind.category()
    }

    /// Characters per element of a string dtype.
    pub(in crate::python) fn chars(self) -> usize {
        self.chars as usize
    }

    pub(in crate::python) fn itemsize(self) -> usize {
        match self.kind {
            Kind::Str => self.chars as usize * 4,
            kind => kind.info().itemsize,
        }
    }

    pub(in crate::python) fn is_numeric(self) -> bool {
        !matches!(self.category(), Category::Str | Category::Object)
    }

    pub(in crate::python) fn is_integer(self) -> bool {
        matches!(self.category(), Category::Signed | Category::Unsigned)
    }

    pub(in crate::python) fn is_inexact(self) -> bool {
        matches!(self.category(), Category::Float | Category::Complex)
    }

    /// NumPy's `dtype.name`, e.g. `int64` or `str160` for `<U5`.
    pub(in crate::python) fn name(self) -> String {
        match self.kind {
            Kind::Str => format!("str{}", self.itemsize() * 8),
            kind => kind.name().to_string(),
        }
    }

    /// NumPy's `dtype.char`.
    pub(in crate::python) fn char(self) -> char {
        self.kind.info().char
    }

    /// NumPy's `dtype.kind` character.
    pub(in crate::python) fn kind_char(self) -> char {
        match self.category() {
            Category::Bool => 'b',
            Category::Unsigned => 'u',
            Category::Signed => 'i',
            Category::Float => 'f',
            Category::Complex => 'c',
            Category::Str => 'U',
            Category::Object => 'O',
        }
    }

    /// NumPy's `dtype.str` array-protocol descriptor, e.g. `<i8`, `|b1`, `<U5`, `|O`.
    pub(in crate::python) fn descr(self) -> String {
        match self.kind {
            Kind::Str => format!("<U{}", self.chars),
            Kind::Object => "|O".to_string(),
            _ => {
                let order = match self.byte_order() {
                    '=' => '<',
                    order => order,
                };
                format!("{order}{}{}", self.kind_char(), self.itemsize())
            }
        }
    }

    /// `str(dtype)`: the name for native numbers and object, the descriptor otherwise.
    pub(in crate::python) fn display(self) -> String {
        match self.kind {
            Kind::Str => self.descr(),
            _ if self.big_endian => self.descr(),
            kind => kind.name().to_string(),
        }
    }

    /// `repr(dtype)`, e.g. `dtype('int64')`, `dtype('<U5')`, `dtype('O')`.
    pub(in crate::python) fn repr(self) -> String {
        match self.kind {
            Kind::Object => "dtype('O')".to_string(),
            _ => format!("dtype('{}')", self.display()),
        }
    }

    /// The opaque storage description handed to the runtime.
    pub(in crate::python) fn storage(self) -> PyArrayDtype {
        let tag = self.pack() as u32;
        if self.kind == Kind::Object {
            PyArrayDtype::values(tag)
        } else {
            PyArrayDtype::bytes(tag, self.itemsize() as u32)
        }
    }

    /// Recover the dtype recorded by [`DType::storage`].
    pub(in crate::python) fn from_storage(storage: PyArrayDtype) -> Self {
        Self::unpack(u64::from(storage.tag())).expect("arrays carry a valid dtype tag")
    }

    /// Payload of a `numpy.dtype` value.
    pub(in crate::python) fn pack(self) -> u64 {
        let order = if self.big_endian { BIG_ENDIAN_BIT } else { 0 };
        u64::from(self.kind as u32 | order | self.chars << 8)
    }

    pub(in crate::python) fn unpack(payload: u64) -> Option<Self> {
        let kind = Kind::from_index((payload & u64::from(BIG_ENDIAN_BIT - 1)) as u8)?;
        let chars = u32::try_from(payload >> 8).ok()?;
        let dtype = Self {
            kind,
            chars,
            big_endian: payload & u64::from(BIG_ENDIAN_BIT) != 0,
        };
        let valid =
            (kind == Kind::Str || chars == 0) && (!dtype.big_endian || dtype.has_byte_order());
        valid.then_some(dtype)
    }

    /// Parse a dtype string such as `int8`, `<i8`, `f`, `U5`, or `object`.
    pub(in crate::python) fn parse(text: &str) -> PyResult<Self> {
        let not_understood = || PyError::type_error(format!("data type '{text}' not understood"));
        if let Some(body) = text.strip_prefix('>').filter(|body| !body.is_empty()) {
            let dtype = Self::parse(body)?;
            if dtype.kind == Kind::Str {
                return Err(PyError::not_implemented_error(format!(
                    "big-endian dtype '{text}' is not supported"
                )));
            }
            return Ok(dtype.big_endian());
        }
        let body = match text.as_bytes().first() {
            Some(b'<' | b'=' | b'|') => &text[1..],
            _ => text,
        };
        if let Some(width) = body.strip_prefix('U') {
            let chars = if width.is_empty() {
                0
            } else {
                width.parse::<usize>().map_err(|_| not_understood())?
            };
            return Self::str(chars);
        }
        let kind = match body {
            "bool" | "?" | "b1" | "bool_" => Kind::Bool,
            "int8" | "i1" | "b" | "byte" => Kind::Int8,
            "int16" | "i2" | "h" | "short" => Kind::Int16,
            "int32" | "i4" | "i" | "intc" => Kind::Int32,
            "int64" | "i8" | "l" | "q" | "int" | "int_" | "long" | "longlong" | "intp" | "p" => {
                Kind::Int64
            }
            "uint8" | "u1" | "B" | "ubyte" => Kind::UInt8,
            "uint16" | "u2" | "H" | "ushort" => Kind::UInt16,
            "uint32" | "u4" | "I" | "uintc" => Kind::UInt32,
            "uint64" | "u8" | "L" | "Q" | "uint" | "ulong" | "ulonglong" | "uintp" | "P" => {
                Kind::UInt64
            }
            "float16" | "f2" | "e" | "half" => Kind::Float16,
            "float32" | "f4" | "f" | "single" => Kind::Float32,
            "float64" | "f8" | "d" | "float" | "double" => Kind::Float64,
            "complex64" | "c8" | "F" | "csingle" => Kind::Complex64,
            "complex128" | "c16" | "D" | "complex" | "cdouble" => Kind::Complex128,
            "object" | "O" | "O8" | "object_" => Kind::Object,
            "str" | "str_" | "unicode" => return Self::str(0),
            "g" | "G" | "f16" | "c32" | "longdouble" | "clongdouble" | "S" | "a" | "V"
            | "bytes" | "bytes_" | "void" | "M" | "m" | "M8" | "m8" | "datetime64"
            | "timedelta64" => {
                return Err(PyError::not_implemented_error(format!(
                    "NumPy dtype '{text}' is not supported"
                )))
            }
            // Comma-separated strings are structured dtypes, and `V<n>` is raw void storage.
            _ if body.starts_with('S')
                || body.starts_with('V')
                || body.contains(',')
                || body.starts_with("M8[")
                || body.starts_with("m8[")
                || body.starts_with("datetime64[")
                || body.starts_with("timedelta64[") =>
            {
                return Err(PyError::not_implemented_error(format!(
                    "NumPy dtype '{text}' is not supported"
                )))
            }
            _ => return Err(not_understood()),
        };
        Ok(Self::of(kind))
    }

    /// Smallest float dtype that holds every value of an integer or bool dtype, as NumPy's
    /// float ufuncs choose it: 8-bit → float16, 16-bit → float32, wider → float64.
    pub(in crate::python) fn smallest_float_for(self) -> Self {
        match self.category() {
            Category::Bool => Self::FLOAT16,
            Category::Signed | Category::Unsigned => match self.kind.bits() {
                8 => Self::FLOAT16,
                16 => Self::FLOAT32,
                _ => Self::FLOAT64,
            },
            _ => self,
        }
    }

    /// The real dtype underlying a complex dtype (`complex64` → `float32`).
    pub(in crate::python) fn real_part(self) -> Self {
        match self.kind {
            Kind::Complex64 => Self::FLOAT32,
            Kind::Complex128 => Self::FLOAT64,
            _ => self,
        }
    }

    /// The complex dtype whose parts have this float dtype's precision.
    pub(in crate::python) fn complex_for(self) -> Self {
        match self.kind {
            Kind::Float16 | Kind::Float32 | Kind::Complex64 => Self::COMPLEX64,
            _ => Self::COMPLEX128,
        }
    }
}

fn signed_with_bits(bits: u32) -> DType {
    match bits {
        8 => DType::INT8,
        16 => DType::INT16,
        32 => DType::INT32,
        _ => DType::INT64,
    }
}

fn wider(left: DType, right: DType) -> DType {
    if left.itemsize() >= right.itemsize() {
        left
    } else {
        right
    }
}

/// `np.promote_types` for two strong dtypes. Strings promote only with strings; mixing a string
/// with a number has no common dtype and raises `TypeError`.
pub(in crate::python) fn promote(left: DType, right: DType) -> PyResult<DType> {
    use Category::*;
    let (left, right) = (left.native(), right.native());
    let (a, b) = if left.category() <= right.category() {
        (left, right)
    } else {
        (right, left)
    };
    Ok(match (a.category(), b.category()) {
        (_, Object) => DType::OBJECT,
        (Str, Str) => DType::str(a.chars().max(b.chars()))?,
        (_, Str) => {
            return Err(PyError::type_error(format!(
                "{} cannot be promoted with a string dtype",
                a.repr()
            )))
        }
        (Bool, _) => b,
        (Unsigned, Unsigned) | (Signed, Signed) | (Float, Float) | (Complex, Complex) => {
            wider(a, b)
        }
        (Unsigned, Signed) => {
            if b.kind.bits() > a.kind.bits() {
                b
            } else if a.kind.bits() < 64 {
                signed_with_bits(a.kind.bits() * 2)
            } else {
                DType::FLOAT64
            }
        }
        (Unsigned | Signed, Float) => wider(a.smallest_float_for(), b),
        (Unsigned | Signed, Complex) => {
            let float = a.smallest_float_for();
            wider(float.complex_for(), b)
        }
        (Float, Complex) => wider(a.complex_for(), b),
        _ => unreachable!("categories are ordered before matching"),
    })
}

/// Category of a weak Python scalar operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::python) enum Weak {
    Bool,
    Int,
    Float,
    Complex,
}

impl Weak {
    /// The dtype a weak scalar takes on its own, as in `np.result_type(3)`.
    pub(in crate::python) fn default_dtype(self) -> DType {
        match self {
            Self::Bool => DType::BOOL,
            Self::Int => DType::INT64,
            Self::Float => DType::FLOAT64,
            Self::Complex => DType::COMPLEX128,
        }
    }
}

/// NEP 50: the result dtype of a strong dtype combined with a weak Python scalar.
pub(in crate::python) fn promote_weak(strong: DType, weak: Weak) -> PyResult<DType> {
    use Category::*;
    let strong = strong.native();
    Ok(match (strong.category(), weak) {
        (Object, _) => DType::OBJECT,
        (Str, _) => return promote(strong, weak.default_dtype()),
        (_, Weak::Bool) => strong,
        (Bool, Weak::Int) => DType::INT64,
        (Unsigned | Signed, Weak::Int) => strong,
        (Bool | Unsigned | Signed, Weak::Float) => DType::FLOAT64,
        (Float | Complex, Weak::Int | Weak::Float) => strong,
        (Bool | Unsigned | Signed, Weak::Complex) => DType::COMPLEX128,
        (Float | Complex, Weak::Complex) => strong.complex_for(),
    })
}

/// NEP 50 result type of strong dtypes (arrays, NumPy scalars, dtype specifications) and weak
/// Python scalars, as used by ufuncs and `np.result_type`.
pub(in crate::python) fn result_type(strong: &[DType], weak: &[Weak]) -> PyResult<DType> {
    let Some((first, rest)) = strong.split_first() else {
        return Ok(weak
            .iter()
            .max()
            .map_or(DType::FLOAT64, |weak| weak.default_dtype()));
    };
    let mut result = first.native();
    for dtype in rest {
        result = promote(result, *dtype)?;
    }
    for weak in weak {
        result = promote_weak(result, *weak)?;
    }
    Ok(result)
}

/// NumPy casting rules for `can_cast`, `astype(casting=...)`, and ufunc outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::python) enum Casting {
    No,
    Equiv,
    Safe,
    SameKind,
    Unsafe,
}

impl Casting {
    pub(in crate::python) fn parse(text: &str) -> PyResult<Self> {
        Ok(match text {
            "no" => Self::No,
            "equiv" => Self::Equiv,
            "safe" => Self::Safe,
            "same_kind" => Self::SameKind,
            "unsafe" => Self::Unsafe,
            _ => {
                return Err(PyError::value_error(
                    "casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe'",
                ))
            }
        })
    }

    pub(in crate::python) fn name(self) -> &'static str {
        match self {
            Self::No => "no",
            Self::Equiv => "equiv",
            Self::Safe => "safe",
            Self::SameKind => "same_kind",
            Self::Unsafe => "unsafe",
        }
    }
}

/// Characters needed to print any value of a numeric dtype, used for number → str casts.
pub(in crate::python) fn str_width_for(dtype: DType) -> usize {
    match dtype.kind {
        Kind::Bool => 5,
        Kind::Int8 => 4,
        Kind::UInt8 => 3,
        Kind::Int16 => 6,
        Kind::UInt16 => 5,
        Kind::Int32 => 11,
        Kind::UInt32 => 10,
        Kind::Int64 => 21,
        Kind::UInt64 => 20,
        Kind::Float16 | Kind::Float32 | Kind::Float64 => 32,
        Kind::Complex64 | Kind::Complex128 => 64,
        Kind::Str => dtype.chars(),
        Kind::Object => 0,
    }
}

/// Whether every value of `from` converts to `to` without loss. Byte order never loses values.
fn safe_cast(from: DType, to: DType) -> bool {
    use Category::*;
    let (from, to) = (from.native(), to.native());
    if from == to || to.category() == Object {
        return true;
    }
    match (from.category(), to.category()) {
        (Str, Str) => to.chars() >= from.chars(),
        (Object, _) | (Str, _) => false,
        (_, Str) => to.chars() >= str_width_for(from),
        _ => promote(from, to).is_ok_and(|promoted| promoted == to),
    }
}

/// `np.can_cast(from, to, casting)`.
pub(in crate::python) fn can_cast(from: DType, to: DType, casting: Casting) -> bool {
    match casting {
        Casting::No => from == to,
        Casting::Equiv => from.native() == to.native(),
        Casting::Safe => safe_cast(from, to),
        Casting::SameKind => {
            safe_cast(from, to)
                || (from.is_numeric() && to.is_numeric() && from.category() <= to.category())
        }
        Casting::Unsafe => true,
    }
}

/// Default result dtype of `sum`, `prod`, `cumsum`, and `cumprod`: small integers and bool
/// accumulate in the platform integer so they do not wrap.
pub(in crate::python) fn accumulator(dtype: DType) -> DType {
    match dtype.category() {
        Category::Bool | Category::Signed => DType::INT64,
        Category::Unsigned => DType::UINT64,
        _ => dtype,
    }
}

/// Result dtype of true division and of float-valued ufuncs such as `sqrt`.
pub(in crate::python) fn true_divide_dtype(dtype: DType) -> DType {
    match dtype.category() {
        Category::Bool | Category::Signed | Category::Unsigned => DType::FLOAT64,
        _ => dtype,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dtype(text: &str) -> DType {
        DType::parse(text).unwrap()
    }

    #[test]
    fn metadata_comes_from_the_table() {
        assert_eq!(dtype("i4").name(), "int32");
        assert_eq!(dtype("<i8").descr(), "<i8");
        assert_eq!(dtype("?").descr(), "|b1");
        assert_eq!(dtype("U5").itemsize(), 20);
        assert_eq!(dtype("U5").name(), "str160");
        assert_eq!(dtype("U5").repr(), "dtype('<U5')");
        assert_eq!(dtype("O").repr(), "dtype('O')");
        assert_eq!(dtype("F").name(), "complex64");
        for kind in KINDS {
            assert_eq!(
                kind as usize,
                KINDS.iter().position(|k| *k == kind).unwrap()
            );
        }
    }

    #[test]
    fn packing_round_trips() {
        for text in ["bool", "u2", "f2", "c16", "U7", "O"] {
            let dtype = dtype(text);
            assert_eq!(DType::unpack(dtype.pack()), Some(dtype));
            assert_eq!(DType::from_storage(dtype.storage()), dtype);
        }
        assert_eq!(DType::unpack(0xff), None);
    }

    #[test]
    fn unknown_and_unsupported_names_fail_differently() {
        assert_eq!(
            DType::parse("i3").unwrap_err().message(),
            "data type 'i3' not understood"
        );
        assert!(DType::parse("M8[ns]")
            .unwrap_err()
            .is_exception("NotImplementedError"));
    }

    #[test]
    fn strong_promotion_matches_numpy() {
        let cases = [
            ("int8", "uint8", "int16"),
            ("uint32", "int32", "int64"),
            ("uint8", "int64", "int64"),
            ("int64", "uint64", "float64"),
            ("bool", "float16", "float16"),
            ("int16", "float16", "float32"),
            ("int32", "float32", "float64"),
            ("int16", "complex64", "complex64"),
            ("int32", "complex64", "complex128"),
            ("float64", "complex64", "complex128"),
            ("U3", "U5", "U5"),
            ("int64", "O", "O"),
        ];
        for (left, right, expected) in cases {
            assert_eq!(promote(dtype(left), dtype(right)).unwrap(), dtype(expected));
            assert_eq!(promote(dtype(right), dtype(left)).unwrap(), dtype(expected));
        }
        assert!(promote(dtype("U1"), dtype("i8")).is_err());
    }

    #[test]
    fn weak_scalars_follow_nep_50() {
        assert_eq!(promote_weak(DType::INT8, Weak::Int).unwrap(), DType::INT8);
        assert_eq!(promote_weak(DType::BOOL, Weak::Int).unwrap(), DType::INT64);
        assert_eq!(
            promote_weak(DType::INT8, Weak::Float).unwrap(),
            DType::FLOAT64
        );
        assert_eq!(
            promote_weak(DType::FLOAT32, Weak::Complex).unwrap(),
            DType::COMPLEX64
        );
        assert_eq!(
            promote_weak(DType::FLOAT16, Weak::Float).unwrap(),
            DType::FLOAT16
        );
    }

    #[test]
    fn casting_rules_match_numpy() {
        assert!(can_cast(DType::INT8, DType::INT16, Casting::Safe));
        assert!(!can_cast(DType::UINT8, DType::INT8, Casting::Safe));
        assert!(can_cast(DType::INT64, DType::FLOAT64, Casting::Safe));
        assert!(can_cast(DType::INT64, DType::COMPLEX128, Casting::Safe));
        assert!(!can_cast(DType::FLOAT64, DType::FLOAT32, Casting::Safe));
        assert!(can_cast(DType::FLOAT64, DType::FLOAT32, Casting::SameKind));
        assert!(!can_cast(DType::FLOAT64, DType::INT64, Casting::SameKind));
        assert!(!can_cast(DType::INT8, DType::UINT8, Casting::SameKind));
        assert!(can_cast(DType::UINT64, DType::INT8, Casting::SameKind));
        assert!(!can_cast(DType::INT16, DType::FLOAT16, Casting::Safe));
    }
}
