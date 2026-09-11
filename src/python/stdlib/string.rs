//! Capability-free constants from Python's :mod:`string` module.
//!
//! The values are ASCII literals, matching CPython's documented module constants.  Keeping the
//! constants static avoids locale and host-environment dependence; formatting helpers can be
//! added separately when a VM adapter needs them.

use super::super::native::{ModuleDef, PyConstant, ValueDef};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "string",
    functions: &[],
    values: &[
        ValueDef::Constant {
            name: "ascii_lowercase",
            value: PyConstant::String(ASCII_LOWERCASE),
        },
        ValueDef::Constant {
            name: "ascii_uppercase",
            value: PyConstant::String(ASCII_UPPERCASE),
        },
        ValueDef::Constant {
            name: "ascii_letters",
            value: PyConstant::String(ASCII_LETTERS),
        },
        ValueDef::Constant {
            name: "digits",
            value: PyConstant::String(DIGITS),
        },
        ValueDef::Constant {
            name: "hexdigits",
            value: PyConstant::String(HEXDIGITS),
        },
        ValueDef::Constant {
            name: "octdigits",
            value: PyConstant::String(OCTDIGITS),
        },
        ValueDef::Constant {
            name: "punctuation",
            value: PyConstant::String(PUNCTUATION),
        },
        ValueDef::Constant {
            name: "whitespace",
            value: PyConstant::String(WHITESPACE),
        },
        ValueDef::Constant {
            name: "printable",
            value: PyConstant::String(PRINTABLE),
        },
    ],
};

pub const ASCII_LOWERCASE: &str = "abcdefghijklmnopqrstuvwxyz";
pub const ASCII_UPPERCASE: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
pub const ASCII_LETTERS: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
pub const DIGITS: &str = "0123456789";
pub const HEXDIGITS: &str = "0123456789abcdefABCDEF";
pub const OCTDIGITS: &str = "01234567";
pub const PUNCTUATION: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
pub const WHITESPACE: &str = " \t\n\r\x0b\x0c";
pub const PRINTABLE: &str = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c";

/// Resolve a Python `string` module constant by its public name.
#[cfg(test)]
pub fn constant(name: &str) -> Option<&'static str> {
    match name {
        "ascii_lowercase" => Some(ASCII_LOWERCASE),
        "ascii_uppercase" => Some(ASCII_UPPERCASE),
        "ascii_letters" => Some(ASCII_LETTERS),
        "digits" => Some(DIGITS),
        "hexdigits" => Some(HEXDIGITS),
        "octdigits" => Some(OCTDIGITS),
        "punctuation" => Some(PUNCTUATION),
        "whitespace" => Some(WHITESPACE),
        "printable" => Some(PRINTABLE),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        constant, ASCII_LETTERS, ASCII_LOWERCASE, DIGITS, HEXDIGITS, OCTDIGITS, PRINTABLE,
        PUNCTUATION, WHITESPACE,
    };

    #[test]
    fn constants_match_python_ascii_definitions() {
        assert_eq!(ASCII_LOWERCASE, "abcdefghijklmnopqrstuvwxyz");
        assert_eq!(ASCII_LETTERS.len(), 52);
        assert_eq!(DIGITS, "0123456789");
        assert_eq!(HEXDIGITS.len(), 22);
        assert_eq!(OCTDIGITS, "01234567");
        assert_eq!(PUNCTUATION.len(), 32);
        assert_eq!(WHITESPACE, " \t\n\r\x0b\x0c");
        assert_eq!(PRINTABLE.len(), 100);
    }

    #[test]
    fn constant_dispatch_is_closed() {
        assert_eq!(constant("ascii_lowercase"), Some(ASCII_LOWERCASE));
        assert_eq!(constant("not_a_constant"), None);
    }
}
