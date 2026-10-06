//! Python string and bytes storage, borrowing, indexing and quoting.
//!
//! Python-visible strings are distinct from interned source identifiers. Short values stay
//! inline in [`Value`]; longer values use [`PyString`] in the heap. Consumers inspect either
//! representation through [`PyStringRef`] without allocating. The `repr()` quoting of text and
//! bytes lives here too, so every renderer escapes the same way.

use std::ops::Deref;

use super::heap::{Heap, Object};
use super::Value;

/// Heap storage for a Python string.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyString {
    text: Box<str>,
    is_ascii: bool,
}

impl PyString {
    pub fn new(text: String) -> Self {
        let is_ascii = text.is_ascii();
        Self {
            text: text.into_boxed_str(),
            is_ascii,
        }
    }

    pub fn as_str(&self) -> &str {
        &self.text
    }

    pub fn is_ascii(&self) -> bool {
        self.is_ascii
    }
}

impl Deref for PyString {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl From<String> for PyString {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for PyString {
    fn from(value: &str) -> Self {
        Self::new(value.to_owned())
    }
}

impl PartialEq<str> for PyString {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for PyString {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

/// Stack storage used while borrowing a short string from a compact [`Value`].
#[derive(Clone, Copy, Debug)]
pub struct InlineString {
    bytes: [u8; 15],
    len: u8,
}

impl InlineString {
    pub(super) fn from_parts(bytes: [u8; 15], len: usize) -> Self {
        Self {
            bytes,
            len: u8::try_from(len).expect("inline string length is bounded"),
        }
    }

    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.bytes[..usize::from(self.len)])
            .expect("inline strings originate from UTF-8")
    }
}

#[derive(Debug)]
enum StringStorage<'a> {
    Inline(InlineString),
    Heap(&'a PyString),
}

/// A non-allocating view over either physical Python string representation.
#[derive(Debug)]
pub struct PyStringRef<'a> {
    storage: StringStorage<'a>,
}

impl PyStringRef<'_> {
    pub fn as_str(&self) -> &str {
        match &self.storage {
            StringStorage::Inline(value) => value.as_str(),
            StringStorage::Heap(value) => value.as_str(),
        }
    }

    pub fn byte_len(&self) -> usize {
        self.as_str().len()
    }

    pub fn is_ascii(&self) -> bool {
        match &self.storage {
            StringStorage::Inline(value) => value.as_str().is_ascii(),
            StringStorage::Heap(value) => value.is_ascii(),
        }
    }
}

/// Borrow a Python string without materializing an owned Rust string.
pub fn string_ref<'a>(heap: &'a Heap, value: Value) -> Result<Option<PyStringRef<'a>>, String> {
    if let Some(value) = value.inline_string_ref() {
        return Ok(Some(PyStringRef {
            storage: StringStorage::Inline(value),
        }));
    }
    if !value.is_object() {
        return Ok(None);
    }
    Ok(match heap.get(value)? {
        Object::String(value) => Some(PyStringRef {
            storage: StringStorage::Heap(value),
        }),
        _ => None,
    })
}

/// Copy a Python string for an API that must outlive its heap borrow.
pub fn string_value(heap: &Heap, value: Value) -> Result<Option<String>, String> {
    Ok(string_ref(heap, value)?.map(|value| value.as_str().to_owned()))
}

/// The outcome of `owner[index]` when `owner` may be a string.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StringIndex {
    NotString,
    Character(char),
    /// CPython raises `TypeError` for a non-integer index.
    NotInteger,
    /// CPython raises `IndexError`.
    OutOfRange,
}

/// Return one Python string code point without materializing the complete string as characters.
///
/// ASCII strings, including the large text buffers used by the frozen I/O layer, support direct
/// byte indexing. Non-ASCII strings still index by Unicode code point to match Python semantics.
pub fn string_index(heap: &Heap, owner: Value, index: Value) -> Result<StringIndex, String> {
    let Some(text) = string_ref(heap, owner)? else {
        return Ok(StringIndex::NotString);
    };
    let Some(index) = super::number::int_value(heap, index) else {
        return Ok(StringIndex::NotInteger);
    };
    Ok(indexed_char(text.as_str(), index, text.is_ascii())
        .map_or(StringIndex::OutOfRange, StringIndex::Character))
}

/// Return a string's Python length without cloning its arena payload.
pub fn string_length(heap: &Heap, value: Value) -> Result<Option<usize>, String> {
    let Some(text) = string_ref(heap, value)? else {
        return Ok(None);
    };
    Ok(Some(if text.is_ascii() {
        text.byte_len()
    } else {
        text.as_str().chars().count()
    }))
}

fn indexed_char(value: &str, index: i64, is_ascii: bool) -> Option<char> {
    if is_ascii {
        let index = normalize_index(value.len(), index)?;
        return value.as_bytes().get(index).copied().map(char::from);
    }

    let index = normalize_index(value.chars().count(), index)?;
    value.chars().nth(index)
}

fn normalize_index(length: usize, index: i64) -> Option<usize> {
    let index = if index < 0 {
        length.checked_sub(usize::try_from(index.unsigned_abs()).ok()?)?
    } else {
        usize::try_from(index).ok()?
    };
    (index < length).then_some(index)
}

/// Copy the contents of a `bytes` or `bytearray`, or `None` for any other value.
pub fn bytes_value(heap: &Heap, value: Value) -> Result<Option<Vec<u8>>, String> {
    Ok(bytes_ref(heap, value)?.map(<[u8]>::to_vec))
}

/// Borrow the contents of a `bytes` or `bytearray` without copying them.
pub fn bytes_ref(heap: &Heap, value: Value) -> Result<Option<&[u8]>, String> {
    if !value.is_object() {
        return Ok(None);
    }
    Ok(match heap.get(value)? {
        Object::Bytes(value) | Object::ByteArray(value) => Some(value),
        _ => None,
    })
}

/// CPython's `repr(str)`: single quotes unless the text contains a single quote and no double
/// quote, with backslash escapes for the quote, backslash, control characters and characters
/// Python does not consider printable.
pub fn quote_string(value: &str) -> String {
    let quote = if value.contains('\'') && !value.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut rendered = String::with_capacity(value.len() + 2);
    rendered.push(quote);
    for character in value.chars() {
        match character {
            '\\' => rendered.push_str("\\\\"),
            '\n' => rendered.push_str("\\n"),
            '\r' => rendered.push_str("\\r"),
            '\t' => rendered.push_str("\\t"),
            character if character == quote => {
                rendered.push('\\');
                rendered.push(character);
            }
            character if is_printable(character) => rendered.push(character),
            character => {
                let code = u32::from(character);
                if code <= 0xff {
                    rendered.push_str(&format!("\\x{code:02x}"));
                } else if code <= 0xffff {
                    rendered.push_str(&format!("\\u{code:04x}"));
                } else {
                    rendered.push_str(&format!("\\U{code:08x}"));
                }
            }
        }
    }
    rendered.push(quote);
    rendered
}

/// Python's `str.isprintable` for one character, approximated without Unicode category tables:
/// controls, separators other than the ASCII space, and the common format characters are not
/// printable.
fn is_printable(character: char) -> bool {
    !(character.is_control()
        || (character.is_whitespace() && character != ' ')
        || matches!(
            character,
            '\u{ad}' | '\u{200b}'..='\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2060}'..='\u{2064}' | '\u{feff}'
        ))
}

/// CPython's bytes repr: single quotes unless the value contains `'` but no `"`, escaping only
/// the chosen quote.
pub fn quote_bytes(value: &[u8]) -> String {
    let quote = if value.contains(&b'\'') && !value.contains(&b'"') {
        b'"'
    } else {
        b'\''
    };
    let mut rendered = String::from("b");
    rendered.push(char::from(quote));
    for byte in value {
        match byte {
            b'\\' => rendered.push_str("\\\\"),
            byte if *byte == quote => {
                rendered.push('\\');
                rendered.push(char::from(quote));
            }
            b'\n' => rendered.push_str("\\n"),
            b'\r' => rendered.push_str("\\r"),
            b'\t' => rendered.push_str("\\t"),
            0x20..=0x7e => rendered.push(char::from(*byte)),
            _ => rendered.push_str(&format!("\\x{byte:02x}")),
        }
    }
    rendered.push(char::from(quote));
    rendered
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::{Limits, Resources};

    #[test]
    fn one_view_covers_inline_and_heap_strings() {
        let inline = Value::inline_string("small").unwrap();
        let mut heap = Heap::default();
        let mut resources = Resources::new(Limits::unlimited());
        let stored = heap
            .alloc(
                Object::String(PyString::new("snowman: ☃".into())),
                &(),
                &mut resources,
            )
            .unwrap();

        let inline = string_ref(&heap, inline).unwrap().unwrap();
        assert_eq!(inline.as_str(), "small");
        assert!(inline.is_ascii());

        let stored = string_ref(&heap, stored).unwrap().unwrap();
        assert_eq!(stored.as_str(), "snowman: ☃");
        assert!(!stored.is_ascii());
    }

    #[test]
    fn bytes_repr_switches_quotes_like_cpython() {
        assert_eq!(quote_bytes(b"it's"), r#"b"it's""#);
        assert_eq!(quote_bytes(br#"it's "x""#), r#"b'it\'s "x"'"#);
        assert_eq!(quote_bytes(b"\"\\\n\x00"), r#"b'"\\\n\x00'"#);
    }

    #[test]
    fn string_protocols_handle_inline_heap_ascii_and_unicode_values() {
        let mut heap = Heap::default();
        let mut resources = Resources::new(Limits::unlimited());
        let inline = Value::inline_string("café").unwrap();
        let ascii = heap
            .alloc(
                Object::String("a long ASCII string".into()),
                &(),
                &mut resources,
            )
            .unwrap();
        let unicode = heap
            .alloc(Object::String("☃ snow".into()), &(), &mut resources)
            .unwrap();

        let inline_ref = string_ref(&heap, inline).unwrap().unwrap();
        assert_eq!(inline_ref.as_str(), "café");
        assert!(!inline_ref.is_ascii());
        let ascii_ref = string_ref(&heap, ascii).unwrap().unwrap();
        assert_eq!(ascii_ref.as_str(), "a long ASCII string");
        assert!(ascii_ref.is_ascii());
        assert_eq!(
            string_index(&heap, inline, Value::Int(3)).unwrap(),
            StringIndex::Character('é')
        );
        assert_eq!(
            string_index(&heap, inline, Value::Int(-4)).unwrap(),
            StringIndex::Character('c')
        );
        assert_eq!(
            string_index(&heap, ascii, Value::Int(7)).unwrap(),
            StringIndex::Character('A')
        );
        assert_eq!(
            string_index(&heap, unicode, Value::Int(-6)).unwrap(),
            StringIndex::Character('☃')
        );
        assert_eq!(string_length(&heap, inline).unwrap(), Some(4));
        assert_eq!(string_length(&heap, ascii).unwrap(), Some(19));
        assert_eq!(string_length(&heap, unicode).unwrap(), Some(6));
        assert_eq!(
            string_index(&heap, Value::Int(1), Value::Int(0)).unwrap(),
            StringIndex::NotString
        );
        assert_eq!(
            string_index(&heap, ascii, Value::Int(99)).unwrap(),
            StringIndex::OutOfRange
        );
        assert_eq!(
            string_index(&heap, ascii, Value::None).unwrap(),
            StringIndex::NotInteger
        );
    }
}
