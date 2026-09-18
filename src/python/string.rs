//! Python string storage and borrowing.
//!
//! Python-visible strings are distinct from interned source identifiers. Short values stay
//! inline in [`Value`]; longer values use [`PyString`] in the heap. Consumers inspect either
//! representation through [`PyStringRef`] without allocating.

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
pub fn string_ref<'a>(heap: &'a Heap, value: &Value) -> Result<Option<PyStringRef<'a>>, String> {
    if let Some(value) = value.inline_string_ref() {
        return Ok(Some(PyStringRef {
            storage: StringStorage::Inline(value),
        }));
    }
    let Some(id) = value.object_id() else {
        return Ok(None);
    };
    Ok(match heap.get(id)? {
        Object::String(value) => Some(PyStringRef {
            storage: StringStorage::Heap(value),
        }),
        _ => None,
    })
}

/// Copy a Python string for an API that must outlive its heap borrow.
pub fn string_value(heap: &Heap, value: &Value) -> Result<Option<String>, String> {
    Ok(string_ref(heap, value)?.map(|value| value.as_str().to_owned()))
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
            .allocate(
                Object::String(PyString::new("snowman: ☃".into())),
                &mut resources,
            )
            .unwrap();

        let inline = string_ref(&heap, &inline).unwrap().unwrap();
        assert_eq!(inline.as_str(), "small");
        assert!(inline.is_ascii());

        let stored = string_ref(&heap, &stored).unwrap().unwrap();
        assert_eq!(stored.as_str(), "snowman: ☃");
        assert!(!stored.is_ascii());
    }
}
