//! Heap payloads of an ndarray: the view object and the element storage it addresses.
//!
//! An array is two heap objects. [`ArrayStorage`] holds the elements, packed little-endian bytes
//! for numeric and string dtypes or traced references for `object`, and may be shared by many
//! views. [`ArrayObject`] is one view: byte strides and an offset into its storage plus the
//! array that owns the storage, for `ndarray.base`. Both are module-owned payloads behind
//! `Object::Native`; the runtime bridge reaches them by downcast and the kernels in [`array`]
//! (`super::array`) only ever see borrowed element data through the runtime's closures.

use super::super::super::heap::{NativeObject, Ref, MODELED_VALUE_BYTES};
use super::super::super::native::PyArrayView;
use super::super::super::object_model::{BuiltinType, TypeId};
use crate::python::error::{PyError, PyResult};

/// Flat element storage shared by one or more array views.
#[derive(Debug)]
pub(in crate::python) enum ArrayStorage {
    Bytes(Vec<u8>),
    Values(Vec<Ref>),
}

impl ArrayStorage {
    /// Length of the addressable storage in bytes.
    pub fn byte_len(&self) -> usize {
        match self {
            Self::Bytes(bytes) => bytes.len(),
            Self::Values(values) => values.len().saturating_mul(16),
        }
    }
}

impl NativeObject for ArrayStorage {
    fn python_type(&self) -> TypeId {
        BuiltinType::Native.id()
    }

    fn modeled_bytes(&self) -> PyResult<u64> {
        match self {
            Self::Bytes(bytes) => u64::try_from(bytes.len()).map_err(|_| overflow()),
            Self::Values(values) => u64::try_from(values.len())
                .ok()
                .and_then(|count| count.checked_mul(MODELED_VALUE_BYTES))
                .ok_or_else(overflow),
        }
    }

    fn visit_refs(&self, visit: &mut dyn FnMut(&Ref)) {
        if let Self::Values(values) = self {
            for slot in values {
                visit(slot);
            }
        }
    }

    fn dup(&self) -> Box<dyn NativeObject> {
        Box::new(match self {
            Self::Bytes(bytes) => Self::Bytes(bytes.clone()),
            Self::Values(values) => Self::Values(values.iter().map(Ref::dup).collect()),
        })
    }

    fn repr(&self, _: &mut dyn FnMut(&Ref) -> PyResult<String>) -> PyResult<String> {
        Ok("<array storage>".into())
    }
}

/// An ndarray view over [`ArrayStorage`].
#[derive(Debug)]
pub(in crate::python) struct ArrayObject {
    pub storage: Ref,
    pub view: PyArrayView,
    /// The array that owns the storage, for `ndarray.base`; `None` for owners.
    pub base: Option<Ref>,
}

impl NativeObject for ArrayObject {
    fn python_type(&self) -> TypeId {
        BuiltinType::Array.id()
    }

    fn modeled_bytes(&self) -> PyResult<u64> {
        let slots = self
            .view
            .shape
            .len()
            .checked_add(self.view.strides.len())
            .and_then(|size| size.checked_add(3))
            .ok_or_else(overflow)?;
        u64::try_from(slots)
            .ok()
            .and_then(|slots| slots.checked_mul(MODELED_VALUE_BYTES))
            .ok_or_else(overflow)
    }

    fn visit_refs(&self, visit: &mut dyn FnMut(&Ref)) {
        visit(&self.storage);
        if let Some(base) = &self.base {
            visit(base);
        }
    }

    fn dup(&self) -> Box<dyn NativeObject> {
        Box::new(Self {
            storage: self.storage.dup(),
            view: self.view.clone(),
            base: self.base.as_ref().map(Ref::dup),
        })
    }

    fn repr(&self, _: &mut dyn FnMut(&Ref) -> PyResult<String>) -> PyResult<String> {
        Ok(format!("array(shape={:?})", self.view.shape))
    }
}

fn overflow() -> PyError {
    "modeled object size overflow".into()
}
