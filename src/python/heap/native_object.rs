//! Heap payloads that belong to one stdlib module rather than to the language core.
//!
//! A compiled regex, a match, an argument parser or a `pytest.raises` context is data the module
//! that created it reads back; the rest of the interpreter only needs to trace, size, copy and
//! render it. Such payloads live behind [`Object::Native`](super::Object) and implement this
//! trait instead of adding a variant to `Object`, so the heap, the collector and the generic
//! protocols stay closed over the language-level representations.
//!
//! The module that owns a payload type reaches it by downcasting through [`Object::native`]
//! (`super::Object::native`), which fails for any other payload. Stored references inside a
//! payload are [`Ref`]s like everywhere else: the owner creates them with the [`Builder`]
//! (`super::Builder`) of the allocation or mutation that stores them, and the collector sees them
//! through [`NativeObject::visit_refs`].

use std::any::Any;
use std::fmt::Debug;

use super::super::object_model::TypeId;
use super::Ref;

/// Behaviour the heap needs from a module-owned payload.
///
/// Implementations are plain data structs. Every stored reference must be reported by
/// `visit_refs` and duplicated by `dup`, or the collector frees it while the payload still
/// points at it.
pub trait NativeObject: Any + Debug + Send {
    /// The Python type of every object carrying this payload.
    fn python_type(&self) -> TypeId;

    /// The payload's modeled size in bytes, excluding the object header. Text counts its length
    /// and each stored reference counts `MODELED_VALUE_BYTES` (`super::MODELED_VALUE_BYTES`),
    /// like the heap's own payloads.
    fn modeled_bytes(&self) -> Result<u64, String>;

    /// Report every stored reference to the collector.
    fn visit_refs(&mut self, visit: &mut dyn FnMut(&mut Ref)) {
        let _ = visit;
    }

    /// A deep copy for a heap snapshot, duplicating stored references with [`Ref::dup`].
    fn dup(&self) -> Box<dyn NativeObject>;

    /// The builtin `repr()` text. `nested` renders a stored reference with the caller's cycle
    /// tracking.
    fn repr(
        &self,
        nested: &mut dyn FnMut(&Ref) -> Result<String, String>,
    ) -> Result<String, String>;

    /// A stored reference the object exposes as a read-only attribute, such as the fields of an
    /// `argparse.Namespace`. Types with no such attributes keep the default.
    fn attribute(&self, name: &str) -> Option<&Ref> {
        let _ = name;
        None
    }
}
