//! The `hash()` builtin: dispatch from runtime values to CPython's hash algorithms.
//!
//! Builtin values hash by value with the algorithms in `python::hash`, registered numbers such
//! as NumPy scalars hash as the Python number they equal, and user instances follow CPython's
//! rules: an explicit `__hash__` wins, `__hash__ = None` or an inherited `__eq__` without
//! `__hash__` makes the class unhashable, and everything else hashes by identity.

use super::super::hash;
use super::super::heap::DictViewKind;
use super::super::number::{self, NumberRef};
use super::super::object_model::SlotValue;
use super::{Object, Slot, Value, Vm};

/// Combines the hashes of a container's items into the container's hash.
type Combine = fn(&[i64]) -> i64;

/// Nesting bound for tuples and frozensets, matching the VM's call-depth limit.
const MAX_HASH_DEPTH: usize = 256;

impl<'s> Vm<'s> {
    /// `hash(value)` with CPython's results for builtin values.
    pub(super) fn hash_value(&mut self, value: &Value<'s>) -> Result<i64, String> {
        self.hash_nested(value, 0)
    }

    fn hash_nested(&mut self, value: &Value<'s>, depth: usize) -> Result<i64, String> {
        if depth == MAX_HASH_DEPTH {
            return Err(self.raise_exception("RecursionError", "maximum recursion depth exceeded"));
        }
        self.charge_cpu(1)?;
        if value.is_none() {
            return Ok(hash::NONE);
        }
        // Instances of `int` or `str` subclasses hold a builtin payload but may define their own
        // `__hash__`, so they take the class path before the payload is consulted.
        if let Some(class) = self.instance_class(*value)? {
            return self.instance_hash(value, class, depth);
        }
        self.payload_hash(value, depth)
    }

    /// The hash of a builtin value or payload, ignoring any user class the object belongs to.
    /// This is what `tuple.__hash__(instance)` and the other builtin hash slots compute.
    pub(super) fn payload_hash(&mut self, value: &Value<'s>, depth: usize) -> Result<i64, String> {
        if let Some(number) = number::view(self.heap(), value) {
            return Ok(number::number_hash(number));
        }
        if let Some(text) = super::super::string::string_ref(self.heap(), *value)? {
            let text = text.as_str().to_owned();
            self.charge_cpu(u64::try_from(text.len() / 32).unwrap_or(u64::MAX))?;
            return Ok(hash::string(&text));
        }
        if let Some(native) = value.native_value() {
            let (payload, tag) = native.encode();
            return Ok(hash::identity(payload ^ u64::from(tag)));
        }
        if let Some((kind, payload)) = value.registered_parts() {
            return Ok(hash::identity(payload ^ u64::from(kind) << 56));
        }
        if !value.is_object() {
            return Err(format!("hash() does not support {value:?}"));
        }
        let object = self.get(*value)?;
        let (items, combine): (Vec<Value<'s>>, Combine) = match object {
            Object::Bytes(bytes) => {
                let bytes = bytes.clone();
                self.charge_cpu(u64::try_from(bytes.len() / 32).unwrap_or(u64::MAX))?;
                return Ok(hash::bytes(&bytes));
            }
            Object::Tuple(items) => (self.handles(items), hash::tuple),
            Object::GenericAlias { origin, arguments } => (
                self.handles(std::iter::once(origin).chain(arguments)),
                hash::tuple,
            ),
            Object::FrozenSet(items) => (self.handles(items.iter()), hash::frozenset),
            Object::Range { start, stop, step } => {
                let (start, stop, step) = (*start, *stop, *step);
                return range_hash(start, stop, step);
            }
            Object::Slice { start, stop, step } => (self.handles([start, stop, step]), hash::slice),
            Object::WideValue { payload, .. } => {
                return Ok(hash::identity(payload[0] ^ payload[1].rotate_left(32)))
            }
            Object::List(_)
            | Object::Dict(_)
            | Object::DefaultDict { .. }
            | Object::Set(_)
            | Object::ByteArray(_)
            | Object::NamespaceDict(_)
            | Object::MappingProxy(_)
            | Object::DictView {
                kind: DictViewKind::Keys | DictViewKind::Items,
                ..
            } => return Err(self.unhashable(value)),
            _ => return Ok(hash::identity(self.identity_bits(value)?)),
        };
        let mut hashes = Vec::with_capacity(items.len());
        for item in &items {
            hashes.push(self.hash_nested(item, depth + 1)?);
        }
        Ok(combine(&hashes))
    }

    /// `hash()` of an instance of a user class, resolved through the class's hash slot. A class
    /// that defines `__eq__` without `__hash__` carries `__hash__ = None` and is unhashable, as
    /// is a `@dataclass` with the default `eq=True`.
    fn instance_hash(
        &mut self,
        value: &Value<'s>,
        class: Value<'s>,
        depth: usize,
    ) -> Result<i64, String> {
        let Object::Class(class_object) = self.get(class)? else {
            return Err("instance has an invalid class".into());
        };
        let is_dataclass = class_object.is_dataclass;
        match self.state.types.slot(self.type_id(value)?, Slot::Hash)? {
            Some(SlotValue::Descriptor {
                value: descriptor, ..
            }) if self.handle(&descriptor).is_none() => {
                return Err(self.unhashable(value));
            }
            Some(SlotValue::Descriptor { .. } | SlotValue::NativeUnary(_)) => {
                let result = self
                    .invoke_slot(value, Slot::Hash, "__hash__", Vec::new())?
                    .ok_or("__hash__ slot disappeared during lookup")?;
                return self.hash_result(&result);
            }
            _ => {}
        }
        // `@dataclass` defaults to `eq=True, frozen=False`, which also sets `__hash__ = None`.
        if is_dataclass {
            return Err(self.unhashable(value));
        }
        // An `int` or `tuple` subclass hashes as the value it holds; a `dict` or `list` subclass
        // is unhashable like its base, and the error names the subclass.
        self.payload_hash(value, depth)
    }

    /// The stable identity of a heap object, as identity hashing consumes it.
    fn identity_bits(&self, value: &Value<'s>) -> Result<u64, String> {
        self.identity(*value)?
            .map(u64::from)
            .ok_or_else(|| "identity hash of an immediate value".into())
    }

    /// Convert a `__hash__` result as CPython's `slot_tp_hash` does: machine-sized integers are
    /// kept, larger ones are reduced with the integer hash, and `-1` becomes `-2`.
    fn hash_result(&mut self, result: &Value<'s>) -> Result<i64, String> {
        match number::index(self.heap(), result) {
            Some(NumberRef::Int(-1)) => Ok(-2),
            Some(NumberRef::Int(value)) => Ok(value),
            Some(number @ (NumberRef::BigInt(_) | NumberRef::UInt(_))) => Ok(hash::big_integer(
                &number.to_bigint().expect("integer view"),
            )),
            _ => Err(self.raise_exception("TypeError", "__hash__ method should return an integer")),
        }
    }

    fn unhashable(&mut self, value: &Value<'s>) -> String {
        let message = match self.type_name_of(value) {
            Ok(name) => format!("unhashable type: '{name}'"),
            Err(error) => return error,
        };
        self.raise_exception("TypeError", message)
    }
}

/// CPython hashes a range by the sequence it produces: `(len, start, step)`, with `None` for the
/// parts that do not affect an empty or one-element range.
fn range_hash(start: i64, stop: i64, step: i64) -> Result<i64, String> {
    let length = super::range_length(start, stop, step)?;
    let parts = match length {
        0 => [hash::integer(0), hash::NONE, hash::NONE],
        1 => [hash::integer(1), hash::integer(start), hash::NONE],
        _ => [
            hash::integer(i64::try_from(length).unwrap_or(i64::MAX)),
            hash::integer(start),
            hash::integer(step),
        ],
    };
    Ok(hash::tuple(&parts))
}
