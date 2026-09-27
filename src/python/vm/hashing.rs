//! The `hash()` builtin: dispatch from runtime values to CPython's hash algorithms.
//!
//! Builtin values hash by value with the algorithms in `python::hash`, registered numbers such
//! as NumPy scalars hash as the Python number they equal, and user instances follow CPython's
//! rules: an explicit `__hash__` wins, `__hash__ = None` or an inherited `__eq__` without
//! `__hash__` makes the class unhashable, and everything else hashes by identity.

use super::super::hash;
use super::super::number::{self, NumberRef};
use super::{protocol, Object, Slot, Value, ValueTag, Vm};

/// Combines the hashes of a container's items into the container's hash.
type Combine = fn(&[i64]) -> i64;

/// Nesting bound for tuples and frozensets, matching the VM's call-depth limit.
const MAX_HASH_DEPTH: usize = 256;

impl Vm<'_> {
    /// `hash(value)` with CPython's results for builtin values.
    pub(super) fn hash_value(&mut self, value: &Value) -> Result<i64, String> {
        self.hash_nested(value, 0)
    }

    fn hash_nested(&mut self, value: &Value, depth: usize) -> Result<i64, String> {
        if depth == MAX_HASH_DEPTH {
            return Err(self.raise_exception("RecursionError", "maximum recursion depth exceeded"));
        }
        self.charge_cpu(1)?;
        if value.is_none() {
            return Ok(hash::NONE);
        }
        // Instances of `int` subclasses have a numeric view but may define their own `__hash__`.
        if !self.is_user_instance(value)? {
            if let Some(number) = number::view(&self.state.heap, value) {
                return Ok(number_hash(number));
            }
        }
        if let Some(text) = super::super::string::string_ref(&self.state.heap, value)? {
            let text = text.as_str().to_owned();
            self.charge_cpu(u64::try_from(text.len() / 32).unwrap_or(u64::MAX))?;
            return Ok(hash::string(&text));
        }
        match value.tag() {
            ValueTag::Native => return Ok(hash::identity(value.payload ^ u64::from(value.aux[0]))),
            ValueTag::Registered => {
                return Ok(hash::identity(
                    value.payload ^ u64::from(value.aux[0]) << 56,
                ))
            }
            _ => {}
        }
        let Some(id) = value.object_id() else {
            return Err(format!("hash() does not support {value:?}"));
        };
        let object = self.state.heap.get(id)?;
        let (items, combine): (Vec<Value>, Combine) = match object {
            Object::Bytes(bytes) => {
                let bytes = bytes.clone();
                self.charge_cpu(u64::try_from(bytes.len() / 32).unwrap_or(u64::MAX))?;
                return Ok(hash::bytes(&bytes));
            }
            Object::Tuple(items) => (items.clone(), hash::tuple),
            Object::FrozenSet(items) => (items.clone(), hash::frozenset),
            Object::Range { start, stop, step } => {
                let (start, stop, step) = (*start, *stop, *step);
                return range_hash(start, stop, step);
            }
            Object::Slice { start, stop, step } => (vec![*start, *stop, *step], hash::slice),
            Object::EnumMember { name, .. } => return Ok(hash::string(name)),
            Object::WideValue { payload, .. } => {
                return Ok(hash::identity(payload[0] ^ payload[1].rotate_left(32)))
            }
            Object::List(_)
            | Object::Dict(_)
            | Object::DefaultDict { .. }
            | Object::Set(_)
            | Object::ByteArray(_) => return Err(self.unhashable(value)),
            Object::Instance { class, .. } => {
                let class = *class;
                return self.instance_hash(value, class);
            }
            _ => return Ok(hash::identity(id.as_raw() as u64)),
        };
        let mut hashes = Vec::with_capacity(items.len());
        for item in &items {
            hashes.push(self.hash_nested(item, depth + 1)?);
        }
        Ok(combine(&hashes))
    }

    fn is_user_instance(&self, value: &Value) -> Result<bool, String> {
        let Some(id) = value.object_id() else {
            return Ok(false);
        };
        Ok(matches!(self.state.heap.get(id)?, Object::Instance { .. }))
    }

    /// CPython's `object.__hash__` resolution for an instance of a user class.
    fn instance_hash(
        &mut self,
        value: &Value,
        class: super::super::heap::ObjectId,
    ) -> Result<i64, String> {
        let Object::Class {
            mro, is_dataclass, ..
        } = self.state.heap.get(class)?
        else {
            return Err("instance has an invalid class".into());
        };
        let is_dataclass = *is_dataclass;
        let lineage: Vec<_> = std::iter::once(class).chain(mro.iter().copied()).collect();
        for ancestor in lineage {
            self.charge_cpu(1)?;
            let Object::Class { attributes, .. } = self.state.heap.get(ancestor)? else {
                return Err("class MRO contains a non-class object".into());
            };
            match attributes.get("__hash__") {
                Some(method) if method.is_none() => return Err(self.unhashable(value)),
                Some(_) => {
                    let result = self
                        .invoke_slot(value, Slot::Hash, "__hash__", Vec::new())?
                        .ok_or("__hash__ slot disappeared during lookup")?;
                    return self.hash_result(&result);
                }
                // A class that defines `__eq__` without `__hash__` gets `__hash__ = None`.
                None if attributes.contains_key("__eq__") => return Err(self.unhashable(value)),
                None => {}
            }
        }
        // `@dataclass` defaults to `eq=True, frozen=False`, which also sets `__hash__ = None`.
        if is_dataclass {
            return Err(self.unhashable(value));
        }
        // An `int` or `tuple` subclass hashes as the value it holds; a `dict` subclass is
        // unhashable like `dict`, and the error names the subclass.
        if let Some(payload) = protocol::builtin_payload(&self.state.heap, value)? {
            let holds_dict = match payload.object_id() {
                Some(id) => matches!(self.state.heap.get(id)?, Object::Dict(_)),
                None => false,
            };
            if holds_dict {
                return Err(self.unhashable(value));
            }
            return self.hash_value(&payload);
        }
        let id = value.object_id().expect("instances are arena objects");
        Ok(hash::identity(id.as_raw() as u64))
    }

    /// Convert a `__hash__` result as CPython's `slot_tp_hash` does: machine-sized integers are
    /// kept, larger ones are reduced with the integer hash, and `-1` becomes `-2`.
    fn hash_result(&mut self, result: &Value) -> Result<i64, String> {
        match number::index(&self.state.heap, result) {
            Some(NumberRef::Int(-1)) => Ok(-2),
            Some(NumberRef::Int(value)) => Ok(value),
            Some(number @ (NumberRef::BigInt(_) | NumberRef::UInt(_))) => Ok(hash::big_integer(
                &number.to_bigint().expect("integer view"),
            )),
            _ => Err(self.raise_exception("TypeError", "__hash__ method should return an integer")),
        }
    }

    fn unhashable(&mut self, value: &Value) -> String {
        let message = match self.type_name_of(value) {
            Ok(name) => format!("unhashable type: '{name}'"),
            Err(error) => return error,
        };
        self.raise_exception("TypeError", message)
    }
}

/// The numeric hash shared by every number type, so equal numbers hash alike.
fn number_hash(number: NumberRef<'_>) -> i64 {
    match number {
        NumberRef::Int(value) => hash::integer(value),
        NumberRef::UInt(value) => hash::big_integer(&value.into()),
        NumberRef::BigInt(value) => hash::big_integer(value),
        NumberRef::Float(value) => hash::float(value),
        NumberRef::Complex(real, imag) => hash::complex(real, imag),
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
