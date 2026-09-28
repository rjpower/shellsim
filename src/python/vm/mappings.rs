//! The mapping protocol: how `dict(m)`, `dict.update(m)`, `f(**m)` and `{**m}` read a mapping.
//!
//! CPython accepts any object with a `keys()` method there, reading each value through
//! `__getitem__`. Dicts, dict subclasses, namespace views and mapping proxies are read straight
//! from storage; every other mapping, including user classes, goes through `keys()` and
//! subscription, so its own methods run.

use super::{Object, Value, Vm};

impl Vm<'_> {
    /// The `(key, value)` entries of `value` if it is a mapping, or `None` when it has no `keys`
    /// method. A dict subclass is read from the dict it holds, as CPython's `dict_merge` reads a
    /// subclass that keeps `dict.__iter__`.
    pub(super) fn mapping_items(
        &mut self,
        value: Value,
    ) -> Result<Option<Vec<(Value, Value)>>, String> {
        let stored = self.builtin_view(value)?;
        if let Some(id) = stored.object_id() {
            let length = match self.state.heap.get(id)? {
                Object::Dict(entries) | Object::DefaultDict { entries, .. } => Some(entries.len()),
                _ => None,
            };
            if let Some(length) = length {
                self.reserve_result(length.saturating_mul(std::mem::size_of::<(Value, Value)>()))?;
                let (Object::Dict(entries) | Object::DefaultDict { entries, .. }) =
                    self.state.heap.get(id)?
                else {
                    unreachable!("dict kind was checked above")
                };
                return Ok(Some(entries.to_vec()));
            }
            match *self.state.heap.get(id)? {
                Object::NamespaceDict(target) => return self.namespace_items(target).map(Some),
                Object::MappingProxy(target) => return self.proxy_items(target).map(Some),
                _ => {}
            }
        }
        let Some(keys) = self.resolve_attribute(value, "keys")? else {
            return Ok(None);
        };
        let keys = self.invoke_value(keys, Vec::new())?;
        let keys = self.iterable_values(&keys)?;
        let mut entries = Vec::with_capacity(keys.len());
        for key in keys {
            self.charge_cpu(1)?;
            let item = self.subscript_value(value, key)?;
            entries.push((key, item));
        }
        Ok(Some(entries))
    }
}
