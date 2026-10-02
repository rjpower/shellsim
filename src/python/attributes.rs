//! Instance attribute storage: shared shapes for the common case, a dictionary after deletions
//! or many attributes.
//!
//! Instances of one class that assign the same attributes in the same order share a shape, so an
//! attribute read is a slot index guarded by the shape, and the VM's inline caches can reuse a
//! lookup. Shapes only grow; deleting an attribute converts the instance to dictionary storage.
//!
//! This module is a client of the heap: it reads instances through handles and writes them
//! through [`Heap::modify`], and the shape tables are interpreter metadata kept beside the type
//! registry rather than inside the heap.

use std::collections::HashMap;

use crate::resources::Resources;

use super::heap::{Heap, InstanceAttributes, Object, Ref, Roots, Value};
use super::symbols::{SymbolId, Symbols};

const MAX_SHAPED_ATTRIBUTES: usize = 32;
const INSTANCE_SLOT_BYTES: u64 = 16;
const INSTANCE_DICT_ENTRY_BYTES: u64 = 48;
const SHAPE_BYTES: u64 = 24;

/// Runtime-local identity for one append-only instance storage layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShapeId(u32);

impl ShapeId {
    /// The empty shape every instance starts with.
    pub const ROOT: ShapeId = ShapeId(0);
}

/// Guard and slot for one shaped instance attribute, as stored in an inline cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InstanceAttributeSlot {
    shape: ShapeId,
    slot: usize,
}

#[derive(Clone, Debug)]
struct Shape {
    parent: Option<ShapeId>,
    added: Option<SymbolId>,
    slots: u32,
}

/// The interpreter's shape tree and transition table.
#[derive(Clone, Debug)]
pub struct Shapes {
    shapes: Vec<Shape>,
    transitions: HashMap<(ShapeId, SymbolId), ShapeId>,
    modeled_bytes: u64,
}

impl Default for Shapes {
    fn default() -> Self {
        Self {
            shapes: vec![Shape {
                parent: None,
                added: None,
                slots: 0,
            }],
            transitions: HashMap::new(),
            modeled_bytes: 0,
        }
    }
}

fn instance_attributes<'h>(
    heap: &'h Heap,
    instance: Value<'_>,
) -> Result<&'h InstanceAttributes, String> {
    match heap.get(instance)? {
        Object::Instance { attributes, .. } => Ok(attributes),
        _ => Err("object does not have instance attributes".into()),
    }
}

impl Shapes {
    /// Transfer the tables' accounting to the caller when an interpreter is discarded.
    pub fn take_modeled_bytes(&mut self) -> u64 {
        std::mem::take(&mut self.modeled_bytes)
    }

    /// Snapshot the names stored directly on an instance in either attribute representation.
    /// Shaped instances list names in the order they were first assigned; dictionary instances
    /// (after a deletion or many attributes) list them sorted, since their storage is unordered.
    pub fn attribute_names(
        &self,
        heap: &Heap,
        symbols: &Symbols,
        instance: Value<'_>,
    ) -> Result<Vec<String>, String> {
        match instance_attributes(heap, instance)? {
            InstanceAttributes::Shaped { shape, values } => (0..values.len())
                .map(|slot| {
                    let symbol = self
                        .attribute_at(*shape, slot)
                        .ok_or("invalid instance shape slot")?;
                    symbols
                        .name(symbol)
                        .map(str::to_string)
                        .ok_or_else(|| "invalid instance attribute symbol".into())
                })
                .collect(),
            InstanceAttributes::Dictionary(values) => {
                let mut names = values
                    .keys()
                    .map(|symbol| {
                        symbols
                            .name(*symbol)
                            .map(str::to_string)
                            .ok_or_else(|| String::from("invalid instance attribute symbol"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                names.sort();
                Ok(names)
            }
        }
    }

    /// Every attribute stored directly on an instance, as (name, value) pairs.
    pub fn attribute_values<'s>(
        &self,
        heap: &Heap,
        symbols: &Symbols,
        instance: Value<'_>,
    ) -> Result<Vec<(String, Value<'s>)>, String> {
        self.attribute_names(heap, symbols, instance)?
            .into_iter()
            .map(|name| {
                let symbol = symbols
                    .id(&name)
                    .ok_or("invalid instance attribute symbol")?;
                let value = self
                    .attribute_by_symbol(heap, instance, symbol)?
                    .ok_or("instance attribute vanished")?;
                Ok((name, value))
            })
            .collect()
    }

    pub fn attribute_by_symbol<'s>(
        &self,
        heap: &Heap,
        instance: Value<'_>,
        symbol: SymbolId,
    ) -> Result<Option<Value<'s>>, String> {
        let slot = match instance_attributes(heap, instance)? {
            InstanceAttributes::Dictionary(values) => values.get(&symbol),
            InstanceAttributes::Shaped { shape, values } => {
                self.slot(*shape, symbol).and_then(|slot| values.get(slot))
            }
        };
        Ok(heap.handle_optional(slot))
    }

    /// The cacheable location of a shaped attribute, or `None` when the instance uses a
    /// dictionary or lacks the attribute.
    pub fn slot_by_symbol(
        &self,
        heap: &Heap,
        instance: Value<'_>,
        symbol: SymbolId,
    ) -> Result<Option<InstanceAttributeSlot>, String> {
        let Object::Instance { attributes, .. } = heap.get(instance)? else {
            return Ok(None);
        };
        let InstanceAttributes::Shaped { shape, values } = attributes else {
            return Ok(None);
        };
        Ok(self
            .slot(*shape, symbol)
            .filter(|slot| *slot < values.len())
            .map(|slot| InstanceAttributeSlot {
                shape: *shape,
                slot,
            }))
    }

    fn slot(&self, mut shape: ShapeId, attribute: SymbolId) -> Option<usize> {
        while shape.0 != 0 {
            let current = self.shapes.get(shape.0 as usize)?;
            if current.added == Some(attribute) {
                return usize::try_from(current.slots.checked_sub(1)?).ok();
            }
            shape = current.parent?;
        }
        None
    }

    fn attribute_at(&self, mut shape: ShapeId, slot: usize) -> Option<SymbolId> {
        while shape.0 != 0 {
            let current = self.shapes.get(shape.0 as usize)?;
            if usize::try_from(current.slots.checked_sub(1)?).ok()? == slot {
                return current.added;
            }
            shape = current.parent?;
        }
        None
    }

    /// The shape reached by adding `symbol` to `shape`, creating and charging it on first use.
    fn transition(
        &mut self,
        shape: ShapeId,
        symbol: SymbolId,
        resources: &mut Resources,
    ) -> Result<ShapeId, String> {
        if let Some(next) = self.transitions.get(&(shape, symbol)) {
            return Ok(*next);
        }
        let next =
            ShapeId(u32::try_from(self.shapes.len()).map_err(|_| "too many instance shapes")?);
        let slots = self
            .shapes
            .get(shape.0 as usize)
            .ok_or("invalid instance shape")?
            .slots
            .checked_add(1)
            .ok_or("too many shaped instance attributes")?;
        let modeled_bytes = self
            .modeled_bytes
            .checked_add(SHAPE_BYTES)
            .ok_or("modeled shape table size overflow")?;
        if !resources.reserve_memory(SHAPE_BYTES) {
            return Err("memory limit exceeded".into());
        }
        self.modeled_bytes = modeled_bytes;
        self.shapes.push(Shape {
            parent: Some(shape),
            added: Some(symbol),
            slots,
        });
        self.transitions.insert((shape, symbol), next);
        Ok(next)
    }
}

/// Read a shaped attribute through an inline cache: the value when `instance` still has class
/// `class` and shape `location.shape`, otherwise `None` so the caller falls back to a full lookup.
pub fn cached_attribute<'s>(
    heap: &Heap,
    instance: Value<'_>,
    class: &Ref,
    location: InstanceAttributeSlot,
) -> Result<Option<Value<'s>>, String> {
    let Object::Instance {
        class: actual_class,
        attributes,
        ..
    } = heap.get(instance)?
    else {
        return Ok(None);
    };
    if actual_class != class {
        return Ok(None);
    }
    let InstanceAttributes::Shaped { shape, values } = attributes else {
        return Ok(None);
    };
    if *shape != location.shape {
        return Ok(None);
    }
    Ok(heap.handle_optional(values.get(location.slot)))
}

/// Everything an attribute write needs: the heap and its roots for growth, the shape tables,
/// and the symbol table.
pub struct AttributeStore<'a> {
    pub heap: &'a mut Heap,
    pub shapes: &'a mut Shapes,
    pub symbols: &'a mut Symbols,
    pub roots: &'a mut dyn Roots,
    pub resources: &'a mut Resources,
}

impl AttributeStore<'_> {
    pub fn insert(
        &mut self,
        instance: Value<'_>,
        name: &str,
        value: Value<'_>,
    ) -> Result<(), String> {
        instance_attributes(self.heap, instance)?;
        let symbol = self.symbols.intern(name, self.resources)?;
        self.insert_by_symbol(instance, symbol, value)
    }

    pub fn extend<'v>(
        &mut self,
        instance: Value<'_>,
        values: impl IntoIterator<Item = (String, Value<'v>)>,
    ) -> Result<(), String> {
        for (name, value) in values {
            self.insert(instance, &name, value)?;
        }
        Ok(())
    }

    pub fn insert_by_symbol(
        &mut self,
        instance: Value<'_>,
        symbol: SymbolId,
        value: Value<'_>,
    ) -> Result<(), String> {
        match instance_attributes(self.heap, instance)? {
            InstanceAttributes::Dictionary(values) => {
                let growth = if values.contains_key(&symbol) {
                    0
                } else {
                    INSTANCE_DICT_ENTRY_BYTES
                };
                self.heap
                    .reserve_object_growth(instance, growth, self.roots, self.resources)?;
                self.heap.modify(instance, |builder, object| {
                    let Object::Instance {
                        attributes: InstanceAttributes::Dictionary(values),
                        ..
                    } = object
                    else {
                        unreachable!("instance representation changed without yielding")
                    };
                    values.insert(symbol, builder.store(value));
                })
            }
            InstanceAttributes::Shaped { shape, values } => {
                let shape = *shape;
                if let Some(slot) = self.shapes.slot(shape, symbol) {
                    return self.heap.modify(instance, |builder, object| {
                        let Object::Instance {
                            attributes: InstanceAttributes::Shaped { values, .. },
                            ..
                        } = object
                        else {
                            unreachable!("instance representation changed without yielding")
                        };
                        values[slot] = builder.store(value);
                    });
                }
                if values.len() >= MAX_SHAPED_ATTRIBUTES {
                    return self.insert_dictionary(instance, symbol, value);
                }
                self.append_shaped(instance, shape, symbol, value)
            }
        }
    }

    fn append_shaped(
        &mut self,
        instance: Value<'_>,
        shape: ShapeId,
        symbol: SymbolId,
        value: Value<'_>,
    ) -> Result<(), String> {
        // Grow the instance first: that reservation may collect, while the shape table's own
        // small charge cannot.
        self.heap.reserve_object_growth(
            instance,
            INSTANCE_SLOT_BYTES,
            self.roots,
            self.resources,
        )?;
        let next_shape = self.shapes.transition(shape, symbol, self.resources)?;
        self.heap.modify(instance, |builder, object| {
            let Object::Instance {
                attributes: InstanceAttributes::Shaped { shape, values },
                ..
            } = object
            else {
                unreachable!("instance representation changed without yielding")
            };
            *shape = next_shape;
            values.push(builder.store(value));
        })
    }

    /// Remove one instance attribute and return its value, or `None` when the instance does
    /// not have it. A shaped instance first converts to dictionary storage, because shapes only
    /// grow; attribute caches never match dictionary instances, so they stay valid.
    pub fn remove_by_symbol<'s>(
        &mut self,
        instance: Value<'_>,
        symbol: SymbolId,
    ) -> Result<Option<Value<'s>>, String> {
        if let InstanceAttributes::Shaped { shape, .. } = instance_attributes(self.heap, instance)?
        {
            if self.shapes.slot(*shape, symbol).is_none() {
                return Ok(None);
            }
            self.convert_to_dictionary(instance, 0)?;
        }
        let removed = self.heap.modify(instance, |_, object| {
            let Object::Instance {
                attributes: InstanceAttributes::Dictionary(values),
                ..
            } = object
            else {
                unreachable!("instance was converted to dictionary storage")
            };
            values.remove(&symbol)
        })?;
        Ok(self.heap.handle_optional(removed.as_ref()))
    }

    fn insert_dictionary(
        &mut self,
        instance: Value<'_>,
        symbol: SymbolId,
        value: Value<'_>,
    ) -> Result<(), String> {
        if !self.convert_to_dictionary(instance, INSTANCE_DICT_ENTRY_BYTES)? {
            return self.insert_by_symbol(instance, symbol, value);
        }
        self.heap.modify(instance, |builder, object| {
            let Object::Instance {
                attributes: InstanceAttributes::Dictionary(values),
                ..
            } = object
            else {
                unreachable!("instance was converted to dictionary storage")
            };
            values.insert(symbol, builder.store(value));
        })
    }

    /// Move a shaped instance's attributes into dictionary storage, reserving `extra` more
    /// bytes. Returns `false` when the instance already uses a dictionary.
    fn convert_to_dictionary(&mut self, instance: Value<'_>, extra: u64) -> Result<bool, String> {
        let (shape, shaped_len) = match instance_attributes(self.heap, instance)? {
            InstanceAttributes::Shaped { shape, values } => (*shape, values.len()),
            InstanceAttributes::Dictionary(_) => return Ok(false),
        };
        let existing = u64::try_from(shaped_len)
            .unwrap_or(u64::MAX)
            .saturating_mul(INSTANCE_DICT_ENTRY_BYTES.saturating_sub(INSTANCE_SLOT_BYTES));
        self.heap.reserve_object_growth(
            instance,
            existing.saturating_add(extra),
            self.roots,
            self.resources,
        )?;
        let names = (0..shaped_len)
            .map(|slot| {
                self.shapes
                    .attribute_at(shape, slot)
                    .ok_or("invalid instance shape slot")
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.heap.modify(instance, |_, object| {
            let Object::Instance { attributes, .. } = object else {
                unreachable!("instance representation changed without yielding")
            };
            let InstanceAttributes::Shaped { values, .. } = attributes else {
                unreachable!("instance representation changed without yielding")
            };
            let values = names
                .into_iter()
                .zip(std::mem::take(values))
                .collect::<HashMap<_, _>>();
            *attributes = InstanceAttributes::Dictionary(Box::new(values));
        })?;
        Ok(true)
    }
}
