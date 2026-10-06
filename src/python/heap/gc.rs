//! Minor and major collection.
//!
//! Objects never move. A minor collection marks every young object reachable from the roots:
//! the pin stack, the slots the VM reports through [`Roots`], the object under construction,
//! and the old objects remembered since the last minor collection. It follows references only
//! into young objects, frees the unmarked young objects, and makes the survivors old.
//!
//! A major collection first runs a minor one, then marks the whole heap from the same roots and
//! frees every unmarked object. It runs when the old generation has doubled since the previous
//! major collection (or exceeds a floor), so tracing work stays proportional to allocation.
//!
//! Both collections charge one CPU unit per object visited or swept, like the guest work that
//! produced those objects.

use crate::resources::Resources;

use super::value::Raw;
use super::{
    ClassObject, FunctionObject, GeneratorObject, Heap, HeapObject, InstanceAttributes,
    NamespaceTarget, Object, ObjectId, ProxyTarget, Ref, ScopeObject, DIRTY_FLAG, MARK_FLAG,
    YOUNG_FLAG,
};

/// Stored references held outside the heap, which the collector must trace.
///
/// The VM implements this for its resumable state (operand stack, frames, pending exceptions,
/// globals, modules, type tables). Missing a root is a bug that the slot generation check turns
/// into an error on the next access instead of silent aliasing.
pub trait Roots {
    fn visit_refs(&self, visitor: &mut dyn FnMut(&Ref));
}

impl Roots for () {
    fn visit_refs(&self, _visitor: &mut dyn FnMut(&Ref)) {}
}

impl Roots for Vec<Ref> {
    fn visit_refs(&self, visitor: &mut dyn FnMut(&Ref)) {
        for slot in self {
            visitor(slot);
        }
    }
}

impl Roots for Option<Ref> {
    fn visit_refs(&self, visitor: &mut dyn FnMut(&Ref)) {
        if let Some(slot) = self {
            visitor(slot);
        }
    }
}

/// Modeled bytes the young generation holds before a minor collection: a sixteenth of the memory
/// limit, between 256 KiB and 4 MiB. Small enough that survivors are few, large enough that a
/// collection amortizes over tens of thousands of allocations.
pub(super) fn young_budget(memory_limit: u64) -> u64 {
    (memory_limit / 16).clamp(256 * 1024, 4 * 1024 * 1024)
}

/// Smallest old-space size that triggers a major collection, so tiny heaps are not retraced
/// after every promotion.
pub(super) fn major_floor(memory_limit: u64) -> u64 {
    (memory_limit / 8).clamp(1024 * 1024, 32 * 1024 * 1024)
}

/// Visit every reference stored in `object`: its instance attributes, then its payload.
pub(super) fn for_each_object_ref(object: &HeapObject, f: &mut dyn FnMut(&Raw)) {
    if let Some(attributes) = &object.attributes {
        match &**attributes {
            InstanceAttributes::Shaped { values, .. } => slots(values, f),
            InstanceAttributes::Dictionary(values) => {
                for slot in values.values() {
                    f(&slot.0);
                }
            }
        }
    }
    for_each_ref(&object.payload, f);
}

/// Visit every reference stored in a payload.
pub(super) fn for_each_ref(object: &Object, f: &mut dyn FnMut(&Raw)) {
    match object {
        Object::List(items)
        | Object::Tuple(items)
        | Object::Iterator { values: items, .. }
        | Object::Exception(items) => slots(items, f),
        Object::Set(members) | Object::FrozenSet(members) => members.visit_refs(f),
        Object::Dict(entries) => entries.visit_refs(f),
        Object::DefaultDict { factory, entries } => {
            f(&factory.0);
            entries.visit_refs(f);
        }
        Object::Slice { start, stop, step } => {
            f(&start.0);
            f(&stop.0);
            f(&step.0);
        }
        Object::Function(function) => {
            let FunctionObject {
                closure,
                defaults,
                defining_class,
                attributes,
                ..
            } = &**function;
            optional(closure, f);
            slots(defaults, f);
            optional(defining_class, f);
            for slot in attributes.values() {
                f(&slot.0);
            }
        }
        Object::Class(class) => {
            let ClassObject {
                bases,
                mro,
                metaclass,
                attributes,
                dataclass_fields,
                enum_members,
                ..
            } = &**class;
            slots(bases, f);
            slots(mro, f);
            f(&metaclass.0);
            for slot in attributes.values() {
                f(&slot.0);
            }
            for (_, slot) in dataclass_fields {
                optional(slot, f);
            }
            slots(enum_members, f);
        }
        Object::DescriptorBoundMethod {
            receiver,
            descriptor,
            owner,
        } => {
            f(&receiver.0);
            f(&descriptor.0);
            optional(owner, f);
        }
        Object::GenericAlias { origin, arguments } => {
            f(&origin.0);
            slots(arguments, f);
        }
        Object::SequenceIterator { owner, .. } | Object::ReverseIterator { owner, .. } => {
            f(&owner.0)
        }
        Object::CallableIterator {
            callable, sentinel, ..
        } => {
            f(&callable.0);
            f(&sentinel.0);
        }
        Object::Generator(generator) => {
            let GeneratorObject {
                function,
                scope,
                contexts,
                exceptions,
                stack,
                return_value,
                ..
            } = &**generator;
            f(&function.0);
            f(&scope.0);
            slots(contexts, f);
            for (_, slot) in exceptions {
                f(&slot.0);
            }
            slots(stack, f);
            f(&return_value.0);
        }
        Object::Module { scope, .. } => f(&scope.0),
        Object::Scope(scope) => {
            let ScopeObject {
                parent,
                locals,
                values,
                ..
            } = &**scope;
            optional(parent, f);
            for slot in locals.iter().flatten() {
                f(&slot.0);
            }
            for slot in values.values() {
                f(&slot.0);
            }
        }
        Object::NamespaceDict(NamespaceTarget::Scope(target))
        | Object::NamespaceDict(NamespaceTarget::Instance(target))
        | Object::DictView {
            mapping: target, ..
        }
        | Object::MappingProxy(ProxyTarget::Class(target)) => f(&target.0),
        // The REPL/script global table is a VM root, so this view owns nothing further.
        Object::NamespaceDict(NamespaceTarget::Repl)
        | Object::MappingProxy(ProxyTarget::NativeModule(_) | ProxyTarget::RegisteredType(_)) => {}
        Object::Native(native) => native.visit_refs(&mut |slot| f(&slot.0)),
        Object::Property { getter, setter } => {
            f(&getter.0);
            optional(setter, f);
        }
        Object::StaticMethod { callable } | Object::ClassMethod { callable } => f(&callable.0),
        Object::Super {
            start_class,
            receiver,
        } => {
            f(&start_class.0);
            f(&receiver.0);
        }
        Object::Bare
        | Object::String(_)
        | Object::Bytes(_)
        | Object::ByteArray(_)
        | Object::WideValue { .. }
        | Object::BigInt(_)
        | Object::Float(_)
        | Object::Complex { .. }
        | Object::Range { .. }
        | Object::RangeIterator { .. }
        | Object::CountIterator { .. }
        | Object::StreamIterator { .. } => {}
    }
}

fn slots(slots: &[Ref], f: &mut dyn FnMut(&Raw)) {
    for slot in slots {
        f(&slot.0);
    }
}

fn optional(slot: &Option<Ref>, f: &mut dyn FnMut(&Raw)) {
    if let Some(slot) = slot {
        f(&slot.0);
    }
}

impl Heap {
    /// Free the young objects no root reaches and make the rest old.
    pub fn collect_young(
        &mut self,
        roots: &dyn Roots,
        pending: Option<&Object>,
        resources: &mut Resources,
    ) -> Result<(), String> {
        let young_count = self.young.len();
        charge_collection(young_count, resources)?;
        let mut work = Vec::new();
        let mark = |raw: &Raw, work: &mut Vec<ObjectId>| {
            if let Some(id) = raw.object_id() {
                if self.mark(id, true) {
                    work.push(id);
                }
            }
        };
        for &id in self.pins.borrow().iter() {
            mark(&Raw::object(id), &mut work);
        }
        roots.visit_refs(&mut |slot| mark(&slot.0, &mut work));
        if let Some(object) = pending {
            for_each_ref(object, &mut |raw| mark(raw, &mut work));
        }
        for &index in &self.remembered {
            if let Some(Some(object)) = self.slots.get(index as usize) {
                object.set(DIRTY_FLAG, false);
                for_each_object_ref(object, &mut |raw| mark(raw, &mut work));
            }
        }
        self.trace(&mut work, true)?;
        self.remembered.clear();

        let mut released = 0u64;
        let mut promoted_bytes = 0u64;
        let mut promoted = 0u64;
        for index in std::mem::take(&mut self.young) {
            let Some(object) = self.slots[index as usize].as_ref() else {
                continue;
            };
            if object.has(MARK_FLAG) {
                object.set(MARK_FLAG | YOUNG_FLAG, false);
                promoted_bytes = promoted_bytes.saturating_add(object.modeled_bytes);
                promoted += 1;
            } else {
                released = released.saturating_add(self.free_slot(index));
            }
        }
        self.young_bytes = 0;
        self.old_bytes = self.old_bytes.saturating_add(promoted_bytes);
        self.modeled_bytes = self.modeled_bytes.saturating_sub(released);
        resources.release_memory(released);
        self.stats.minor_collections += 1;
        self.stats.promoted_objects += promoted;
        charge_collection(usize::try_from(promoted).unwrap_or(usize::MAX), resources)?;
        Ok(())
    }

    /// Promote the live young objects, then reclaim every unreachable object.
    /// Returns the modeled bytes released.
    pub fn collect_full(
        &mut self,
        roots: &dyn Roots,
        pending: Option<&Object>,
        resources: &mut Resources,
    ) -> Result<u64, String> {
        self.collect_young(roots, pending, resources)?;
        charge_collection(self.slots.len(), resources)?;
        let mut work = Vec::new();
        let mark = |raw: &Raw, work: &mut Vec<ObjectId>| {
            if let Some(id) = raw.object_id() {
                if self.mark(id, false) {
                    work.push(id);
                }
            }
        };
        for &id in self.pins.borrow().iter() {
            mark(&Raw::object(id), &mut work);
        }
        roots.visit_refs(&mut |slot| mark(&slot.0, &mut work));
        if let Some(object) = pending {
            for_each_ref(object, &mut |raw| mark(raw, &mut work));
        }
        self.trace(&mut work, false)?;

        let mut released = 0u64;
        for index in 0..self.slots.len() {
            let Some(object) = self.slots[index].as_ref() else {
                continue;
            };
            if object.has(MARK_FLAG) {
                object.set(MARK_FLAG, false);
            } else {
                released = released.saturating_add(self.free_slot(index as u32));
            }
        }
        self.old_bytes = self.old_bytes.saturating_sub(released);
        self.modeled_bytes = self.modeled_bytes.saturating_sub(released);
        self.major_threshold = self.old_bytes.saturating_mul(2);
        resources.release_memory(released);
        self.stats.major_collections += 1;
        Ok(released)
    }

    /// Mark the object `id` names, returning whether it was newly marked and so needs tracing.
    /// A minor collection marks only young objects; old ones count as live.
    fn mark(&self, id: ObjectId, young_only: bool) -> bool {
        let Some(Some(object)) = self.slots.get(id.index()) else {
            return false;
        };
        if object.generation() != id.generation()
            || object.has(MARK_FLAG)
            || (young_only && !object.has(YOUNG_FLAG))
        {
            return false;
        }
        object.set(MARK_FLAG, true);
        true
    }

    /// Mark everything reachable from the objects in `work`.
    fn trace(&self, work: &mut Vec<ObjectId>, young_only: bool) -> Result<(), String> {
        while let Some(id) = work.pop() {
            let Some(Some(object)) = self.slots.get(id.index()) else {
                return Err("invalid object reference during collection".into());
            };
            for_each_object_ref(object, &mut |raw| {
                if let Some(target) = raw.object_id() {
                    if self.mark(target, young_only) {
                        work.push(target);
                    }
                }
            });
        }
        Ok(())
    }

    /// Free one slot, advancing its generation so ids of the freed object go stale. Returns the
    /// modeled bytes the object held.
    fn free_slot(&mut self, index: u32) -> u64 {
        let Some(object) = self.slots[index as usize].take() else {
            return 0;
        };
        let generation = (object.generation() + 1) & super::GENERATION_MASK;
        self.free.push((index, generation));
        object.modeled_bytes
    }

    #[cfg(test)]
    pub(super) fn live_objects(&self) -> usize {
        self.slots.iter().flatten().count()
    }
}

fn charge_collection(objects: usize, resources: &mut Resources) -> Result<(), String> {
    if resources.charge_cpu(u64::try_from(objects).unwrap_or(u64::MAX)) {
        Ok(())
    } else {
        Err("resource limit exceeded while executing Python".into())
    }
}

impl HeapObject {
    /// A deep copy of one object for heap snapshots.
    pub(super) fn dup(&self) -> Self {
        Self {
            type_id: self.type_id,
            flags: self.flags.clone(),
            payload: super::snapshot::dup_object(&self.payload),
            attributes: self
                .attributes
                .as_deref()
                .map(|attributes| Box::new(super::snapshot::dup_attributes(attributes))),
            modeled_bytes: self.modeled_bytes,
        }
    }
}
