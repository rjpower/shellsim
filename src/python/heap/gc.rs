//! Minor and major collection.
//!
//! A minor collection copies every young object reachable from the roots into the old space
//! and rewrites each reference it followed: the handle stack, the slots the VM reports through
//! [`Roots`], the remembered old objects, the object under construction, and the promoted
//! objects themselves (scanned Cheney-style until no young reference remains). Everything left in
//! the young space is dead and is dropped; the young epoch then advances so a stale address
//! fails loudly.
//!
//! A major collection first promotes the live young objects, then marks the old space from the
//! same roots and sweeps the unmarked objects onto the free list. It runs when the old space has
//! doubled since the previous major collection (or exceeds a floor), so tracing work stays
//! proportional to allocation.
//!
//! Both collections charge one CPU unit per object visited or swept, like the guest work that
//! produced those objects.

use crate::resources::Resources;

use super::value::Raw;
use super::{
    ArrayStorage, ClassObject, FunctionObject, GeneratorObject, Heap, HeapObject,
    InstanceAttributes, NamespaceTarget, Object, ObjectId, ProxyTarget, Ref, ScopeObject,
};

/// Stored references held outside the heap, which the collector must trace and may rewrite.
///
/// The VM implements this for its resumable state (operand stack, frames, pending exceptions,
/// globals, modules, type tables). Missing a root is a bug that the young epoch check turns into
/// an error on the next access instead of silent aliasing.
pub trait Roots {
    fn visit_refs(&mut self, visitor: &mut dyn FnMut(&mut Ref));
}

impl Roots for () {
    fn visit_refs(&mut self, _visitor: &mut dyn FnMut(&mut Ref)) {}
}

impl Roots for Vec<Ref> {
    fn visit_refs(&mut self, visitor: &mut dyn FnMut(&mut Ref)) {
        for slot in self {
            visitor(slot);
        }
    }
}

impl Roots for Option<Ref> {
    fn visit_refs(&mut self, visitor: &mut dyn FnMut(&mut Ref)) {
        if let Some(slot) = self {
            visitor(slot);
        }
    }
}

/// Modeled bytes the young space holds before a minor collection: a sixteenth of the memory
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
pub(super) fn for_each_object_ref(object: &mut HeapObject, f: &mut dyn FnMut(&mut Raw)) {
    if let Some(attributes) = &mut object.attributes {
        match &mut **attributes {
            InstanceAttributes::Shaped { values, .. } => slots(values, f),
            InstanceAttributes::Dictionary(values) => {
                for slot in values.values_mut() {
                    f(&mut slot.0);
                }
            }
        }
    }
    for_each_ref(&mut object.payload, f);
}

/// Visit every reference stored in a payload.
pub(super) fn for_each_ref(object: &mut Object, f: &mut dyn FnMut(&mut Raw)) {
    match object {
        Object::List(items)
        | Object::Tuple(items)
        | Object::ArrayStorage(ArrayStorage::Values(items))
        | Object::Iterator { values: items, .. }
        | Object::Exception(items) => slots(items, f),
        Object::Set(members) | Object::FrozenSet(members) => members.visit_refs(f),
        Object::Dict(entries) => entries.visit_refs(f),
        Object::DefaultDict { factory, entries } => {
            f(&mut factory.0);
            entries.visit_refs(f);
        }
        Object::Slice { start, stop, step } => {
            f(&mut start.0);
            f(&mut stop.0);
            f(&mut step.0);
        }
        Object::Function(function) => {
            let FunctionObject {
                closure,
                defaults,
                defining_class,
                attributes,
                ..
            } = &mut **function;
            optional(closure, f);
            slots(defaults, f);
            optional(defining_class, f);
            for slot in attributes.values_mut() {
                f(&mut slot.0);
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
            } = &mut **class;
            slots(bases, f);
            slots(mro, f);
            f(&mut metaclass.0);
            for slot in attributes.values_mut() {
                f(&mut slot.0);
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
            f(&mut receiver.0);
            f(&mut descriptor.0);
            optional(owner, f);
        }
        Object::GenericAlias { origin, arguments } => {
            f(&mut origin.0);
            slots(arguments, f);
        }
        Object::SequenceIterator { owner, .. } | Object::ReverseIterator { owner, .. } => {
            f(&mut owner.0)
        }
        Object::CallableIterator {
            callable, sentinel, ..
        } => {
            f(&mut callable.0);
            f(&mut sentinel.0);
        }
        Object::Generator(generator) => {
            let GeneratorObject {
                scope,
                exceptions,
                stack,
                return_value,
                ..
            } = &mut **generator;
            f(&mut scope.0);
            for (_, slot) in exceptions {
                f(&mut slot.0);
            }
            slots(stack, f);
            f(&mut return_value.0);
        }
        Object::Module { scope, .. } => f(&mut scope.0),
        Object::Scope(scope) => {
            let ScopeObject {
                parent,
                locals,
                values,
                ..
            } = &mut **scope;
            optional(parent, f);
            for slot in locals.iter_mut().flatten() {
                f(&mut slot.0);
            }
            for slot in values.values_mut() {
                f(&mut slot.0);
            }
        }
        Object::NamespaceDict(NamespaceTarget::Scope(target))
        | Object::NamespaceDict(NamespaceTarget::Instance(target))
        | Object::DictView {
            mapping: target, ..
        }
        | Object::MappingProxy(ProxyTarget::Class(target)) => f(&mut target.0),
        // The REPL/script global table is a VM root, so this view owns nothing further.
        Object::NamespaceDict(NamespaceTarget::Repl)
        | Object::MappingProxy(ProxyTarget::NativeModule(_) | ProxyTarget::RegisteredType(_)) => {}
        Object::Array { storage, base, .. } => {
            f(&mut storage.0);
            optional(base, f);
        }
        Object::Native(native) => native.visit_refs(&mut |slot| f(&mut slot.0)),
        Object::Property { getter, setter } => {
            f(&mut getter.0);
            optional(setter, f);
        }
        Object::StaticMethod { callable } | Object::ClassMethod { callable } => f(&mut callable.0),
        Object::Super {
            start_class,
            receiver,
        } => {
            f(&mut start_class.0);
            f(&mut receiver.0);
        }
        Object::Bare
        | Object::String(_)
        | Object::Bytes(_)
        | Object::ByteArray(_)
        | Object::ArrayStorage(ArrayStorage::Bytes(_))
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

fn slots(slots: &mut [Ref], f: &mut dyn FnMut(&mut Raw)) {
    for slot in slots {
        f(&mut slot.0);
    }
}

fn optional(slot: &mut Option<Ref>, f: &mut dyn FnMut(&mut Raw)) {
    if let Some(slot) = slot {
        f(&mut slot.0);
    }
}

/// Promotion bookkeeping for one minor collection.
struct Promotion {
    /// Old index each young object was copied to, by young index.
    forward: Vec<Option<usize>>,
    /// Promoted objects whose own references have not been scanned yet.
    pending: Vec<usize>,
    error: Option<String>,
}

impl Heap {
    /// Copy the reachable young objects into the old space and drop the rest.
    pub fn collect_young(
        &mut self,
        roots: &mut dyn Roots,
        pending: Option<&mut Object>,
        resources: &mut Resources,
    ) -> Result<(), String> {
        let young_count = self.young.len();
        charge_collection(young_count, resources)?;
        let mut promotion = Promotion {
            forward: vec![None; young_count],
            pending: Vec::new(),
            error: None,
        };

        let mut handles = std::mem::take(self.handles.get_mut());
        for raw in &mut handles {
            self.forward(raw, &mut promotion);
        }
        *self.handles.get_mut() = handles;

        roots.visit_refs(&mut |slot| self.forward(&mut slot.0, &mut promotion));
        if let Some(object) = pending {
            for_each_ref(object, &mut |raw| self.forward(raw, &mut promotion));
        }
        let remembered = std::mem::take(&mut self.remembered);
        for index in remembered {
            self.scan_old(index, &mut promotion, true);
        }
        while let Some(index) = promotion.pending.pop() {
            self.scan_old(index, &mut promotion, false);
        }
        if let Some(error) = promotion.error {
            return Err(error);
        }

        let mut released = 0u64;
        let mut promoted_bytes = 0u64;
        for object in self.young.drain(..).flatten() {
            released = released.saturating_add(object.modeled_bytes);
        }
        for index in promotion.forward.iter().flatten() {
            if let Some(object) = self.old.get(*index).and_then(Option::as_ref) {
                promoted_bytes = promoted_bytes.saturating_add(object.modeled_bytes);
            }
        }
        self.young_epoch = self.young_epoch.wrapping_add(1) & 0x7fff_ffff;
        self.young_bytes = 0;
        self.old_bytes = self.old_bytes.saturating_add(promoted_bytes);
        self.modeled_bytes = self.modeled_bytes.saturating_sub(released);
        resources.release_memory(released);
        self.stats.minor_collections += 1;
        self.stats.promoted_objects += promotion
            .forward
            .iter()
            .filter(|target| target.is_some())
            .count() as u64;
        charge_collection(promotion.forward.len(), resources)?;
        Ok(())
    }

    /// Rewrite one reference to a young object so it names the object's promoted copy, copying
    /// the object first if this is the first reference to reach it.
    fn forward(&mut self, raw: &mut Raw, promotion: &mut Promotion) {
        let Some(id) = raw.object_id() else {
            return;
        };
        if !id.is_young() {
            return;
        }
        if id.epoch() != self.young_epoch {
            promotion
                .error
                .get_or_insert_with(|| "stale reference to a moved young object".into());
            return;
        }
        let index = id.index();
        let target = match promotion.forward.get(index).copied().flatten() {
            Some(target) => target,
            None => {
                let Some(object) = self.young.get_mut(index).and_then(Option::take) else {
                    promotion
                        .error
                        .get_or_insert_with(|| "invalid young object reference".into());
                    return;
                };
                let target = if let Some(free) = self.free_old.pop() {
                    self.old[free] = Some(object);
                    free
                } else {
                    self.old.push(Some(object));
                    self.old.len() - 1
                };
                promotion.forward[index] = Some(target);
                promotion.pending.push(target);
                target
            }
        };
        *raw = Raw::object(ObjectId::old(target));
    }

    /// Forward every young reference stored in the old object at `index`.
    fn scan_old(&mut self, index: usize, promotion: &mut Promotion, clear_dirty: bool) {
        let Some(mut object) = self.old.get_mut(index).and_then(Option::take) else {
            return;
        };
        if clear_dirty {
            object.set_dirty(false);
        }
        for_each_object_ref(&mut object, &mut |raw| self.forward(raw, promotion));
        self.old[index] = Some(object);
    }

    /// Promote the live young objects, then reclaim every unreachable old object.
    /// Returns the modeled bytes released.
    pub fn collect_full(
        &mut self,
        roots: &mut dyn Roots,
        mut pending: Option<&mut Object>,
        resources: &mut Resources,
    ) -> Result<u64, String> {
        self.collect_young(roots, pending.as_deref_mut(), resources)?;
        charge_collection(self.old.len(), resources)?;
        let mut marked = vec![false; self.old.len()];
        let mut work: Vec<usize> = Vec::new();
        let push = |raw: &Raw, work: &mut Vec<usize>| {
            if let Some(id) = raw.object_id() {
                work.push(id.index());
            }
        };
        for raw in self.handles.borrow().iter() {
            push(raw, &mut work);
        }
        roots.visit_refs(&mut |slot| push(&slot.0, &mut work));
        if let Some(object) = pending {
            for_each_ref(object, &mut |raw| push(raw, &mut work));
        }
        while let Some(index) = work.pop() {
            let Some(flag) = marked.get_mut(index) else {
                return Err("invalid object reference during collection".into());
            };
            if *flag {
                continue;
            }
            *flag = true;
            let Some(mut object) = self.old.get_mut(index).and_then(Option::take) else {
                return Err("invalid object reference during collection".into());
            };
            for_each_object_ref(&mut object, &mut |raw| push(raw, &mut work));
            self.old[index] = Some(object);
        }

        let mut released = 0u64;
        for (index, slot) in self.old.iter_mut().enumerate() {
            if marked[index] {
                continue;
            }
            if let Some(object) = slot.take() {
                released = released.saturating_add(object.modeled_bytes);
                self.free_old.push(index);
            }
        }
        self.old_bytes = self.old_bytes.saturating_sub(released);
        self.modeled_bytes = self.modeled_bytes.saturating_sub(released);
        self.major_threshold = self.old_bytes.saturating_mul(2);
        resources.release_memory(released);
        self.stats.major_collections += 1;
        Ok(released)
    }

    #[cfg(test)]
    pub(super) fn live_objects(&self) -> usize {
        self.young.iter().flatten().count() + self.old.iter().flatten().count()
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
