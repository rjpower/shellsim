//! Deep copies of heap objects for process-state snapshots.
//!
//! Stored references cannot be cloned outside the heap, so object payloads have no `Clone`.
//! This module duplicates them for [`Heap::clone`](super::Heap), where every reference is copied
//! together with the space it points into.

use super::{
    ClassObject, FunctionObject, GeneratorObject, InstanceAttributes, NamespaceTarget, Object,
    ProxyTarget, Ref, ScopeObject,
};

fn slots(slots: &[Ref]) -> Vec<Ref> {
    slots.iter().map(Ref::dup).collect()
}

fn optional(slot: &Option<Ref>) -> Option<Ref> {
    slot.as_ref().map(Ref::dup)
}

pub(super) fn dup_attributes(attributes: &InstanceAttributes) -> InstanceAttributes {
    match attributes {
        InstanceAttributes::Shaped { shape, values } => InstanceAttributes::Shaped {
            shape: *shape,
            values: slots(values),
        },
        InstanceAttributes::Dictionary(values) => {
            InstanceAttributes::Dictionary(Box::new(values.dup()))
        }
    }
}

pub(super) fn dup_object(object: &Object) -> Object {
    match object {
        Object::Bare => Object::Bare,
        Object::Float(value) => Object::Float(*value),
        Object::String(value) => Object::String(value.clone()),
        Object::Bytes(value) => Object::Bytes(value.clone()),
        Object::ByteArray(value) => Object::ByteArray(value.clone()),
        Object::Exception(args) => Object::Exception(slots(args)),
        Object::List(items) => Object::List(slots(items)),
        Object::Tuple(items) => Object::Tuple(slots(items)),
        Object::Slice { start, stop, step } => Object::Slice {
            start: start.dup(),
            stop: stop.dup(),
            step: step.dup(),
        },
        Object::Dict(entries) => Object::Dict(entries.dup()),
        Object::DefaultDict { factory, entries } => Object::DefaultDict {
            factory: factory.dup(),
            entries: entries.dup(),
        },
        Object::Set(members) => Object::Set(members.dup()),
        Object::FrozenSet(members) => Object::FrozenSet(members.dup()),
        Object::BigInt(value) => Object::BigInt(value.clone()),
        Object::Complex { real, imag } => Object::Complex {
            real: *real,
            imag: *imag,
        },
        Object::Range { start, stop, step } => Object::Range {
            start: *start,
            stop: *stop,
            step: *step,
        },
        Object::Function(function) => Object::Function(Box::new(FunctionObject {
            name: function.name.clone(),
            code: function.code.clone(),
            closure: function.closure.dup(),
            globals: function.globals.dup(),
            defaults: slots(&function.defaults),
            defining_class: optional(&function.defining_class),
            attributes: function.attributes.dup(),
        })),
        Object::Class(class) => Object::Class(Box::new(ClassObject {
            instance_type: class.instance_type,
            name: class.name.clone(),
            bases: slots(&class.bases),
            mro: slots(&class.mro),
            metaclass: class.metaclass.dup(),
            layout: class.layout,
            exception_base: class.exception_base,
            attributes: class.attributes.dup(),
            is_dataclass: class.is_dataclass,
            dataclass_fields: class
                .dataclass_fields
                .iter()
                .map(|(name, value)| (name.clone(), optional(value)))
                .collect(),
            enum_members: slots(&class.enum_members),
        })),
        Object::DescriptorBoundMethod {
            receiver,
            descriptor,
            owner,
        } => Object::DescriptorBoundMethod {
            receiver: receiver.dup(),
            descriptor: descriptor.dup(),
            owner: optional(owner),
        },
        Object::GenericAlias { origin, arguments } => Object::GenericAlias {
            origin: origin.dup(),
            arguments: slots(arguments),
        },
        Object::Iterator { values, position } => Object::Iterator {
            values: slots(values),
            position: *position,
        },
        Object::SequenceIterator { owner, position } => Object::SequenceIterator {
            owner: owner.dup(),
            position: *position,
        },
        Object::ReverseIterator { owner, next } => Object::ReverseIterator {
            owner: owner.dup(),
            next: *next,
        },
        Object::RangeIterator {
            current,
            stop,
            step,
            exhausted,
        } => Object::RangeIterator {
            current: *current,
            stop: *stop,
            step: *step,
            exhausted: *exhausted,
        },
        Object::CountIterator { current, step } => Object::CountIterator {
            current: *current,
            step: *step,
        },
        Object::CallableIterator {
            callable,
            sentinel,
            exhausted,
        } => Object::CallableIterator {
            callable: callable.dup(),
            sentinel: sentinel.dup(),
            exhausted: *exhausted,
        },
        Object::StreamIterator { binary } => Object::StreamIterator { binary: *binary },
        Object::Generator(generator) => Object::Generator(Box::new(GeneratorObject {
            function: generator.function.dup(),
            code: generator.code.clone(),
            scope: generator.scope.dup(),
            globals: generator.globals.dup(),
            instruction_pointer: generator.instruction_pointer,
            handlers: generator.handlers.clone(),
            contexts: slots(&generator.contexts),
            exceptions: slots(&generator.exceptions),
            stack: slots(&generator.stack),
            exhausted: generator.exhausted,
            running: generator.running,
            return_value: generator.return_value.dup(),
        })),
        Object::Module { name, scope } => Object::Module {
            name: name.clone(),
            scope: scope.dup(),
        },
        Object::Scope(scope) => Object::Scope(Box::new(ScopeObject {
            parent: optional(&scope.parent),
            local_names: scope.local_names.clone(),
            locals: scope.locals.iter().map(optional).collect(),
            names: scope.names.dup(),
        })),
        Object::NamespaceDict(target) => Object::NamespaceDict(match target {
            NamespaceTarget::Scope(scope) => NamespaceTarget::Scope(scope.dup()),
            NamespaceTarget::Instance(instance) => NamespaceTarget::Instance(instance.dup()),
        }),
        Object::DictView { kind, mapping } => Object::DictView {
            kind: *kind,
            mapping: mapping.dup(),
        },
        Object::MappingProxy(target) => Object::MappingProxy(match target {
            ProxyTarget::Class(class) => ProxyTarget::Class(class.dup()),
            ProxyTarget::RegisteredType(type_id) => ProxyTarget::RegisteredType(*type_id),
            ProxyTarget::NativeModule(module) => ProxyTarget::NativeModule(module),
        }),
        Object::WideValue {
            type_id,
            kind,
            payload,
        } => Object::WideValue {
            type_id: *type_id,
            kind: *kind,
            payload: *payload,
        },
        Object::Native(native) => Object::Native(native.dup()),
        Object::Property { getter, setter } => Object::Property {
            getter: getter.dup(),
            setter: optional(setter),
        },
        Object::StaticMethod { callable } => Object::StaticMethod {
            callable: callable.dup(),
        },
        Object::ClassMethod { callable } => Object::ClassMethod {
            callable: callable.dup(),
        },
        Object::Super {
            start_class,
            receiver,
        } => Object::Super {
            start_class: start_class.dup(),
            receiver: receiver.dup(),
        },
    }
}
