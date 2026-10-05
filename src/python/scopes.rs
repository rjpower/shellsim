//! Name resolution through lexical scopes.
//!
//! A scope ([`ScopeObject`]) is an ordinary heap object; this module implements the lookups and
//! stores the interpreter performs on one: local slots assigned by the compiler, dynamically
//! bound names, and the walk through enclosing scopes. Everything here goes through handles and
//! [`Heap::modify`], so a store into an old scope is remembered for the collector.

use std::collections::HashMap;
use std::sync::Arc;

use crate::resources::Resources;

use super::heap::{Heap, Object, Ref, Roots, ScopeObject, Value, DYNAMIC_NAME_BYTES};

/// The fixed part of a new scope: where name lookup continues and which names have slots.
pub struct ScopeLayout<'a> {
    pub parent: Option<Value<'a>>,
    pub uses_repl_globals: bool,
    pub local_names: Arc<[String]>,
}

/// Allocate a scope whose compiler-assigned local slots are already populated.
pub fn alloc_scope<'s>(
    heap: &mut Heap,
    layout: ScopeLayout<'_>,
    locals: Vec<Option<Value<'_>>>,
    values: HashMap<String, Value<'_>>,
    roots: &mut dyn Roots,
    resources: &mut Resources,
) -> Result<Value<'s>, String> {
    if locals.len() != layout.local_names.len() {
        return Err("local slot metadata mismatch".into());
    }
    heap.alloc_with(roots, resources, |builder| {
        Object::Scope(Box::new(ScopeObject {
            parent: builder.optional(layout.parent),
            uses_repl_globals: layout.uses_repl_globals,
            local_names: layout.local_names,
            locals: locals
                .into_iter()
                .map(|value| builder.optional(value))
                .collect(),
            order: Vec::new(),
            values: builder.named(values),
        }))
    })
}

/// Allocate a scope, placing each of `values` that names a local slot into that slot.
pub fn alloc_scope_named<'s>(
    heap: &mut Heap,
    layout: ScopeLayout<'_>,
    mut values: HashMap<String, Value<'_>>,
    roots: &mut dyn Roots,
    resources: &mut Resources,
) -> Result<Value<'s>, String> {
    let locals = layout
        .local_names
        .iter()
        .map(|name| values.remove(name))
        .collect::<Vec<_>>();
    alloc_scope(heap, layout, locals, values, roots, resources)
}

fn scope<'h>(heap: &'h Heap, scope: Value<'_>) -> Result<&'h ScopeObject, String> {
    match heap.get(scope)? {
        Object::Scope(scope) => Ok(scope),
        _ => Err("invalid scope reference".into()),
    }
}

fn modify_scope<R>(
    heap: &mut Heap,
    scope: Value<'_>,
    f: impl FnOnce(&super::heap::Builder<'_>, &mut ScopeObject) -> R,
) -> Result<R, String> {
    heap.modify(scope, |builder, object| match object {
        Object::Scope(scope) => Ok(f(builder, scope)),
        _ => Err(String::from("invalid scope reference")),
    })?
}

/// Look a name up through this scope and its ancestors.
pub fn get<'s>(heap: &Heap, start: Value<'s>, name: &str) -> Result<Option<Value<'s>>, String> {
    let mut current = start;
    loop {
        let object = scope(heap, current)?;
        if let Some(index) = object.local_names.iter().position(|local| local == name) {
            if let Some(slot) = object.locals.get(index).and_then(Option::as_ref) {
                return Ok(Some(heap.handle(slot)));
            }
        }
        if let Some(slot) = object.values.get(name) {
            return Ok(Some(heap.handle(slot)));
        }
        match &object.parent {
            Some(parent) => current = heap.handle(parent),
            None => return Ok(None),
        }
    }
}

/// The outermost ancestor of `start`: a module scope, or a function scope created at the
/// REPL/script top level.
pub fn root<'s>(heap: &Heap, start: Value<'s>) -> Result<Value<'s>, String> {
    let mut current = start;
    loop {
        match &scope(heap, current)?.parent {
            Some(parent) => current = heap.handle(parent),
            None => return Ok(current),
        }
    }
}

pub fn uses_repl_globals(heap: &Heap, start: Value<'_>) -> Result<bool, String> {
    Ok(scope(heap, root(heap, start)?)?.uses_repl_globals)
}

pub fn parent<'s>(heap: &Heap, start: Value<'_>) -> Result<Option<Value<'s>>, String> {
    Ok(heap.handle_optional(scope(heap, start)?.parent.as_ref()))
}

/// The dynamic names bound in this scope in the order they were first bound, which a class
/// body's namespace keeps for its fields and enum members.
pub fn bound_names(heap: &Heap, start: Value<'_>) -> Result<Vec<String>, String> {
    Ok(scope(heap, start)?.order.clone())
}

/// Every name bound directly in this scope, slots and dynamic names together.
pub fn values<'s>(heap: &Heap, start: Value<'_>) -> Result<HashMap<String, Value<'s>>, String> {
    let object = scope(heap, start)?;
    let mut values = object
        .values
        .iter()
        .map(|(name, slot)| (name.clone(), heap.handle(slot)))
        .collect::<HashMap<_, _>>();
    for (name, slot) in object.local_names.iter().zip(&object.locals) {
        if let Some(slot) = slot {
            values.insert(name.clone(), heap.handle(slot));
        }
    }
    Ok(values)
}

/// The stored reference in one local slot, for pushing onto the operand stack directly.
pub fn local_ref<'h>(
    heap: &'h Heap,
    start: Value<'_>,
    slot: usize,
) -> Result<Option<&'h Ref>, String> {
    scope(heap, start)?
        .locals
        .get(slot)
        .map(Option::as_ref)
        .ok_or_else(|| "invalid local slot".into())
}

pub fn store_local(
    heap: &mut Heap,
    start: Value<'_>,
    slot: usize,
    value: Value<'_>,
) -> Result<(), String> {
    modify_scope(heap, start, |builder, scope| {
        match scope.locals.get_mut(slot) {
            Some(local) => {
                *local = Some(builder.store(value));
                Ok(())
            }
            None => Err(String::from("invalid local slot")),
        }
    })?
}

/// Store a reference popped from the operand stack into a local slot.
pub fn store_local_ref(
    heap: &mut Heap,
    start: Value<'_>,
    slot: usize,
    value: Ref,
) -> Result<(), String> {
    modify_scope(heap, start, |_, scope| match scope.locals.get_mut(slot) {
        Some(local) => {
            *local = Some(value);
            Ok(())
        }
        None => Err(String::from("invalid local slot")),
    })?
}

pub fn remove_local<'s>(
    heap: &mut Heap,
    start: Value<'_>,
    slot: usize,
) -> Result<Option<Value<'s>>, String> {
    let removed = modify_scope(heap, start, |_, scope| {
        scope
            .locals
            .get_mut(slot)
            .map(Option::take)
            .ok_or_else(|| String::from("invalid local slot"))
    })??;
    Ok(heap.handle_optional(removed.as_ref()))
}

/// Bind `name` in this scope: in its local slot when the compiler assigned one, otherwise as a
/// dynamic name, charging the new entry.
pub fn insert(
    heap: &mut Heap,
    start: Value<'_>,
    name: String,
    value: Value<'_>,
    roots: &mut dyn Roots,
    resources: &mut Resources,
) -> Result<(), String> {
    let object = scope(heap, start)?;
    if let Some(slot) = object.local_names.iter().position(|local| local == &name) {
        return store_local(heap, start, slot, value);
    }
    let new_name = !object.values.contains_key(&name);
    if new_name {
        heap.reserve_object_growth(start, DYNAMIC_NAME_BYTES, roots, resources)?;
    }
    modify_scope(heap, start, |builder, scope| {
        if new_name {
            scope.order.push(name.clone());
        }
        scope.values.insert(name, builder.store(value));
    })
}

/// Rebind `name` in the nearest scope, starting at `start`, that already binds it
/// (`nonlocal`). `start` is the scope enclosing the one that declared the name.
pub fn store_nonlocal(
    heap: &mut Heap,
    start: Value<'_>,
    name: &str,
    value: Value<'_>,
) -> Result<(), String> {
    let missing = || format!("no binding for nonlocal {name:?} found");
    let mut current: Value<'_> = start;
    loop {
        let candidate = scope(heap, current)?;
        let next = heap.handle_optional(candidate.parent.as_ref());
        if let Some(slot) = candidate.local_names.iter().position(|local| local == name) {
            if candidate.locals[slot].is_some() {
                return store_local(heap, current, slot, value);
            }
        }
        if candidate.values.contains_key(name) {
            return modify_scope(heap, current, |builder, scope| {
                scope.values.insert(name.to_string(), builder.store(value));
            });
        }
        current = next.ok_or_else(missing)?;
    }
}

pub fn remove<'s>(
    heap: &mut Heap,
    start: Value<'_>,
    name: &str,
) -> Result<Option<Value<'s>>, String> {
    let removed = modify_scope(heap, start, |_, scope| {
        match scope.local_names.iter().position(|local| local == name) {
            Some(slot) => scope.locals[slot].take(),
            None => {
                scope.order.retain(|bound| bound != name);
                scope.values.remove(name)
            }
        }
    })?;
    Ok(heap.handle_optional(removed.as_ref()))
}
