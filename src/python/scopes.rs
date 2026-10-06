//! Name resolution through lexical scopes.
//!
//! A scope ([`ScopeObject`]) is an ordinary heap object; this module implements the lookups and
//! stores the interpreter performs on one: local slots assigned by the compiler, dynamically
//! bound names, and the walk through enclosing scopes. Stores go through [`Heap::modify`], so a
//! store into an old scope is remembered for the collector.

use std::collections::HashMap;
use std::sync::Arc;

use crate::resources::Resources;

use super::heap::{Heap, Object, Ref, Roots, ScopeObject, Value, DYNAMIC_NAME_BYTES};

/// The fixed part of a new scope: where name lookup continues and which names have slots.
pub struct ScopeLayout {
    pub parent: Option<Value>,
    pub uses_repl_globals: bool,
    pub local_names: Arc<[String]>,
}

/// Allocate a scope whose compiler-assigned local slots are already populated.
pub fn alloc_scope(
    heap: &mut Heap,
    layout: ScopeLayout,
    locals: Vec<Option<Value>>,
    values: HashMap<String, Value>,
    roots: &dyn Roots,
    resources: &mut Resources,
) -> Result<Value, String> {
    if locals.len() != layout.local_names.len() {
        return Err("local slot metadata mismatch".into());
    }
    let object = Object::Scope(Box::new(ScopeObject {
        parent: Ref::optional(layout.parent),
        uses_repl_globals: layout.uses_repl_globals,
        local_names: layout.local_names,
        locals: locals.into_iter().map(Ref::optional).collect(),
        order: Vec::new(),
        values: Ref::named(values),
    }));
    heap.alloc(object, roots, resources)
}

/// Allocate a scope, placing each of `values` that names a local slot into that slot.
pub fn alloc_scope_named(
    heap: &mut Heap,
    layout: ScopeLayout,
    mut values: HashMap<String, Value>,
    roots: &dyn Roots,
    resources: &mut Resources,
) -> Result<Value, String> {
    let locals = layout
        .local_names
        .iter()
        .map(|name| values.remove(name))
        .collect::<Vec<_>>();
    alloc_scope(heap, layout, locals, values, roots, resources)
}

fn scope(heap: &Heap, scope: Value) -> Result<&ScopeObject, String> {
    match heap.get(scope)? {
        Object::Scope(scope) => Ok(scope),
        _ => Err("invalid scope reference".into()),
    }
}

fn modify_scope<R>(
    heap: &mut Heap,
    scope: Value,
    f: impl FnOnce(&mut ScopeObject) -> R,
) -> Result<R, String> {
    heap.modify(scope, |object| match object {
        Object::Scope(scope) => Ok(f(scope)),
        _ => Err(String::from("invalid scope reference")),
    })?
}

/// Look a name up through this scope and its ancestors.
pub fn get(heap: &Heap, start: Value, name: &str) -> Result<Option<Value>, String> {
    let mut current = start;
    loop {
        let object = scope(heap, current)?;
        if let Some(index) = object.local_names.iter().position(|local| local == name) {
            if let Some(slot) = object.locals.get(index).and_then(Option::as_ref) {
                return Ok(Some(heap.value(slot)));
            }
        }
        if let Some(slot) = object.values.get(name) {
            return Ok(Some(heap.value(slot)));
        }
        match &object.parent {
            Some(parent) => current = heap.value(parent),
            None => return Ok(None),
        }
    }
}

/// The outermost ancestor of `start`: a module scope, or a function scope created at the
/// REPL/script top level.
pub fn root(heap: &Heap, start: Value) -> Result<Value, String> {
    let mut current = start;
    loop {
        match &scope(heap, current)?.parent {
            Some(parent) => current = heap.value(parent),
            None => return Ok(current),
        }
    }
}

pub fn uses_repl_globals(heap: &Heap, start: Value) -> Result<bool, String> {
    Ok(scope(heap, root(heap, start)?)?.uses_repl_globals)
}

pub fn parent(heap: &Heap, start: Value) -> Result<Option<Value>, String> {
    Ok(heap.value_optional(scope(heap, start)?.parent.as_ref()))
}

/// The dynamic names bound in this scope in the order they were first bound, which a class
/// body's namespace keeps for its fields and enum members.
pub fn bound_names(heap: &Heap, start: Value) -> Result<Vec<String>, String> {
    Ok(scope(heap, start)?.order.clone())
}

/// Every name bound directly in this scope, slots and dynamic names together.
pub fn values(heap: &Heap, start: Value) -> Result<HashMap<String, Value>, String> {
    let object = scope(heap, start)?;
    let mut values = object
        .values
        .iter()
        .map(|(name, slot)| (name.clone(), heap.value(slot)))
        .collect::<HashMap<_, _>>();
    for (name, slot) in object.local_names.iter().zip(&object.locals) {
        if let Some(slot) = slot {
            values.insert(name.clone(), heap.value(slot));
        }
    }
    Ok(values)
}

/// The stored reference in one local slot, for pushing onto the operand stack directly.
pub fn local_ref(heap: &Heap, start: Value, slot: usize) -> Result<Option<&Ref>, String> {
    scope(heap, start)?
        .locals
        .get(slot)
        .map(Option::as_ref)
        .ok_or_else(|| "invalid local slot".into())
}

pub fn store_local(heap: &mut Heap, start: Value, slot: usize, value: Value) -> Result<(), String> {
    modify_scope(heap, start, |scope| match scope.locals.get_mut(slot) {
        Some(local) => {
            *local = Some(Ref::from(value));
            Ok(())
        }
        None => Err(String::from("invalid local slot")),
    })?
}

/// Store a reference popped from the operand stack into a local slot.
pub fn store_local_ref(
    heap: &mut Heap,
    start: Value,
    slot: usize,
    value: Ref,
) -> Result<(), String> {
    modify_scope(heap, start, |scope| match scope.locals.get_mut(slot) {
        Some(local) => {
            *local = Some(value);
            Ok(())
        }
        None => Err(String::from("invalid local slot")),
    })?
}

pub fn remove_local(heap: &mut Heap, start: Value, slot: usize) -> Result<Option<Value>, String> {
    let removed = modify_scope(heap, start, |scope| {
        scope
            .locals
            .get_mut(slot)
            .map(Option::take)
            .ok_or_else(|| String::from("invalid local slot"))
    })??;
    Ok(heap.value_optional(removed.as_ref()))
}

/// Bind `name` in this scope: in its local slot when the compiler assigned one, otherwise as a
/// dynamic name, charging the new entry.
pub fn insert(
    heap: &mut Heap,
    start: Value,
    name: String,
    value: Value,
    roots: &dyn Roots,
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
    modify_scope(heap, start, |scope| {
        if new_name {
            scope.order.push(name.clone());
        }
        scope.values.insert(name, Ref::from(value));
    })
}

/// Rebind `name` in the nearest scope, starting at `start`, that already binds it
/// (`nonlocal`). `start` is the scope enclosing the one that declared the name.
pub fn store_nonlocal(
    heap: &mut Heap,
    start: Value,
    name: &str,
    value: Value,
) -> Result<(), String> {
    let missing = || format!("no binding for nonlocal {name:?} found");
    let mut current: Value = start;
    loop {
        let candidate = scope(heap, current)?;
        let next = heap.value_optional(candidate.parent.as_ref());
        if let Some(slot) = candidate.local_names.iter().position(|local| local == name) {
            if candidate.locals[slot].is_some() {
                return store_local(heap, current, slot, value);
            }
        }
        if candidate.values.contains_key(name) {
            return modify_scope(heap, current, |scope| {
                scope.values.insert(name.to_string(), Ref::from(value));
            });
        }
        current = next.ok_or_else(missing)?;
    }
}

pub fn remove(heap: &mut Heap, start: Value, name: &str) -> Result<Option<Value>, String> {
    let removed = modify_scope(heap, start, |scope| {
        match scope.local_names.iter().position(|local| local == name) {
            Some(slot) => scope.locals[slot].take(),
            None => {
                scope.order.retain(|bound| bound != name);
                scope.values.remove(name)
            }
        }
    })?;
    Ok(heap.value_optional(removed.as_ref()))
}
