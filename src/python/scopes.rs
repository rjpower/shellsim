//! Name resolution through lexical scopes.
//!
//! A scope ([`ScopeObject`]) is an ordinary heap object; this module implements the lookups and
//! stores the interpreter performs on one: local slots assigned by the compiler, dynamically
//! bound names, and the walk through enclosing scopes. Names are symbols. Every chain ends at a
//! module's scope, whose dynamic names are that module's globals, so the main script is no
//! different from an imported module. Stores go through [`Heap::modify`], so a store into an
//! old scope is remembered for the collector.

use std::sync::Arc;

use crate::resources::Resources;

use super::heap::{Heap, Namespace, Object, Ref, Roots, ScopeObject, Value, DYNAMIC_NAME_BYTES};
use super::symbols::SymbolId;
use crate::python::error::{PyError, PyResult};

/// The fixed part of a new scope: where name lookup continues and which names have slots.
pub struct ScopeLayout {
    pub parent: Option<Value>,
    pub local_names: Arc<[SymbolId]>,
}

/// Allocate a scope whose compiler-assigned local slots are already populated, with `names`
/// bound dynamically in that order.
pub fn alloc_scope(
    heap: &mut Heap,
    layout: ScopeLayout,
    locals: Vec<Option<Value>>,
    names: Vec<(SymbolId, Value)>,
    roots: &dyn Roots,
    resources: &mut Resources,
) -> PyResult<Value> {
    if locals.len() != layout.local_names.len() {
        return Err("local slot metadata mismatch".into());
    }
    let object = Object::Scope(Box::new(ScopeObject {
        parent: Ref::optional(layout.parent),
        local_names: layout.local_names,
        locals: locals.into_iter().map(Ref::optional).collect(),
        names: names
            .into_iter()
            .map(|(symbol, value)| (symbol, Ref::from(value)))
            .collect(),
    }));
    heap.alloc(object, roots, resources)
}

fn scope(heap: &Heap, scope: Value) -> PyResult<&ScopeObject> {
    match heap.get(scope)? {
        Object::Scope(scope) => Ok(scope),
        _ => Err("invalid scope reference".into()),
    }
}

fn modify_scope<R>(
    heap: &mut Heap,
    scope: Value,
    f: impl FnOnce(&mut ScopeObject) -> R,
) -> PyResult<R> {
    heap.modify(scope, |object| match object {
        Object::Scope(scope) => Ok(f(scope)),
        _ => Err(PyError::from("invalid scope reference")),
    })?
}

fn slot(scope: &ScopeObject, symbol: SymbolId) -> Option<usize> {
    scope.local_names.iter().position(|local| *local == symbol)
}

/// The binding of `symbol` in this one scope, slot or dynamic name.
fn bound(scope: &ScopeObject, symbol: SymbolId) -> Option<&Ref> {
    match slot(scope, symbol) {
        Some(index) => scope.locals[index].as_ref(),
        None => scope.names.get(symbol),
    }
}

/// Look a name up through this scope and its ancestors.
pub fn get(heap: &Heap, start: Value, symbol: SymbolId) -> PyResult<Option<Value>> {
    let mut current = start;
    loop {
        let object = scope(heap, current)?;
        if let Some(value) = bound(object, symbol) {
            return Ok(Some(heap.value(value)));
        }
        match &object.parent {
            Some(parent) => current = heap.value(parent),
            None => return Ok(None),
        }
    }
}

/// The stored binding of a module global, read through the stored reference a frame keeps to
/// its module's scope so that the hot path pins nothing.
#[inline]
pub fn global<'h>(heap: &'h Heap, globals: &Ref, symbol: SymbolId) -> PyResult<Option<&'h Ref>> {
    match heap.stored(globals)? {
        Object::Scope(scope) => Ok(scope.names.get(symbol)),
        _ => Err("invalid scope reference".into()),
    }
}

/// Rebind a module global that is already bound, in place; returns `false` when the module
/// does not bind `symbol` yet, so the caller charges the new entry through [`insert`].
#[inline]
pub fn replace_global(
    heap: &mut Heap,
    globals: &Ref,
    symbol: SymbolId,
    value: Value,
) -> PyResult<bool> {
    heap.modify_stored(globals, |object| match object {
        Object::Scope(scope) => Ok(match scope.names.get_mut(symbol) {
            Some(slot) => {
                *slot = Ref::from(value);
                true
            }
            None => false,
        }),
        _ => Err(PyError::from("invalid scope reference")),
    })?
}

pub fn parent(heap: &Heap, start: Value) -> PyResult<Option<Value>> {
    Ok(heap.value_optional(scope(heap, start)?.parent.as_ref()))
}

/// Every name bound directly in this scope: bound slots in slot order, then dynamic names in
/// the order they were first bound, which a class body keeps for its namespace.
pub fn entries(heap: &Heap, start: Value) -> PyResult<Vec<(SymbolId, Value)>> {
    let object = scope(heap, start)?;
    let slots = object
        .local_names
        .iter()
        .zip(&object.locals)
        .filter_map(|(symbol, slot)| Some((*symbol, heap.value(slot.as_ref()?))));
    let names = object
        .names
        .iter()
        .map(|(symbol, value)| (symbol, heap.value(value)));
    Ok(slots.chain(names).collect())
}

/// The stored reference in one local slot, for pushing onto the operand stack directly.
pub fn local_ref(heap: &Heap, start: Value, slot: usize) -> PyResult<Option<&Ref>> {
    scope(heap, start)?
        .locals
        .get(slot)
        .map(Option::as_ref)
        .ok_or_else(|| "invalid local slot".into())
}

pub fn store_local(heap: &mut Heap, start: Value, slot: usize, value: Value) -> PyResult<()> {
    store_local_ref(heap, start, slot, Ref::from(value))
}

/// Store a reference popped from the operand stack into a local slot.
pub fn store_local_ref(heap: &mut Heap, start: Value, slot: usize, value: Ref) -> PyResult<()> {
    modify_scope(heap, start, |scope| match scope.locals.get_mut(slot) {
        Some(local) => {
            *local = Some(value);
            Ok(())
        }
        None => Err(PyError::from("invalid local slot")),
    })?
}

pub fn remove_local(heap: &mut Heap, start: Value, slot: usize) -> PyResult<Option<Value>> {
    let removed = modify_scope(heap, start, |scope| {
        scope
            .locals
            .get_mut(slot)
            .map(Option::take)
            .ok_or_else(|| PyError::from("invalid local slot"))
    })??;
    Ok(heap.value_optional(removed.as_ref()))
}

/// Bind `symbol` in this scope: in its local slot when the compiler assigned one, otherwise as
/// a dynamic name, charging the new entry.
pub fn insert(
    heap: &mut Heap,
    start: Value,
    symbol: SymbolId,
    value: Value,
    roots: &dyn Roots,
    resources: &mut Resources,
) -> PyResult<()> {
    let object = scope(heap, start)?;
    if let Some(slot) = slot(object, symbol) {
        return store_local(heap, start, slot, value);
    }
    if !object.names.contains(symbol) {
        heap.reserve_object_growth(start, DYNAMIC_NAME_BYTES, roots, resources)?;
    }
    modify_scope(heap, start, |scope| {
        scope.names.insert(symbol, Ref::from(value));
    })
}

/// Rebind `symbol` in the nearest scope, starting at `start`, that already binds it
/// (`nonlocal`). `start` is the scope enclosing the one that declared the name. Returns whether
/// any scope bound it.
pub fn store_nonlocal(
    heap: &mut Heap,
    start: Value,
    symbol: SymbolId,
    value: Value,
) -> PyResult<bool> {
    let mut current = start;
    loop {
        let candidate = scope(heap, current)?;
        if bound(candidate, symbol).is_some() {
            modify_scope(heap, current, |scope| match slot(scope, symbol) {
                Some(index) => scope.locals[index] = Some(Ref::from(value)),
                None => {
                    scope.names.insert(symbol, Ref::from(value));
                }
            })?;
            return Ok(true);
        }
        match heap.value_optional(candidate.parent.as_ref()) {
            Some(parent) => current = parent,
            None => return Ok(false),
        }
    }
}

pub fn remove(heap: &mut Heap, start: Value, symbol: SymbolId) -> PyResult<Option<Value>> {
    let removed = modify_scope(heap, start, |scope| match slot(scope, symbol) {
        Some(index) => scope.locals[index].take(),
        None => scope.names.remove(symbol),
    })?;
    Ok(heap.value_optional(removed.as_ref()))
}

/// The scope's dynamic names, for callers that read a module's globals in order.
pub fn names(heap: &Heap, start: Value) -> PyResult<&Namespace> {
    Ok(&scope(heap, start)?.names)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::python::error::PyErrorKind;
    use crate::python::symbols::Symbols;
    use crate::resources::Limits;

    #[test]
    fn a_new_name_is_charged_before_the_scope_binds_it() {
        let mut resources = Resources::new(Limits {
            memory: 1 << 20,
            ..Limits::unlimited()
        });
        let mut symbols = Symbols::default();
        let (bound, unbound) = (
            symbols.intern("bound", &mut resources).unwrap(),
            symbols.intern("unbound", &mut resources).unwrap(),
        );
        let mut heap = Heap::default();
        let layout = ScopeLayout {
            parent: None,
            local_names: Arc::from([]),
        };
        let scope = alloc_scope(
            &mut heap,
            layout,
            Vec::new(),
            vec![(bound, Value::Int(1))],
            &(),
            &mut resources,
        )
        .unwrap();
        assert!(resources.reserve_memory(resources.memory_remaining() - (DYNAMIC_NAME_BYTES - 1)));

        // Rebinding a name the scope already has needs no memory; a new name does.
        insert(&mut heap, scope, bound, Value::Int(2), &(), &mut resources).unwrap();
        let error = insert(
            &mut heap,
            scope,
            unbound,
            Value::Int(3),
            &(),
            &mut resources,
        )
        .unwrap_err();
        assert_eq!(error.kind(), Some(&PyErrorKind::Resource));
        let entries = entries(&heap, scope)
            .unwrap()
            .into_iter()
            .map(|(symbol, value)| (symbol, value.as_int()))
            .collect::<Vec<_>>();
        assert_eq!(entries, [(bound, Some(2))]);
    }
}
