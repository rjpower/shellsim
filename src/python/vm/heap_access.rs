//! Heap access for the interpreter and native runtime.
//!
//! Every value these methods return is pinned on the heap's pin stack. A `Vm` records the pin
//! count when it opens and releases everything pinned since then when it is reset or dropped:
//! the dispatch loop resets after each instruction, and nested work that makes many temporaries
//! runs inside [`Vm::scope`], a child `Vm` whose pins are released when it ends. A value that
//! must outlive the scope that made it moves out as a stored reference in a root ([`Vm::store`])
//! or is read again in the outer scope.
//!
//! Allocation can collect, so every allocating method reports the VM's roots to the heap through
//! [`VmRoots`]: the operand stack, frames, pending exceptions, builtins, modules and type tables.

use std::collections::HashMap;

use crate::resources::Resources;

use super::super::attributes::{AttributeStore, InstanceAttributeSlot};
use super::super::heap::Namespace;
use super::super::heap::{Heap, Object, Ref, Roots, Value};
use super::super::object_model::{TypeId, TypeRegistry};
use super::super::scopes;
use super::super::symbols::SymbolId;
use super::super::ReplState;
use super::{CodeTable, Flow, Vm, VmState};
use crate::python::error::PyResult;

/// Every stored reference the VM keeps outside the heap.
struct VmRoots<'a> {
    execution: &'a VmState,
    codes: &'a CodeTable,
    builtins: &'a Namespace,
    types: &'a TypeRegistry,
    modules: &'a HashMap<String, Ref>,
    sys_path: &'a Option<Ref>,
}

impl Roots for VmRoots<'_> {
    fn visit_refs(&self, visitor: &mut dyn FnMut(&Ref)) {
        self.execution.visit_refs(visitor);
        self.codes.visit_refs(visitor);
        for slot in self.builtins.refs() {
            visitor(slot);
        }
        self.types.visit_refs(visitor);
        for slot in self.modules.values() {
            visitor(slot);
        }
        self.sys_path.visit_refs(visitor);
    }
}

impl Drop for Vm<'_> {
    fn drop(&mut self) {
        self.reset_scope();
    }
}

impl<'s> Vm<'s> {
    // ----- scopes --------------------------------------------------------------------------

    /// Open a child scope. Values pinned through it are released when it is dropped; the
    /// parent's values stay pinned inside it.
    pub(super) fn scope(&mut self) -> Vm<'_> {
        let pin_base = self.state.heap.pin_count();
        Vm {
            interp: &mut *self.interp,
            argv: self.argv,
            stdin: self.stdin,
            state: &mut *self.state,
            execution: &mut *self.execution,
            mode: self.mode,
            out: &mut *self.out,
            err: &mut *self.err,
            pin_base,
        }
    }

    /// Release every pin and every byte of host scratch this scope created so far. Only for a
    /// loop that owns its scope and keeps nothing across iterations, such as the bytecode
    /// dispatch loop; dropping the scope does the same.
    pub(super) fn reset_scope(&mut self) {
        self.state
            .heap
            .truncate_pins(self.pin_base, &mut self.interp.resources);
    }

    /// [`Self::reset_scope`] when the scope holds a pin or scratch; the dispatch loop calls
    /// this once per instruction, so an instruction that works in place on the operand stack
    /// pays one comparison.
    #[inline(always)]
    pub(super) fn reset_scope_if_used(&mut self) {
        if self.state.heap.pin_count() != self.pin_base {
            self.reset_scope();
        }
    }

    // ----- heap access ---------------------------------------------------------------------

    pub(super) fn heap(&self) -> &Heap {
        &self.state.heap
    }

    /// The pinned value of a stored reference.
    #[inline(always)]
    pub(super) fn value(&self, slot: &Ref) -> Value {
        self.state.heap.value(slot)
    }

    pub(super) fn value_optional(&self, slot: Option<&Ref>) -> Option<Value> {
        self.state.heap.value_optional(slot)
    }

    pub(super) fn values<'a>(&self, slots: impl IntoIterator<Item = &'a Ref>) -> Vec<Value> {
        self.state.heap.values(slots)
    }

    /// The stored form of a value, for the VM's root containers. Place it in a root before the
    /// value's pin is released.
    #[inline(always)]
    pub(super) fn store(&self, value: Value) -> Ref {
        Ref::from(value)
    }

    /// Run `f` with the heap, the VM's complete root set, and the resource meter.
    pub(super) fn with_heap<R>(
        &mut self,
        f: impl FnOnce(&mut Heap, &dyn Roots, &mut Resources) -> R,
    ) -> R {
        let ReplState {
            heap,
            builtins,
            types,
            modules,
            sys_path,
            codes,
            ..
        } = &mut *self.state;
        let roots = VmRoots {
            execution: &*self.execution,
            codes,
            builtins,
            types,
            modules,
            sys_path,
        };
        f(heap, &roots, &mut self.interp.resources)
    }

    /// Allocate an object whose payload holds no references.
    pub(super) fn alloc(&mut self, object: Object) -> PyResult<Value> {
        self.with_heap(|heap, roots, resources| heap.alloc(object, roots, resources))
    }

    /// Allocate `object` as an instance of the user class registered as `type_id`.
    pub(super) fn allocate_typed(&mut self, type_id: TypeId, object: Object) -> PyResult<Value> {
        self.with_heap(|heap, roots, resources| heap.alloc_typed(type_id, object, roots, resources))
    }

    pub(super) fn get(&self, value: Value) -> PyResult<&Object> {
        self.state.heap.get(value)
    }

    /// Mutable payload access. The heap remembers a mutated old object, so references may be
    /// stored through it.
    pub(super) fn get_mut(&mut self, value: Value) -> PyResult<&mut Object> {
        self.state.heap.get_mut(value)
    }

    pub(super) fn modify<R>(
        &mut self,
        value: Value,
        f: impl FnOnce(&mut Object) -> R,
    ) -> PyResult<R> {
        self.state.heap.modify(value, f)
    }

    pub(super) fn replace_payload(&mut self, value: Value, payload: Object) -> PyResult<()> {
        self.with_heap(|heap, roots, resources| {
            heap.replace_payload(value, payload, roots, resources)
        })
    }

    pub(super) fn reserve_object_growth(&mut self, value: Value, bytes: u64) -> PyResult<()> {
        self.with_heap(|heap, roots, resources| {
            heap.reserve_object_growth(value, bytes, roots, resources)
        })
    }

    pub(super) fn release_object_shrink(&mut self, value: Value, bytes: u64) -> PyResult<()> {
        self.state
            .heap
            .release_object_shrink(value, bytes, &mut self.interp.resources)
    }

    /// Whether two values are the same object or the same immediate (`is`).
    pub(super) fn identical(&self, left: Value, right: Value) -> bool {
        left.is(right)
    }

    /// A stable identity for `id()` and identity hashing; `None` for immediates.
    pub(super) fn identity(&self, value: Value) -> PyResult<Option<u32>> {
        self.state.heap.identity(value)
    }

    pub(super) fn object_type_id(&self, value: Value) -> PyResult<TypeId> {
        self.state.heap.type_id(value)
    }

    /// The type object registered for `id`.
    pub(super) fn type_value(&self, id: TypeId) -> PyResult<Value> {
        Ok(self.value(self.state.types.value_ref(id)?))
    }

    // ----- operand stack -------------------------------------------------------------------

    #[inline(always)]
    pub(super) fn push(&mut self, value: Value) {
        self.execution.stack.push(value);
    }

    /// Pop the top stored reference without pinning it, for moving a value between roots
    /// or discarding it.
    #[inline(always)]
    pub(super) fn pop_ref(&mut self) -> PyResult<Ref> {
        if self.frame_stack_len() == 0 {
            return Err("invalid bytecode stack effect".into());
        }
        Ok(self
            .execution
            .stack
            .pop_ref()
            .expect("non-empty frame stack was checked"))
    }

    /// The stored reference `depth` entries below the top of the stack (0 is the top), for
    /// instructions that work in place.
    #[inline(always)]
    pub(super) fn peek_ref(&self, depth: usize) -> PyResult<&Ref> {
        if self.frame_stack_len() <= depth {
            return Err("stack underflow".into());
        }
        self.execution
            .stack
            .top(depth)
            .ok_or_else(|| "stack underflow".into())
    }

    /// Leave a call's result on the operand stack, where the caller expects it.
    pub(super) fn produce(&mut self, value: Value) -> Flow {
        self.push(value);
        Flow::Next
    }

    /// The value `depth` entries below the top of the stack (0 is the top).
    #[inline(always)]
    pub(super) fn peek(&self, depth: usize) -> PyResult<Value> {
        self.execution
            .stack
            .peek(&self.state.heap, depth)
            .ok_or_else(|| "stack underflow".into())
    }

    /// Pop `count` values, bottom-first.
    pub(super) fn pop_many(&mut self, count: usize) -> PyResult<Vec<Value>> {
        self.execution
            .stack
            .pop_many(&self.state.heap, count)
            .ok_or_else(|| "stack underflow".into())
    }

    // ----- symbols -------------------------------------------------------------------------

    pub(super) fn intern_symbol(&mut self, name: &str) -> PyResult<SymbolId> {
        self.state.symbols.intern(name, &mut self.interp.resources)
    }

    pub(super) fn symbol_id(&self, name: &str) -> Option<SymbolId> {
        self.state.symbols.id(name)
    }

    /// The name of a symbol this interpreter issued, such as one compiled code carries.
    pub(super) fn symbol_name(&self, symbol: SymbolId) -> &str {
        self.state.symbols.issued(symbol)
    }

    // ----- lexical scopes ------------------------------------------------------------------

    /// Allocate a scope enclosed by `parent` whose slots hold `locals` and which binds `names`
    /// dynamically in that order.
    pub(super) fn alloc_scope(
        &mut self,
        parent: Option<Value>,
        local_names: std::sync::Arc<[SymbolId]>,
        locals: Vec<Option<Value>>,
        names: Vec<(SymbolId, Value)>,
    ) -> PyResult<Value> {
        let layout = scopes::ScopeLayout {
            parent,
            local_names,
        };
        self.with_heap(|heap, roots, resources| {
            scopes::alloc_scope(heap, layout, locals, names, roots, resources)
        })
    }

    pub(super) fn scope_get(&self, scope: Value, symbol: SymbolId) -> PyResult<Option<Value>> {
        scopes::get(&self.state.heap, scope, symbol)
    }

    /// [`Self::scope_get`] for a name native code holds as a string. A name the interpreter
    /// never interned is bound in no scope.
    pub(super) fn scope_get_name(&self, scope: Value, name: &str) -> PyResult<Option<Value>> {
        match self.symbol_id(name) {
            Some(symbol) => self.scope_get(scope, symbol),
            None => Ok(None),
        }
    }

    pub(super) fn scope_insert(
        &mut self,
        scope: Value,
        symbol: SymbolId,
        value: Value,
    ) -> PyResult<()> {
        self.with_heap(|heap, roots, resources| {
            scopes::insert(heap, scope, symbol, value, roots, resources)
        })
    }

    pub(super) fn scope_insert_name(
        &mut self,
        scope: Value,
        name: &str,
        value: Value,
    ) -> PyResult<()> {
        let symbol = self.intern_symbol(name)?;
        self.scope_insert(scope, symbol, value)
    }

    // ----- instance attributes -------------------------------------------------------------

    /// Run `f` with everything an attribute write needs.
    pub(super) fn with_attributes<R>(&mut self, f: impl FnOnce(&mut AttributeStore<'_>) -> R) -> R {
        let ReplState {
            heap,
            builtins,
            types,
            modules,
            sys_path,
            shapes,
            symbols,
            codes,
            ..
        } = &mut *self.state;
        let roots = VmRoots {
            execution: &*self.execution,
            codes,
            builtins,
            types,
            modules,
            sys_path,
        };
        let mut store = AttributeStore {
            heap,
            shapes,
            symbols,
            roots: &roots,
            resources: &mut self.interp.resources,
        };
        f(&mut store)
    }

    pub(super) fn insert_attribute(
        &mut self,
        instance: Value,
        name: &str,
        value: Value,
    ) -> PyResult<()> {
        self.with_attributes(|store| store.insert(instance, name, value))
    }

    pub(super) fn insert_attribute_by_symbol(
        &mut self,
        instance: Value,
        symbol: SymbolId,
        value: Value,
    ) -> PyResult<()> {
        self.with_attributes(|store| store.insert_by_symbol(instance, symbol, value))
    }

    pub(super) fn remove_attribute_by_symbol(
        &mut self,
        instance: Value,
        symbol: SymbolId,
    ) -> PyResult<Option<Value>> {
        self.with_attributes(|store| store.remove_by_symbol(instance, symbol))
    }

    pub(super) fn attribute_by_symbol(
        &self,
        instance: Value,
        symbol: SymbolId,
    ) -> PyResult<Option<Value>> {
        self.state
            .shapes
            .attribute_by_symbol(&self.state.heap, instance, symbol)
    }

    pub(super) fn instance_attribute_slot_by_symbol(
        &self,
        instance: Value,
        symbol: SymbolId,
    ) -> PyResult<Option<InstanceAttributeSlot>> {
        self.state
            .shapes
            .slot_by_symbol(&self.state.heap, instance, symbol)
    }

    pub(super) fn instance_attribute_names(&self, instance: Value) -> PyResult<Vec<String>> {
        self.state
            .shapes
            .attribute_names(&self.state.heap, &self.state.symbols, instance)
    }
}
