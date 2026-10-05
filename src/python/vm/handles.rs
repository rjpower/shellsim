//! The VM as a handle scope: every heap access the interpreter and native runtime perform goes
//! through these methods, which bind the resulting handles to the VM's scope lifetime `'s`.
//!
//! A `Vm<'s>` owns the handle-stack entries created since it opened. Nested work that allocates
//! many temporaries runs inside [`Vm::scope`] or [`Vm::nested`], a child `Vm<'c>` whose handles
//! are released when it ends; a `Value<'c>` cannot be used after that because the parent is
//! mutably borrowed for exactly `'c`. Values move out of a child scope either as a stored
//! reference in a root ([`Vm::store`]) or through [`Vm::nested_value`].
//!
//! Allocation can collect, so every allocating method reports the VM's roots to the heap through
//! [`VmRoots`]: the operand stack, frames, pending exceptions, globals, modules and type tables.

use std::collections::HashMap;

use crate::resources::Resources;

use super::super::attributes::{AttributeStore, InstanceAttributeSlot};
use super::super::heap::{Builder, Heap, Object, Ref, Roots, Value};
use super::super::object_model::{TypeId, TypeRegistry};
use super::super::scopes;
use super::super::symbols::SymbolId;
use super::super::{GlobalBindings, ReplState};
use super::{Vm, VmState};

/// Every stored reference the VM keeps outside the heap.
struct VmRoots<'a> {
    execution: &'a mut VmState,
    globals: &'a mut GlobalBindings,
    types: &'a mut TypeRegistry,
    modules: &'a mut HashMap<String, Ref>,
    sys_path: &'a mut Option<Ref>,
}

impl Roots for VmRoots<'_> {
    fn visit_refs(&mut self, visitor: &mut dyn FnMut(&mut Ref)) {
        self.execution.visit_refs(visitor);
        self.globals.visit_refs(visitor);
        self.types.visit_refs(visitor);
        for slot in self.modules.values_mut() {
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

    /// Open a child scope. Handles created through it are released when it is dropped; the
    /// parent's handles stay usable inside it.
    pub(super) fn scope(&mut self) -> Vm<'_> {
        let handle_base = self.state.heap.handle_count();
        let transient_base = self.execution.transient_memory;
        Vm {
            interp: &mut *self.interp,
            argv: self.argv,
            stdin: self.stdin,
            state: &mut *self.state,
            execution: &mut *self.execution,
            mode: self.mode,
            out: &mut *self.out,
            err: &mut *self.err,
            handle_base,
            transient_base,
        }
    }

    /// Release every handle and every byte of host scratch this scope created so far. Only for a
    /// loop that owns its scope and keeps nothing across iterations, such as the bytecode
    /// dispatch loop; dropping the scope does the same.
    pub(super) fn reset_scope(&mut self) {
        self.state.heap.truncate_handles(self.handle_base);
        self.release_transient_memory();
    }

    // ----- heap access ---------------------------------------------------------------------

    pub(super) fn heap(&self) -> &Heap {
        &self.state.heap
    }

    /// A handle for a stored reference.
    pub(super) fn handle(&self, slot: &Ref) -> Value<'s> {
        self.state.heap.handle(slot)
    }

    pub(super) fn handle_optional(&self, slot: Option<&Ref>) -> Option<Value<'s>> {
        self.state.heap.handle_optional(slot)
    }

    pub(super) fn handles<'a>(&self, slots: impl IntoIterator<Item = &'a Ref>) -> Vec<Value<'s>> {
        self.state.heap.handles(slots)
    }

    /// The stored form of a handle, for the VM's root containers. Place it in a root before the
    /// next allocation.
    pub(super) fn store(&self, value: Value<'_>) -> Ref {
        self.state.heap.store(value)
    }

    /// Run `f` with the heap, the VM's complete root set, and the resource meter.
    pub(super) fn with_heap<R>(
        &mut self,
        f: impl FnOnce(&mut Heap, &mut dyn Roots, &mut Resources) -> R,
    ) -> R {
        let ReplState {
            heap,
            globals,
            types,
            modules,
            sys_path,
            ..
        } = &mut *self.state;
        let mut roots = VmRoots {
            execution: &mut *self.execution,
            globals,
            types,
            modules,
            sys_path,
        };
        f(heap, &mut roots, &mut self.interp.resources)
    }

    /// Allocate an object whose payload holds no references.
    pub(super) fn alloc(&mut self, object: Object) -> Result<Value<'s>, String> {
        self.with_heap(|heap, roots, resources| heap.alloc(object, roots, resources))
    }

    /// Allocate an object with references as an instance of the class registered as `type_id`.
    pub(super) fn alloc_with_typed(
        &mut self,
        type_id: TypeId,
        build: impl FnOnce(&Builder<'_>) -> Object,
    ) -> Result<Value<'s>, String> {
        self.with_heap(|heap, roots, resources| {
            heap.alloc_with_typed(type_id, roots, resources, build)
        })
    }

    /// Allocate `object` as an instance of the user class registered as `type_id`.
    pub(super) fn allocate_typed(
        &mut self,
        type_id: TypeId,
        object: Object,
    ) -> Result<Value<'s>, String> {
        self.with_heap(|heap, roots, resources| heap.alloc_typed(type_id, object, roots, resources))
    }

    /// Allocate an object, turning handles into stored references with the builder.
    pub(super) fn alloc_with(
        &mut self,
        build: impl FnOnce(&Builder<'_>) -> Object,
    ) -> Result<Value<'s>, String> {
        self.with_heap(|heap, roots, resources| heap.alloc_with(roots, resources, build))
    }

    pub(super) fn get(&self, value: Value<'_>) -> Result<&Object, String> {
        self.state.heap.get(value)
    }

    /// Mutable payload access for fields that hold no references; use [`Self::modify`] to store
    /// one.
    pub(super) fn get_mut(&mut self, value: Value<'_>) -> Result<&mut Object, String> {
        self.state.heap.get_mut(value)
    }

    pub(super) fn modify<R>(
        &mut self,
        value: Value<'_>,
        f: impl FnOnce(&Builder<'_>, &mut Object) -> R,
    ) -> Result<R, String> {
        self.state.heap.modify(value, f)
    }

    pub(super) fn replace_payload(
        &mut self,
        value: Value<'_>,
        payload: Object,
    ) -> Result<(), String> {
        self.with_heap(|heap, roots, resources| {
            heap.replace_payload(value, payload, roots, resources)
        })
    }

    pub(super) fn reserve_object_growth(
        &mut self,
        value: Value<'_>,
        bytes: u64,
    ) -> Result<(), String> {
        self.with_heap(|heap, roots, resources| {
            heap.reserve_object_growth(value, bytes, roots, resources)
        })
    }

    pub(super) fn release_object_shrink(
        &mut self,
        value: Value<'_>,
        bytes: u64,
    ) -> Result<(), String> {
        self.state
            .heap
            .release_object_shrink(value, bytes, &mut self.interp.resources)
    }

    /// Whether two values are the same object or the same immediate (`is`).
    pub(super) fn identical(&self, left: Value<'_>, right: Value<'_>) -> bool {
        self.state.heap.identical(left, right)
    }

    /// A stable identity for `id()` and identity hashing; `None` for immediates.
    pub(super) fn identity(&self, value: Value<'_>) -> Result<Option<u32>, String> {
        self.state.heap.identity(value)
    }

    pub(super) fn object_type_id(&self, value: Value<'_>) -> Result<TypeId, String> {
        self.state.heap.type_id(value)
    }

    /// The type object registered for `id`.
    pub(super) fn type_value(&self, id: TypeId) -> Result<Value<'s>, String> {
        Ok(self.handle(self.state.types.value_ref(id)?))
    }

    // ----- operand stack -------------------------------------------------------------------

    pub(super) fn push(&mut self, value: Value<'_>) {
        self.execution.stack.push(&self.state.heap, value);
    }

    /// The value `depth` entries below the top of the stack (0 is the top).
    pub(super) fn peek(&self, depth: usize) -> Result<Value<'s>, String> {
        self.execution
            .stack
            .peek(&self.state.heap, depth)
            .ok_or_else(|| "stack underflow".into())
    }

    /// Pop `count` values, bottom-first.
    pub(super) fn pop_many(&mut self, count: usize) -> Result<Vec<Value<'s>>, String> {
        self.execution
            .stack
            .pop_many(&self.state.heap, count)
            .ok_or_else(|| "stack underflow".into())
    }

    // ----- symbols -------------------------------------------------------------------------

    pub(super) fn intern_symbol(&mut self, name: &str) -> Result<SymbolId, String> {
        self.state.symbols.intern(name, &mut self.interp.resources)
    }

    pub(super) fn symbol_id(&self, name: &str) -> Option<SymbolId> {
        self.state.symbols.id(name)
    }

    pub(super) fn symbol_name(&self, symbol: SymbolId) -> Option<&str> {
        self.state.symbols.name(symbol)
    }

    // ----- lexical scopes ------------------------------------------------------------------

    pub(super) fn alloc_scope(
        &mut self,
        parent: Option<Value<'_>>,
        uses_repl_globals: bool,
        local_names: std::sync::Arc<[String]>,
        locals: Vec<Option<Value<'_>>>,
        values: HashMap<String, Value<'_>>,
    ) -> Result<Value<'s>, String> {
        let layout = scopes::ScopeLayout {
            parent,
            uses_repl_globals,
            local_names,
        };
        self.with_heap(|heap, roots, resources| {
            scopes::alloc_scope(heap, layout, locals, values, roots, resources)
        })
    }

    pub(super) fn alloc_scope_named(
        &mut self,
        parent: Option<Value<'_>>,
        uses_repl_globals: bool,
        local_names: std::sync::Arc<[String]>,
        values: HashMap<String, Value<'_>>,
    ) -> Result<Value<'s>, String> {
        let layout = scopes::ScopeLayout {
            parent,
            uses_repl_globals,
            local_names,
        };
        self.with_heap(|heap, roots, resources| {
            scopes::alloc_scope_named(heap, layout, values, roots, resources)
        })
    }

    pub(super) fn scope_get(
        &self,
        scope: Value<'s>,
        name: &str,
    ) -> Result<Option<Value<'s>>, String> {
        scopes::get(&self.state.heap, scope, name)
    }

    pub(super) fn scope_insert(
        &mut self,
        scope: Value<'_>,
        name: String,
        value: Value<'_>,
    ) -> Result<(), String> {
        self.with_heap(|heap, roots, resources| {
            scopes::insert(heap, scope, name, value, roots, resources)
        })
    }

    // ----- instance attributes -------------------------------------------------------------

    /// Run `f` with everything an attribute write needs.
    pub(super) fn with_attributes<R>(&mut self, f: impl FnOnce(&mut AttributeStore<'_>) -> R) -> R {
        let ReplState {
            heap,
            globals,
            types,
            modules,
            sys_path,
            shapes,
            symbols,
            ..
        } = &mut *self.state;
        let mut roots = VmRoots {
            execution: &mut *self.execution,
            globals,
            types,
            modules,
            sys_path,
        };
        let mut store = AttributeStore {
            heap,
            shapes,
            symbols,
            roots: &mut roots,
            resources: &mut self.interp.resources,
        };
        f(&mut store)
    }

    pub(super) fn insert_attribute(
        &mut self,
        instance: Value<'_>,
        name: &str,
        value: Value<'_>,
    ) -> Result<(), String> {
        self.with_attributes(|store| store.insert(instance, name, value))
    }

    pub(super) fn insert_attribute_by_symbol(
        &mut self,
        instance: Value<'_>,
        symbol: SymbolId,
        value: Value<'_>,
    ) -> Result<(), String> {
        self.with_attributes(|store| store.insert_by_symbol(instance, symbol, value))
    }

    pub(super) fn remove_attribute_by_symbol(
        &mut self,
        instance: Value<'_>,
        symbol: SymbolId,
    ) -> Result<Option<Value<'s>>, String> {
        self.with_attributes(|store| store.remove_by_symbol(instance, symbol))
    }

    pub(super) fn attribute_by_symbol(
        &self,
        instance: Value<'_>,
        symbol: SymbolId,
    ) -> Result<Option<Value<'s>>, String> {
        self.state
            .shapes
            .attribute_by_symbol(&self.state.heap, instance, symbol)
    }

    pub(super) fn instance_attribute_slot_by_symbol(
        &self,
        instance: Value<'_>,
        symbol: SymbolId,
    ) -> Result<Option<InstanceAttributeSlot>, String> {
        self.state
            .shapes
            .slot_by_symbol(&self.state.heap, instance, symbol)
    }

    pub(super) fn instance_attribute_names(
        &self,
        instance: Value<'_>,
    ) -> Result<Vec<String>, String> {
        self.state
            .shapes
            .attribute_names(&self.state.heap, &self.state.symbols, instance)
    }
}
