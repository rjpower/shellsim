//! Reconstruct side instances with the current Store's handles and prepaid TLS.

use super::{reserve_store, Loaded, Record};
use crate::commands::wasm::{build_linker, fibers, Host};
use std::sync::Arc;
use wasmtime::{
    AsContextMut, Error, Extern, ExternType, Global, Mutability, Ref, StoreContextMut, Val, ValType,
};

fn library_export(
    mut store: StoreContextMut<'_, Host>,
    handle: u32,
    name: &str,
) -> Option<(Extern, u32, Option<u32>)> {
    let library = store
        .data()
        .threaded_dynamic
        .libraries
        .get(handle.checked_sub(1)? as usize)?;
    let instance = library.instance;
    let base = library.record.memory_base;
    let slot = library.record.function_slots.get(name).copied();
    instance
        .get_export(&mut store, name)
        .map(|value| (value, base, slot))
}

// A fixed queue visits each process module once, including diamond graphs.
// Local dependency scope never changes another module's global visibility.
fn lookup_dependencies(
    mut store: StoreContextMut<'_, Host>,
    name: &str,
    roots: &[u32],
) -> Result<Option<(Extern, u32, Option<u32>)>, Error> {
    let mut pending = [0u32; super::MAX_MODULES];
    let mut visited = [false; super::MAX_MODULES];
    let mut count = 0;
    for handle in roots {
        let index = handle
            .checked_sub(1)
            .filter(|index| (*index as usize) < super::MAX_MODULES)
            .ok_or_else(|| Error::msg("invalid threaded dependency handle"))?
            as usize;
        if !visited[index] {
            visited[index] = true;
            pending[count] = *handle;
            count += 1;
        }
    }
    let mut current = 0;
    while current < count {
        let handle = pending[current];
        current += 1;
        let library = store
            .data()
            .threaded_dynamic
            .libraries
            .get(handle as usize - 1)
            .ok_or_else(|| Error::msg("threaded dependency not reconstructed"))?;
        if !store.data().machine.get().resources.charge_cpu(
            (library.record.dependencies.len() as u64)
                .saturating_mul(16)
                .saturating_add(name.len() as u64 + 32),
        ) {
            return Err(super::super::exhausted());
        }
        for child in &library.record.dependencies {
            let index = child
                .checked_sub(1)
                .filter(|index| (*index as usize) < super::MAX_MODULES)
                .ok_or_else(|| Error::msg("invalid threaded dependency handle"))?
                as usize;
            if !visited[index] {
                visited[index] = true;
                pending[count] = *child;
                count += 1;
            }
        }
        if let Some(value) = library_export(store.as_context_mut(), handle, name) {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

fn lookup(
    mut store: StoreContextMut<'_, Host>,
    name: &str,
    dependencies: &[u32],
) -> Result<Option<(Extern, u32, Option<u32>)>, Error> {
    let main = store
        .data()
        .threaded_dynamic
        .main
        .ok_or_else(|| Error::msg("threaded main unavailable"))?;
    if let Some(value) = main.get_export(&mut store, name) {
        let slot = store
            .data()
            .threaded_dynamic
            .main_symbols
            .get(name)
            .copied();
        let thread = store.data().thread.as_ref().expect("thread host");
        let base = if thread.main_tls.contains(name) {
            let global = main
                .get_global(&mut store, "__tls_base")
                .ok_or_else(|| Error::msg("main TLS classification requires TLS base"))?;
            global
                .get(&mut store)
                .i32()
                .ok_or_else(|| Error::msg("main TLS base requires i32"))? as u32
        } else {
            0
        };
        return Ok(Some((value, base, slot)));
    }
    if super::super::dynamic::canonical_runtime_symbol(name) {
        return Ok(None);
    }
    let count = store.data().threaded_dynamic.libraries.len();
    if !store
        .data()
        .machine
        .get()
        .resources
        .charge_cpu((count as u64).saturating_mul(name.len() as u64 + 32))
    {
        return Err(super::super::exhausted());
    }
    for index in 0..count {
        if store.data().threaded_dynamic.libraries[index]
            .record
            .global
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            if let Some(value) = library_export(store.as_context_mut(), index as u32 + 1, name) {
                return Ok(Some(value));
            }
        }
    }
    lookup_dependencies(store, name, dependencies)
}

fn address(
    mut store: StoreContextMut<'_, Host>,
    value: Extern,
    base: u32,
    slot: Option<u32>,
) -> Result<u32, Error> {
    match value {
        Extern::Global(global) => {
            let value = global
                .get(&mut store)
                .i32()
                .ok_or_else(|| Error::msg("dynamic data requires an i32 address"))?
                as u32;
            if global.ty(&store).mutability() == Mutability::Var {
                Ok(value)
            } else {
                base.checked_add(value)
                    .ok_or_else(|| Error::msg("dynamic data address overflow"))
            }
        }
        Extern::Func(_) => {
            slot.ok_or_else(|| Error::msg("dynamic function has no process table slot"))
        }
        _ => Err(Error::msg("unsupported dynamic symbol kind")),
    }
}

pub(super) fn symbol(
    mut store: StoreContextMut<'_, Host>,
    handle: u32,
    name: &str,
) -> Result<u32, Error> {
    if !store
        .data()
        .machine
        .get()
        .resources
        .charge_cpu((name.len() as u64).saturating_add(super::MAX_MODULES as u64 * 32))
    {
        return Err(super::super::exhausted());
    }
    let value = if handle == 0 || handle == u32::MAX {
        lookup(store.as_context_mut(), name, &[])?
    } else {
        lookup_dependencies(store.as_context_mut(), name, &[handle])?
    }
    .ok_or_else(|| Error::msg(format!("missing dynamic symbol: {name}")))?;
    address(store, value.0, value.1, value.2)
}

/// Initialization thunks may yield fuel, but must not block or reenter the
/// loader. No allocator call occurs here, even when replay interrupts malloc.
pub(super) async fn install(
    mut store: StoreContextMut<'_, Host>,
    record: Arc<Record>,
    initialize_process: bool,
) -> Result<(), Error> {
    let main = store
        .data()
        .threaded_dynamic
        .main
        .ok_or_else(|| Error::msg("threaded main instance unavailable"))?;
    let thread = store.data().thread.as_ref().expect("thread host").clone();
    let table = main
        .get_table(&mut store, "__indirect_function_table")
        .ok_or_else(|| Error::msg("threaded table unavailable"))?;
    let table_end = record
        .function_slots
        .values()
        .copied()
        .max()
        .map_or(record.table_base + record.layout.table_size, |index| {
            index + 1
        });
    let growth = u64::from(table_end).saturating_sub(table.size(&store));
    if !store.data().machine.get().resources.charge_cpu(
        growth
            .saturating_mul(16)
            .saturating_add(record.layout.tls_size as u64)
            .saturating_add(record.store_cost),
    ) {
        return Err(super::super::exhausted());
    }
    reserve_store(&mut store, record.store_cost)?;
    if growth != 0 {
        table.grow(&mut store, growth, Ref::Func(None))?;
    }
    let mut linker = build_linker(store.engine());
    super::super::threads::register(&mut linker);
    super::super::posix_exec::register(&mut linker);
    super::super::posix_open::register(&mut linker);
    super::register(&mut linker);
    let mut own_got = Vec::new();
    let mut imported_memory = false;
    for import in record.module.imports() {
        let namespace = import.module();
        let name = import.name();
        let value = match (namespace, name) {
            (
                "wasi_snapshot_preview1"
                | "shellsim_posix_v1"
                | "shellsim_threads_v2"
                | "shellsim_dylink_v3"
                | "shellsim_ffi_v1",
                _,
            ) => continue,
            ("env", "memory") => {
                imported_memory = true;
                Extern::SharedMemory(thread.memory())
            }
            ("env", "__indirect_function_table") => Extern::Table(table),
            ("env", "__stack_pointer") => Extern::Global(
                main.get_global(&mut store, name)
                    .ok_or_else(|| Error::msg("threaded stack unavailable"))?,
            ),
            ("env", "__memory_base" | "__table_base") => {
                let ExternType::Global(ty) = import.ty() else {
                    return Err(Error::msg("invalid side base import"));
                };
                let base = if name == "__memory_base" {
                    record.memory_base
                } else {
                    record.table_base
                };
                Extern::Global(Global::new(&mut store, ty, Val::I32(base as i32))?)
            }
            ("GOT.mem" | "GOT.func", _) => {
                let ExternType::Global(ty) = import.ty() else {
                    return Err(Error::msg("invalid GOT import"));
                };
                if !ValType::eq(ty.content(), &ValType::I32) || ty.mutability() != Mutability::Var {
                    return Err(Error::msg("GOT imports require mutable i32 globals"));
                }
                if let Some(symbol) = lookup(store.as_context_mut(), name, &record.dependencies)? {
                    if (namespace == "GOT.mem" && !matches!(symbol.0, Extern::Global(_)))
                        || (namespace == "GOT.func" && !matches!(symbol.0, Extern::Func(_)))
                    {
                        return Err(Error::msg("dynamic GOT symbol kind mismatch"));
                    }
                    if let Extern::Global(global) = symbol.0 {
                        if global.ty(&store).mutability() == Mutability::Var {
                            Extern::Global(global)
                        } else {
                            let value =
                                address(store.as_context_mut(), symbol.0, symbol.1, symbol.2)?;
                            Extern::Global(Global::new(&mut store, ty, Val::I32(value as i32))?)
                        }
                    } else {
                        let value = address(store.as_context_mut(), symbol.0, symbol.1, symbol.2)?;
                        Extern::Global(Global::new(&mut store, ty, Val::I32(value as i32))?)
                    }
                } else if record
                    .layout
                    .weak_imports
                    .contains(&(namespace.to_owned(), name.to_owned()))
                {
                    Extern::Global(Global::new(&mut store, ty, Val::I32(0))?)
                } else {
                    let own = record
                        .module
                        .exports()
                        .find(|export| export.name() == name)
                        .ok_or_else(|| Error::msg(format!("missing dynamic GOT symbol: {name}")))?;
                    if super::super::dynamic::canonical_runtime_symbol(name)
                        || record.layout.has_start
                        || (namespace == "GOT.mem" && !matches!(own.ty(), ExternType::Global(_)))
                        || (namespace == "GOT.func" && !matches!(own.ty(), ExternType::Func(_)))
                    {
                        return Err(Error::msg("unsupported self GOT before side start"));
                    }
                    let global = Global::new(&mut store, ty, Val::I32(0))?;
                    own_got.push((name.to_owned(), global));
                    Extern::Global(global)
                }
            }
            ("env", _) => {
                lookup(store.as_context_mut(), name, &record.dependencies)?
                    .ok_or_else(|| Error::msg(format!("missing dynamic symbol: {name}")))?
                    .0
            }
            _ => {
                return Err(Error::msg(format!(
                    "unsupported threaded side import: {namespace}.{name}"
                )))
            }
        };
        linker.define(&store, namespace, name, value)?;
    }
    if !imported_memory {
        return Err(Error::msg("threaded side must import process memory"));
    }
    store.data_mut().threaded_dynamic.initializing = true;
    let result = async {
        let instance = {
            let _fiber = fibers::begin(store.as_context_mut())?;
            linker
                .instantiate_async(store.as_context_mut(), &record.module)
                .await?
        };
        // Constructors can call preemptible own functions through GOT slots.
        // Install these Store-local handles before any relocation/TLS/ctor call.
        for (name, index) in &record.function_slots {
            let function = instance
                .get_func(store.as_context_mut(), name)
                .ok_or_else(|| Error::msg("missing side function export"))?;
            table.set(
                store.as_context_mut(),
                u64::from(*index),
                Ref::Func(Some(function)),
            )?;
        }
        for (name, global) in &own_got {
            let value = if let Some(slot) = record.function_slots.get(name) {
                *slot
            } else {
                let offset = *record.layout.relative_data.get(name).ok_or_else(|| {
                    Error::msg("self GOT data requires a constant relative export")
                })?;
                let (base, size) = if record.layout.tls_exports.contains(name) {
                    (record.tls[thread.slot()], record.layout.tls_size)
                } else {
                    (record.memory_base, record.layout.memory_size)
                };
                if offset >= size {
                    return Err(Error::msg("self GOT data offset exceeds assigned segment"));
                }
                base.checked_add(offset)
                    .ok_or_else(|| Error::msg("self GOT address overflow"))?
            };
            global.set(store.as_context_mut(), Val::I32(value as i32))?;
        }
        // Preserve LLD's generated start order. Process memory initialization is
        // omitted on replay; global relocations use only this Store's globals.
        for name in ["__wasm_apply_global_relocs", "__wasm_init_memory"] {
            if name == "__wasm_init_memory" && !initialize_process {
                continue;
            }
            if let Some(function) = instance.get_func(store.as_context_mut(), name) {
                let _fiber = fibers::begin(store.as_context_mut())?;
                function
                    .typed::<(), ()>(&store)?
                    .call_async(store.as_context_mut(), ())
                    .await?;
            }
        }
        if record.layout.tls_size != 0 {
            let initialize =
                instance.get_typed_func::<u32, ()>(store.as_context_mut(), "__wasm_init_tls")?;
            let _fiber = fibers::begin(store.as_context_mut())?;
            initialize
                .call_async(store.as_context_mut(), record.tls[thread.slot()])
                .await?;
        }
        for (name, global) in own_got {
            let value = instance
                .get_export(store.as_context_mut(), &name)
                .ok_or_else(|| Error::msg("missing own GOT export"))?;
            let slot = record.function_slots.get(&name).copied();
            let value = address(store.as_context_mut(), value, record.memory_base, slot)?;
            global.set(store.as_context_mut(), Val::I32(value as i32))?;
        }
        if initialize_process {
            for name in ["__wasm_apply_data_relocs", "__wasm_call_ctors"] {
                if let Some(function) = instance.get_func(store.as_context_mut(), name) {
                    let _fiber = fibers::begin(store.as_context_mut())?;
                    function
                        .typed::<(), ()>(&store)?
                        .call_async(store.as_context_mut(), ())
                        .await?;
                }
            }
        }
        store
            .data_mut()
            .threaded_dynamic
            .libraries
            .push(Loaded { instance, record });
        Ok(())
    }
    .await;
    store.data_mut().threaded_dynamic.initializing = false;
    result
}
