//! Bind versioned threaded executable imports before entering guest startup.
//!
//! The main owns libc and its allocator. Typed forwarding imports let it be
//! instantiated before VFS dependencies, but remain unusable until every strong
//! binding has been checked. All handles and GOT cells are local to one Store.

use super::{layout, loading, replay, reserve_store};
use crate::commands::wasm::{fibers, Host};
use std::collections::BTreeMap;
use wasmtime::{
    AsContextMut, Error, Extern, ExternType, Func, FuncType, Global, Linker, Module, Mutability,
    StoreContextMut, Val, ValType,
};

#[derive(Default)]
pub(super) struct Bindings {
    functions: BTreeMap<String, Func>,
    got: Vec<(String, String, Global)>,
    pub(super) roots: Vec<u32>,
}

/// Install typed forwarding imports and mutable GOT cells. They expose no host
/// capability; their targets must come from the admitted VFS dependency graph.
pub(in crate::commands::wasm) fn define(
    mut store: StoreContextMut<'_, Host>,
    module: &Module,
    linker: &mut Linker<Host>,
) -> Result<(), Error> {
    let cost = module.imports().fold(0u64, |cost, import| {
        cost.saturating_add(import.name().len() as u64 * 2 + 256)
    });
    if !store.data().machine.get().resources.charge_cpu(cost) {
        return Err(super::super::exhausted());
    }
    reserve_store(&mut store, cost)?;
    for import in module.imports() {
        let namespace = import.module();
        let name = import.name();
        match (namespace, import.ty()) {
            ("env", ExternType::Func(ty)) => {
                let name = name.to_owned();
                let key = name.clone();
                let function = Func::new_async(
                    store.as_context_mut(),
                    ty,
                    move |mut caller, args, results| {
                        let name = name.clone();
                        Box::new(async move {
                            if !caller
                                .data()
                                .machine
                                .get()
                                .resources
                                .charge_cpu(name.len() as u64 + 64)
                            {
                                return Err(super::super::exhausted());
                            }
                            if !caller.data().threaded_dynamic.ready
                                && caller.data().thread.as_ref().expect("thread host").id() != 0
                            {
                                return Err(Error::msg(
                                    "executable dependency called before worker TLS readiness",
                                ));
                            }
                            let target = caller
                                .data()
                                .threaded_dynamic
                                .executable
                                .functions
                                .get(&name)
                                .copied()
                                .ok_or_else(|| {
                                    Error::msg(format!("unbound executable function: {name}"))
                                })?;
                            let _fiber = fibers::begin(&mut caller)?;
                            target.call_async(&mut caller, args, results).await
                        })
                    },
                );
                linker.define(&store, namespace, &key, function)?;
            }
            ("GOT.mem" | "GOT.func", ExternType::Global(ty)) => {
                if !ValType::eq(ty.content(), &ValType::I32) || ty.mutability() != Mutability::Var {
                    return Err(Error::msg("executable GOT requires mutable i32 globals"));
                }
                let global = Global::new(store.as_context_mut(), ty, Val::I32(0))?;
                linker.define(&store, namespace, name, global)?;
                store.data_mut().threaded_dynamic.executable.got.push((
                    namespace.to_owned(),
                    name.to_owned(),
                    global,
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

fn bind(mut store: StoreContextMut<'_, Host>, module: &Module) -> Result<(), Error> {
    let thread = store.data().thread.as_ref().expect("thread host").clone();
    for import in module.imports().filter(|import| import.module() == "env") {
        let ExternType::Func(expected) = import.ty() else {
            continue;
        };
        let name = import.name();
        let target = match replay::lookup(store.as_context_mut(), name, &[])? {
            Some((Extern::Func(function), _, _)) => {
                if !FuncType::eq(&expected, &function.ty(&store)) {
                    return Err(Error::msg(format!(
                        "executable function type mismatch: {name}"
                    )));
                }
                function
            }
            None if thread
                .executable
                .weak_imports
                .contains(&("env".to_owned(), name.to_owned())) =>
            {
                let name = name.to_owned();
                Func::new(store.as_context_mut(), expected, move |_, _, _| {
                    Err(Error::msg(format!(
                        "unresolved weak function called: {name}"
                    )))
                })
            }
            _ => return Err(Error::msg(format!("missing executable function: {name}"))),
        };
        store
            .data_mut()
            .threaded_dynamic
            .executable
            .functions
            .insert(name.to_owned(), target);
    }
    bind_got(store)
}

fn bind_got(mut store: StoreContextMut<'_, Host>) -> Result<(), Error> {
    let thread = store.data().thread.as_ref().expect("thread host").clone();
    let cost = store.data().threaded_dynamic.executable.got.iter().fold(
        0u64,
        |cost, (namespace, name, _)| {
            cost.saturating_add(namespace.len() as u64 + name.len() as u64 + 64)
        },
    );
    if !store.data().machine.get().resources.charge_cpu(cost) {
        return Err(super::super::exhausted());
    }
    let got = store.data().threaded_dynamic.executable.got.clone();
    for (namespace, name, global) in got {
        let value = match replay::lookup(store.as_context_mut(), &name, &[])? {
            Some(symbol) => {
                if (namespace == "GOT.mem" && !matches!(symbol.0, Extern::Global(_)))
                    || (namespace == "GOT.func" && !matches!(symbol.0, Extern::Func(_)))
                {
                    return Err(Error::msg(format!(
                        "executable GOT symbol kind mismatch: {name}"
                    )));
                }
                replay::address(store.as_context_mut(), symbol.0, symbol.1, symbol.2)?
            }
            None if thread
                .executable
                .weak_imports
                .contains(&(namespace.clone(), name.clone()))
                || (namespace == "GOT.func"
                    && thread
                        .executable
                        .weak_imports
                        .contains(&("env".to_owned(), name.clone()))) =>
            {
                0
            }
            None => return Err(Error::msg(format!("missing executable GOT symbol: {name}"))),
        };
        global.set(store.as_context_mut(), Val::I32(value as i32))?;
    }
    Ok(())
}

/// The main memory template precedes guest malloc. Its start and relocations
/// follow dependency binding, and side constructors precede main `_start`.
/// Workers never initialize shared data or rerun process constructors.
/// The pinned LLD memory initializer guards shared data with an atomic once flag:
/// calling it early for malloc and again through the original start is safe, as
/// is the worker's original start. Main data relocations need a separate replay
/// because the early initializer sees the initially unbound GOT cells.
pub(in crate::commands::wasm) async fn startup(
    mut store: StoreContextMut<'_, Host>,
    module: &Module,
    path: &str,
) -> Result<(), Error> {
    let thread = store.data().thread.as_ref().expect("thread host").clone();
    let main = store.data().threaded_dynamic.main.expect("main instance");
    const BOOTSTRAP: &str = "__wasm_call_runtime_ctors";
    if thread.executable.forwarded_exports.contains(BOOTSTRAP) {
        return Err(Error::msg(
            "executable runtime constructors must be main-owned",
        ));
    }
    let runtime_ctors = main
        .get_func(store.as_context_mut(), BOOTSTRAP)
        .ok_or_else(|| Error::msg("missing executable runtime constructors"))?
        .typed::<(), ()>(&store)
        .map_err(|_| Error::msg("executable runtime constructors require () -> ()"))?;
    store.data_mut().threaded_dynamic.initializing = true;
    let result = async {
        if thread.id() == 0 {
            if let Some(function) = main.get_func(store.as_context_mut(), "__wasm_init_memory") {
                let _fiber = fibers::begin(store.as_context_mut())?;
                function
                    .typed::<(), ()>(&store)?
                    .call_async(store.as_context_mut(), ())
                    .await?;
            }
        }
        let pending = loading::startup(store.as_context_mut(), path).await?;
        // replay::install temporarily changes this flag for its own thunks.
        store.data_mut().threaded_dynamic.initializing = true;
        replay::bind_graph(store.as_context_mut())?;
        bind(store.as_context_mut(), module)?;
        if thread.id() == 0 {
            if let Some(function) =
                main.get_func(store.as_context_mut(), "__wasm_apply_global_tls_relocs")
            {
                let _fiber = fibers::begin(store.as_context_mut())?;
                function
                    .typed::<(), ()>(&store)?
                    .call_async(store.as_context_mut(), ())
                    .await?;
            }
            refresh_main(store.as_context_mut()).await?;
            replay::prepare_graph(store.as_context_mut(), true).await?;
            refresh_main(store.as_context_mut()).await?;
        }
        if let Some(function) = main.get_func(store.as_context_mut(), layout::START_EXPORT) {
            let _fiber = fibers::begin(store.as_context_mut())?;
            function
                .typed::<(), ()>(&store)?
                .call_async(store.as_context_mut(), ())
                .await?;
        }
        if let Some((initialization, records)) = pending {
            if let Some(function) =
                main.get_func(store.as_context_mut(), "__wasm_apply_data_relocs")
            {
                let _fiber = fibers::begin(store.as_context_mut())?;
                function
                    .typed::<(), ()>(&store)?
                    .call_async(store.as_context_mut(), ())
                    .await?;
            }
            replay::relocate_data(store.as_context_mut()).await?;
            // The admitted linker splits implementation-priority constructors
            // into a process-once thunk. Ordinary main ctors delegate to the same
            // guard later, after side ctors, before main application ctors.
            {
                let _fiber = fibers::begin(store.as_context_mut())?;
                runtime_ctors.call_async(store.as_context_mut(), ()).await?;
            }
            let installed = store.data().threaded_dynamic.libraries.len();
            for index in installed - records.len()..installed {
                replay::initialize(store.as_context_mut(), index).await?;
            }
            {
                let process = thread.dynamic_process();
                let mut state = process.0.lock().expect("threaded dynamic registry");
                state.startup_roots = store.data().threaded_dynamic.executable.roots.clone();
                state.startup_modules = records.len();
            }
            initialization.publish(records)?;
            store.data_mut().threaded_dynamic.generation = thread.dynamic_process().generation().0;
        }
        Ok(())
    }
    .await;
    store.data_mut().threaded_dynamic.initializing = false;
    result
}

/// Complete early worker instances after libc installs the worker's main TLS.
pub(in crate::commands::wasm) async fn ready(
    mut store: StoreContextMut<'_, Host>,
) -> Result<(), Error> {
    store.data_mut().threaded_dynamic.initializing = true;
    let result = async {
        replay::prepare_graph(store.as_context_mut(), false).await?;
        refresh_main(store.as_context_mut()).await?;
        Ok(())
    }
    .await;
    store.data_mut().threaded_dynamic.initializing = false;
    result
}

async fn refresh_main(mut store: StoreContextMut<'_, Host>) -> Result<(), Error> {
    bind_got(store.as_context_mut())?;
    let main = store.data().threaded_dynamic.main.expect("main instance");
    if let Some(function) = main.get_func(store.as_context_mut(), "__wasm_apply_global_relocs") {
        let _fiber = fibers::begin(store.as_context_mut())?;
        function
            .typed::<(), ()>(&store)?
            .call_async(store.as_context_mut(), ())
            .await?;
    }
    Ok(())
}
