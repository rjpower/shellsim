//! Shared engine semantics with portable CPU code generation only for published companions.

use std::sync::OnceLock;
use wasmtime::{Collector, Config, Engine, ModuleVersionStrategy};

pub(super) const ASYNC_STACK_BYTES: usize = 2 * 1024 * 1024;

// Bump only when our Wasmtime patches break serialized metadata, generated-code/runtime
// calling conventions, or VM layouts. Wasmtime checks target, ISA and engine settings itself.
// Package metadata and compatible source, dependency or toolchain changes are not ABI revisions.
const PATCHED_RUNTIME_ABI: u32 = 1;

/// Identify load compatibility independently of the shellsim release or build machine.
/// Keep Wasmtime's own serialization version so an upstream ABI change still invalidates it.
pub(super) fn compilation_compatibility_id() -> &'static str {
    static ID: OnceLock<String> = OnceLock::new();
    ID.get_or_init(|| {
        format!(
            "shellsim-wasmtime-{}-abi{PATCHED_RUNTIME_ABI}",
            ModuleVersionStrategy::WasmtimeVersion.as_str()
        )
    })
}

pub(super) fn configure_command_engine(config: &mut Config) {
    config
        .consume_fuel(true)
        .async_stack_size(ASYNC_STACK_BYTES)
        .wasm_exceptions(true)
        .wasm_threads(true)
        .shared_memory(true)
        .wasm_gc(false)
        .collector(Collector::DeferredReferenceCounting);
    config
        .module_version(ModuleVersionStrategy::Custom(
            compilation_compatibility_id().into(),
        ))
        .expect("bounded patched Wasmtime identity");
}

pub(super) fn precompile_engine() -> &'static Engine {
    static ENGINE: OnceLock<Engine> = OnceLock::new();
    ENGINE.get_or_init(|| {
        let mut config = Config::default();
        configure_command_engine(&mut config);
        // Host-inferred ISA flags would tie a published artifact to the worker's CPU.
        // Runtime JIT keeps native features; Wasmtime admits baseline companions there.
        config
            .target(env!("SHELLSIM_WASMTIME_TARGET"))
            .expect("supported host target");
        Engine::new(&config).expect("valid precompile configuration")
    })
}
