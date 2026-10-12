//! Shared engine semantics with portable CPU code generation only for published companions.

use std::sync::OnceLock;
use wasmtime::{Collector, Config, Engine, ModuleVersionStrategy};

use super::compilation_cache;

pub(super) const ASYNC_STACK_BYTES: usize = 2 * 1024 * 1024;

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
        .module_version(ModuleVersionStrategy::Custom(format!(
            "shellsim-wasmtime-49-{}",
            compilation_cache::BUILD_ID
        )))
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
