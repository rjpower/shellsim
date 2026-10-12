//! Host-only native companions for exact Wasm sources.
//!
//! Companions are ordinary Wasmtime serialization. Only a trusted host caller may associate
//! them with source bytes and deserialize them. Guests use ordinary admission, metering,
//! instantiation, and the bounded module cache; no guest path loads native code.

use wasmtime::{Error, Module};

use super::{
    command_engine, command_module_cache, compilation_cache, executable_admission,
    executable_compilation_source, retain_executable_admission, threads, validate_imports,
    ExecutableAdmission, MAX_CACHED_MODULE_BYTES, MAX_WASM_BYTES,
};

/// Select ordinary module compilation or main-executable admission and startup rewriting.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WasmCompilationKind {
    /// A side module or other module, without main-executable profile requirements.
    Module,
    /// A main executable, including its scheduler ABI and deferred-start requirements.
    Executable,
}

/// Whether a trusted companion entered the runtime cache or the caller should retain JIT use.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrecompiledWasmOutcome {
    /// The companion is compatible and cached under its exact compilation source.
    Registered,
    /// The format, runtime identity, target, settings, or cache size bounds are incompatible.
    Incompatible,
}

/// Compile binary Wasm on the host using the command engine and return a native companion.
///
/// Executables undergo the same profile and import admission as guest execution, including
/// deferred-start rewriting. This does not execute the module or grant host capabilities.
/// The returned bytes contain native code and must remain under trusted host control.
pub fn precompile_wasm(source: &[u8], kind: WasmCompilationKind) -> Result<Vec<u8>, Error> {
    check_source(source)?;
    let admission = match kind {
        WasmCompilationKind::Module => {
            threads::reject_raw_waits(source)?;
            None
        }
        WasmCompilationKind::Executable => Some(executable_admission(source)?),
    };
    let rewritten = admission
        .as_ref()
        .map(|admission| executable_compilation_source(source, admission.profile.as_ref()))
        .transpose()?
        .flatten();
    let module = Module::new(command_engine(), rewritten.as_deref().unwrap_or(source))?;
    if let Some(admission) = &admission {
        admit_imports(&module, admission)?;
    }
    module.serialize()
}

/// Identify the patched runtime used to produce native companions.
///
/// CPU and configuration incompatibility is handled by Wasmtime when loading the companion.
pub fn precompiled_wasm_build_id() -> &'static str {
    compilation_cache::BUILD_ID
}

/// Register a trusted host companion for the exact source used by subsequent guest compilation.
///
/// Incompatibility leaves the ordinary JIT path available. Invalid source or failed executable
/// admission returns an error. Registration does not bypass guest resource charges or instantiate
/// a module. Cache eviction can later require recompilation.
///
/// # Safety
///
/// `companion` must be trusted, untouched Wasmtime serialization compiled from exactly `source`
/// with `kind`, including executable startup rewriting. [`precompile_wasm`] supplies this output;
/// other runtime versions are allowed. The caller guarantees this association. Neither raw source
/// admission nor compatibility checks verify native code. Never pass guest VFS bytes or untrusted
/// native artifacts.
#[allow(unsafe_code)]
pub unsafe fn register_trusted_precompiled_wasm(
    source: &[u8],
    companion: &[u8],
    kind: WasmCompilationKind,
) -> Result<PrecompiledWasmOutcome, Error> {
    check_source(source)?;
    if companion.len() > MAX_CACHED_MODULE_BYTES {
        return Ok(PrecompiledWasmOutcome::Incompatible);
    }
    let admission = match kind {
        WasmCompilationKind::Module => {
            threads::reject_raw_waits(source)?;
            None
        }
        WasmCompilationKind::Executable => Some(executable_admission(source)?),
    };
    let rewritten = admission
        .as_ref()
        .map(|admission| executable_compilation_source(source, admission.profile.as_ref()))
        .transpose()?
        .flatten();
    let compilation_source = rewritten.as_deref().unwrap_or(source);
    if command_module_cache()
        .lock()
        .expect("module cache lock")
        .get(compilation_source)
        .is_some()
    {
        return Ok(PrecompiledWasmOutcome::Registered);
    }
    // SAFETY: the host caller guarantees untouched Wasmtime output compiled for this source/kind.
    // The command engine checks compatibility, including the patched fuel-resume runtime identity.
    let module = match unsafe { Module::deserialize(command_engine(), companion) } {
        Ok(module) => module,
        Err(_) => return Ok(PrecompiledWasmOutcome::Incompatible),
    };
    if let Some(admission) = &admission {
        admit_imports(&module, admission)?;
    }
    let image = module.image_range();
    if compilation_source
        .len()
        .saturating_add((image.end as usize).saturating_sub(image.start as usize))
        > MAX_CACHED_MODULE_BYTES
    {
        return Ok(PrecompiledWasmOutcome::Incompatible);
    }
    command_module_cache()
        .lock()
        .expect("module cache lock")
        .insert(compilation_source, module);
    if let Some(admission) = admission {
        retain_executable_admission(source, admission);
    }
    Ok(PrecompiledWasmOutcome::Registered)
}

fn check_source(source: &[u8]) -> Result<(), Error> {
    if source.len() > MAX_WASM_BYTES {
        return Err(Error::msg("wasm module exceeds size limit"));
    }
    if !source.starts_with(b"\0asm") {
        return Err(Error::msg("not a binary Wasm module"));
    }
    Ok(())
}

fn admit_imports(module: &Module, admission: &ExecutableAdmission) -> Result<(), Error> {
    validate_imports(
        module,
        admission.profile.is_some(),
        admission
            .profile
            .as_ref()
            .is_some_and(|profile| profile.dynamic),
        admission
            .profile
            .as_ref()
            .is_some_and(|profile| !profile.executable.needed.is_empty()),
    )
}

#[cfg(test)]
mod tests {
    // A private test can supply a genuine foreign-engine artifact without tampering with native
    // bytes, and observe a cached export directly before any JIT path could hide a bad load.
    use super::*;
    use wasmtime::{Config, Engine, Instance, ModuleVersionStrategy, Store};

    #[test]
    #[allow(unsafe_code)]
    fn module_round_trip_and_foreign_runtime_fallback() {
        let source =
            wat::parse_str("(module (func (export \"answer\") (result i32) (i32.const 739)))")
                .unwrap();
        let bytes = precompile_wasm(&source, WasmCompilationKind::Module).unwrap();
        // SAFETY: bytes came directly from our host compiler.
        assert_eq!(
            unsafe {
                register_trusted_precompiled_wasm(&source, &bytes, WasmCompilationKind::Module)
            }
            .unwrap(),
            PrecompiledWasmOutcome::Registered
        );
        let module = command_module_cache().lock().unwrap().get(&source).unwrap();
        let mut store = Store::new(command_engine(), ());
        store.set_fuel(1000).unwrap();
        let instance = Instance::new(&mut store, &module, &[]).unwrap();
        assert_eq!(
            instance
                .get_typed_func::<(), i32>(&mut store, "answer")
                .unwrap()
                .call(&mut store, ())
                .unwrap(),
            739
        );

        let mut config = Config::new();
        super::super::configure_command_engine(&mut config);
        config
            .module_version(ModuleVersionStrategy::Custom("foreign-runtime".into()))
            .unwrap();
        let engine = Engine::new(&config).unwrap();
        let source =
            wat::parse_str("(module (func (export \"_start\") (i32.const 17) drop))").unwrap();
        let module = Module::new(&engine, &source).unwrap();
        let bytes = module.serialize().unwrap();
        // SAFETY: the native payload is genuine output from a differently configured Wasmtime.
        assert_eq!(
            unsafe {
                register_trusted_precompiled_wasm(&source, &bytes, WasmCompilationKind::Executable)
            }
            .unwrap(),
            PrecompiledWasmOutcome::Incompatible
        );
        let mut environment = crate::Environment::new();
        environment
            .vfs
            .write("/", "/foreign", &source, 0o755)
            .unwrap();
        assert_eq!(environment.run_script_capture("/foreign").0.exit_status, 0);
    }
}
