//! Host-only native companions for exact Wasm sources.
//!
//! Companions are ordinary Wasmtime serialization. Only a trusted host caller may associate
//! them with source bytes and deserialize them. Guests use ordinary admission, metering,
//! instantiation, and the bounded module cache; no guest path loads native code.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use wasmtime::{Error, Module};

use super::{
    command_engine, command_module_cache, compilation_cache, executable_admission,
    executable_compilation_source, retain_executable_admission, threads, validate_imports,
    CachedCompilation, ExecutableAdmission, MAX_CACHED_MODULE_BYTES, MAX_WASM_BYTES,
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
    /// The trusted companion is retained; compatibility is checked on its first use.
    Registered,
    /// The companion exceeds cache bounds and ordinary JIT remains available.
    Incompatible,
}

#[derive(Clone)]
pub(super) enum Companion {
    Bytes(Arc<[u8]>),
    File(PathBuf),
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
    let module = Module::new(
        super::engine::precompile_engine(),
        rewritten.as_deref().unwrap_or(source),
    )?;
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

/// Check retained compilation before a host adapter reads a large companion file.
/// A missing executable admission profile is a miss; this lookup never admits new source.
fn precompiled_wasm_registered(source: &[u8], kind: WasmCompilationKind) -> Result<bool, Error> {
    check_source(source)?;
    let admission = match kind {
        WasmCompilationKind::Module => None,
        WasmCompilationKind::Executable => {
            let cache = command_module_cache().lock().expect("module cache lock");
            let Some(admission) = cache
                .entries
                .iter()
                .find(|entry| super::same_module_source(entry.source.as_slice(), source))
                .and_then(|entry| entry.admission.clone())
            else {
                return Ok(false);
            };
            Some(admission)
        }
    };
    let rewritten = admission
        .as_ref()
        .map(|admission| executable_compilation_source(source, admission.profile.as_ref()))
        .transpose()?
        .flatten();
    let compilation_source = rewritten.as_deref().unwrap_or(source);
    Ok(command_module_cache()
        .lock()
        .expect("module cache lock")
        .entries
        .iter()
        .any(|entry| super::same_module_source(entry.source.as_slice(), compilation_source)))
}

/// Register a trusted host companion for the exact source used by subsequent guest compilation.
///
/// Deserialization and compatibility checks happen on the first compilation cache miss, outside
/// the cache lock. An incompatible companion falls back to JIT. Registration checks source
/// admission but does not instantiate a module or bypass guest resource charges.
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
    register(source, kind, companion.len(), || {
        Companion::Bytes(companion.into())
    })
}

/// Retain a trusted companion path and map it only when its module is first used.
///
/// # Safety
/// The host must control the file and keep its contents unchanged for the lifetime of all
/// cached modules and live instances using it, which can span the host process lifetime.
/// Its bytes must have the exact trusted source/kind association of the bytes-based API.
#[allow(unsafe_code)]
pub unsafe fn register_trusted_precompiled_wasm_file(
    source: &[u8],
    companion: &Path,
    kind: WasmCompilationKind,
) -> Result<PrecompiledWasmOutcome, Error> {
    if precompiled_wasm_registered(source, kind)? {
        return Ok(PrecompiledWasmOutcome::Registered);
    }
    let size = usize::try_from(std::fs::metadata(companion)?.len())
        .map_err(|_| Error::msg("companion exceeds cache bounds"))?;
    register(source, kind, size, || {
        Companion::File(companion.to_path_buf())
    })
}

fn register(
    source: &[u8],
    kind: WasmCompilationKind,
    companion_size: usize,
    companion: impl FnOnce() -> Companion,
) -> Result<PrecompiledWasmOutcome, Error> {
    check_source(source)?;
    if companion_size > MAX_CACHED_MODULE_BYTES {
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
    let bytes = compilation_source.len().saturating_add(companion_size);
    if bytes > MAX_CACHED_MODULE_BYTES {
        return Ok(PrecompiledWasmOutcome::Incompatible);
    }
    let mut cache = command_module_cache().lock().expect("module cache lock");
    if !cache
        .entries
        .iter()
        .any(|entry| super::same_module_source(entry.source.as_slice(), compilation_source))
    {
        cache.insert_entry(
            &Arc::new(compilation_source.to_vec()),
            CachedCompilation::Precompiled(companion()),
            bytes,
        );
    }
    drop(cache);
    if let Some(admission) = admission {
        retain_executable_admission(source, admission);
    }
    Ok(PrecompiledWasmOutcome::Registered)
}

#[allow(unsafe_code)]
pub(super) fn load(companion: Option<&Companion>) -> Option<Module> {
    // SAFETY: only the unsafe trusted-host registration API can create pending companions.
    // File registration requires immutable contents throughout the mapped module's lifetime.
    // Guest resource charges and source admission precede this lookup. Incompatibility falls back.
    companion.and_then(|companion| match companion {
        Companion::Bytes(bytes) => unsafe { Module::deserialize(command_engine(), bytes).ok() },
        Companion::File(path) => unsafe { Module::deserialize_file(command_engine(), path).ok() },
    })
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
        let path = std::env::temp_dir().join(format!(
            "shellsim-context-companion-{}.cwasm",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();
        // SAFETY: the file contains only our host compiler's output and remains unchanged.
        assert_eq!(
            unsafe {
                register_trusted_precompiled_wasm_file(&source, &path, WasmCompilationKind::Module)
            }
            .unwrap(),
            PrecompiledWasmOutcome::Registered
        );
        let module = super::super::compiled_command_module(&Arc::new(source.clone())).unwrap();
        assert_eq!(module.serialize().unwrap(), bytes);
        std::fs::remove_file(&path).unwrap();
        // SAFETY: this module is already loaded; repeated setup must not reopen its file.
        unsafe {
            register_trusted_precompiled_wasm_file(&source, &path, WasmCompilationKind::Module)
        }
        .unwrap();
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
            PrecompiledWasmOutcome::Registered
        );
        let mut environment = crate::Environment::new();
        environment
            .vfs
            .write("/", "/foreign", &source, 0o755)
            .unwrap();
        assert_eq!(environment.run_script_capture("/foreign").0.exit_status, 0);
    }
}
