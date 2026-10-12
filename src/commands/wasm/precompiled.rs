//! Host-only native companions for exact Wasm sources.
//!
//! Companions carry the original source and compilation kind, then untouched Wasmtime output.
//! Only an explicitly trusted host caller may deserialize them. Guests use ordinary admission,
//! metering, instantiation, and the bounded module cache; no guest path loads native code.

use wasmtime::{Error, Module};

use super::{
    command_engine, command_module_cache, compiled_command_module, compiled_executable_module,
    executable_admission, executable_compilation_source, retain_executable_admission, threads,
    validate_imports, ExecutableAdmission, MAX_CACHED_MODULE_BYTES, MAX_WASM_BYTES,
};

const MAGIC: &[u8; 8] = b"SSCWASM1";
const HEADER_BYTES: usize = MAGIC.len() + 1 + 8;

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
    /// The companion belongs to different source bytes or a different compilation kind.
    SourceMismatch,
}

/// Compile binary Wasm on the host using the command engine and return a native companion.
///
/// Executables undergo the same profile and import admission as guest execution, including
/// deferred-start rewriting. This does not execute the module or grant host capabilities.
/// The returned bytes contain native code and must remain under trusted host control.
pub fn precompile_wasm(source: &[u8], kind: WasmCompilationKind) -> Result<Vec<u8>, Error> {
    check_source(source)?;
    let module = match kind {
        WasmCompilationKind::Module => compiled_command_module(source)?,
        WasmCompilationKind::Executable => {
            let admission = executable_admission(source)?;
            let module = compiled_executable_module(source, admission.profile.as_ref())?;
            admit_imports(&module, &admission)?;
            retain_executable_admission(source, admission);
            module
        }
    };
    companion(source, kind, &module.serialize()?)
}

/// Register a trusted host companion for the exact source used by subsequent guest compilation.
///
/// Incompatibility or a source/kind mismatch leaves the ordinary JIT path available. Invalid
/// source or failed executable admission returns an error. Registration does not bypass guest
/// resource charges or instantiate a module. Cache eviction can later require recompilation.
///
/// # Safety
///
/// `companion` must come from a trusted producer using [`precompile_wasm`]'s envelope and contain
/// untouched Wasmtime serialization for its recorded source and compilation kind. Output from
/// other runtime versions is allowed. Never pass guest VFS bytes or untrusted native artifacts.
/// Source matching and compatibility checks do not authenticate code or make arbitrary input safe.
#[allow(unsafe_code)]
pub unsafe fn register_trusted_precompiled_wasm(
    source: &[u8],
    companion: &[u8],
    kind: WasmCompilationKind,
) -> Result<PrecompiledWasmOutcome, Error> {
    check_source(source)?;
    let Some(header) = companion.get(..HEADER_BYTES) else {
        return Ok(PrecompiledWasmOutcome::Incompatible);
    };
    if !header.starts_with(MAGIC) || companion.len() > MAX_CACHED_MODULE_BYTES {
        return Ok(PrecompiledWasmOutcome::Incompatible);
    }
    let length = u64::from_le_bytes(header[MAGIC.len() + 1..].try_into().expect("header length"));
    if header[MAGIC.len()] != kind_byte(kind) || length != source.len() as u64 {
        return Ok(PrecompiledWasmOutcome::SourceMismatch);
    }
    let source_end = HEADER_BYTES + source.len();
    if companion.get(HEADER_BYTES..source_end) != Some(source) {
        return Ok(PrecompiledWasmOutcome::SourceMismatch);
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
    // SAFETY: the caller guarantees the payload is untouched, trusted Wasmtime output.
    // The envelope selects its exact source/kind, and the command engine checks compatibility,
    // including our patched runtime identity and fuel-resume behavior.
    let module = match unsafe { Module::deserialize(command_engine(), &companion[source_end..]) } {
        Ok(module) => module,
        Err(_) => return Ok(PrecompiledWasmOutcome::Incompatible),
    };
    if let Some(admission) = &admission {
        admit_imports(&module, admission)?;
    }
    let compilation_source = rewritten.as_deref().unwrap_or(source);
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

fn kind_byte(kind: WasmCompilationKind) -> u8 {
    match kind {
        WasmCompilationKind::Module => 0,
        WasmCompilationKind::Executable => 1,
    }
}

fn companion(
    source: &[u8],
    kind: WasmCompilationKind,
    serialized: &[u8],
) -> Result<Vec<u8>, Error> {
    let size = HEADER_BYTES
        .saturating_add(source.len())
        .saturating_add(serialized.len());
    if size > MAX_CACHED_MODULE_BYTES {
        return Err(Error::msg("precompiled companion exceeds cache size limit"));
    }
    let mut bytes = Vec::with_capacity(size);
    bytes.extend_from_slice(MAGIC);
    bytes.push(kind_byte(kind));
    bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
    bytes.extend_from_slice(source);
    bytes.extend_from_slice(serialized);
    Ok(bytes)
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
        // Remove only this source's JIT entry so a no-op registration cannot pass the round trip.
        {
            let mut cache = command_module_cache().lock().unwrap();
            let index = cache
                .entries
                .iter()
                .position(|entry| entry.source == source)
                .unwrap();
            let entry = cache.entries.remove(index).unwrap();
            cache.bytes -= entry.bytes;
        }
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
        let bytes = companion(
            &source,
            WasmCompilationKind::Executable,
            &module.serialize().unwrap(),
        )
        .unwrap();
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
