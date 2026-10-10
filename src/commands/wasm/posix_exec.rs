//! Same-PID exec over copied, bounded guest arguments.
//!
//! Resolution happens before the old image is stopped. A successful host call unwinds
//! Wasmtime; the process owner drops every old Store before applying the replacement.

use super::*;
use std::collections::BTreeMap;

const MAX_STRINGS: usize = 131_072;
const DECODE_MEMORY: u64 = 3 * MAX_STRINGS as u64 + 32_768;

/// Validated replacement plus its independently retained decode reservation.
#[derive(Debug)]
pub(super) struct ExecSpec {
    executable: String,
    argv: Vec<String>,
    environment: BTreeMap<String, String>,
    pub(super) reserved_bytes: u64,
}

/// Successful exec is control transfer, never a return to the old guest.
#[derive(Debug)]
pub(super) struct GuestExec(pub(super) ExecSpec);

impl std::fmt::Display for GuestExec {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("virtual process image replacement")
    }
}

impl std::error::Error for GuestExec {}

fn string(
    memory: &GuestMemory,
    caller: &Caller<'_, Host>,
    pointer: u32,
    remaining: &mut usize,
) -> Result<String, i32> {
    if pointer == 0 || pointer as usize >= memory.data_size(caller) {
        return Err(ERRNO_FAULT);
    }
    let bytes = memory
        .c_string(caller, pointer as usize, *remaining)
        .map_err(|_| ERRNO_INVAL)?;
    *remaining = remaining.checked_sub(bytes.len() + 1).ok_or(ERRNO_INVAL)?;
    match bytes {
        std::borrow::Cow::Borrowed(bytes) => std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| ERRNO_INVAL),
        std::borrow::Cow::Owned(bytes) => String::from_utf8(bytes).map_err(|_| ERRNO_INVAL),
    }
}

fn vector(
    memory: &GuestMemory,
    caller: &Caller<'_, Host>,
    pointer: u32,
    count: u32,
    remaining: &mut usize,
) -> Result<Vec<String>, i32> {
    if pointer == 0 && count == 0 {
        return Ok(Vec::new());
    }
    if count > 256 || pointer == 0 {
        return Err(ERRNO_INVAL);
    }
    let entries = memory
        .bytes(caller, pointer as usize, (count as usize + 1) * 4)
        .map_err(|_| ERRNO_FAULT)?;
    if entries[count as usize * 4..] != [0; 4] {
        return Err(ERRNO_INVAL);
    }
    entries[..count as usize * 4]
        .chunks_exact(4)
        .map(|entry| {
            let pointer = u32::from_le_bytes(entry.try_into().expect("fixed-width pointer"));
            string(memory, caller, pointer, remaining)
        })
        .collect()
}

fn environment(values: Vec<String>) -> Result<BTreeMap<String, String>, i32> {
    let mut environment = BTreeMap::new();
    for value in values {
        let (name, value) = value.split_once('=').ok_or(ERRNO_INVAL)?;
        if name.is_empty() {
            return Err(ERRNO_INVAL);
        }
        environment.insert(name.to_owned(), value.to_owned());
    }
    Ok(environment)
}

fn resolve(
    interp: &mut Interp,
    requested: String,
    argv: Vec<String>,
    environment: BTreeMap<String, String>,
    search: bool,
) -> Result<ExecSpec, SyscallError> {
    if requested.is_empty() || requested.len() > 4096 || argv.is_empty() {
        return Err(SyscallError::InvalidArgument);
    }
    let requested = if search || requested.contains('/') {
        requested
    } else {
        format!("./{requested}")
    };
    let path_value = environment
        .get("PATH")
        .map(String::as_str)
        .unwrap_or("/bin:/usr/bin");
    let work = (path_value.len() as u64).saturating_add(
        (path_value.split(':').count() as u64)
            .saturating_mul((requested.len() + interp.process.cwd.len() + 16) as u64),
    );
    if !interp.resources.charge_cpu(work) {
        return Err(SyscallError::ResourceExhausted);
    }
    let executable = match crate::commands::util::resolve_executable_in(
        &interp.vfs,
        &interp.process.cwd,
        Some(path_value),
        &requested,
    ) {
        crate::commands::util::ExecutableLookup::Found(path) => path,
        crate::commands::util::ExecutableLookup::NotFound => {
            return Err(SyscallError::File(VfsError::NotFound(requested)));
        }
        _ => return Err(SyscallError::Permission),
    };
    let (executable, argv) = interp.resolve_spawn_image(executable, argv)?;
    if crate::commands::is_wasm_executable(&interp.vfs, &executable) {
        let size = interp.vfs.file_len("/", &executable)?;
        if size > MAX_WASM_BYTES {
            return Err(SyscallError::ExecutableFormat);
        }
        // Main startup rewriting temporarily retains a second executable image.
        let scratch = (size as u64).saturating_mul(66).saturating_add(4096);
        if !interp
            .resources
            .charge_cpu((size as u64).saturating_mul(10))
            || !interp.resources.reserve_memory(scratch)
        {
            return Err(SyscallError::ResourceExhausted);
        }
        let validation = (|| {
            let bytes = interp.vfs.read_limited("/", &executable, MAX_WASM_BYTES)?;
            threads::reject_raw_waits(&bytes).map_err(|_| SyscallError::ExecutableFormat)?;
            let profile = threads::profile(&bytes).map_err(|_| SyscallError::ExecutableFormat)?;
            let module = compiled_executable_module(&bytes, profile.as_ref())
                .map_err(|_| SyscallError::ExecutableFormat)?;
            validate_imports(
                &module,
                profile.is_some(),
                profile.as_ref().is_some_and(|profile| profile.dynamic),
                profile
                    .as_ref()
                    .is_some_and(|profile| !profile.executable.needed.is_empty()),
            )
            .map_err(|_| SyscallError::ExecutableFormat)
        })();
        interp.resources.release_memory(scratch);
        validation?;
    }
    Ok(ExecSpec {
        executable,
        argv,
        environment,
        reserved_bytes: DECODE_MEMORY,
    })
}

fn exec(
    mut caller: Caller<'_, Host>,
    (path, argv, argc, env, envc, flags): (u32, u32, u32, u32, u32, u32),
) -> Result<i32, Error> {
    if flags & !1 != 0 {
        return Ok(ERRNO_INVAL);
    }
    if matches!(caller.data().stdio, Stdio::Buffered { .. }) {
        return Ok(ERRNO_NOTSUP);
    }
    let Some(memory) = memory(&mut caller) else {
        return Ok(ERRNO_FAULT);
    };
    {
        let mut machine = caller.data_mut().machine.get();
        if !machine.resources.charge_cpu(2 * MAX_STRINGS as u64)
            || !machine.resources.reserve_memory(DECODE_MEMORY)
        {
            return Err(exhausted());
        }
    }
    caller
        .data()
        .retained
        .fetch_add(DECODE_MEMORY, Ordering::Relaxed);
    let decoded = (|| {
        let mut remaining = MAX_STRINGS;
        let requested = string(&memory, &caller, path, &mut remaining)?;
        let argv = vector(&memory, &caller, argv, argc, &mut remaining)?;
        let environment = environment(vector(&memory, &caller, env, envc, &mut remaining)?)?;
        resolve(
            &mut caller.data_mut().machine.get(),
            requested,
            argv,
            environment,
            flags == 1,
        )
        .map_err(|error| match error {
            SyscallError::ResourceExhausted => -1,
            error => syscall_errno(&error),
        })
    })();
    match decoded {
        Ok(mut spec) => {
            // Image argv, exported bindings and labels can retain separate copies. Shebang
            // expansion adds bounded interpreter arguments beyond the input string budget.
            let retained = spec
                .argv
                .iter()
                .map(|value| value.len() as u64 + 32)
                .chain(
                    spec.environment
                        .iter()
                        .map(|(name, value)| (name.len() + value.len()) as u64 + 128),
                )
                .fold(spec.executable.len() as u64, u64::saturating_add)
                .saturating_mul(3)
                .saturating_add(65_536)
                .max(DECODE_MEMORY);
            let extra = retained - DECODE_MEMORY;
            if !caller
                .data_mut()
                .machine
                .get()
                .resources
                .reserve_memory(extra)
            {
                caller
                    .data()
                    .retained
                    .fetch_sub(DECODE_MEMORY, Ordering::Relaxed);
                caller
                    .data_mut()
                    .machine
                    .get()
                    .resources
                    .release_memory(DECODE_MEMORY);
                return Err(exhausted());
            }
            caller.data().retained.fetch_add(extra, Ordering::Relaxed);
            spec.reserved_bytes = retained;
            Err(Error::new(GuestExec(spec)))
        }
        Err(errno) => {
            caller
                .data()
                .retained
                .fetch_sub(DECODE_MEMORY, Ordering::Relaxed);
            caller
                .data_mut()
                .machine
                .get()
                .resources
                .release_memory(DECODE_MEMORY);
            if errno == -1 || caller.data().machine.get().resources.is_stopped() {
                Err(exhausted())
            } else {
                Ok(errno)
            }
        }
    }
}

/// Consume the replacement after the old guest Stores and reservations are dropped.
pub(super) fn apply(spec: ExecSpec, interp: &mut Interp) -> Result<(), SyscallError> {
    let reserved = spec.reserved_bytes;
    let result = (|| {
        let pid = interp.process.pid;
        let supplied_pwd = spec.environment.get("PWD").cloned();
        interp
            .configure_process(pid, None, Some(spec.environment))
            .map_err(SyscallError::Process)?;
        match supplied_pwd {
            Some(value) => {
                interp.process.vars.insert("PWD".into(), value);
            }
            None => {
                interp.process.exported.remove("PWD");
            }
        }
        let label = crate::process::command_label(&spec.argv);
        interp
            .load_argv_program_from(pid, spec.argv, &spec.executable, true)
            .map_err(SyscallError::Process)?;
        interp.processes.set_command(pid, label);
        Ok(())
    })();
    if result.is_ok() {
        let prior = std::mem::replace(&mut interp.process.exec_allocation_bytes, reserved);
        interp.resources.release_memory(prior);
    } else {
        interp.resources.release_memory(reserved);
    }
    result
}

pub(super) fn register(linker: &mut Linker<Host>) {
    linker
        .func_wrap(
            "shellsim_posix_v1",
            "process_exec",
            |caller: Caller<'_, Host>,
             path: u32,
             argv: u32,
             argc: u32,
             env: u32,
             envc: u32,
             flags: u32| { exec(caller, (path, argv, argc, env, envc, flags)) },
        )
        .expect("unique exec import");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_preflight_keeps_environment_descriptors_and_program() {
        let mut interp = Interp::default();
        interp
            .process
            .vars
            .insert("SENTINEL".into(), "original".into());
        interp.process.exported.insert("SENTINEL".into());
        let descriptions = interp.process.fds.iter().collect::<Vec<_>>();
        interp
            .vfs
            .write("/", "/malformed", b"\0asm\x01", 0o755)
            .unwrap();
        interp
            .vfs
            .write("/", "/denied", b"\0asm\x01", 0o644)
            .unwrap();
        for name in ["/missing", "/malformed", "/denied"] {
            assert!(resolve(
                &mut interp,
                name.into(),
                vec!["custom argv zero".into()],
                BTreeMap::from([("SENTINEL".into(), "replacement".into())]),
                false,
            )
            .is_err());
            assert_eq!(interp.process.vars["SENTINEL"], "original");
            assert_eq!(interp.process.fds.iter().collect::<Vec<_>>(), descriptions);
        }
    }

    #[test]
    fn path_search_uses_supplied_environment_and_preserves_argv_zero() {
        let mut interp = Interp::default();
        interp.vfs.mkdir_all("/", "/alternate").unwrap();
        let bytes = wat::parse_str("(module (func (export \"_start\")))").unwrap();
        interp
            .vfs
            .write("/", "/alternate/image", &bytes, 0o755)
            .unwrap();
        let spec = resolve(
            &mut interp,
            "image".into(),
            vec!["independent argv zero".into(), "argument".into()],
            BTreeMap::from([("PATH".into(), "/alternate".into())]),
            true,
        )
        .unwrap();
        assert_eq!(spec.executable, "/alternate/image");
        assert_eq!(spec.argv, ["independent argv zero", "argument"]);
        assert!(resolve(
            &mut interp,
            "image".into(),
            vec!["image".into()],
            BTreeMap::new(),
            false
        )
        .is_err());
    }
}
