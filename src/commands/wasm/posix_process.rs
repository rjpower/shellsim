//! Versioned POSIX process imports over the virtual kernel.
//!
//! Guest pointers, counts and strings are copied and bounded before launch. Blocking waits
//! suspend the Wasmtime stack on the cooperative scheduler; no host process or FD is exposed.

use super::*;
use crate::process::Signal;
use crate::syscalls::{ProcessSpawn, SignalTarget, SpawnFdAction};

pub(super) const MAX_ARGUMENTS: u32 = 4096;
pub(super) const MAX_ENVIRONMENT: u32 = 256;
pub(super) const MAX_STRINGS: usize = 131_072;
const O_CLOEXEC: u32 = 0x0008_0000;
const ERRNO_CHILD: i32 = 12;
const ERRNO_SRCH: i32 = 71;

/// Reserve the decode peak before copying any guest input. String payloads,
/// argv String slots and shared pointer copies coexist with environment tree
/// nodes and their key/value copies. Fixed slack covers paths and FD actions.
/// Oversized counts are clamped for charging, then rejected by the decoder.
pub(super) fn decode_memory(argc: u32, envc: u32) -> u64 {
    (3 * MAX_STRINGS as u64 + 32_768)
        .saturating_add(u64::from(argc.min(MAX_ARGUMENTS)).saturating_mul(32))
        .saturating_add(u64::from(envc.min(MAX_ENVIRONMENT)).saturating_mul(160))
}

/// Prepay bounded byte scans and per-entry allocation/UTF-8/map work.
pub(super) fn decode_cost(argc: u32, envc: u32) -> u64 {
    (2 * MAX_STRINGS as u64)
        .saturating_add(u64::from(argc.min(MAX_ARGUMENTS)).saturating_mul(64))
        .saturating_add(u64::from(envc.min(MAX_ENVIRONMENT)).saturating_mul(128))
}

/// Borrow ordinary ranges and copy bounded shared ranges without exposing a heap slice.
pub(super) trait ReadGuest {
    fn range(&self, pointer: u32, length: usize) -> Result<std::borrow::Cow<'_, [u8]>, i32>;
    fn c_string(&self, pointer: u32, limit: usize) -> Result<std::borrow::Cow<'_, [u8]>, i32>;
}

impl<T: AsRef<[u8]>> ReadGuest for T {
    fn range(&self, pointer: u32, length: usize) -> Result<std::borrow::Cow<'_, [u8]>, i32> {
        let start = pointer as usize;
        let end = start.checked_add(length).ok_or(ERRNO_FAULT)?;
        self.as_ref()
            .get(start..end)
            .map(std::borrow::Cow::Borrowed)
            .ok_or(ERRNO_FAULT)
    }

    fn c_string(&self, pointer: u32, limit: usize) -> Result<std::borrow::Cow<'_, [u8]>, i32> {
        let available = self.as_ref().get(pointer as usize..).ok_or(ERRNO_FAULT)?;
        let bytes = &available[..available.len().min(limit)];
        let end = bytes
            .iter()
            .position(|byte| *byte == 0)
            .ok_or(ERRNO_INVAL)?;
        Ok(std::borrow::Cow::Borrowed(&bytes[..end]))
    }
}

pub(super) struct GuestReader<'a, 'b> {
    pub(super) memory: &'a GuestMemory,
    pub(super) caller: &'a Caller<'b, Host>,
}

impl ReadGuest for GuestReader<'_, '_> {
    fn range(&self, pointer: u32, length: usize) -> Result<std::borrow::Cow<'_, [u8]>, i32> {
        self.memory
            .bytes(self.caller, pointer as usize, length)
            .map_err(|_| ERRNO_FAULT)
    }

    fn c_string(&self, pointer: u32, limit: usize) -> Result<std::borrow::Cow<'_, [u8]>, i32> {
        if pointer as usize > self.memory.data_size(self.caller) {
            return Err(ERRNO_FAULT);
        }
        self.memory
            .c_string(self.caller, pointer as usize, limit)
            .map_err(|_| ERRNO_INVAL)
    }
}

fn string(bytes: &impl ReadGuest, pointer: u32, remaining: &mut usize) -> Result<String, i32> {
    let value = bytes.c_string(pointer, *remaining)?;
    *remaining -= value.len() + 1;
    match value {
        std::borrow::Cow::Borrowed(bytes) => std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| ERRNO_INVAL),
        std::borrow::Cow::Owned(bytes) => String::from_utf8(bytes).map_err(|_| ERRNO_INVAL),
    }
}

pub(super) fn path(
    bytes: &impl ReadGuest,
    pointer: u32,
    remaining: &mut usize,
) -> Result<String, i32> {
    let value = string(bytes, pointer, remaining)?;
    if value.len() > 4096 {
        return Err(ERRNO_INVAL);
    }
    Ok(value)
}

pub(super) fn vector(
    bytes: &impl ReadGuest,
    pointer: u32,
    count: u32,
    maximum: u32,
    remaining: &mut usize,
) -> Result<Vec<String>, i32> {
    if pointer == 0 && count == 0 {
        return Ok(Vec::new());
    }
    if count > maximum {
        return Err(ERRNO_INVAL);
    }
    let entries = bytes.range(pointer, (count as usize + 1) * 4)?;
    if entries[count as usize * 4..] != [0, 0, 0, 0] {
        return Err(ERRNO_INVAL);
    }
    // Keep String-slot capacity exact so the prepaid per-entry bound holds.
    let mut values = Vec::with_capacity(count as usize);
    for entry in entries[..count as usize * 4].chunks_exact(4) {
        let pointer = u32::from_le_bytes(entry.try_into().expect("fixed entry"));
        if pointer == 0 {
            return Err(ERRNO_INVAL);
        }
        values.push(string(bytes, pointer, remaining)?);
    }
    Ok(values)
}

fn signal(number: i32) -> Option<Signal> {
    Signal::parse(&number.to_string())
}

/// Probe virtual identities without queuing a signal or changing dispositions.
fn signal_probe(processes: &crate::process::ProcessTable, pid: i32, current_group: u32) -> i32 {
    let exists = if pid > 0 {
        processes.get(pid as u32).is_some()
    } else if pid == -1 {
        return ERRNO_NOTSUP;
    } else {
        let group = if pid == 0 {
            current_group
        } else {
            pid.unsigned_abs()
        };
        processes
            .iter()
            .any(|process| process.process_group == group)
    };
    if exists {
        ERRNO_SUCCESS
    } else {
        ERRNO_SRCH
    }
}

type SpawnParams = (u32, u32, u32, u32, u32, u32, u32, u32, u32, u32);

fn decode_spawn(bytes: &impl ReadGuest, params: SpawnParams) -> Result<ProcessSpawn, i32> {
    let (executable_path, argv, argc, env, envc, actions, actionc, options, defaults, output) =
        params;
    if argc == 0 || actionc > 64 || options & !3 != 0 || (options & 2 == 0 && defaults != 0) {
        return Err(ERRNO_INVAL);
    }
    bytes.range(output, 4)?;
    let mut remaining = MAX_STRINGS;
    let executable = path(bytes, executable_path, &mut remaining)?;
    if executable.is_empty() {
        return Err(ERRNO_NOENT);
    }
    let argv = vector(bytes, argv, argc, MAX_ARGUMENTS, &mut remaining)?;
    let environment = vector(bytes, env, envc, MAX_ENVIRONMENT, &mut remaining)?
        .into_iter()
        .map(|entry| {
            let (name, value) = entry.split_once('=').ok_or(ERRNO_INVAL)?;
            if name.is_empty() {
                return Err(ERRNO_INVAL);
            }
            Ok((name.to_owned(), value.to_owned()))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let records = bytes.range(actions, actionc as usize * 24)?;
    let mut decoded = Vec::with_capacity(actionc as usize);
    let valid_fd = |fd: i32| (0..MAX_FDS_PER_PROCESS as i32 + 5).contains(&fd);
    for record in records.chunks_exact(24) {
        let field = |offset| {
            u32::from_le_bytes(
                record[offset..offset + 4]
                    .try_into()
                    .expect("fixed action field"),
            )
        };
        let kind = field(0);
        let fd = field(4) as i32;
        let target = field(8) as i32;
        let flags = field(12);
        let mode = field(16);
        let path_pointer = field(20);
        let action = match kind {
            1 if valid_fd(fd) => SpawnFdAction::Close(fd),
            2 if valid_fd(fd) && valid_fd(target) && target != 3 && target != 4 => {
                SpawnFdAction::Dup2 {
                    source: fd,
                    destination: target,
                }
            }
            3 if fd >= 0 => SpawnFdAction::CloseFrom(fd),
            4 if valid_fd(fd) && fd != 3 && fd != 4 => {
                let supported =
                    0x0400_0000 | 0x1000_0000 | 0x1000 | 0x4000 | 0x8000 | 1 | 4 | O_CLOEXEC;
                if flags & !supported != 0
                    || flags & (0x0400_0000 | 0x1000_0000) == 0
                    || mode & !0o7777 != 0
                {
                    return Err(ERRNO_INVAL);
                }
                SpawnFdAction::Open {
                    fd,
                    path: path(bytes, path_pointer, &mut remaining)?,
                    options: OpenFile {
                        readable: flags & 0x0400_0000 != 0,
                        writable: flags & 0x1000_0000 != 0,
                        create: flags & 0x1000 != 0,
                        exclusive: flags & 0x4000 != 0,
                        truncate: flags & 0x8000 != 0,
                        append: flags & 1 != 0,
                    },
                    mode,
                    close_on_exec: flags & O_CLOEXEC != 0,
                    nonblocking: flags & 4 != 0,
                }
            }
            5 => SpawnFdAction::Chdir(path(bytes, path_pointer, &mut remaining)?),
            _ => return Err(ERRNO_INVAL),
        };
        decoded.push(action);
    }
    let mut signal_defaults = Vec::new();
    for number in 0..32 {
        if defaults & (1 << number) != 0 {
            let signal = signal(number).ok_or(ERRNO_NOTSUP)?;
            if matches!(signal, Signal::Kill | Signal::Stop) {
                return Err(ERRNO_INVAL);
            }
            signal_defaults.push(signal);
        }
    }
    Ok(ProcessSpawn {
        executable,
        search_path: options & 1 != 0,
        argv,
        environment,
        actions: decoded,
        signal_defaults,
    })
}

fn spawn(mut caller: Caller<'_, Host>, params: SpawnParams) -> Result<i32, Error> {
    let Some(memory) = memory(&mut caller) else {
        return Ok(ERRNO_FAULT);
    };
    // Charge the bounded parser before scanning or copying any guest strings.
    let mut machine = caller.data_mut().machine.get();
    let reserved = decode_memory(params.2, params.4);
    if !machine
        .resources
        .charge_cpu(decode_cost(params.2, params.4))
    {
        return Err(exhausted());
    }
    if !machine.resources.reserve_memory(reserved) {
        return Err(exhausted());
    }
    drop(machine);
    let decoded = decode_spawn(
        &GuestReader {
            memory: &memory,
            caller: &caller,
        },
        params,
    );
    let outcome = match decoded {
        Ok(spec) => ActiveSystem::new(&mut caller.data_mut().machine.get())
            .spawn_process(spec)
            .map_err(|error| syscall_errno(&error)),
        Err(error) => Err(error),
    };
    ActiveSystem::new(&mut caller.data_mut().machine.get()).release_memory(reserved);
    Ok(match outcome {
        Ok(pid) if write_u32(&mut caller, params.9, pid) => ERRNO_SUCCESS,
        Ok(_) => ERRNO_FAULT,
        Err(error) => error,
    })
}

fn pipe(mut caller: Caller<'_, Host>, flags: i32, reader: u32, writer: u32) -> Result<i32, Error> {
    if flags as u32 & !(O_CLOEXEC | 4) != 0 {
        return Ok(ERRNO_INVAL);
    }
    let Some(memory) = memory(&mut caller) else {
        return Ok(ERRNO_FAULT);
    };
    if memory.range(&caller, reader as usize, 4).is_err()
        || memory.range(&caller, writer as usize, 4).is_err()
        || reader.abs_diff(writer) < 4
    {
        return Ok(ERRNO_FAULT);
    }
    let outcome = ActiveSystem::new(&mut caller.data_mut().machine.get())
        .pipe(flags as u32 & O_CLOEXEC != 0, flags & 4 != 0);
    Ok(match outcome {
        Ok((read_fd, write_fd)) => {
            caller.data_mut().open_files.extend([read_fd, write_fd]);
            if write_u32(&mut caller, reader, read_fd as u32)
                && write_u32(&mut caller, writer, write_fd as u32)
            {
                ERRNO_SUCCESS
            } else {
                ERRNO_FAULT
            }
        }
        Err(error) => syscall_errno(&error),
    })
}

async fn wait(
    mut caller: Caller<'_, Host>,
    (pid, options, status, observed): (i32, i32, u32, u32),
) -> Result<i32, Error> {
    if options & !1 != 0 || pid == 0 || pid < -1 {
        return Ok(ERRNO_NOTSUP);
    }
    let Some(memory) = memory(&mut caller) else {
        return Ok(ERRNO_FAULT);
    };
    if memory.range(&caller, status as usize, 4).is_err()
        || memory.range(&caller, observed as usize, 4).is_err()
        || status.abs_diff(observed) < 4
    {
        return Ok(ERRNO_FAULT);
    }
    loop {
        let (children, completed) = {
            let mut machine = caller.data_mut().machine.get();
            let owner = machine.process.pid;
            let children = machine
                .processes
                .iter()
                .filter(|child| child.ppid == owner && (pid == -1 || child.pid == pid as u32))
                .map(|child| child.pid)
                .collect::<Vec<_>>();
            if children.is_empty() {
                return Ok(ERRNO_CHILD);
            }
            if !machine.resources.charge_cpu(children.len() as u64) {
                return Err(exhausted());
            }
            let mut system = ActiveSystem::new(&mut machine);
            let mut completed = None;
            for child in &children {
                if let Some(completion) = system
                    .child_completion(*child)
                    .map_err(|error| Error::msg(error.to_string()))?
                {
                    system
                        .reap_child(*child)
                        .map_err(|error| Error::msg(error.to_string()))?;
                    completed = Some((*child, completion));
                    break;
                }
            }
            (children, completed)
        };
        if let Some((child, completion)) = completed {
            return Ok(
                if write_u32(&mut caller, status, completion.wait_status() as u32)
                    && write_u32(&mut caller, observed, child)
                {
                    ERRNO_SUCCESS
                } else {
                    ERRNO_FAULT
                },
            );
        }
        if options & 1 != 0 {
            return Ok(if write_u32(&mut caller, observed, 0) {
                ERRNO_SUCCESS
            } else {
                ERRNO_FAULT
            });
        }
        super::threaded_dynamic::can_block(caller.data())?;
        caller
            .data()
            .machine
            .clone()
            .suspend(Suspension::Blocked(WaitReason::Any(
                children.into_iter().map(WaitReason::Child).collect(),
            )))
            .await;
    }
}

pub(super) fn register(linker: &mut Linker<Host>) {
    linker
        .func_wrap(
            "shellsim_posix_v1",
            "process_identity",
            |mut caller: Caller<'_, Host>, operation: i32| -> i32 {
                let mut machine = caller.data_mut().machine.get();
                let system = ActiveSystem::new(&mut machine);
                match operation {
                    0 => system.pid() as i32,
                    1 => system.parent_pid() as i32,
                    _ => -ERRNO_INVAL,
                }
            },
        )
        .expect("unique process import");
    linker
        .func_wrap("shellsim_posix_v1", "process_pipe", pipe)
        .expect("unique process import");
    linker
        .func_wrap(
            "shellsim_posix_v1",
            "process_spawn",
            |caller: Caller<'_, Host>,
             path: u32,
             argv: u32,
             argc: u32,
             env: u32,
             envc: u32,
             actions: u32,
             actionc: u32,
             options: u32,
             defaults: u32,
             output: u32| {
                spawn(
                    caller,
                    (
                        path, argv, argc, env, envc, actions, actionc, options, defaults, output,
                    ),
                )
            },
        )
        .expect("unique process import");
    linker
        .func_wrap_async("shellsim_posix_v1", "process_wait", |caller, params| {
            Box::new(wait(caller, params))
        })
        .expect("unique process import");
    linker
        .func_wrap(
            "shellsim_posix_v1",
            "process_kill",
            |mut caller: Caller<'_, Host>, pid: i32, number: i32| -> Result<i32, Error> {
                let mut machine = caller.data_mut().machine.get();
                if number == 0 {
                    if !machine
                        .resources
                        .charge_cpu(crate::process::MAX_PROCESSES as u64)
                    {
                        return Err(exhausted());
                    }
                    return Ok(signal_probe(
                        &machine.processes,
                        pid,
                        machine.process.process_group,
                    ));
                }
                let Some(signal) = signal(number) else {
                    return Ok(ERRNO_INVAL);
                };
                let target = if pid > 0 {
                    SignalTarget::Process(pid as u32)
                } else if pid == 0 {
                    SignalTarget::Group(machine.process.process_group)
                } else if pid < -1 {
                    SignalTarget::Group(pid.unsigned_abs())
                } else {
                    return Ok(ERRNO_NOTSUP);
                };
                Ok(match ActiveSystem::new(&mut machine).kill(target, signal) {
                    Ok(()) => ERRNO_SUCCESS,
                    Err(_) => ERRNO_SRCH,
                })
            },
        )
        .expect("unique process import");
    linker
        .func_wrap(
            "shellsim_posix_v1",
            "process_signal_disposition",
            |mut caller: Caller<'_, Host>, number: i32, mode: i32| -> i32 {
                if !(0..=1).contains(&mode) {
                    return ERRNO_INVAL;
                }
                let Some(signal) = signal(number) else {
                    return ERRNO_NOTSUP;
                };
                match ActiveSystem::new(&mut caller.data_mut().machine.get())
                    .signal_disposition(signal, mode == 1)
                {
                    Ok(()) => ERRNO_SUCCESS,
                    Err(error) => syscall_errno(&error),
                }
            },
        )
        .expect("unique process import");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn larger_argument_decode_exhausts_before_reading_guest_pointers() {
        use crate::resources::StopReason;

        // Set the remaining budget after image startup. Legacy guest startup
        // otherwise leaves a fixed I/O allowance, obscuring decode exhaustion.
        // One argument reaches the invalid path; 1500 exhaust the same budget
        // before any pointer reads or process creation.
        for spawn in [true, false] {
            for reason in [StopReason::CpuExhausted, StopReason::MemoryExhausted] {
                for argc in [1, 1500] {
                    let (signature, arguments, import) = if spawn {
                        ("i32 i32 i32 i32 i32 i32 i32 i32 i32 i32",
                         "(i32.const 0) (i32.const 0) (i32.const 0) (i32.const 0) (i32.const 64)",
                         "process_spawn")
                    } else {
                        ("i32 i32 i32 i32 i32 i32", "(i32.const 0)", "process_exec")
                    };
                    let bytes = wat::parse_str(format!(
                        r#"(module
                        (import "shellsim_posix_v1" "{import}"
                            (func $launch (param {signature}) (result i32)))
                        (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
                        (memory (export "memory") 1)
                        (func (export "_start")
                            (call $exit (call $launch (i32.const -1) (i32.const -1)
                                (i32.const {argc}) (i32.const 0) (i32.const 0) {arguments}))))"#
                    ))
                    .unwrap();
                    let mut environment = Interp::new();
                    environment
                        .vfs
                        .write("/", "/caller", &bytes, 0o755)
                        .unwrap();
                    let mut guest = start_guest(
                        &mut environment,
                        Launch {
                            path: "/caller",
                            argv: vec![b"/caller".to_vec()],
                            stdio: Stdio::Descriptors,
                            interaction: None,
                        },
                    )
                    .unwrap();
                    if reason == StopReason::CpuExhausted {
                        let consumed = environment.resources.cpu_remaining() - 300_000;
                        assert!(environment.resources.charge_cpu(consumed));
                    } else {
                        let occupied = environment.resources.memory_remaining() - 450_000;
                        assert!(environment.resources.reserve_memory(occupied));
                    }
                    let GuestPoll::Ready(outcome) = guest.poll(&mut environment) else {
                        panic!("bounded launch must finish in one poll");
                    };
                    let GuestOutcome::Exit { status, .. } = *outcome else {
                        panic!("invalid input must not replace the image");
                    };
                    assert_eq!(status, if argc == 1 { ERRNO_FAULT } else { 137 });
                    assert_eq!(
                        environment.resources.stop_reason(),
                        (argc == 1500).then_some(reason)
                    );
                    assert_eq!(environment.processes.iter().count(), 1);
                    release_guest_resources(&mut environment, &mut guest.reserved);
                }
            }
        }
    }

    #[test]
    fn zero_signal_probes_virtual_groups_without_changing_processes() {
        let mut processes = crate::process::ProcessTable::new(12, "/".into(), BTreeMap::new());
        let child = processes
            .spawn(
                12,
                crate::process::ChildPlacement::NewProcessGroup,
                "child",
                "/",
                BTreeMap::new(),
            )
            .unwrap();
        let before = processes.iter().cloned().collect::<Vec<_>>();
        assert_eq!(signal_probe(&processes, 0, 12), ERRNO_SUCCESS);
        assert_eq!(signal_probe(&processes, -(child as i32), 12), ERRNO_SUCCESS);
        assert_eq!(signal_probe(&processes, child as i32, 12), ERRNO_SUCCESS);
        assert_eq!(signal_probe(&processes, -99, 12), ERRNO_SRCH);
        assert_eq!(signal_probe(&processes, 0, 99), ERRNO_SRCH);
        assert_eq!(signal_probe(&processes, i32::MIN, 12), ERRNO_SRCH);
        assert_eq!(signal_probe(&processes, -1, 12), ERRNO_NOTSUP);
        assert_eq!(processes.iter().cloned().collect::<Vec<_>>(), before);
    }

    #[test]
    fn near_limit_shared_string_consumes_owned_storage_without_copying() {
        struct OwnedReader {
            bytes: Vec<u8>,
            allocated: std::cell::Cell<*const u8>,
        }
        impl ReadGuest for OwnedReader {
            fn range(
                &self,
                pointer: u32,
                length: usize,
            ) -> Result<std::borrow::Cow<'_, [u8]>, i32> {
                self.bytes.range(pointer, length)
            }
            fn c_string(
                &self,
                pointer: u32,
                limit: usize,
            ) -> Result<std::borrow::Cow<'_, [u8]>, i32> {
                let bytes = self.bytes.c_string(pointer, limit)?.into_owned();
                self.allocated.set(bytes.as_ptr());
                Ok(std::borrow::Cow::Owned(bytes))
            }
        }
        let mut bytes = vec![b'a'; MAX_STRINGS];
        bytes[MAX_STRINGS - 1] = 0;
        let reader = OwnedReader {
            bytes,
            allocated: std::cell::Cell::new(std::ptr::null()),
        };
        let mut remaining = MAX_STRINGS;
        let value = string(&reader, 0, &mut remaining).unwrap();
        assert_eq!(value.len(), MAX_STRINGS - 1);
        assert_eq!(value.as_ptr(), reader.allocated.get());
        assert_eq!(remaining, 0);
        assert_eq!(string(&reader, 0, &mut remaining), Err(ERRNO_INVAL));
        let unterminated = vec![b'a'; MAX_STRINGS + 1];
        let mut remaining = MAX_STRINGS;
        assert_eq!(string(&unterminated, 0, &mut remaining), Err(ERRNO_INVAL));
        assert_eq!(remaining, MAX_STRINGS);
    }

    #[test]
    fn decoder_bounds_vectors_strings_actions_and_outputs() {
        let mut bytes = vec![0; 256];
        bytes[100..114].copy_from_slice(b"/usr/bin/true\0");
        bytes[32..36].copy_from_slice(&100u32.to_le_bytes());
        let valid = (100, 32, 1, 0, 0, 0, 0, 2, (1 << 13) | (1 << 25), 48);
        let spec = decode_spawn(&bytes, valid).unwrap();
        assert_eq!(spec.signal_defaults, vec![Signal::Pipe, Signal::FileSize]);
        assert!(spec.environment.is_empty());
        assert_eq!(
            decode_spawn(&bytes, (100, 32, 4097, 0, 0, 0, 0, 0, 0, 48)).err(),
            Some(ERRNO_INVAL)
        );
        assert_eq!(
            decode_spawn(&bytes, (100, 32, 1, 0, 0, 0, 65, 0, 0, 48)).err(),
            Some(ERRNO_INVAL)
        );
        assert_eq!(
            decode_spawn(&bytes, (100, 32, 1, 0, 0, 0, 0, 0, 0, 254)).err(),
            Some(ERRNO_FAULT)
        );
        let mut remaining = 3;
        assert_eq!(
            string(b"xxxx\0", 0, &mut remaining).err(),
            Some(ERRNO_INVAL)
        );
        assert_eq!(remaining, 3);
        let mut remaining = MAX_STRINGS;
        assert_eq!(
            path(
                &vec![b'x'; 4097].into_iter().chain([0]).collect::<Vec<_>>(),
                0,
                &mut remaining
            )
            .err(),
            Some(ERRNO_INVAL)
        );
    }
}
