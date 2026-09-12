//! Built-in commands and coreutils, implemented natively against the VFS.
//!
//! Each command is a plain function with the uniform signature
//! [`CmdFn`] = `fn(&mut CommandContext, &[String], &mut Io) -> i32`, where the slice is `argv[1..]`
//! and all I/O flows through the [`Io`] context (read `io.stdin`, write `io.out` / `io.err`).
//! Builtins that mutate environment state (cd, export, set, …) do so through the context.
//!
//! Commands are looked up in a [`OnceLock`]-backed registry that records each command's
//! [`Trust`] level, so a run can report whether it stayed inside the faithfully-simulated
//! envelope. Unknown commands fall through to the VFS-script / shebang fallback.

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::OnceLock;

use crate::interp::Interp;

mod awk;
mod builtins;
mod echo;
mod fs;
mod git;
mod hashing;
mod makecmd;
mod net;
pub(crate) mod pkg;
mod printf;
mod proc;
mod sort;
mod system;
mod text;
pub mod util;

/// Bundled standard I/O for a command invocation.
pub struct Io<'a> {
    pub stdin: Vec<u8>,
    pub out: &'a mut Vec<u8>,
    pub err: &'a mut Vec<u8>,
}

/// Uniform command signature. `args` is `argv[1..]`.
pub type CmdFn = fn(&mut CommandContext<'_>, &[String], &mut Io) -> i32;

/// The only environment handle handed to command implementations.
///
/// `Deref` keeps the initial port mechanical; quota-sensitive primitives are enforced by the
/// environment-owned VFS and resource meter even while older bodies use field syntax.
pub struct CommandContext<'a> {
    env: &'a mut Interp,
}

impl CommandContext<'_> {
    pub fn charge_cpu(&mut self, units: u64) -> bool {
        self.env.resources.charge_cpu(units)
    }

    pub fn reserve_memory(&mut self, bytes: u64) -> bool {
        self.env.resources.reserve_memory(bytes)
    }
}

impl Deref for CommandContext<'_> {
    type Target = Interp;

    fn deref(&self) -> &Self::Target {
        self.env
    }
}

impl DerefMut for CommandContext<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.env
    }
}

/// How faithfully a command is simulated.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Trust {
    /// Faithful implementation (coreutils / builtins we fully model).
    Real,
    /// A subset of the real behavior (jq, sed, grep, uv, …).
    Partial,
    /// Ignored / pretend-success (apt-get, pip, …).
    NoOp,
}

/// A registered command: its implementation and trust level.
pub struct CommandSpec {
    pub run: CmdFn,
    pub trust: Trust,
    pub base_cpu: u64,
    pub base_memory: u64,
}

static REGISTRY: OnceLock<HashMap<&'static str, CommandSpec>> = OnceLock::new();

fn registry() -> &'static HashMap<&'static str, CommandSpec> {
    REGISTRY.get_or_init(build_registry)
}

/// Register `f` under every name in `names` with trust `t`.
fn reg(map: &mut HashMap<&'static str, CommandSpec>, names: &[&'static str], t: Trust, f: CmdFn) {
    reg_costed(map, names, t, 100, 10 * 1024, f);
}

/// Register a command with deterministic, deliberately-coarse base resource costs.
fn reg_costed(
    map: &mut HashMap<&'static str, CommandSpec>,
    names: &[&'static str],
    t: Trust,
    base_cpu: u64,
    base_memory: u64,
    f: CmdFn,
) {
    for &n in names {
        map.insert(
            n,
            CommandSpec {
                run: f,
                trust: t,
                base_cpu,
                base_memory,
            },
        );
    }
}

fn build_registry() -> HashMap<&'static str, CommandSpec> {
    let mut m = HashMap::new();
    builtins::register(&mut m);
    awk::register(&mut m);
    echo::register(&mut m);
    printf::register(&mut m);
    sort::register(&mut m);
    system::register(&mut m);
    text::register(&mut m);
    fs::register(&mut m);
    git::register(&mut m);
    hashing::register(&mut m);
    makecmd::register(&mut m);
    net::register(&mut m);
    proc::register(&mut m);
    pkg::register(&mut m);
    m
}

/// Dispatch entry point: look up `argv[0]`, record its trust, and run it.
///
/// Unknown commands fall through to the legacy fallback: try to execute a script that lives
/// in the VFS (shell or `#!`-python), otherwise record it as unsupported and return 127.
pub fn run(
    interp: &mut Interp,
    argv: &[String],
    stdin: Vec<u8>,
    out: &mut Vec<u8>,
    err: &mut Vec<u8>,
) -> i32 {
    let requested = argv[0].as_str();
    // Agents frequently use explicit paths or `/usr/bin/env` shebangs. Standard utility paths
    // resolve to the same in-process command without pretending arbitrary host paths exist.
    let cmd = standard_utility_name(requested).unwrap_or(requested);
    let args = &argv[1..];
    if let Some(spec) = registry().get(cmd) {
        match spec.trust {
            Trust::NoOp => {
                // NoOp commands (package managers and native compilers) are recorded as unsupported,
                // preserving the legacy `note_unsupported(cmd)` behavior for that arm.
                interp.note_unsupported(cmd);
                interp.trust_noop.insert(cmd.to_string());
            }
            Trust::Partial => {
                interp.trust_partial.insert(cmd.to_string());
            }
            Trust::Real => {}
        }
        let cpu_before = interp.resources.cpu_used();
        let disk_before = interp.vfs.disk_used();
        let fs_read_before = interp.vfs.read_bytes();
        let input_bytes = stdin.len() as u64;
        let arg_bytes = args.iter().map(|a| a.len() as u64).sum::<u64>();
        let working_memory = spec.base_memory.saturating_add(input_bytes);
        if !interp
            .resources
            .charge_cpu(spec.base_cpu.saturating_add(arg_bytes))
        {
            return interp
                .resources
                .stop_reason()
                .map_or(137, |r| r.exit_status());
        }
        if !interp.resources.reserve_memory(working_memory) {
            return interp
                .resources
                .stop_reason()
                .map_or(137, |r| r.exit_status());
        }

        let out_before = out.len();
        let err_before = err.len();
        let output_meter_before = interp.resources.output_bytes();
        interp.sync_vfs_time();
        let mut io = Io { stdin, out, err };
        let mut context = CommandContext { env: interp };
        let mut status = (spec.run)(&mut context, args, &mut io);
        let out_bytes = io.out.len().saturating_sub(out_before);
        let err_bytes = io.err.len().saturating_sub(err_before);
        let output_bytes = out_bytes.saturating_add(err_bytes) as u64;
        // Nested dispatches already charged their own output. Only meter bytes produced directly
        // by this frame, otherwise `xargs`/`timeout` style commands would double-count them.
        let nested_output = interp
            .resources
            .output_bytes()
            .saturating_sub(output_meter_before);
        let unaccounted_output = output_bytes.saturating_sub(nested_output);
        let output_remaining = interp.resources.output_remaining();
        if unaccounted_output > output_remaining {
            let allowed_total = output_bytes.saturating_sub(unaccounted_output - output_remaining);
            let allowed_out = out_bytes.min(allowed_total as usize);
            io.out.truncate(out_before + allowed_out);
            let remaining = allowed_total.saturating_sub(allowed_out as u64) as usize;
            io.err.truncate(err_before + err_bytes.min(remaining));
        }
        let _ = interp.resources.charge_cpu(
            input_bytes
                .saturating_add(output_bytes)
                .saturating_add(interp.vfs.read_bytes().saturating_sub(fs_read_before)),
        );
        let _ = interp.resources.charge_output(unaccounted_output);
        interp.resources.release_memory(working_memory);
        interp
            .resources
            .record_command(cmd, cpu_before, disk_before, interp.vfs.disk_used());
        if let Some(reason) = interp.resources.stop_reason() {
            status = reason.exit_status();
        }
        return status;
    }

    // ---- fallback: maybe it's an executable script in the VFS ----
    interp.sync_vfs_time();
    if let Some(code) = util::try_exec_script(interp, requested, args, &stdin, out, err) {
        return code;
    }
    // An unknown command the task actually invoked (a missing tool, a compiled binary we can't
    // run, …) is a genuine simulation gap — record it so the trust verdict reflects it.
    interp.note_unsupported(requested);
    interp.trust_noop.insert(requested.to_string());
    util::ewln(err, &format!("{requested}: command not found"));
    127
}

fn standard_utility_name(path: &str) -> Option<&str> {
    [
        "/bin/",
        "/sbin/",
        "/usr/bin/",
        "/usr/sbin/",
        "/usr/local/bin/",
    ]
    .into_iter()
    .find_map(|prefix| path.strip_prefix(prefix).filter(|name| !name.contains('/')))
}

/// Provide `run_script_into` for nested execution (source, eval, scripts).
impl Interp {
    pub fn run_script_into(&mut self, src: &str, out: &mut Vec<u8>, err: &mut Vec<u8>) -> i32 {
        let parser_memory = 8 * 1024 + (src.len() as u64).saturating_mul(2);
        if !self.resources.reserve_memory(parser_memory) {
            return self
                .resources
                .stop_reason()
                .map_or(137, |r| r.exit_status());
        }
        if !self.resources.charge_cpu(src.len() as u64) {
            self.resources.release_memory(parser_memory);
            return self
                .resources
                .stop_reason()
                .map_or(137, |r| r.exit_status());
        }
        let ast = match crate::shell::parse(src) {
            Ok(ast) => ast,
            Err(error) => {
                err.extend_from_slice(format!("shellsim: syntax error: {error}\n").as_bytes());
                self.last_status = 2;
                self.resources.release_memory(parser_memory);
                return 2;
            }
        };
        let r = self.returning.take();
        let code = crate::exec::exec(self, &ast, Vec::new(), out, err);
        self.resources.release_memory(parser_memory);
        if r.is_some() {
            self.returning = r;
        }
        code
    }
}
