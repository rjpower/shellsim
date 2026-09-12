//! Interpreter state shared across the shell executor and all commands.

use std::collections::{BTreeMap, HashMap};
use std::ops::{Deref, DerefMut};

use crate::clock::{Clock, EventId};
use crate::descriptors::{DescriptorArena, FdTable};
use crate::net::VirtualNet;
use crate::process::{ProcessId, ProcessTable};
use crate::resources::{Limits, Resources, RunOutcome};
use crate::scheduler::{Scheduler, WaitReason};
use crate::vfs::Vfs;

/// A bash array value. Indexed arrays are sparse (`arr[5]=x` on an empty array is legal),
/// so unset slots are `None`. Associative arrays preserve sorted key order (bash uses an
/// unspecified hash order; sorted is deterministic and good enough for our checks).
#[derive(Clone, Debug)]
pub enum ArrayVal {
    Indexed(Vec<Option<String>>),
    Assoc(BTreeMap<String, String>),
}

/// A simulated background job (started with `&`). Because there is no real concurrency,
/// a job is just a captured AST that will be run-to-completion when the scheduler decides
/// to. Jobs currently run immediately and synchronously, after which `jobs` and `wait` expose
/// their modeled identifier and status. This is indistinguishable for many file-state checks but
/// does not model overlapping work.
#[derive(Clone, Debug)]
pub struct Job {
    pub id: u32,
    pub pid: ProcessId,
    pub cmd: String,
    pub done: bool,
    pub status: i32,
}

/// Machine-wide state and limits. A future scheduler can attach multiple process states to the
/// same machine; the initial implementation intentionally runs one process synchronously.
pub struct Environment {
    pub vfs: Vfs,
    pub clock: Clock,
    pub net: VirtualNet,
    /// Deterministic CPU, transient-memory, and output accounting for this environment.
    pub resources: Resources,
    /// Machine-wide logical process identities and retained child statuses.
    pub processes: ProcessTable,
    /// Deterministic runnable/blocked lifecycle for logical process execution.
    pub scheduler: Scheduler,
    /// Machine-owned open descriptions shared by forked process descriptor tables.
    pub descriptors: DescriptorArena,
    pub process: ProcessState,
    next_temp_id: u64,
    /// Trace of every external command name executed.
    pub cmd_trace: Vec<String>,
    /// Commands requested that shellsim does not implement.
    pub unsupported: Vec<String>,
    /// Commands that deliberately used a successful compatibility no-op.
    pub trust_noop: std::collections::BTreeSet<String>,
    /// Commands that implement a documented subset.
    pub trust_partial: std::collections::BTreeSet<String>,
    /// Packages recorded by lightweight package-manager compatibility commands.
    pub packages: std::collections::BTreeSet<String>,
}

/// Shell-local state for the single process currently executing in an [`Environment`].
pub struct ProcessState {
    /// PID of the currently executing logical process.
    pub pid: ProcessId,
    /// PID of the logical parent process.
    pub ppid: ProcessId,
    /// PID expanded by `$$`; preserved across Bash subshells.
    pub shell_pid: ProcessId,
    /// shell + environment variables (we don't distinguish exported vs not for simplicity,
    /// except that `env`/child python only sees exported ones, tracked in `exported`)
    pub vars: HashMap<String, String>,
    /// bash arrays (indexed + associative), keyed by variable name. A name present here is an
    /// array; it shadows any scalar `vars` entry of the same name for `${name[...]}` access.
    pub arrays: HashMap<String, ArrayVal>,
    pub exported: std::collections::BTreeSet<String>,
    pub cwd: String,
    pub funcs: HashMap<String, crate::shell::Node>,
    /// `$?`
    pub last_status: i32,
    /// positional parameters `$1 $2 ... $@`
    pub positional: Vec<String>,
    /// `set -e` / `set -u` / `set -x`
    pub opt_errexit: bool,
    pub opt_nounset: bool,
    pub opt_xtrace: bool,
    /// `set -o pipefail`
    pub opt_pipefail: bool,
    pub jobs: Vec<Job>,
    next_job_id: u32,
    /// Recursion and loop-control signaling.
    pub loop_break: u32,
    pub loop_continue: u32,
    pub returning: Option<i32>,
    pub exiting: Option<i32>,
    /// depth of "condition" contexts (if/while/&&/||/!) where `set -e` is suppressed
    pub cond_depth: u32,
    /// Deadline event currently unwinding the synchronous executor.  A future resumable
    /// scheduler will keep this on each task frame; today there is one shell process.
    pub(crate) deadline_interrupt: Option<EventId>,
    /// effective uid (for permission-ish checks / `id`)
    pub uid: u32,
    /// persistent input stream + cursor for `read` inside `while read…; done < file`
    pub input_stream: Vec<u8>,
    pub input_pos: usize,
    /// A foreground Python REPL, when `python` was invoked without a program. Keeping this in
    /// process state lets an agent enter Python in one shell action and continue it in later
    /// actions without giving the shim access to host stdin.
    pub python_repl: Option<crate::python::ReplState>,
    /// Unix-style descriptor map inherited by logical children.
    pub(crate) fds: FdTable,
}

const MAX_FORK_STATE_BYTES: u64 = 32 * 1024 * 1024;

/// Parent state retained while one logical child runs synchronously.
pub(crate) struct ParentProcess {
    state: ProcessState,
    memory_mark: u64,
}

impl ProcessState {
    fn fork_for_child(&self, pid: ProcessId, new_shell: bool, fds: FdTable) -> Self {
        Self {
            pid,
            ppid: self.pid,
            shell_pid: if new_shell { pid } else { self.shell_pid },
            vars: self.vars.clone(),
            arrays: self.arrays.clone(),
            exported: self.exported.clone(),
            cwd: self.cwd.clone(),
            funcs: self.funcs.clone(),
            last_status: self.last_status,
            positional: self.positional.clone(),
            opt_errexit: self.opt_errexit,
            opt_nounset: self.opt_nounset,
            opt_xtrace: self.opt_xtrace,
            opt_pipefail: self.opt_pipefail,
            jobs: Vec::new(),
            next_job_id: 1,
            loop_break: 0,
            loop_continue: 0,
            returning: None,
            exiting: None,
            cond_depth: self.cond_depth,
            deadline_interrupt: self.deadline_interrupt,
            uid: self.uid,
            input_stream: self.input_stream.clone(),
            input_pos: self.input_pos,
            python_repl: None,
            fds,
        }
    }

    fn fork_memory_bytes(&self) -> u64 {
        let string = |value: &String| (value.len() as u64).saturating_add(24);
        let mut bytes = self
            .vars
            .iter()
            .fold(0_u64, |total, (name, value)| {
                total
                    .saturating_add(string(name))
                    .saturating_add(string(value))
            })
            .saturating_add(
                self.exported
                    .iter()
                    .fold(0, |total, value| total.saturating_add(string(value))),
            )
            .saturating_add(string(&self.cwd))
            .saturating_add(
                self.positional
                    .iter()
                    .fold(0, |total, value| total.saturating_add(string(value))),
            )
            .saturating_add(self.input_stream.len() as u64);
        bytes = bytes.saturating_add((self.fds.iter().count() as u64).saturating_mul(16));
        for (name, value) in &self.arrays {
            bytes = bytes.saturating_add(string(name));
            bytes = bytes.saturating_add(match value {
                ArrayVal::Indexed(values) => values.iter().fold(0, |total, value| {
                    total.saturating_add(value.as_ref().map_or(0, string))
                }),
                ArrayVal::Assoc(values) => values.iter().fold(0, |total, (name, value)| {
                    total
                        .saturating_add(string(name))
                        .saturating_add(string(value))
                }),
            });
        }
        for (name, body) in &self.funcs {
            bytes = bytes
                .saturating_add(string(name))
                .saturating_add(body.estimated_bytes());
        }
        bytes
    }
}

impl Deref for Environment {
    type Target = ProcessState;

    fn deref(&self) -> &Self::Target {
        &self.process
    }
}

impl DerefMut for Environment {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.process
    }
}

impl Environment {
    pub fn new() -> Self {
        Self::with_limits(Limits::default())
    }

    pub fn with_limits(limits: Limits) -> Self {
        let mut vars: HashMap<String, String> = HashMap::new();
        vars.insert("HOME".into(), "/root".into());
        vars.insert(
            "PATH".into(),
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin".into(),
        );
        vars.insert("PWD".into(), "/".into());
        vars.insert("SHELL".into(), "/bin/bash".into());
        vars.insert("TERM".into(), "xterm-256color".into());
        vars.insert("USER".into(), "root".into());
        vars.insert("HOSTNAME".into(), "sandbox".into());
        vars.insert("LANG".into(), "C.UTF-8".into());
        vars.insert("IFS".into(), " \t\n".into());
        let mut exported = std::collections::BTreeSet::new();
        for k in [
            "HOME", "PATH", "PWD", "SHELL", "TERM", "USER", "HOSTNAME", "LANG",
        ] {
            exported.insert(k.to_string());
        }
        let clock = Clock::new();
        let mut descriptors = DescriptorArena::new();
        let mut fds = FdTable::new();
        for (fd, description) in [
            (
                0,
                descriptors
                    .open_input(Vec::new())
                    .expect("stdin descriptor"),
            ),
            (1, descriptors.open_capture().expect("stdout descriptor")),
            (2, descriptors.open_capture().expect("stderr descriptor")),
        ] {
            fds.install(fd, description, &mut descriptors)
                .expect("standard descriptor table");
        }
        let mut vfs = Vfs::with_disk_limit(limits.disk);
        vfs.set_mutation_time(clock.unix_ms());
        vfs.seed_dirs(["/root", "/tmp", "/work"]);
        const ROOT_PID: ProcessId = 1_234;
        let process_environment = exported
            .iter()
            .filter_map(|name| vars.get(name).map(|value| (name.clone(), value.clone())))
            .collect();
        let descriptor_snapshot = fds
            .iter()
            .filter_map(|(fd, id)| descriptors.label(id).ok().map(|label| (fd, label)))
            .collect();
        let mut processes = ProcessTable::new(ROOT_PID, "/".to_string(), process_environment);
        processes.update_descriptors(ROOT_PID, descriptor_snapshot);
        Environment {
            vfs,
            clock,
            net: VirtualNet::new(),
            resources: Resources::new(limits),
            processes,
            scheduler: Scheduler::new(ROOT_PID),
            descriptors,
            process: ProcessState {
                pid: ROOT_PID,
                ppid: 0,
                shell_pid: ROOT_PID,
                vars,
                arrays: HashMap::new(),
                exported,
                cwd: "/".to_string(),
                funcs: HashMap::new(),
                last_status: 0,
                positional: Vec::new(),
                opt_errexit: false,
                opt_nounset: false,
                opt_xtrace: false,
                opt_pipefail: false,
                jobs: Vec::new(),
                next_job_id: 1,
                loop_break: 0,
                loop_continue: 0,
                returning: None,
                exiting: None,
                cond_depth: 0,
                deadline_interrupt: None,
                uid: 0,
                input_stream: Vec::new(),
                input_pos: 0,
                python_repl: None,
                fds,
            },
            next_temp_id: 0,
            cmd_trace: Vec::new(),
            unsupported: Vec::new(),
            trust_noop: std::collections::BTreeSet::new(),
            trust_partial: std::collections::BTreeSet::new(),
            packages: std::collections::BTreeSet::new(),
        }
    }

    /// Enter a synchronous logical child while retaining the parent shell state.
    ///
    /// Machine capabilities remain on `Environment`; only process-local state is copied. The
    /// caller must pair a successful call with [`Environment::finish_child`].
    pub(crate) fn start_child(
        &mut self,
        command: &str,
        new_shell: bool,
    ) -> Result<(ProcessId, ParentProcess), String> {
        let fork_bytes = self.process.fork_memory_bytes();
        if fork_bytes > MAX_FORK_STATE_BYTES {
            return Err("shell state exceeds the 32 MiB fork limit".to_string());
        }
        let memory_mark = self.resources.memory_mark();
        if !self
            .resources
            .reserve_memory(fork_bytes.saturating_add(256))
        {
            return Err("memory limit exceeded while creating child process".to_string());
        }
        let environment: BTreeMap<String, String> = self.child_env().into_iter().collect();
        let mut child_fds = match self.process.fds.fork(&mut self.descriptors) {
            Ok(fds) => fds,
            Err(error) => {
                self.resources.restore_memory(memory_mark);
                return Err(format!("unable to inherit descriptors: {error:?}"));
            }
        };
        self.processes
            .update_current(self.process.pid, &self.process.cwd, environment.clone());
        let Some(pid) =
            self.processes
                .spawn(self.process.pid, command, &self.process.cwd, environment)
        else {
            child_fds.close_all(&mut self.descriptors);
            self.resources.restore_memory(memory_mark);
            return Err("logical process limit exceeded".to_string());
        };
        if let Err(error) = self.scheduler.block_current(WaitReason::Child(pid)) {
            child_fds.close_all(&mut self.descriptors);
            self.processes.exit(pid, 125, &self.process.cwd);
            self.processes.reap(pid);
            self.resources.restore_memory(memory_mark);
            return Err(format!("unable to suspend parent process: {error:?}"));
        }
        if let Err(error) = self.scheduler.spawn(pid) {
            child_fds.close_all(&mut self.descriptors);
            let _ = self.scheduler.wake(self.process.pid);
            let _ = self.scheduler.dispatch();
            self.processes.exit(pid, 125, &self.process.cwd);
            self.processes.reap(pid);
            self.resources.restore_memory(memory_mark);
            return Err(format!("unable to schedule child process: {error:?}"));
        }
        match self.scheduler.dispatch() {
            Ok(Some(scheduled)) if scheduled == pid => {}
            result => {
                child_fds.close_all(&mut self.descriptors);
                self.processes.exit(pid, 125, &self.process.cwd);
                self.resources.restore_memory(memory_mark);
                return Err(format!("unexpected child dispatch result: {result:?}"));
            }
        }
        let child = self.process.fork_for_child(pid, new_shell, child_fds);
        let state = std::mem::replace(&mut self.process, child);
        self.refresh_descriptor_snapshot(pid);
        Ok((pid, ParentProcess { state, memory_mark }))
    }

    /// Restore the parent after a synchronous child and optionally retain the exited record.
    pub(crate) fn finish_child(
        &mut self,
        pid: ProcessId,
        parent: ParentProcess,
        status: i32,
        retain: bool,
    ) {
        let child_deadline_interrupt = self.process.deadline_interrupt;
        self.process.fds.close_all(&mut self.descriptors);
        self.processes.update_descriptors(pid, BTreeMap::new());
        self.processes.exit(pid, status, &self.process.cwd);
        let _ = self.scheduler.exit_current(status);
        let _ = self.scheduler.wake(parent.state.pid);
        let _ = self.scheduler.dispatch();
        if !retain {
            self.processes.reap(pid);
            let _ = self.scheduler.reap(pid);
        }
        self.process = parent.state;
        if child_deadline_interrupt.is_some() {
            self.process.deadline_interrupt = child_deadline_interrupt;
        }
        self.resources.restore_memory(parent.memory_mark);
    }

    /// Synchronize the active descriptor table into generated process metadata.
    pub(crate) fn refresh_descriptor_snapshot(&mut self, pid: ProcessId) {
        let descriptors = self
            .process
            .fds
            .iter()
            .filter_map(|(fd, id)| self.descriptors.label(id).ok().map(|label| (fd, label)))
            .collect();
        self.processes.update_descriptors(pid, descriptors);
    }

    /// Record a package name installed by a compatibility command.
    pub fn install_package(&mut self, import_name: &str) {
        let name = import_name.trim();
        if name.is_empty() {
            return;
        }
        self.packages.insert(name.to_string());
        let deps: &[&str] = match name {
            "pandas" => &["numpy"],
            "scipy" => &["numpy"],
            "sklearn" => &["numpy", "scipy"],
            _ => &[],
        };
        for d in deps {
            self.packages.insert((*d).to_string());
        }
    }

    /// Make the wall-clock/VFS boundary explicit immediately before a filesystem effect.
    pub fn sync_vfs_time(&mut self) {
        self.vfs.set_mutation_time(self.clock.unix_ms());
    }

    /// Read through the generated pseudo-filesystem before consulting persistent VFS state.
    pub fn fs_read(&self, cwd: &str, path: &str) -> crate::vfs::Result<Vec<u8>> {
        crate::pseudo_fs::read(self, cwd, path).unwrap_or_else(|| self.vfs.read(cwd, path))
    }

    /// Return a generated or persistent file length without exposing its backing implementation.
    pub fn fs_file_len(&self, cwd: &str, path: &str) -> crate::vfs::Result<usize> {
        if let Some(result) = crate::pseudo_fs::read(self, cwd, path) {
            result.map(|data| data.len())
        } else {
            self.vfs.file_len(cwd, path)
        }
    }

    /// Read a finite generated or persistent file after enforcing a caller-provided bound.
    pub fn fs_read_limited(
        &self,
        cwd: &str,
        path: &str,
        limit: usize,
    ) -> crate::vfs::Result<Vec<u8>> {
        if let Some(result) = crate::pseudo_fs::read(self, cwd, path) {
            let data = result?;
            if data.len() > limit {
                return Err(crate::vfs::VfsError::TooLarge {
                    path: path.to_string(),
                    limit,
                });
            }
            Ok(data)
        } else {
            self.vfs.read_limited(cwd, path, limit)
        }
    }

    /// Inspect generated or persistent filesystem metadata through one capability boundary.
    pub fn fs_metadata(
        &self,
        cwd: &str,
        path: &str,
        follow: bool,
    ) -> crate::vfs::Result<crate::vfs::Node> {
        crate::pseudo_fs::metadata(self, cwd, path, follow)
            .unwrap_or_else(|| self.vfs.metadata(cwd, path, follow))
    }

    /// List a generated or persistent directory.
    pub fn fs_list_dir(&self, cwd: &str, path: &str) -> crate::vfs::Result<Vec<String>> {
        crate::pseudo_fs::list_dir(self, cwd, path).unwrap_or_else(|| self.vfs.list_dir(cwd, path))
    }

    /// Read a generated or persistent symbolic link.
    pub fn fs_read_link(&self, cwd: &str, path: &str) -> crate::vfs::Result<String> {
        crate::pseudo_fs::read_link(self, cwd, path)
            .unwrap_or_else(|| self.vfs.read_link(cwd, path))
    }

    pub fn get_var(&self, name: &str) -> Option<String> {
        match name {
            "?" => Some(self.last_status.to_string()),
            "$" => Some(self.shell_pid.to_string()),
            "PPID" => Some(self.ppid.to_string()),
            "BASHPID" => Some(self.pid.to_string()),
            "#" => Some(self.positional.len().to_string()),
            "PWD" => Some(self.cwd.clone()),
            "@" | "*" => Some(self.positional.join(" ")),
            _ => {
                if let Ok(n) = name.parse::<usize>() {
                    if n == 0 {
                        return Some("shellsim".to_string());
                    }
                    return self.positional.get(n - 1).cloned();
                }
                // A bare reference to an array name yields element 0 (`$arr` == `${arr[0]}`).
                if self.arrays.contains_key(name) {
                    return Some(self.array_get(name, "0").unwrap_or_default());
                }
                self.vars.get(name).cloned()
            }
        }
    }

    pub fn set_var(&mut self, name: &str, val: impl Into<String>) {
        let val = val.into();
        if name == "PWD" {
            self.cwd = val.clone();
        }
        // A plain scalar assignment to an array name in bash sets element 0; we instead treat it
        // as a fresh scalar (drop the array) — the common case in our scripts and lower-risk.
        self.arrays.remove(name);
        self.vars.insert(name.to_string(), val);
    }

    pub fn export(&mut self, name: &str) {
        self.exported.insert(name.to_string());
    }

    // ===================== arrays =====================

    pub fn is_array(&self, name: &str) -> bool {
        self.arrays.contains_key(name)
    }

    /// Ensure `name` exists as an *indexed* array. If it was a plain scalar, bash promotes it
    /// so that the old scalar becomes element 0 (`x=1; x[2]=3` ⇒ x[0]=1).
    fn ensure_indexed(&mut self, name: &str) {
        if !self.arrays.contains_key(name) {
            let mut v: Vec<Option<String>> = Vec::new();
            if let Some(s) = self.vars.get(name).cloned() {
                v.push(Some(s));
            }
            self.arrays.insert(name.to_string(), ArrayVal::Indexed(v));
        }
    }

    /// Ensure `name` exists as an *associative* array (created empty if absent).
    pub fn declare_assoc(&mut self, name: &str) {
        if !matches!(self.arrays.get(name), Some(ArrayVal::Assoc(_))) {
            self.arrays
                .insert(name.to_string(), ArrayVal::Assoc(BTreeMap::new()));
        }
    }

    /// Ensure `name` exists as an indexed array (created empty if absent).
    pub fn declare_indexed(&mut self, name: &str) {
        self.ensure_indexed(name);
    }

    /// Replace the whole array at `name` with the given element list (indexed).
    pub fn set_array(&mut self, name: &str, elems: Vec<String>) {
        self.vars.remove(name);
        self.arrays.insert(
            name.to_string(),
            ArrayVal::Indexed(elems.into_iter().map(Some).collect()),
        );
    }

    /// Append elements to the end of an indexed array (creating/promoting as needed). For an
    /// associative array, callers should use `array_set` per key; this is indexed-only.
    pub fn array_append(&mut self, name: &str, elems: Vec<String>) {
        self.ensure_indexed(name);
        if let Some(ArrayVal::Indexed(v)) = self.arrays.get_mut(name) {
            for e in elems {
                v.push(Some(e));
            }
        }
    }

    /// Assign `value` to a subscript. For an associative array `key` is the literal key; for an
    /// indexed array `key` is parsed as an integer index (sparse — gaps become `None`).
    pub fn array_set(&mut self, name: &str, key: &str, value: String) {
        match self.arrays.get_mut(name) {
            Some(ArrayVal::Assoc(m)) => {
                m.insert(key.to_string(), value);
            }
            _ => {
                self.ensure_indexed(name);
                if let Some(ArrayVal::Indexed(v)) = self.arrays.get_mut(name) {
                    let idx = key.trim().parse::<usize>().unwrap_or(0);
                    if idx >= v.len() {
                        v.resize(idx + 1, None);
                    }
                    v[idx] = Some(value);
                }
            }
        }
    }

    /// Look up one subscript value.
    pub fn array_get(&self, name: &str, key: &str) -> Option<String> {
        match self.arrays.get(name) {
            Some(ArrayVal::Assoc(m)) => m.get(key).cloned(),
            Some(ArrayVal::Indexed(v)) => {
                let idx = key.trim().parse::<usize>().ok()?;
                v.get(idx).and_then(|o| o.clone())
            }
            None => None,
        }
    }

    /// All set values in order (`${arr[@]}` / `${arr[*]}`).
    pub fn array_all(&self, name: &str) -> Vec<String> {
        match self.arrays.get(name) {
            Some(ArrayVal::Assoc(m)) => m.values().cloned().collect(),
            Some(ArrayVal::Indexed(v)) => v.iter().filter_map(|o| o.clone()).collect(),
            None => Vec::new(),
        }
    }

    /// Keys/indices of set elements (`${!arr[@]}`).
    pub fn array_keys(&self, name: &str) -> Vec<String> {
        match self.arrays.get(name) {
            Some(ArrayVal::Assoc(m)) => m.keys().cloned().collect(),
            Some(ArrayVal::Indexed(v)) => v
                .iter()
                .enumerate()
                .filter_map(|(i, o)| o.as_ref().map(|_| i.to_string()))
                .collect(),
            None => Vec::new(),
        }
    }

    /// Count of set elements (`${#arr[@]}`).
    pub fn array_len(&self, name: &str) -> usize {
        match self.arrays.get(name) {
            Some(ArrayVal::Assoc(m)) => m.len(),
            Some(ArrayVal::Indexed(v)) => v.iter().filter(|o| o.is_some()).count(),
            None => 0,
        }
    }

    /// Unset one subscript (an element); returns true if the name remained an array.
    pub fn array_unset_elem(&mut self, name: &str, key: &str) {
        match self.arrays.get_mut(name) {
            Some(ArrayVal::Assoc(m)) => {
                m.remove(key);
            }
            Some(ArrayVal::Indexed(v)) => {
                if let Ok(idx) = key.trim().parse::<usize>() {
                    if idx < v.len() {
                        v[idx] = None;
                    }
                }
            }
            None => {}
        }
    }

    /// Environment map visible to a child process (e.g. the python engine).
    pub fn child_env(&self) -> HashMap<String, String> {
        let mut env = HashMap::new();
        for k in &self.exported {
            if let Some(v) = self.vars.get(k) {
                env.insert(k.clone(), v.clone());
            }
        }
        env.insert("PWD".to_string(), self.cwd.clone());
        env
    }

    pub fn new_job(&mut self, pid: ProcessId, cmd: String) -> Option<u32> {
        if self.jobs.len() >= crate::process::MAX_PROCESSES {
            return None;
        }
        let id = self.next_job_id;
        self.next_job_id = self.next_job_id.checked_add(1)?;
        self.jobs.push(Job {
            id,
            pid,
            cmd,
            done: false,
            status: 0,
        });
        Some(id)
    }

    pub(crate) fn next_temp_id(&mut self) -> Option<u64> {
        let id = self.next_temp_id;
        self.next_temp_id = self.next_temp_id.checked_add(1)?;
        Some(id)
    }

    pub fn note_unsupported(&mut self, what: &str) {
        self.unsupported.push(what.to_string());
    }

    pub fn outcome(&self, exit_status: i32) -> RunOutcome {
        self.resources
            .outcome(exit_status, self.vfs.disk_used(), self.vfs.disk_peak())
    }

    /// Whether this persistent shell can accept another action.
    pub fn is_terminated(&self) -> bool {
        self.resources.is_stopped() || self.exiting.is_some()
    }

    /// Whether subsequent session actions are currently interpreted by the minimal Python REPL.
    pub fn in_python_repl(&self) -> bool {
        self.python_repl.is_some()
    }

    /// Sticky terminal status after `exit`, `set -e`, or resource exhaustion.
    pub fn termination_status(&self) -> Option<i32> {
        self.resources
            .stop_reason()
            .map(crate::resources::StopReason::exit_status)
            .or(self.exiting)
    }
}

impl Default for Environment {
    fn default() -> Self {
        Self::new()
    }
}

/// Compatibility name for callers written against the original shell simulator API.
pub type Interp = Environment;
