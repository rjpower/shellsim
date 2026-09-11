//! Interpreter state shared across the shell executor and all commands.

use std::collections::{BTreeMap, HashMap};
use std::ops::{Deref, DerefMut};

use crate::clock::{Clock, EventId};
use crate::net::VirtualNet;
use crate::resources::{Limits, Resources, RunOutcome};
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
pub struct Job {
    pub id: u32,
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
    pub process: ProcessState,
}

/// Shell-local state for the single process currently executing in an [`Environment`].
pub struct ProcessState {
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
    /// Deterministic identity source kept separate from virtual time.
    next_temp_id: u64,
    /// recursion / loop-control signaling
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
    /// trace of every external command name executed (telemetry for "what did we cover")
    pub cmd_trace: Vec<String>,
    /// commands requested that we don't implement (coverage gaps)
    pub unsupported: Vec<String>,
    /// command names that ran as `Trust::NoOp` (ignored — e.g. apt-get) during this run
    pub trust_noop: std::collections::BTreeSet<String>,
    /// command names that ran as `Trust::Partial` (subset impl — e.g. jq/sed) during this run
    pub trust_partial: std::collections::BTreeSet<String>,
    /// Package names recorded by the lightweight `pip`/`uv`/`conda` compatibility commands.
    pub packages: std::collections::BTreeSet<String>,
    /// A foreground Python REPL, when `python` was invoked without a program. Keeping this in
    /// process state lets an agent enter Python in one shell action and continue it in later
    /// actions without giving the shim access to host stdin.
    pub python_repl: Option<crate::python::ReplState>,
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
        let mut vars = HashMap::new();
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
        let mut vfs = Vfs::with_disk_limit(limits.disk);
        vfs.set_mutation_time(clock.unix_ms());
        vfs.seed_dirs(["/root", "/tmp", "/work"]);
        Environment {
            vfs,
            clock,
            net: VirtualNet::new(),
            resources: Resources::new(limits),
            process: ProcessState {
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
                next_temp_id: 0,
                loop_break: 0,
                loop_continue: 0,
                returning: None,
                exiting: None,
                cond_depth: 0,
                deadline_interrupt: None,
                uid: 0,
                input_stream: Vec::new(),
                input_pos: 0,
                cmd_trace: Vec::new(),
                unsupported: Vec::new(),
                trust_noop: std::collections::BTreeSet::new(),
                trust_partial: std::collections::BTreeSet::new(),
                packages: std::collections::BTreeSet::new(),
                python_repl: None,
            },
        }
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

    pub fn get_var(&self, name: &str) -> Option<String> {
        match name {
            "?" => Some(self.last_status.to_string()),
            "$" => Some("1234".to_string()), // deterministic fake PID
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

    pub fn new_job(&mut self, cmd: String) -> u32 {
        let id = self.next_job_id;
        self.next_job_id += 1;
        self.jobs.push(Job {
            id,
            cmd,
            done: false,
            status: 0,
        });
        id
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
