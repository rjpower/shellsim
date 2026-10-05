//! Safe, deterministic Python 3.14 compatibility for shellsim.
//!
//! The implementation is growing in auditable vertical slices: source is tokenized and parsed,
//! compiled to shellsim's own semantic bytecode, then executed by a metered VM. Unsupported
//! syntax fails explicitly; there is no host-Python or ad-hoc evaluation fallback.

mod ast;
mod attributes;
mod bytecode;
mod compiler;
mod complex;
mod cpython_names;
mod definitions;
mod exception_types;
mod filesystem;
mod float_text;
mod hash;
mod heap;
mod http;
mod lexer;
mod native;
mod number;
mod object_model;
mod parser;
mod process;
mod protocol;
mod pytest_options;
mod scopes;
mod slice;
mod sort;
mod source;
mod stdlib;
mod string;
mod symbols;
mod token;
mod unicode;
mod vm;

use std::collections::HashMap;

use crate::interp::Interp;

use ast::StatementKind;
use heap::Value;

type Out<'a> = &'a mut Vec<u8>;

// Runner input is untrusted VFS data. Keep parser and wrapper construction bounded before any
// source is copied into an aggregate string or handed to the front end.
const MAX_RUNNER_FILES: usize = 128;
const MAX_RUNNER_FILE_BYTES: usize = 256 * 1024;
const MAX_RUNNER_SOURCE_BYTES: usize = 512 * 1024;
const MAX_RUNNER_WRAPPER_BYTES: usize = 1024 * 1024;

/// Return whether an offline package command can supply this third-party distribution.
pub(crate) fn is_bundled_distribution(name: &str) -> bool {
    matches!(name, "numpy" | "pytest" | "pytest_json_ctrf")
}

/// Persistent locals for the deliberately-small foreground Python REPL.
#[derive(Default, Debug)]
pub struct ReplState {
    globals: GlobalBindings,
    heap: heap::Heap,
    symbols: symbols::Symbols,
    shapes: attributes::Shapes,
    types: object_model::TypeRegistry,
    modules: HashMap<String, heap::Ref>,
    import_paths: Vec<String>,
    sys_path: Option<heap::Ref>,
    temporary_import_paths: Vec<String>,
    original_cwd: Option<String>,
    type_memory: u64,
}

impl Clone for ReplState {
    fn clone(&self) -> Self {
        Self {
            globals: self.globals.clone(),
            heap: self.heap.clone(),
            symbols: self.symbols.clone(),
            shapes: self.shapes.clone(),
            types: self.types.clone(),
            modules: self
                .modules
                .iter()
                .map(|(name, module)| (name.clone(), module.dup()))
                .collect(),
            import_paths: self.import_paths.clone(),
            sys_path: self.sys_path.as_ref().map(heap::Ref::dup),
            temporary_import_paths: self.temporary_import_paths.clone(),
            original_cwd: self.original_cwd.clone(),
            type_memory: self.type_memory,
        }
    }
}

/// The flat global table of the entry-point script or REPL, indexed by symbol.
#[derive(Default, Debug)]
struct GlobalBindings {
    values: Vec<Option<heap::Ref>>,
    modeled_bytes: u64,
}

impl Clone for GlobalBindings {
    fn clone(&self) -> Self {
        Self {
            values: self
                .values
                .iter()
                .map(|value| value.as_ref().map(heap::Ref::dup))
                .collect(),
            modeled_bytes: self.modeled_bytes,
        }
    }
}

impl heap::Roots for GlobalBindings {
    fn visit_refs(&mut self, visitor: &mut dyn FnMut(&mut heap::Ref)) {
        for value in self.values.iter_mut().flatten() {
            visitor(value);
        }
    }
}

impl GlobalBindings {
    #[inline(always)]
    fn get<'s>(&self, heap: &heap::Heap, symbol: symbols::SymbolId) -> Option<Value<'s>> {
        heap.handle_optional(self.values.get(symbol.index()).and_then(Option::as_ref))
    }

    /// The stored reference of a binding, for pushing onto the operand stack directly.
    #[inline(always)]
    fn get_ref(&self, symbol: symbols::SymbolId) -> Option<&heap::Ref> {
        self.values.get(symbol.index()).and_then(Option::as_ref)
    }

    #[inline(always)]
    fn insert(
        &mut self,
        heap: &heap::Heap,
        symbol: symbols::SymbolId,
        value: Value<'_>,
        resources: &mut crate::resources::Resources,
    ) -> Result<(), String> {
        if self.values.len() <= symbol.index() {
            self.grow(symbol, resources)?;
        }
        self.values[symbol.index()] = Some(heap.store(value));
        Ok(())
    }

    #[cold]
    #[inline(never)]
    fn grow(
        &mut self,
        symbol: symbols::SymbolId,
        resources: &mut crate::resources::Resources,
    ) -> Result<(), String> {
        let required_len = symbol
            .index()
            .checked_add(1)
            .ok_or("global binding count overflow")?;
        let added = required_len - self.values.len();
        let bytes = u64::try_from(added)
            .unwrap_or(u64::MAX)
            .checked_mul(std::mem::size_of::<heap::Ref>() as u64)
            .ok_or("global binding size overflow")?;
        let modeled_bytes = self
            .modeled_bytes
            .checked_add(bytes)
            .ok_or("modeled global binding size overflow")?;
        if !resources.reserve_memory(bytes) {
            return Err("memory limit exceeded".into());
        }
        self.values.resize_with(required_len, || None);
        self.modeled_bytes = modeled_bytes;
        Ok(())
    }

    fn remove<'s>(&mut self, heap: &heap::Heap, symbol: symbols::SymbolId) -> Option<Value<'s>> {
        let removed = self.values.get_mut(symbol.index()).and_then(Option::take);
        heap.handle_optional(removed.as_ref())
    }

    /// Every populated `(name, value)` binding, resolved through the heap's symbol table, for
    /// `globals()` on the entry-point script or REPL. Slot order here just follows `SymbolId`
    /// allocation order, not Python's per-dict insertion order; callers that need a stable order
    /// sort the result themselves.
    fn entries<'s>(
        &self,
        heap: &heap::Heap,
        symbols: &symbols::Symbols,
    ) -> Vec<(String, Value<'s>)> {
        self.values
            .iter()
            .enumerate()
            .filter_map(|(index, value)| {
                let value = value.as_ref()?;
                let symbol = symbols::SymbolId::from_index(index)?;
                let name = symbols.name(symbol)?.to_string();
                Some((name, heap.handle(value)))
            })
            .collect()
    }

    fn take_modeled_bytes(&mut self) -> u64 {
        std::mem::take(&mut self.modeled_bytes)
    }
}

impl ReplState {
    fn sync_type_memory(&mut self, resources: &mut crate::resources::Resources) -> bool {
        let current = self.types.modeled_bytes();
        if current > self.type_memory {
            let growth = current - self.type_memory;
            if !resources.reserve_memory(growth) {
                return false;
            }
        } else {
            resources.release_memory(self.type_memory - current);
        }
        self.type_memory = current;
        true
    }

    fn release_owned_memory(&mut self, resources: &mut crate::resources::Resources) {
        let heap = self.heap.take_modeled_bytes();
        resources.release_memory(
            heap.saturating_add(self.globals.take_modeled_bytes())
                .saturating_add(self.symbols.take_modeled_bytes())
                .saturating_add(self.shapes.take_modeled_bytes())
                .saturating_add(std::mem::take(&mut self.type_memory)),
        );
    }
}

enum ExecResult {
    Continue,
    Exit(i32),
    Unsupported(String),
}

/// Owned Python command state retained by a shell continuation between scheduler quanta.
#[derive(Clone)]
pub(crate) struct PythonContinuation {
    argv: Vec<String>,
    stdin: Vec<u8>,
    state: ReplState,
    program: vm::VmProgram,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    original_cwd: String,
}

/// Result of parsing and starting a Python command.
pub(crate) enum PythonCommandStart {
    Ready(i32),
    Running(Box<PythonContinuation>),
}

pub(crate) enum PythonPoll {
    Runnable,
    Blocked(crate::scheduler::WaitReason),
    Ready(i32),
}

impl PythonContinuation {
    /// Run one bounded VM quantum. Completed output remains owned until the shell installs its
    /// ordinary descriptor-write frames.
    pub(crate) fn poll(&mut self, interp: &mut Interp) -> PythonPoll {
        self.poll_with_mode(interp, vm::VmMode::scheduled())
    }

    fn poll_synchronous(&mut self, interp: &mut Interp) -> PythonPoll {
        self.poll_with_mode(interp, vm::VmMode::synchronous(false))
    }

    fn poll_with_mode(&mut self, interp: &mut Interp, mode: vm::VmMode) -> PythonPoll {
        let result = match self.program.poll(
            interp,
            vm::ProcessInput {
                argv: &self.argv,
                stdin: &self.stdin,
            },
            &mut self.state,
            mode,
            &mut self.stdout,
            &mut self.stderr,
        ) {
            vm::VmPoll::Runnable => return PythonPoll::Runnable,
            vm::VmPoll::Blocked(reason) => return PythonPoll::Blocked(reason),
            vm::VmPoll::Ready(result) => result,
        };
        if interp.cwd != self.original_cwd {
            interp.set_var("PWD", self.original_cwd.clone());
        }
        let status = match result {
            ExecResult::Continue => 0,
            ExecResult::Exit(status) => status,
            ExecResult::Unsupported(feature) => unsupported(interp, &feature, &mut self.stderr),
        };
        self.state.release_owned_memory(&mut interp.resources);
        PythonPoll::Ready(status)
    }

    pub(crate) fn into_output(self) -> (Vec<u8>, Vec<u8>) {
        (self.stdout, self.stderr)
    }
}

pub fn run_python(interp: &mut Interp, argv: &[String], stdin: Vec<u8>, out: Out, err: Out) -> i32 {
    match start_python(interp, argv, stdin, out, err) {
        PythonCommandStart::Ready(status) => status,
        PythonCommandStart::Running(mut continuation) => loop {
            match continuation.poll_synchronous(interp) {
                PythonPoll::Runnable => {}
                PythonPoll::Blocked(_) => unreachable!("synchronous Python cannot suspend"),
                PythonPoll::Ready(status) => {
                    let (stdout, stderr) = (*continuation).into_output();
                    out.extend_from_slice(&stdout);
                    err.extend_from_slice(&stderr);
                    return status;
                }
            }
        },
    }
}

/// Prepare a Python command without borrowing the environment across execution quanta.
pub(crate) fn start_python(
    interp: &mut Interp,
    argv: &[String],
    stdin: Vec<u8>,
    out: Out,
    err: Out,
) -> PythonCommandStart {
    let args = argv.get(1..).unwrap_or_default();
    if args
        .first()
        .is_some_and(|arg| arg == "--version" || arg == "-V")
    {
        out.extend_from_slice(b"Python 3.14.0\n");
        return PythonCommandStart::Ready(0);
    }

    if args.first().map(String::as_str) == Some("-m") {
        return PythonCommandStart::Ready(run_module(interp, &args[1..], out, err));
    }

    let (source, py_argv, execution_stdin) = if args.first().map(String::as_str) == Some("-c") {
        let Some(source) = args.get(1).cloned() else {
            err.extend_from_slice(b"python: argument expected for the -c option\n");
            return PythonCommandStart::Ready(2);
        };
        let mut py_argv = vec!["-c".to_string()];
        py_argv.extend_from_slice(args.get(2..).unwrap_or_default());
        (source, py_argv, stdin)
    } else if args.first().map(String::as_str) == Some("-")
        || (args.is_empty() && !stdin.is_empty())
    {
        let mut py_argv = vec![if args.is_empty() { "" } else { "-" }.to_string()];
        py_argv.extend_from_slice(args.get(1..).unwrap_or_default());
        (
            String::from_utf8_lossy(&stdin).into_owned(),
            py_argv,
            Vec::new(),
        )
    } else if args.is_empty() {
        let mut state = ReplState {
            original_cwd: Some(interp.cwd.clone()),
            ..ReplState::default()
        };
        state.import_paths.push(interp.cwd.clone());
        interp.python_repl = Some(state);
        out.extend_from_slice(
            b"Python 3.14.0 (shellsim)\nType exit() or quit() to return to the shell.\n>>> ",
        );
        return PythonCommandStart::Ready(0);
    } else if args[0].starts_with('-') {
        return PythonCommandStart::Ready(unsupported(interp, &format!("option {}", args[0]), err));
    } else {
        let script = &args[0];
        let source = match interp.vfs.read_string(&interp.cwd, script) {
            Ok(source) => source,
            Err(error) => {
                err.extend_from_slice(
                    format!("python: can't open file {script:?}: {error}\n").as_bytes(),
                );
                return PythonCommandStart::Ready(2);
            }
        };
        let mut py_argv = vec![script.clone()];
        py_argv.extend_from_slice(&args[1..]);
        (source, py_argv, stdin)
    };

    let scratch = 10 * 1024 + source.len() as u64;
    if !interp.resources.reserve_memory(scratch) {
        return PythonCommandStart::Ready(137);
    }
    if !interp.resources.charge_cpu(100 + source.len() as u64) {
        interp.resources.release_memory(scratch);
        return PythonCommandStart::Ready(137);
    }

    let program = vm::VmProgram::compile(&source);
    interp.resources.release_memory(scratch);
    let program = match program {
        Ok(program) => program,
        Err(ExecResult::Unsupported(feature)) => {
            return PythonCommandStart::Ready(unsupported(interp, &feature, err))
        }
        Err(ExecResult::Exit(status)) => return PythonCommandStart::Ready(status),
        Err(ExecResult::Continue) => unreachable!("compilation cannot complete execution"),
    };

    let mut state = ReplState::default();
    let import_root = match py_argv.first().map(String::as_str) {
        Some("-c" | "-" | "") | None => interp.cwd.clone(),
        Some(script) => {
            let path = crate::vfs::resolve_against(&interp.cwd, script);
            path.rsplit_once('/').map_or_else(
                || "/".to_string(),
                |(parent, _)| {
                    if parent.is_empty() {
                        "/".to_string()
                    } else {
                        parent.to_string()
                    }
                },
            )
        }
    };
    state.import_paths.push(import_root);
    let name_symbol = match state.symbols.intern("__name__", &mut interp.resources) {
        Ok(symbol) => symbol,
        Err(_) => return PythonCommandStart::Ready(137),
    };
    if state
        .globals
        .insert(
            &state.heap,
            name_symbol,
            Value::inline_string("__main__").expect("short builtin string"),
            &mut interp.resources,
        )
        .is_err()
    {
        return PythonCommandStart::Ready(137);
    }
    if let Some(script) = py_argv
        .first()
        .filter(|script| !matches!(script.as_str(), "-c" | "-" | ""))
    {
        // No VM scope exists yet, so release the handle once the global table holds the value.
        let handle_base = state.heap.handle_count();
        let file = match Value::inline_string(script) {
            Some(value) => value,
            // The fresh state's only references are its globals.
            None => match state.heap.alloc(
                heap::Object::String(script.clone().into()),
                &mut state.globals,
                &mut interp.resources,
            ) {
                Ok(value) => value,
                Err(_) => return PythonCommandStart::Ready(137),
            },
        };
        let file_symbol = match state.symbols.intern("__file__", &mut interp.resources) {
            Ok(symbol) => symbol,
            Err(_) => return PythonCommandStart::Ready(137),
        };
        let inserted = state
            .globals
            .insert(&state.heap, file_symbol, file, &mut interp.resources);
        state.heap.truncate_handles(handle_base);
        if inserted.is_err() {
            return PythonCommandStart::Ready(137);
        }
    }
    PythonCommandStart::Running(Box::new(PythonContinuation {
        argv: py_argv,
        stdin: execution_stdin,
        state,
        program,
        stdout: Vec::new(),
        stderr: Vec::new(),
        original_cwd: interp.cwd.clone(),
    }))
}

/// Execute one action while Python owns the foreground session. The caller temporarily removes
/// `state` from the environment to avoid aliasing it with `interp`.
pub fn run_repl_line(
    interp: &mut Interp,
    state: &mut ReplState,
    source: &str,
    out: Out,
    err: Out,
) -> (i32, bool) {
    let scratch = 10 * 1024 + source.len() as u64;
    if !interp.resources.reserve_memory(scratch) {
        return (137, false);
    }
    if !interp.resources.charge_cpu(100 + source.len() as u64) {
        interp.resources.release_memory(scratch);
        return (137, false);
    }

    let result = execute_source(
        interp,
        source,
        &["<stdin>".to_string()],
        state,
        true,
        out,
        err,
    );
    let (status, stay) = match result {
        ExecResult::Continue => (0, true),
        ExecResult::Exit(status) => (status, false),
        ExecResult::Unsupported(feature) => {
            interp.note_unsupported(&format!("python:{feature}"));
            err.extend_from_slice(
                format!("python: unsupported by minimal REPL: {feature}\n").as_bytes(),
            );
            (2, true)
        }
    };
    interp.resources.release_memory(scratch);
    if stay {
        out.extend_from_slice(b">>> ");
    } else {
        if let Some(original_cwd) = state.original_cwd.take() {
            interp.set_var("PWD", original_cwd);
        }
        state.release_owned_memory(&mut interp.resources);
    }
    (status, stay)
}

fn execute_source(
    interp: &mut Interp,
    source: &str,
    argv: &[String],
    state: &mut ReplState,
    interactive: bool,
    out: Out,
    err: Out,
) -> ExecResult {
    vm::execute(
        interp,
        source,
        argv,
        state,
        interactive,
        &mut *out,
        &mut *err,
    )
}

fn run_module(interp: &mut Interp, args: &[String], out: Out, err: Out) -> i32 {
    match args.first().map(String::as_str) {
        Some("pip") => {
            let mut io = crate::commands::Io {
                stdin: Vec::new(),
                out,
                err,
            };
            crate::commands::pkg::run_pip(interp, &args[1..], &mut io)
        }
        Some("venv") => {
            let Some(dir) = args.iter().skip(1).find(|arg| !arg.starts_with('-')) else {
                return 1;
            };
            if !crate::commands::pkg::ensure_venv(interp, dir, false) {
                err.extend_from_slice(b"python: venv: No space left on device\n");
                return 1;
            }
            0
        }
        Some("pytest") => run_pytest(interp, &args[1..], out, err),
        Some("unittest") => run_unittest(interp, &args[1..], out, err),
        Some(module) => unsupported(interp, &format!("module {module}"), err),
        None => 1,
    }
}

/// Run the deliberately small, VFS-only pytest compatibility slice.  The collector and wrapper
/// share this interpreter's parser, compiler, exception machinery, and resource limits.
pub fn run_pytest(interp: &mut Interp, args: &[String], out: Out, err: Out) -> i32 {
    let options = match pytest_options::parse(args) {
        Ok(options) => options,
        Err(message) => {
            err.extend_from_slice(format!("pytest: {message}\n").as_bytes());
            return 2;
        }
    };
    let paths = match pytest_options::discover(interp, &options.paths) {
        Ok(paths) => paths,
        Err(message) => {
            err.extend_from_slice(format!("pytest: {message}\n").as_bytes());
            return if interp.resources.is_stopped() {
                resource_exit_status(interp)
            } else {
                2
            };
        }
    };
    if paths.len() > MAX_RUNNER_FILES {
        return runner_limit(err, "pytest", "file count", MAX_RUNNER_FILES);
    }
    if paths.is_empty() {
        err.extend_from_slice(b"pytest: no tests collected\n");
        return 5;
    }

    let mut collected = 0usize;
    let mut source_bytes = 0usize;
    let mut sources = Vec::new();
    let mut collection_errors = 0usize;
    for path in &paths {
        let source = match interp
            .vfs
            .read_string_limited(&interp.cwd, path, MAX_RUNNER_FILE_BYTES)
        {
            Ok(source) => source,
            Err(error) => {
                err.extend_from_slice(format!("pytest: {path:?}: {error}\n").as_bytes());
                return 2;
            }
        };
        source_bytes = match source_bytes.checked_add(source.len()) {
            Some(bytes) if bytes <= MAX_RUNNER_SOURCE_BYTES => bytes,
            _ => return runner_limit(err, "pytest", "combined source", MAX_RUNNER_SOURCE_BYTES),
        };
        if !meter_runner_stage(interp, source.len()) {
            return resource_exit_status(interp);
        }
        let collection = match collect_pytest_functions(&source) {
            Ok(collection) => collection,
            Err(error) => {
                err.extend_from_slice(format!("pytest: {path}: {error}\n").as_bytes());
                if options.continue_collection {
                    collection_errors += 1;
                    continue;
                }
                return 2;
            }
        };
        collected += collection.tests.len();
        sources.push((path, source, collection));
    }
    if collected == 0 {
        err.extend_from_slice(b"pytest: no tests collected\n");
        return if collection_errors > 0 { 1 } else { 5 };
    }

    // The wrapper is compiled by the same parser/compiler/VM as ordinary Python.  This keeps
    // collection source-driven while preserving VM exception handling for each test item.
    let mut wrapper = String::new();
    if let Err(kind) = append_runner_piece(
        interp,
        &mut wrapper,
        "__shellsim_pytest_failed = 0\n__shellsim_pytest_total = 0\nfrom pathlib import Path as __ShellsimPath\nfrom pytest import _parametrized_cases as __shellsim_pytest_cases\n",
    ) {
        return append_failure(interp, err, "pytest", kind);
    }
    let setup = format!("from _pytest import _set_timeout as __shellsim_set_timeout\nimport warnings as __shellsim_warnings\n__shellsim_saved_filters = __shellsim_warnings.filters[:]\n__shellsim_pytest_failed = {collection_errors}\n{}", options.warning_filters.concat());
    if let Err(kind) = append_runner_piece(interp, &mut wrapper, &setup) {
        return append_failure(interp, err, "pytest", kind);
    }
    for (path, source, collection) in sources {
        let source = if options.continue_collection {
            format!("__shellsim_collection_ok = True\ntry:\n{}\nexcept Exception as error:\n    __shellsim_collection_ok = False\n    __shellsim_pytest_failed += 1\n    print({:?}, 'ERROR', error)\n", source.lines().map(|line| format!("    {line}\n")).collect::<String>(), path)
        } else {
            source
        };
        if let Err(kind) = append_runner_piece(interp, &mut wrapper, &source) {
            return append_failure(interp, err, "pytest", kind);
        }
        if !source.ends_with('\n') {
            if let Err(kind) = append_runner_piece(interp, &mut wrapper, "\n") {
                return append_failure(interp, err, "pytest", kind);
            }
        }
        for test in &collection.tests {
            let item = match build_pytest_item(path, test, &collection.fixtures, options.timeout) {
                Ok(item) => item,
                Err(error) => {
                    err.extend_from_slice(format!("pytest: {path}: {error}\n").as_bytes());
                    return 2;
                }
            };
            let item = if options.continue_collection {
                format!(
                    "if __shellsim_collection_ok:\n{}",
                    item.lines()
                        .map(|line| format!("    {line}\n"))
                        .collect::<String>()
                )
            } else {
                item
            };
            if let Err(kind) = append_runner_piece(interp, &mut wrapper, &item) {
                return append_failure(interp, err, "pytest", kind);
            }
        }
    }
    let summary =
        "__shellsim_warnings.filters[:] = __shellsim_saved_filters\nprint('__SHELLSIM_PYTEST_SUMMARY__', __shellsim_pytest_failed, __shellsim_pytest_total)\n";
    if let Err(kind) = append_runner_piece(interp, &mut wrapper, summary) {
        return append_failure(interp, err, "pytest", kind);
    }

    let mut python_out = Vec::new();
    let status = run_python(
        interp,
        &["python3.14".into(), "-c".into(), wrapper],
        Vec::new(),
        &mut python_out,
        err,
    );
    if status != 0 {
        out.extend_from_slice(&python_out);
        return status;
    }
    let marker = b"__SHELLSIM_PYTEST_SUMMARY__ ";
    let Some(marker_start) = python_out
        .windows(marker.len())
        .rposition(|window| window == marker)
    else {
        out.extend_from_slice(&python_out);
        err.extend_from_slice(b"pytest: runner did not produce a summary\n");
        return 2;
    };
    let summary_end = python_out[marker_start..]
        .iter()
        .position(|byte| *byte == b'\n')
        .map_or(python_out.len(), |offset| marker_start + offset + 1);
    let summary = String::from_utf8_lossy(&python_out[marker_start..summary_end]);
    let mut fields = summary.split_whitespace();
    let _marker = fields.next();
    let failed = fields.next().and_then(|value| value.parse::<usize>().ok());
    let total = fields.next().and_then(|value| value.parse::<usize>().ok());
    out.extend_from_slice(&python_out[..marker_start]);
    let (Some(failed), Some(total)) = (failed, total) else {
        err.extend_from_slice(b"pytest: runner produced an invalid summary\n");
        return 2;
    };
    if let Some(path) = options.ctrf {
        if let Err(error) = write_ctrf(interp, &path, total, failed) {
            err.extend_from_slice(
                format!("pytest: cannot write CTRF report: {error}\n").as_bytes(),
            );
            return 2;
        }
    }
    if failed != 0 {
        return 1;
    }
    0
}

fn write_ctrf(
    interp: &mut Interp,
    path: &str,
    total: usize,
    failed: usize,
) -> crate::vfs::Result<()> {
    let path = crate::vfs::resolve_against(&interp.cwd, path);
    if let Some((parent, _)) = path.rsplit_once('/') {
        interp
            .vfs
            .mkdir_all("/", if parent.is_empty() { "/" } else { parent })?;
    }
    let passed = total.saturating_sub(failed);
    let report = format!(
        "{{\"results\":{{\"tool\":{{\"name\":\"pytest\"}},\"summary\":{{\"tests\":{total},\"passed\":{passed},\"failed\":{failed},\"skipped\":0,\"pending\":0,\"other\":0}},\"tests\":[]}}}}\n"
    );
    interp.sync_vfs_time();
    interp.vfs.put_file(&path, report.into_bytes(), 0o644)
}

/// Run the deliberately small, VFS-only unittest compatibility slice.  Discovery is limited to
/// classes whose direct base is ``unittest.TestCase`` and zero-argument test methods (``self`` is
/// implicit).  The generated driver is ordinary Python source, so setup, teardown, assertions,
/// and exception handling still execute through the same parser/compiler/VM as user code.
pub fn run_unittest(interp: &mut Interp, args: &[String], out: Out, err: Out) -> i32 {
    if args.iter().any(|arg| arg.starts_with('-')) {
        let option = args
            .iter()
            .find(|arg| arg.starts_with('-'))
            .expect("option was found");
        return unsupported(interp, &format!("unittest option {option}"), err);
    }
    let Some(path) = args.first() else {
        err.extend_from_slice(b"unittest: an explicit VFS test file is required\n");
        return 2;
    };
    if args.len() != 1 {
        return unsupported(interp, "unittest accepts one explicit VFS test file", err);
    }
    let source = match interp
        .vfs
        .read_string_limited(&interp.cwd, path, MAX_RUNNER_FILE_BYTES)
    {
        Ok(source) => source,
        Err(error) => {
            err.extend_from_slice(format!("unittest: {path:?}: {error}\n").as_bytes());
            return 2;
        }
    };
    if source.len() > MAX_RUNNER_SOURCE_BYTES {
        return runner_limit(err, "unittest", "combined source", MAX_RUNNER_SOURCE_BYTES);
    }
    if !meter_runner_stage(interp, source.len()) {
        return resource_exit_status(interp);
    }
    let classes = match collect_unittest_classes(&source) {
        Ok(classes) => classes,
        Err(error) => {
            err.extend_from_slice(format!("unittest: {path}: {error}\n").as_bytes());
            return 2;
        }
    };
    let total = classes.iter().map(|class| class.tests.len()).sum::<usize>();
    if total == 0 {
        err.extend_from_slice(b"unittest: no tests collected\n");
        return 5;
    }

    let mut wrapper = String::new();
    if let Err(kind) = append_runner_piece(interp, &mut wrapper, "__shellsim_unittest_failed = 0\n")
    {
        return append_failure(interp, err, "unittest", kind);
    }
    if let Err(kind) = append_runner_piece(interp, &mut wrapper, &source) {
        return append_failure(interp, err, "unittest", kind);
    }
    if !source.ends_with('\n') {
        if let Err(kind) = append_runner_piece(interp, &mut wrapper, "\n") {
            return append_failure(interp, err, "unittest", kind);
        }
    }
    for class in classes {
        for test in class.tests {
            let label = format!("{path}::{class_name}.{test}", class_name = class.name);
            let mut item = format!(
                "try:\n    __shellsim_unittest_case = {}()\n    try:\n",
                class.name
            );
            if class.setup {
                item.push_str("        __shellsim_unittest_case.setUp()\n");
            }
            item.push_str(&format!("        __shellsim_unittest_case.{test}()\n"));
            item.push_str("    finally:\n");
            if class.teardown {
                item.push_str("        __shellsim_unittest_case.tearDown()\n");
            } else {
                item.push_str("        pass\n");
            }
            item.push_str(&format!("    print({label:?}, 'ok')\n"));
            item.push_str("except Exception as error:\n");
            item.push_str(&format!("    print({label:?}, 'FAIL', error)\n"));
            item.push_str("    __shellsim_unittest_failed += 1\n");
            if let Err(kind) = append_runner_piece(interp, &mut wrapper, &item) {
                return append_failure(interp, err, "unittest", kind);
            }
        }
    }
    let trailer = format!(
        "print('Ran', {total}, 'tests')\nif __shellsim_unittest_failed == 0:\n    print('OK')\nelse:\n    print('FAILED', __shellsim_unittest_failed)\n"
    );
    if let Err(kind) = append_runner_piece(interp, &mut wrapper, &trailer) {
        return append_failure(interp, err, "unittest", kind);
    }

    let mut python_out = Vec::new();
    let status = run_python(
        interp,
        &["python3.14".into(), "-c".into(), wrapper],
        Vec::new(),
        &mut python_out,
        err,
    );
    out.extend_from_slice(&python_out);
    if status != 0 {
        return status;
    }
    let marker = b"FAILED ";
    if python_out
        .windows(marker.len())
        .any(|window| window == marker)
    {
        return 1;
    }
    0
}

struct UnitTestClass {
    name: String,
    tests: Vec<String>,
    setup: bool,
    teardown: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunnerAppendError {
    Limit,
    Resource,
}

fn runner_limit(err: Out, runner: &str, what: &str, limit: usize) -> i32 {
    err.extend_from_slice(
        format!("{runner}: {what} exceeds the safety limit ({limit})\n").as_bytes(),
    );
    2
}

fn resource_exit_status(interp: &Interp) -> i32 {
    interp
        .resources
        .stop_reason()
        .map_or(137, |reason| reason.exit_status())
}

/// Account for front-end work before lexing/parsing a collected file. The multiplier models the
/// token/AST copies without relying on the host allocator as a safety boundary.
fn meter_runner_stage(interp: &mut Interp, bytes: usize) -> bool {
    let bytes = bytes as u64;
    interp.resources.charge_cpu(100u64.saturating_add(bytes))
        && interp
            .resources
            .reserve_memory(4096u64.saturating_add(bytes.saturating_mul(2)))
}

/// Append one bounded wrapper fragment, charging both construction CPU and its additional working
/// memory before `String` copies the fragment.
fn append_runner_piece(
    interp: &mut Interp,
    wrapper: &mut String,
    piece: &str,
) -> Result<(), RunnerAppendError> {
    let Some(next_len) = wrapper.len().checked_add(piece.len()) else {
        return Err(RunnerAppendError::Limit);
    };
    if next_len > MAX_RUNNER_WRAPPER_BYTES {
        return Err(RunnerAppendError::Limit);
    }
    let bytes = piece.len() as u64;
    if !interp.resources.charge_cpu(bytes)
        || !interp.resources.reserve_memory(bytes.saturating_mul(2))
    {
        return Err(RunnerAppendError::Resource);
    }
    wrapper.push_str(piece);
    Ok(())
}

fn append_failure(interp: &Interp, err: Out, runner: &str, failure: RunnerAppendError) -> i32 {
    match failure {
        RunnerAppendError::Limit => {
            runner_limit(err, runner, "generated wrapper", MAX_RUNNER_WRAPPER_BYTES)
        }
        RunnerAppendError::Resource => resource_exit_status(interp),
    }
}

fn collect_unittest_classes(source: &str) -> Result<Vec<UnitTestClass>, String> {
    let tokens = lexer::lex(source).map_err(|error| {
        format!(
            "{} at line {}, column {}",
            error.message, error.span.line, error.span.column
        )
    })?;
    let program = parser::parse(tokens).map_err(|error| {
        format!(
            "{} at line {}, column {}",
            error.message, error.span.line, error.span.column
        )
    })?;
    let mut classes = Vec::new();
    for statement in program.statements {
        let StatementKind::Class {
            name, bases, body, ..
        } = statement.kind
        else {
            if matches!(statement.kind, StatementKind::Decorated { .. }) {
                return Err("decorators are unsupported by the minimal unittest runner".into());
            }
            continue;
        };
        let is_test_case = bases.len() == 1
            && matches!(
                &bases[0].kind,
                ast::ExpressionKind::Attribute { value, name: base }
                    if base == "TestCase"
                        && matches!(&value.kind, ast::ExpressionKind::Name(module) if module == "unittest")
            );
        if !is_test_case {
            continue;
        }
        let mut tests = Vec::new();
        let mut setup = false;
        let mut teardown = false;
        for member in body {
            let StatementKind::Function {
                name: member_name,
                parameters,
                ..
            } = member.kind
            else {
                if matches!(member.kind, StatementKind::Decorated { .. }) {
                    return Err(format!("decorators are unsupported in test class {name:?}"));
                }
                continue;
            };
            if member_name == "setUpClass"
                || member_name == "tearDownClass"
                || member_name == "load_tests"
            {
                return Err(format!(
                    "unittest class hook {member_name:?} is unsupported"
                ));
            }
            if !member_name.starts_with("test_")
                && member_name != "setUp"
                && member_name != "tearDown"
            {
                continue;
            }
            if parameters.len() != 1 || parameters[0].name != "self" {
                return Err(format!(
                    "method {name}.{member_name} must accept only self; fixtures are unsupported"
                ));
            }
            match member_name.as_str() {
                "setUp" => setup = true,
                "tearDown" => teardown = true,
                _ => tests.push(member_name),
            }
        }
        classes.push(UnitTestClass {
            name,
            tests,
            setup,
            teardown,
        });
    }
    Ok(classes)
}

#[derive(Clone)]
struct PytestFixture {
    name: String,
    parameters: Vec<String>,
    yields: bool,
}

struct PytestFunction {
    name: String,
    parameters: Vec<String>,
    /// Parameters `@pytest.mark.parametrize` supplies; the others name fixtures.
    parametrized: Vec<String>,
    skip: bool,
}

struct PytestCollection {
    tests: Vec<PytestFunction>,
    fixtures: HashMap<String, PytestFixture>,
}

fn collect_pytest_functions(source: &str) -> Result<PytestCollection, String> {
    let tokens = lexer::lex(source).map_err(|error| {
        format!(
            "{} at line {}, column {}",
            error.message, error.span.line, error.span.column
        )
    })?;
    let program = parser::parse(tokens).map_err(|error| {
        format!(
            "{} at line {}, column {}",
            error.message, error.span.line, error.span.column
        )
    })?;
    let mut tests = Vec::new();
    let mut fixtures = HashMap::new();
    for statement in program.statements {
        let (decorators, function) = match statement.kind {
            ast::StatementKind::Function { .. } => (Vec::new(), statement.kind),
            ast::StatementKind::Decorated {
                decorators,
                statement,
            } if matches!(statement.as_ref(), ast::StatementKind::Function { .. }) => {
                (decorators, *statement)
            }
            _ => continue,
        };
        let ast::StatementKind::Function {
            name,
            parameters,
            body,
            ..
        } = function
        else {
            unreachable!()
        };
        let parameter_names = parameters
            .iter()
            .map(|parameter| parameter.name.clone())
            .collect::<Vec<_>>();
        if decorators.iter().any(is_fixture_decorator) {
            fixtures.insert(
                name.clone(),
                PytestFixture {
                    name,
                    parameters: parameter_names,
                    yields: statements_contain_yield(&body),
                },
            );
            continue;
        }
        if !name.starts_with("test_") {
            continue;
        }
        let mut parametrized = Vec::new();
        let mut skip = false;
        for decorator in &decorators {
            skip |= parse_skip_marker(decorator)?;
            parametrized.extend(parametrize_names(decorator)?.into_iter().flatten());
        }
        tests.push(PytestFunction {
            name,
            parameters: parameter_names,
            parametrized,
            skip,
        });
    }
    Ok(PytestCollection { tests, fixtures })
}

fn decorator_path(expression: &ast::Expression) -> Option<String> {
    match &expression.kind {
        ast::ExpressionKind::Name(name) => Some(name.clone()),
        ast::ExpressionKind::Attribute { value, name } => {
            Some(format!("{}.{}", decorator_path(value)?, name))
        }
        _ => None,
    }
}

fn is_fixture_decorator(expression: &ast::Expression) -> bool {
    let target = match &expression.kind {
        ast::ExpressionKind::Call { function, .. } => function.as_ref(),
        _ => expression,
    };
    matches!(
        decorator_path(target).as_deref(),
        Some("fixture" | "pytest.fixture")
    )
}

/// The argument names of a `@pytest.mark.parametrize` decorator, or `None` for other
/// decorators. Only the names are read from source: the facade in `pytest.py` records the
/// evaluated rows on the test function when its module runs, as pytest does.
fn parametrize_names(expression: &ast::Expression) -> Result<Option<Vec<String>>, String> {
    let ast::ExpressionKind::Call {
        function,
        arguments,
    } = &expression.kind
    else {
        return Ok(None);
    };
    if !matches!(
        decorator_path(function).as_deref(),
        Some("parametrize" | "mark.parametrize" | "pytest.mark.parametrize")
    ) {
        return Ok(None);
    }
    let Some(names) = arguments
        .iter()
        .find(|argument| matches!(argument.kind, ast::CallArgumentKind::Positional))
    else {
        return Err("pytest.mark.parametrize requires names and values".into());
    };
    let names = match &names.value.kind {
        ast::ExpressionKind::Constant(ast::Constant::String(names)) => names
            .split(',')
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>(),
        ast::ExpressionKind::List(values) | ast::ExpressionKind::Tuple(values) => values
            .iter()
            .map(|value| match &value.kind {
                ast::ExpressionKind::Constant(ast::Constant::String(name)) => Ok(name.clone()),
                _ => Err("parametrize names must be string literals".to_string()),
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err("parametrize names must be a string or list of strings".into()),
    };
    if names.is_empty() {
        return Err("parametrize names cannot be empty".into());
    }
    Ok(Some(names))
}

fn parse_skip_marker(expression: &ast::Expression) -> Result<bool, String> {
    let (target, arguments) = match &expression.kind {
        ast::ExpressionKind::Call {
            function,
            arguments,
        } => (function.as_ref(), arguments.as_slice()),
        _ => (expression, &[][..]),
    };
    match decorator_path(target).as_deref() {
        Some("mark.skip" | "pytest.mark.skip") => Ok(true),
        Some("mark.skipif" | "pytest.mark.skipif") => {
            let condition = arguments
                .iter()
                .find(|argument| {
                    matches!(
                        &argument.kind,
                        ast::CallArgumentKind::Keyword(name) if name == "condition"
                    )
                })
                .or_else(|| {
                    arguments
                        .iter()
                        .find(|argument| matches!(argument.kind, ast::CallArgumentKind::Positional))
                })
                .ok_or("pytest.mark.skipif requires a condition")?;
            match condition.value.kind {
                ast::ExpressionKind::Constant(ast::Constant::Bool(value)) => Ok(value),
                ast::ExpressionKind::Constant(ast::Constant::Integer(value)) => Ok(value != 0),
                _ => Err("pytest.mark.skipif condition must be a literal in shellsim".into()),
            }
        }
        _ => Ok(false),
    }
}

fn statements_contain_yield(statements: &[ast::Statement]) -> bool {
    statements.iter().any(|statement| match &statement.kind {
        ast::StatementKind::Expression(expression)
        | ast::StatementKind::Return(Some(expression)) => expression_contains_yield(expression),
        ast::StatementKind::Assign { value, .. }
        | ast::StatementKind::AugmentedAssign { value, .. }
        | ast::StatementKind::AnnotatedAssign {
            value: Some(value), ..
        } => expression_contains_yield(value),
        ast::StatementKind::If {
            body, otherwise, ..
        }
        | ast::StatementKind::While {
            body, otherwise, ..
        }
        | ast::StatementKind::For {
            body, otherwise, ..
        } => statements_contain_yield(body) || statements_contain_yield(otherwise),
        ast::StatementKind::Try {
            body,
            handlers,
            otherwise,
            finalbody,
        } => {
            statements_contain_yield(body)
                || handlers
                    .iter()
                    .any(|handler| statements_contain_yield(&handler.body))
                || statements_contain_yield(otherwise)
                || statements_contain_yield(finalbody)
        }
        ast::StatementKind::With { body, .. } => statements_contain_yield(body),
        _ => false,
    })
}

fn expression_contains_yield(expression: &ast::Expression) -> bool {
    matches!(
        expression.kind,
        ast::ExpressionKind::Yield(_) | ast::ExpressionKind::YieldFrom(_)
    )
}

/// The wrapper code that runs every case of one test function. The cases come from
/// `pytest._parametrized_cases` at run time, so parametrized values may be any expression.
fn build_pytest_item(
    path: &str,
    test: &PytestFunction,
    fixtures: &HashMap<String, PytestFixture>,
    timeout: f64,
) -> Result<String, String> {
    let label = if test.parametrized.is_empty() {
        format!("{:?}", format!("{path}::{}", test.name))
    } else {
        format!(
            "{:?} + str(__shellsim_case_index) + \"]\"",
            format!("{path}::{}[", test.name)
        )
    };
    let mut item = format!(
        "for __shellsim_case_index, __shellsim_case in enumerate(__shellsim_pytest_cases({})):\n    __shellsim_pytest_total += 1\n",
        test.name
    );
    if test.skip {
        item.push_str(&format!("    print({label}, 'SKIPPED')\n"));
        return Ok(item);
    }
    let mut setup = String::new();
    let mut teardown = Vec::new();
    let mut cache = HashMap::new();
    let mut active = Vec::new();
    let mut counter = 0usize;
    let mut arguments = Vec::new();
    for parameter in &test.parameters {
        if test.parametrized.contains(parameter) {
            continue;
        }
        let value = resolve_pytest_fixture(
            parameter,
            fixtures,
            &mut cache,
            &mut active,
            &mut counter,
            &mut setup,
            &mut teardown,
        )?;
        arguments.push(format!("{parameter}={value}"));
    }
    arguments.push("**__shellsim_case".into());
    item.push_str(&format!(
        "    __shellsim_set_timeout({timeout})\n    try:\n"
    ));
    for line in setup.lines() {
        item.push_str("        ");
        item.push_str(line);
        item.push('\n');
    }
    item.push_str(&format!(
        "        {}({})\n        print({label}, 'PASSED')\n    except Skipped:\n        print({label}, 'SKIPPED')\n    except Exception as error:\n        print({label}, 'FAILED', error)\n        __shellsim_pytest_failed += 1\n",
        test.name,
        arguments.join(", ")
    ));
    item.push_str("    finally:\n        try:\n            pass\n");
    // As in pytest, teardown resumes each yield fixture past its `yield`; a second `yield` is an
    // error rather than a place to stop.
    for generator in teardown.into_iter().rev() {
        item.push_str(&format!(
            "            for _ in {generator}:\n                raise RuntimeError(\"fixture function has more than one 'yield'\")\n"
        ));
    }
    item.push_str(&format!("        except Exception as error:\n            print({label}, 'ERROR', error)\n            __shellsim_pytest_failed += 1\n        finally:\n            __shellsim_set_timeout(0)\n"));
    Ok(item)
}

#[allow(clippy::too_many_arguments)]
fn resolve_pytest_fixture(
    name: &str,
    fixtures: &HashMap<String, PytestFixture>,
    cache: &mut HashMap<String, String>,
    active: &mut Vec<String>,
    counter: &mut usize,
    setup: &mut String,
    teardown: &mut Vec<String>,
) -> Result<String, String> {
    if let Some(value) = cache.get(name) {
        return Ok(value.clone());
    }
    if name == "tmp_path" || name == "tmpdir" {
        let variable = format!("__shellsim_fixture_{}", *counter);
        *counter += 1;
        setup.push_str(&format!(
            "{variable} = __ShellsimPath(\"/tmp/pytest-\" + str(__shellsim_pytest_total))\n{variable}.mkdir(parents=True, exist_ok=True)\n"
        ));
        cache.insert(name.into(), variable.clone());
        return Ok(variable);
    }
    let fixture = fixtures
        .get(name)
        .ok_or_else(|| format!("fixture {name:?} not found"))?;
    if active.iter().any(|candidate| candidate == name) {
        return Err(format!("recursive fixture dependency involving {name:?}"));
    }
    active.push(name.into());
    let mut arguments = Vec::new();
    for parameter in &fixture.parameters {
        arguments.push(resolve_pytest_fixture(
            parameter, fixtures, cache, active, counter, setup, teardown,
        )?);
    }
    active.pop();
    let variable = format!("__shellsim_fixture_{}", *counter);
    *counter += 1;
    if fixture.yields {
        let generator = format!("{variable}_generator");
        setup.push_str(&format!(
            "{generator} = {}({})\n{variable} = next({generator})\n",
            fixture.name,
            arguments.join(", ")
        ));
        teardown.push(generator);
    } else {
        setup.push_str(&format!(
            "{variable} = {}({})\n",
            fixture.name,
            arguments.join(", ")
        ));
    }
    cache.insert(name.into(), variable.clone());
    Ok(variable)
}

fn unsupported(interp: &mut Interp, feature: &str, err: Out) -> i32 {
    interp.note_unsupported(&format!("python:{feature}"));
    err.extend_from_slice(format!("python: unsupported by minimal shim: {feature}\n").as_bytes());
    2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::{Limits, Resources};

    fn run(source: &str) -> (i32, String, String) {
        let mut env = Interp::new();
        let mut out = Vec::new();
        let mut err = Vec::new();
        let status = run_python(
            &mut env,
            &["python".into(), "-c".into(), source.into()],
            Vec::new(),
            &mut out,
            &mut err,
        );
        (
            status,
            String::from_utf8_lossy(&out).into_owned(),
            String::from_utf8_lossy(&err).into_owned(),
        )
    }

    #[test]
    fn literal_print_and_write() {
        assert_eq!(
            run("import sys; print('hello'); sys.stdout.write(\"!\")"),
            (0, "hello\n!".into(), String::new())
        );
    }

    #[test]
    fn multiline_assignments_and_multiple_print_arguments() {
        assert_eq!(
            run("import sys\nx = 3\nprint('hello', x + 2)"),
            (0, "hello 5\n".into(), String::new())
        );
    }

    #[test]
    fn unsupported_python_fails_loudly() {
        let (status, _, err) = run("breakpoint()");
        assert_eq!(status, 2);
        assert!(err.contains("unsupported"));
    }

    #[test]
    fn starred_assignment_obeys_python_minimum_arity_and_tail_shape() {
        assert_eq!(
            run("head, *tail = [1, 2, 3, 4]\nprint(head, tail)"),
            (0, "1 [2, 3, 4]\n".into(), String::new())
        );
        let (status, _, error) = run("head, *tail, last = [1]\n");
        assert_eq!(status, 1);
        assert!(error
            .ends_with("ValueError: not enough values to unpack (expected at least 2, got 1)\n"));
    }

    #[test]
    fn starred_calls_expand_iterables_in_source_order() {
        assert_eq!(
            run("def join(a, b, c):\n    return a * 100 + b * 10 + c\nargs = [1, 2]\nprint(join(0, *args))"),
            (0, "12\n".into(), String::new())
        );
    }

    #[test]
    fn basic_fstrings_render_expressions_and_escaped_braces() {
        assert_eq!(
            run("name = 'Ada'\nvalue = 7\nprint(f'Hello, {name}: {{{value}}}')"),
            (0, "Hello, Ada: {7}\n".into(), String::new())
        );
    }

    #[test]
    fn global_binding_growth_is_charged_before_mutation() {
        let mut resources = Resources::new(Limits {
            memory: 32,
            ..Limits::unlimited()
        });
        let heap = heap::Heap::default();
        let symbol = symbols::Symbols::default()
            .intern("x", &mut resources)
            .unwrap();
        let mut globals = GlobalBindings::default();

        assert_eq!(
            globals
                .insert(&heap, symbol, Value::Int(1), &mut resources)
                .unwrap_err(),
            "memory limit exceeded"
        );
        assert!(globals.get(&heap, symbol).is_none());
    }
}
