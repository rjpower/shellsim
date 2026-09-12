//! Safe, deterministic Python 3.14 compatibility for shellsim.
//!
//! The implementation is growing in auditable vertical slices: source is tokenized and parsed,
//! compiled to shellsim's own semantic bytecode, then executed by a metered VM. Unsupported
//! syntax fails explicitly; there is no host-Python or ad-hoc evaluation fallback.

mod ast;
mod bytecode;
mod compiler;
mod filesystem;
mod heap;
mod lexer;
mod native;
mod number;
mod object_model;
mod parser;
mod process;
mod protocol;
mod source;
mod stdlib;
mod token;
mod vm;

use std::collections::HashMap;

use crate::interp::Interp;

use ast::StatementKind;

type Out<'a> = &'a mut Vec<u8>;

// Runner input is untrusted VFS data. Keep parser and wrapper construction bounded before any
// source is copied into an aggregate string or handed to the front end.
const MAX_RUNNER_FILES: usize = 128;
const MAX_RUNNER_FILE_BYTES: usize = 256 * 1024;
const MAX_RUNNER_SOURCE_BYTES: usize = 512 * 1024;
const MAX_RUNNER_WRAPPER_BYTES: usize = 1024 * 1024;

/// Physical storage discriminator kept separate from Python's semantic [`object_model::TypeId`].
///
/// Tags describe storage only. Python semantics come from the value's registered `TypeId`, so a
/// short string and a heap string have the same Python type despite different physical tags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum ValueTag {
    SmallString0,
    SmallString1,
    SmallString2,
    SmallString3,
    SmallString4,
    SmallString5,
    SmallString6,
    SmallString7,
    SmallString8,
    SmallString9,
    SmallString10,
    SmallString11,
    SmallString12,
    SmallString13,
    SmallString14,
    SmallString15,
    Int,
    Float,
    Bool,
    None,
    Object,
    Native,
}

/// Compact, copyable Python value used by the VM and native-module ABI.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C)]
struct Value {
    payload: u64,
    aux: [u8; 7],
    tag: ValueTag,
}

impl Value {
    #[allow(non_snake_case)]
    const fn Int(value: i64) -> Self {
        Self {
            payload: value as u64,
            aux: [0; 7],
            tag: ValueTag::Int,
        }
    }

    #[allow(non_snake_case)]
    const fn Float(value: f64) -> Self {
        Self {
            payload: value.to_bits(),
            aux: [0; 7],
            tag: ValueTag::Float,
        }
    }

    #[allow(non_snake_case)]
    const fn Bool(value: bool) -> Self {
        Self {
            payload: value as u64,
            aux: [0; 7],
            tag: ValueTag::Bool,
        }
    }

    #[allow(non_upper_case_globals)]
    const None: Self = Self {
        payload: 0,
        aux: [0; 7],
        tag: ValueTag::None,
    };

    #[allow(non_snake_case)]
    const fn Object(value: heap::ObjectId) -> Self {
        Self {
            payload: value.as_raw() as u64,
            aux: [0; 7],
            tag: ValueTag::Object,
        }
    }

    #[allow(non_snake_case)]
    fn Native(value: vm::NativeValue) -> Self {
        let (payload, native_tag) = value.encode();
        let mut aux = [0; 7];
        aux[0] = native_tag;
        Self {
            payload,
            aux,
            tag: ValueTag::Native,
        }
    }

    fn inline_string(value: &str) -> Option<Self> {
        if value.len() > 15 {
            return None;
        }
        let mut bytes = [0; 15];
        bytes[..value.len()].copy_from_slice(value.as_bytes());
        let mut payload = [0; 8];
        payload.copy_from_slice(&bytes[..8]);
        let mut aux = [0; 7];
        aux.copy_from_slice(&bytes[8..]);
        Some(Self {
            payload: u64::from_ne_bytes(payload),
            aux,
            tag: string_tag(value.len()),
        })
    }

    fn inline_string_value(&self) -> Option<String> {
        let length = self.inline_string_len()?;
        let mut bytes = [0; 15];
        bytes[..8].copy_from_slice(&self.payload.to_ne_bytes());
        bytes[8..].copy_from_slice(&self.aux);
        Some(
            std::str::from_utf8(&bytes[..length])
                .expect("inline strings originate from UTF-8")
                .to_string(),
        )
    }

    const fn inline_string_len(&self) -> Option<usize> {
        let raw = self.tag as u8;
        if raw <= ValueTag::SmallString15 as u8 {
            Some(raw as usize)
        } else {
            None
        }
    }

    const fn tag(&self) -> ValueTag {
        self.tag
    }

    const fn immediate_int(&self) -> Option<i64> {
        match self.tag {
            ValueTag::Int => Some(self.payload as i64),
            ValueTag::Bool => Some(self.payload as i64),
            _ => None,
        }
    }

    const fn float_value(&self) -> Option<f64> {
        match self.tag {
            ValueTag::Float => Some(f64::from_bits(self.payload)),
            _ => None,
        }
    }

    const fn bool_value(&self) -> Option<bool> {
        match self.tag {
            ValueTag::Bool => Some(self.payload != 0),
            _ => None,
        }
    }

    const fn object_id(&self) -> Option<heap::ObjectId> {
        match self.tag {
            ValueTag::Object => Some(heap::ObjectId::from_raw(self.payload as usize)),
            _ => None,
        }
    }

    fn native_value(&self) -> Option<vm::NativeValue> {
        match self.tag {
            ValueTag::Native => Some(vm::NativeValue::decode(self.payload, self.aux[0])),
            _ => None,
        }
    }

    const fn is_none(&self) -> bool {
        matches!(self.tag, ValueTag::None)
    }

    fn as_int(&self) -> Option<i64> {
        if let Some(value) = self.immediate_int() {
            return Some(value);
        }
        if let Some(value) = self.float_value().filter(|value| value.is_finite()) {
            return Some(value as i64);
        }
        self.inline_string_value()?.parse().ok()
    }
}

impl std::fmt::Debug for Value {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(value) = self.inline_string_value() {
            return formatter.debug_tuple("String").field(&value).finish();
        }
        match self.tag {
            ValueTag::Int => formatter
                .debug_tuple("Int")
                .field(&(self.payload as i64))
                .finish(),
            ValueTag::Float => formatter
                .debug_tuple("Float")
                .field(&f64::from_bits(self.payload))
                .finish(),
            ValueTag::Bool => formatter
                .debug_tuple("Bool")
                .field(&(self.payload != 0))
                .finish(),
            ValueTag::None => formatter.write_str("None"),
            ValueTag::Object => formatter
                .debug_tuple("Object")
                .field(&self.object_id())
                .finish(),
            ValueTag::Native => formatter
                .debug_tuple("Native")
                .field(&self.native_value())
                .finish(),
            _ => unreachable!("small strings returned above"),
        }
    }
}

const fn string_tag(length: usize) -> ValueTag {
    match length {
        0 => ValueTag::SmallString0,
        1 => ValueTag::SmallString1,
        2 => ValueTag::SmallString2,
        3 => ValueTag::SmallString3,
        4 => ValueTag::SmallString4,
        5 => ValueTag::SmallString5,
        6 => ValueTag::SmallString6,
        7 => ValueTag::SmallString7,
        8 => ValueTag::SmallString8,
        9 => ValueTag::SmallString9,
        10 => ValueTag::SmallString10,
        11 => ValueTag::SmallString11,
        12 => ValueTag::SmallString12,
        13 => ValueTag::SmallString13,
        14 => ValueTag::SmallString14,
        15 => ValueTag::SmallString15,
        _ => panic!("inline string length exceeds payload"),
    }
}

const _: () = assert!(std::mem::size_of::<Value>() == 16);

#[cfg(test)]
mod value_layout_tests {
    use super::*;

    #[test]
    fn compact_values_are_exactly_sixteen_bytes() {
        assert_eq!(std::mem::size_of::<Value>(), 16);
        assert_eq!(
            Value::inline_string("123456789012345")
                .unwrap()
                .inline_string_len(),
            Some(15)
        );
        assert!(Value::inline_string("1234567890123456").is_none());
    }
}

/// Persistent locals for the deliberately-small foreground Python REPL.
#[derive(Default, Debug)]
pub struct ReplState {
    locals: HashMap<String, Value>,
    heap: heap::Heap,
    types: object_model::TypeRegistry,
    modules: HashMap<String, Value>,
    import_paths: Vec<String>,
}

enum ExecResult {
    Continue,
    Exit(i32),
    Unsupported(String),
}

pub fn run_python(interp: &mut Interp, argv: &[String], stdin: Vec<u8>, out: Out, err: Out) -> i32 {
    let args = argv.get(1..).unwrap_or_default();
    if args
        .first()
        .is_some_and(|arg| arg == "--version" || arg == "-V")
    {
        out.extend_from_slice(b"Python 3.14.0\n");
        return 0;
    }

    if args.first().map(String::as_str) == Some("-m") {
        return run_module(interp, &args[1..], out, err);
    }

    let (source, py_argv) = if args.first().map(String::as_str) == Some("-c") {
        let Some(source) = args.get(1).cloned() else {
            err.extend_from_slice(b"python: argument expected for the -c option\n");
            return 2;
        };
        let mut py_argv = vec!["-c".to_string()];
        py_argv.extend_from_slice(args.get(2..).unwrap_or_default());
        (source, py_argv)
    } else if args.first().map(String::as_str) == Some("-")
        || (args.is_empty() && !stdin.is_empty())
    {
        let mut py_argv = vec![if args.is_empty() { "" } else { "-" }.to_string()];
        py_argv.extend_from_slice(args.get(1..).unwrap_or_default());
        (String::from_utf8_lossy(&stdin).into_owned(), py_argv)
    } else if args.is_empty() {
        let mut state = ReplState::default();
        state.import_paths.push(interp.cwd.clone());
        interp.python_repl = Some(state);
        out.extend_from_slice(
            b"Python 3.14.0 (shellsim)\nType exit() or quit() to return to the shell.\n>>> ",
        );
        return 0;
    } else if args[0].starts_with('-') {
        return unsupported(interp, &format!("option {}", args[0]), err);
    } else {
        let script = &args[0];
        let source = match interp.vfs.read_string(&interp.cwd, script) {
            Ok(source) => source,
            Err(error) => {
                err.extend_from_slice(
                    format!("python: can't open file {script:?}: {error}\n").as_bytes(),
                );
                return 2;
            }
        };
        let mut py_argv = vec![script.clone()];
        py_argv.extend_from_slice(&args[1..]);
        (source, py_argv)
    };

    let scratch = 10 * 1024 + source.len() as u64;
    if !interp.resources.reserve_memory(scratch)
        || !interp.resources.charge_cpu(100 + source.len() as u64)
    {
        return 137;
    }

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
    match execute_source(interp, &source, &py_argv, &mut state, false, out, err) {
        ExecResult::Continue => 0,
        ExecResult::Exit(status) => status,
        ExecResult::Unsupported(feature) => unsupported(interp, &feature, err),
    }
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
            if args.get(1).map(String::as_str) == Some("install") {
                crate::commands::pkg::register_install_args(interp, &args[1..]);
            }
            0
        }
        Some("venv") => {
            let Some(dir) = args.iter().skip(1).find(|arg| !arg.starts_with('-')) else {
                return 1;
            };
            let base = crate::vfs::resolve_against(&interp.cwd, dir);
            if interp.vfs.mkdir_all("/", &format!("{base}/bin")).is_err()
                || interp
                    .vfs
                    .put_file(
                        &format!("{base}/bin/python"),
                        b"#!shellsim-python\n".to_vec(),
                        0o755,
                    )
                    .is_err()
            {
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
    let mut paths = Vec::new();
    for arg in args {
        if arg.starts_with('-') {
            if !matches!(arg.as_str(), "-q" | "-v" | "-rA" | "--tb=short") {
                return unsupported(interp, &format!("pytest option {arg}"), err);
            }
        } else {
            paths.push(arg.clone());
        }
    }
    if paths.len() > MAX_RUNNER_FILES {
        return runner_limit(err, "pytest", "file count", MAX_RUNNER_FILES);
    }
    if paths.is_empty() {
        err.extend_from_slice(b"pytest: an explicit VFS test file is required\n");
        return 2;
    }

    let mut total = 0usize;
    let mut source_bytes = 0usize;
    let mut sources = Vec::new();
    for path in paths {
        let source = match interp
            .vfs
            .read_string_limited(&interp.cwd, &path, MAX_RUNNER_FILE_BYTES)
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
        let tests = match collect_pytest_functions(&source) {
            Ok(tests) => tests,
            Err(error) => {
                err.extend_from_slice(format!("pytest: {path}: {error}\n").as_bytes());
                return 2;
            }
        };
        total += tests.len();
        sources.push((path, source, tests));
    }
    if total == 0 {
        err.extend_from_slice(b"pytest: no tests collected\n");
        return 5;
    }

    // The wrapper is compiled by the same parser/compiler/VM as ordinary Python.  This keeps
    // collection source-driven while preserving VM exception handling for each test item.
    let mut wrapper = String::new();
    if let Err(kind) = append_runner_piece(interp, &mut wrapper, "__shellsim_pytest_failed = 0\n") {
        return append_failure(interp, err, "pytest", kind);
    }
    for (path, source, tests) in sources {
        if let Err(kind) = append_runner_piece(interp, &mut wrapper, &source) {
            return append_failure(interp, err, "pytest", kind);
        }
        if !source.ends_with('\n') {
            if let Err(kind) = append_runner_piece(interp, &mut wrapper, "\n") {
                return append_failure(interp, err, "pytest", kind);
            }
        }
        for name in tests {
            let item = format!(
                "try:\n    {name}()\n    print({:?}, 'PASSED')\nexcept Skipped:\n    print({:?}, 'SKIPPED')\nexcept Exception as error:\n    print({:?}, 'FAILED', error)\n    __shellsim_pytest_failed += 1\n",
                format!("{path}::{name}"),
                format!("{path}::{name}"),
                format!("{path}::{name}"),
            );
            if let Err(kind) = append_runner_piece(interp, &mut wrapper, &item) {
                return append_failure(interp, err, "pytest", kind);
            }
        }
    }
    let summary =
        format!("print('__SHELLSIM_PYTEST_SUMMARY__', __shellsim_pytest_failed, {total})\n");
    if let Err(kind) = append_runner_piece(interp, &mut wrapper, &summary) {
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
    out.extend_from_slice(&python_out[..marker_start]);
    if failed != Some(0) {
        return 1;
    }
    0
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

fn collect_pytest_functions(source: &str) -> Result<Vec<String>, String> {
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
    for statement in program.statements {
        match statement.kind {
            ast::StatementKind::Function {
                name, parameters, ..
            } if name.starts_with("test_") => {
                if !parameters.is_empty() {
                    return Err(format!(
                        "test function {name:?} has parameters; fixtures are unsupported"
                    ));
                }
                tests.push(name);
            }
            ast::StatementKind::Decorated { .. } => {
                return Err("decorators are unsupported by the minimal runner".into())
            }
            _ => {}
        }
    }
    Ok(tests)
}

fn unsupported(interp: &mut Interp, feature: &str, err: Out) -> i32 {
    interp.note_unsupported(&format!("python:{feature}"));
    err.extend_from_slice(format!("python: unsupported by minimal shim: {feature}\n").as_bytes());
    2
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let (status, _, err) = run("print(1 @ 2)");
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
        assert_eq!(status, 2);
        assert!(error.contains("not enough values"));
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
}
