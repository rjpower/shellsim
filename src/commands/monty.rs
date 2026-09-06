//! Opt-in Monty execution against Shellsim's resource meter and virtual files.
//!
//! Monty 0.0.16 is pinned for its custom ResourceTracker API. Its time-check hook
//! charges deterministic fuel, never elapsed host time. No OS calls are serviced.
//! The only external functions are bounded UTF-8 VFS reads and quota-checked writes.

use std::{borrow::Cow, cell::RefCell, rc::Rc};

use ::monty::{
    ExcType, MontyException, MontyObject, MontyRun, PrintWriter, PrintWriterCallback,
    ResourceError, ResourceTracker, RunProgress,
};

use super::{CommandContext, Io};
use crate::interp::Interp;
use crate::vfs::{resolve_against, NodeKind};

const MAX_SOURCE: usize = 256 * 1024;
const MAX_FILE: usize = 1024 * 1024;

#[derive(Clone)]
struct Meter<'a>(Rc<RefCell<&'a mut Interp>>);

impl std::fmt::Debug for Meter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ShellsimMeter")
    }
}

impl ResourceTracker for Meter<'_> {
    fn on_allocate(&self, size: impl FnOnce() -> usize) -> Result<(), ResourceError> {
        self.on_grow(size())
    }

    fn on_grow(&self, bytes: usize) -> Result<(), ResourceError> {
        if self.0.borrow_mut().resources.reserve_memory(bytes as u64) {
            Ok(())
        } else {
            Err(ResourceError::Exception(error("memory limit exceeded")))
        }
    }

    fn on_free(&self, _size: impl FnOnce() -> usize) {
        // Conservatively account cumulative allocation for this invocation. The
        // command dispatcher releases the reservation at the command boundary.
    }

    fn check_time(&self) -> Result<(), ResourceError> {
        if self.0.borrow_mut().resources.charge_cpu(1) {
            Ok(())
        } else {
            Err(ResourceError::Exception(error("CPU exhausted")))
        }
    }

    fn check_recursion_depth(&self, depth: usize) -> Result<(), ResourceError> {
        self.check_time()?;
        if depth >= 100 {
            Err(ResourceError::Recursion { limit: 100, depth })
        } else {
            Ok(())
        }
    }

    fn check_large_result(&self, bytes: usize) -> Result<(), ResourceError> {
        let mut env = self.0.borrow_mut();
        let mark = env.resources.memory_mark();
        let allowed = env.resources.reserve_memory(bytes as u64);
        env.resources.restore_memory(mark);
        if allowed {
            Ok(())
        } else {
            Err(ResourceError::Exception(error("memory limit exceeded")))
        }
    }
}

struct Output<'a, 'b> {
    meter: Meter<'a>,
    out: &'b mut Vec<u8>,
}

impl PrintWriterCallback for Output<'_, '_> {
    fn stdout_write(&mut self, text: Cow<'_, str>) -> Result<(), MontyException> {
        let mut env = self.meter.0.borrow_mut();
        if !env.resources.charge_output(text.len() as u64) {
            return Err(error("output limit exceeded"));
        }
        self.out.extend_from_slice(text.as_bytes());
        Ok(())
    }

    fn stdout_push(&mut self, value: char) -> Result<(), MontyException> {
        let mut bytes = [0; 4];
        self.stdout_write(Cow::Borrowed(value.encode_utf8(&mut bytes)))
    }
}

fn error(message: &str) -> MontyException {
    MontyException::new(ExcType::RuntimeError, Some(message.to_owned()))
}

/// Run `monty -c CODE`, `monty -` (stdin), or a script from the VFS.
/// Each invocation has fresh Python globals; files persist in the shell session.
pub(super) fn run(env: &mut CommandContext<'_>, args: &[String], io: &mut Io<'_>) -> i32 {
    let source = match args {
        [flag, code] if flag == "-c" && code.len() <= MAX_SOURCE => Ok(code.clone()),
        [flag] if flag == "-" && io.stdin.len() <= MAX_SOURCE => {
            String::from_utf8(io.stdin.clone()).map_err(|e| e.to_string())
        }
        [path] if !path.starts_with('-') => env
            .vfs
            .read_string_limited(&env.cwd, path, MAX_SOURCE)
            .map_err(|e| e.to_string()),
        _ => Err("expected monty -c CODE, monty -, or monty SCRIPT (source limit 256 KiB)".into()),
    };
    let source = match source {
        Ok(source) => source,
        Err(message) => {
            io.err
                .extend_from_slice(format!("monty: {message}\n").as_bytes());
            return 2;
        }
    };
    // Bound frontend input and reserve conservative parser/compiler scratch before parsing.
    if !env.resources.charge_cpu(source.len() as u64)
        || !env.resources.reserve_memory(
            (source.len() as u64)
                .saturating_mul(32)
                .saturating_add(16384),
        )
    {
        return 137;
    }
    let names = vec!["read_file".to_owned(), "write_file".to_owned()];
    let runner = match MontyRun::new(source, "<monty>", names.clone()) {
        Ok(runner) => runner,
        Err(exc) => {
            io.err
                .extend_from_slice(format!("monty: {}\n", exc.summary()).as_bytes());
            return 1;
        }
    };
    let inputs = names
        .into_iter()
        .map(|name| MontyObject::Function {
            name,
            docstring: None,
        })
        .collect();
    let meter = Meter(Rc::new(RefCell::new(&mut **env)));
    let mut output = Output {
        meter: meter.clone(),
        out: io.out,
    };
    let mut progress = runner.start(inputs, meter.clone(), PrintWriter::Callback(&mut output));
    loop {
        progress = match progress {
            Ok(RunProgress::Complete(_)) => {
                return if meter.0.borrow().resources.is_stopped() {
                    137
                } else {
                    0
                }
            }
            Ok(RunProgress::FunctionCall(call)) => {
                if meter.check_time().is_err() {
                    return 137;
                }
                let result = if call.kwargs.is_empty() {
                    file_call(&meter, &call.function_name, &call.args)
                } else {
                    Err(error("file tools accept positional arguments only"))
                };
                let result: ::monty::ExtFunctionResult = match result {
                    Ok(value) => value.into(),
                    Err(exc) => exc.into(),
                };
                call.resume(result, PrintWriter::Callback(&mut output))
            }
            Ok(_) => {
                meter
                    .0
                    .borrow_mut()
                    .note_unsupported("monty:host capability");
                io.err.extend_from_slice(
                    b"monty: unsupported OS call, external name, or async operation\n",
                );
                return 2;
            }
            Err(exc) => {
                io.err
                    .extend_from_slice(format!("monty: {}\n", exc.summary()).as_bytes());
                return if meter.0.borrow().resources.is_stopped() {
                    137
                } else {
                    1
                };
            }
        };
    }
}

fn file_call(
    meter: &Meter<'_>,
    name: &str,
    args: &[MontyObject],
) -> Result<MontyObject, MontyException> {
    let mut env = meter.0.borrow_mut();
    let cwd = env.cwd.clone();
    match (name, args) {
        ("read_file", [MontyObject::String(path)]) => {
            let real = env
                .vfs
                .realpath(&resolve_against(&cwd, path), true)
                .map_err(|e| error(&e.to_string()))?;
            if let Some(node) = env.vfs.raw_get(&real) {
                if let NodeKind::File(data) = &node.kind {
                    let bytes = data.len();
                    if bytes > MAX_FILE {
                        return Err(error("file limit 1 MiB exceeded"));
                    }
                    // Lossy UTF-8 decoding can expand each input byte to three bytes.
                    // Reserve before the VFS clones or decodes any file contents.
                    if !env
                        .resources
                        .reserve_memory((bytes as u64).saturating_mul(3))
                    {
                        return Err(error("memory limit exceeded"));
                    }
                    if !env.resources.charge_cpu(bytes as u64) {
                        return Err(error("CPU exhausted"));
                    }
                }
            }
            let text = env
                .vfs
                .read_string_limited(&cwd, path, MAX_FILE)
                .map_err(|e| error(&e.to_string()))?;
            Ok(MontyObject::String(text))
        }
        ("write_file", [MontyObject::String(path), MontyObject::String(text)])
            if text.len() <= MAX_FILE =>
        {
            if !env.resources.charge_cpu(text.len() as u64) {
                return Err(error("CPU exhausted"));
            }
            env.vfs
                .write(&cwd, path, text.as_bytes(), 0o644)
                .map_err(|e| error(&e.to_string()))?;
            Ok(MontyObject::None)
        }
        _ => Err(error(
            "expected read_file(path) or write_file(path, text); file limit 1 MiB",
        )),
    }
}
