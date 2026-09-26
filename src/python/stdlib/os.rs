//! Environment access backed only by shellsim's modeled process-environment capability.
//!
//! `getpid`/`getppid` report the virtual PID and parent PID of the logical process currently
//! running Python. `kill` delivers a signal through the same modeled path the shell's `kill`
//! builtin uses; it never reaches a host process.

use super::super::native::PyValue as Value;
use super::super::native::{
    CallArgs, FunctionDef, MethodDef, ModuleDef, NativeTypeDef, OwnedPyString, PyError, PyMarker,
    PyResult, PyRuntime, PyValueCast, ValueDef,
};

pub(crate) static ENVIRONMENT_TYPE: NativeTypeDef = NativeTypeDef {
    name: "shellsim.environment",
    methods: &[MethodDef {
        type_name: "shellsim.environment",
        name: "get",
        call: environment_get,
    }],
};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "_os",
    functions: &[
        FunctionDef {
            module: "_os",
            name: "getenv",
            call: getenv,
        },
        FunctionDef {
            module: "_os",
            name: "getcwd",
            call: getcwd,
        },
        FunctionDef {
            module: "_os",
            name: "chdir",
            call: chdir,
        },
        FunctionDef {
            module: "_os",
            name: "getpid",
            call: getpid,
        },
        FunctionDef {
            module: "_os",
            name: "getppid",
            call: getppid,
        },
        FunctionDef {
            module: "_os",
            name: "kill",
            call: kill,
        },
    ],
    values: &[ValueDef::Factory {
        name: "environ",
        get: environ,
    }],
};

fn getcwd(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("os.getcwd", 0, 0)?;
    args.reject_keywords("os.getcwd")?;
    let directory = runtime.filesystem().current_dir();
    runtime.new_string(directory)
}

fn chdir(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("os.chdir", 1, 1)?;
    args.reject_keywords("os.chdir")?;
    let OwnedPyString(path) = args.positional()[0].cast(runtime)?;
    runtime.filesystem().change_dir(&path)?;
    Ok(Value::None)
}

fn getenv(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("os.getenv", 1, 2)?;
    args.reject_keywords("os.getenv")?;
    let OwnedPyString(name) = args.positional()[0].cast(runtime)?;
    if let Some(value) = runtime.environment().get(&name) {
        runtime.new_string(value)
    } else {
        Ok(args.positional().get(1).copied().unwrap_or(Value::None))
    }
}

fn getpid(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("os.getpid", 0, 0)?;
    args.reject_keywords("os.getpid")?;
    Ok(Value::Int(i64::from(runtime.current_pid())))
}

fn getppid(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("os.getppid", 0, 0)?;
    args.reject_keywords("os.getppid")?;
    Ok(Value::Int(i64::from(runtime.current_ppid())))
}

fn kill(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("os.kill", 2, 2)?;
    args.reject_keywords("os.kill")?;
    let pid = runtime
        .int_value(&args.positional()[0])
        .ok_or_else(|| PyError::type_error("os.kill pid must be an integer"))?;
    let pid =
        u32::try_from(pid).map_err(|_| PyError::value_error("os.kill pid is out of range"))?;
    let number = runtime
        .int_value(&args.positional()[1])
        .ok_or_else(|| PyError::type_error("os.kill signal must be an integer"))?;
    let signal = crate::process::Signal::parse(&number.to_string())
        .ok_or_else(|| PyError::value_error("unsupported signal"))?;
    runtime.send_os_signal(pid, signal)?;
    Ok(Value::None)
}

fn environ(runtime: &mut dyn PyRuntime) -> PyResult {
    Ok(runtime.marker(PyMarker::Environment))
}

fn environment_get(runtime: &mut dyn PyRuntime, _receiver: Value, args: CallArgs) -> PyResult {
    args.expect_positional("environ.get", 1, 2)?;
    args.reject_keywords("environ.get")?;
    let OwnedPyString(name) = args.positional()[0].cast(runtime)?;
    if let Some(value) = runtime.environment().get(&name) {
        runtime.new_string(value)
    } else {
        Ok(args.positional().get(1).copied().unwrap_or(Value::None))
    }
}
