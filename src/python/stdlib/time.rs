//! Python time functions backed only by shellsim's deterministic virtual clock capability.

use super::super::native::{CallArgs, FunctionDef, ModuleDef, PyError, PyResult, PyRuntime};
use super::super::number::PyNumber;
use super::super::{native::PyValueCast, Value};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "time",
    functions: &[
        FunctionDef {
            module: "time",
            name: "time",
            call: time,
        },
        FunctionDef {
            module: "time",
            name: "time_ns",
            call: time_ns,
        },
        FunctionDef {
            module: "time",
            name: "monotonic",
            call: monotonic,
        },
        FunctionDef {
            module: "time",
            name: "perf_counter",
            call: monotonic,
        },
        FunctionDef {
            module: "time",
            name: "monotonic_ns",
            call: monotonic_ns,
        },
        FunctionDef {
            module: "time",
            name: "perf_counter_ns",
            call: monotonic_ns,
        },
        FunctionDef {
            module: "time",
            name: "process_time",
            call: process_time,
        },
        FunctionDef {
            module: "time",
            name: "process_time_ns",
            call: process_time_ns,
        },
        FunctionDef {
            module: "time",
            name: "sleep",
            call: sleep,
        },
    ],
    values: &[],
};

fn no_args(args: &CallArgs, name: &str) -> PyResult<()> {
    args.expect_positional(name, 0, 0)?;
    args.reject_keywords(name)
}

fn time(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    no_args(&args, "time.time")?;
    runtime.clock().wall_time().map(Value::Float)
}
fn time_ns(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    no_args(&args, "time.time_ns")?;
    runtime.clock().wall_time_ns().map(Value::Int)
}
fn monotonic(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    no_args(&args, "time.monotonic")?;
    Ok(Value::Float(runtime.clock().monotonic()))
}
fn monotonic_ns(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    no_args(&args, "time.monotonic_ns")?;
    runtime.clock().monotonic_ns().map(Value::Int)
}
fn process_time(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    no_args(&args, "time.process_time")?;
    Ok(Value::Float(runtime.clock().process_time()))
}
fn process_time_ns(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    no_args(&args, "time.process_time_ns")?;
    runtime.clock().process_time_ns().map(Value::Int)
}

fn sleep(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("time.sleep", 1, 1)?;
    args.reject_keywords("time.sleep")?;
    let seconds = args.positional()[0].cast::<PyNumber>(runtime)?.into_f64()?;
    if !seconds.is_finite() || seconds < 0.0 {
        return Err(PyError::value_error(
            "time.sleep() length must be a finite non-negative number",
        ));
    }
    runtime.clock().sleep(seconds)?;
    Ok(Value::None)
}
