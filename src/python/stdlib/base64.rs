//! Byte-preserving RFC 4648 primitives for the frozen `base64` facade.

use base64::Engine;

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyBytes, PyError, PyResult, PyRuntime, PyValueCast,
};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "_base64",
    functions: &[
        FunctionDef {
            module: "_base64",
            name: "b64encode",
            call: b64encode,
        },
        FunctionDef {
            module: "_base64",
            name: "b64decode",
            call: b64decode,
        },
        FunctionDef {
            module: "_base64",
            name: "urlsafe_b64encode",
            call: urlsafe_b64encode,
        },
        FunctionDef {
            module: "_base64",
            name: "urlsafe_b64decode",
            call: urlsafe_b64decode,
        },
    ],
    values: &[],
};

fn b64encode(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    encode(
        runtime,
        args,
        &base64::engine::general_purpose::STANDARD,
        "b64encode",
    )
}

fn urlsafe_b64encode(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    encode(
        runtime,
        args,
        &base64::engine::general_purpose::URL_SAFE,
        "urlsafe_b64encode",
    )
}

fn encode(
    runtime: &mut dyn PyRuntime,
    args: CallArgs,
    engine: &base64::engine::GeneralPurpose,
    name: &str,
) -> PyResult {
    args.expect_positional(name, 1, 1)?;
    args.reject_keywords(name)?;
    let PyBytes(value) = args.positional()[0].cast(runtime)?;
    let length = value
        .len()
        .checked_add(2)
        .and_then(|size| size.checked_div(3))
        .and_then(|size| size.checked_mul(4))
        .ok_or_else(|| PyError::resource_error("base64 result is too large"))?;
    runtime.reserve_memory(length)?;
    runtime.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
    runtime.new_bytes(engine.encode(value).into_bytes())
}

fn b64decode(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    decode(
        runtime,
        args,
        &base64::engine::general_purpose::STANDARD,
        "b64decode",
    )
}

fn urlsafe_b64decode(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    decode(
        runtime,
        args,
        &base64::engine::general_purpose::URL_SAFE,
        "urlsafe_b64decode",
    )
}

fn decode(
    runtime: &mut dyn PyRuntime,
    args: CallArgs,
    engine: &base64::engine::GeneralPurpose,
    name: &str,
) -> PyResult {
    args.expect_positional(name, 1, 1)?;
    args.reject_keywords(name)?;
    let PyBytes(value) = args.positional()[0].cast(runtime)?;
    runtime.reserve_memory(value.len())?;
    runtime.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
    let decoded = engine
        .decode(value)
        .map_err(|error| PyError::value_error(format!("invalid base64 data: {error}")))?;
    runtime.new_bytes(decoded)
}
