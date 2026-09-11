//! Exact checksum primitives for the frozen `zlib` facade.
//!
//! Shellsim does not yet expose a byte-preserving `PyBytes` value, so compression and
//! decompression remain explicit frontiers. CRC-32 is exact for the current UTF-8 byte surrogate.

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyError, PyResult, PyRuntime, PyString, PyValueCast,
};
use super::super::Value;

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "_zlib",
    functions: &[FunctionDef {
        module: "_zlib",
        name: "crc32",
        call: crc32,
    }],
    values: &[],
};

fn crc32(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("zlib.crc32", 1, 2)?;
    args.reject_keywords("zlib.crc32")?;
    let PyString(value) = args.positional()[0].cast(runtime)?;
    let initial = args
        .positional()
        .get(1)
        .map(|value| {
            runtime
                .int_value(value)
                .ok_or_else(|| PyError::type_error("zlib.crc32 initial value must be an integer"))
        })
        .transpose()?
        .unwrap_or(0);
    let initial = u32::try_from(initial)
        .map_err(|_| PyError::value_error("zlib.crc32 initial value is out of range"))?;
    runtime.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
    let mut hasher = crc32fast::Hasher::new_with_initial(initial);
    hasher.update(value.as_bytes());
    Ok(Value::Int(i64::from(hasher.finalize())))
}
