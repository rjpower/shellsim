//! Bounded compression and exact checksum primitives for the frozen `zlib` facade.

use std::io::{Read, Write};

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyBytes, PyError, PyResult, PyRuntime, PyValueCast,
};
use super::super::Value;

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "_zlib",
    functions: &[
        FunctionDef {
            module: "_zlib",
            name: "crc32",
            call: crc32,
        },
        FunctionDef {
            module: "_zlib",
            name: "compress",
            call: compress,
        },
        FunctionDef {
            module: "_zlib",
            name: "decompress",
            call: decompress,
        },
    ],
    values: &[],
};

fn crc32(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("zlib.crc32", 1, 2)?;
    args.reject_keywords("zlib.crc32")?;
    let PyBytes(value) = args.positional()[0].cast(runtime)?;
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
    hasher.update(&value);
    Ok(Value::Int(i64::from(hasher.finalize())))
}

const MAX_ZLIB_OUTPUT: usize = 4 * 1024 * 1024;

fn compress(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("zlib.compress", 1, 2)?;
    args.reject_keywords("zlib.compress")?;
    let PyBytes(value) = args.positional()[0].cast(runtime)?;
    let level = args
        .positional()
        .get(1)
        .map(|value| {
            runtime
                .int_value(value)
                .ok_or_else(|| PyError::type_error("compression level must be an integer"))
        })
        .transpose()?
        .unwrap_or(-1);
    if !(-1..=9).contains(&level) {
        return Err(PyError::value_error(
            "compression level must be between -1 and 9",
        ));
    }
    let bound = value
        .len()
        .checked_add(value.len() / 8)
        .and_then(|size| size.checked_add(128))
        .ok_or_else(|| PyError::resource_error("compressed result is too large"))?;
    runtime.reserve_memory(bound)?;
    runtime.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
    let compression = if level < 0 {
        flate2::Compression::default()
    } else {
        flate2::Compression::new(u32::try_from(level).expect("validated level"))
    };
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::with_capacity(bound), compression);
    encoder
        .write_all(&value)
        .map_err(|error| PyError::value_error(error.to_string()))?;
    let result = encoder
        .finish()
        .map_err(|error| PyError::value_error(error.to_string()))?;
    runtime.new_bytes(result)
}

fn decompress(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("zlib.decompress", 1, 1)?;
    args.reject_keywords("zlib.decompress")?;
    let PyBytes(value) = args.positional()[0].cast(runtime)?;
    runtime.reserve_memory(MAX_ZLIB_OUTPUT)?;
    runtime.charge_cpu(u64::try_from(value.len()).unwrap_or(u64::MAX))?;
    let decoder = flate2::read::ZlibDecoder::new(value.as_slice());
    let mut result = Vec::new();
    decoder
        .take(u64::try_from(MAX_ZLIB_OUTPUT + 1).expect("constant fits u64"))
        .read_to_end(&mut result)
        .map_err(|error| PyError::value_error(error.to_string()))?;
    if result.len() > MAX_ZLIB_OUTPUT {
        return Err(PyError::resource_error(
            "decompressed data exceeds the 4 MiB limit",
        ));
    }
    runtime.charge_cpu(u64::try_from(result.len()).unwrap_or(u64::MAX))?;
    runtime.new_bytes(result)
}
