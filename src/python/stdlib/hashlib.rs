//! Hash primitives used by the frozen pure-Python `hashlib` facade.
//!
//! The ABI accepts shellsim's temporary UTF-8 bytes surrogate and returns exact lowercase hex.

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyError, PyResult, PyRuntime, PyString, PyValueCast,
};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "_hashlib",
    functions: &[FunctionDef {
        module: "_hashlib",
        name: "hexdigest",
        call: hexdigest,
    }],
    values: &[],
};

fn hexdigest(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_hashlib.hexdigest", 2, 2)?;
    args.reject_keywords("_hashlib.hexdigest")?;
    let PyString(algorithm) = args.positional()[0].cast(runtime)?;
    let PyString(data) = args.positional()[1].cast(runtime)?;
    runtime.charge_cpu(u64::try_from(data.len()).unwrap_or(u64::MAX))?;
    let digest = match algorithm.as_str() {
        "md5" => crate::hashes::md5_hex(data.as_bytes()),
        "sha1" => crate::hashes::sha1_hex(data.as_bytes()),
        "sha256" => crate::hashes::sha256_hex(data.as_bytes()),
        "sha512" => crate::hashes::sha512_hex(data.as_bytes()),
        _ => return Err(PyError::value_error("unsupported hash algorithm")),
    };
    runtime.new_string(digest)
}
