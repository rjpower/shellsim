//! Unicode normalization for pure-Python package compatibility.
//!
//! The public module is frozen Python. This private native operation uses bundled Unicode tables
//! and charges scratch memory and work before constructing normalized strings.

use unicode_normalization::char::{decompose_canonical, decompose_compatible};
use unicode_normalization::UnicodeNormalization;

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, OwnedPyString, PyError, PyResult, PyRuntime, PyValueCast,
};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "_unicodedata",
    functions: &[FunctionDef {
        module: "_unicodedata",
        name: "normalize",
        call: normalize,
    }],
    values: &[],
};

fn normalize(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("unicodedata.normalize", 2, 2)?;
    args.reject_keywords("unicodedata.normalize")?;
    let OwnedPyString(form) = args.positional()[0].cast(runtime)?;
    let OwnedPyString(input) = args.positional()[1].cast(runtime)?;
    let compatible = match form.as_str() {
        "NFC" | "NFD" => false,
        "NFKC" | "NFKD" => true,
        _ => return Err(PyError::value_error("invalid normalization form")),
    };

    // The normalizer can buffer an entire combining run, including compatibility
    // expansions such as U+FDFA. Count those characters before it can allocate.
    runtime.charge_cpu(
        u64::try_from(input.len())
            .unwrap_or(u64::MAX)
            .saturating_mul(32),
    )?;
    let mut decomposed = 0usize;
    for character in input.chars() {
        let mut count = 0usize;
        let mut emitted = |_| count = count.saturating_add(1);
        if compatible {
            decompose_compatible(character, &mut emitted);
        } else {
            decompose_canonical(character, &mut emitted);
        }
        decomposed = decomposed.checked_add(count).ok_or_else(|| {
            PyError::exception("MemoryError", "normalization output is too large")
        })?;
    }
    let scratch = decomposed
        .checked_mul(16)
        .ok_or_else(|| PyError::exception("MemoryError", "normalization output is too large"))?;
    runtime.reserve_memory(scratch)?;

    let iterator: Box<dyn Iterator<Item = char> + '_> = match form.as_str() {
        "NFC" => Box::new(input.nfc()),
        "NFD" => Box::new(input.nfd()),
        "NFKC" => Box::new(input.nfkc()),
        "NFKD" => Box::new(input.nfkd()),
        _ => unreachable!(),
    };
    let mut bytes = 0usize;
    for character in iterator {
        runtime.charge_cpu(1)?;
        bytes = bytes.checked_add(character.len_utf8()).ok_or_else(|| {
            PyError::exception("MemoryError", "normalization output is too large")
        })?;
    }
    runtime.reserve_memory(bytes)?;
    let result = match form.as_str() {
        "NFC" => input.nfc().collect(),
        "NFD" => input.nfd().collect(),
        "NFKC" => input.nfkc().collect(),
        "NFKD" => input.nfkd().collect(),
        _ => unreachable!(),
    };
    runtime.new_string(result)
}
