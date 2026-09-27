//! Deterministic, explicitly capability-scoped Python standard-library slices.
//!
//! The parent Python runtime deliberately wires these modules into its import table separately.
//! Keeping this directory self-contained makes each shim straightforward to review and test.

pub mod argparse;
mod asyncio;
mod base64;
pub mod bisect;
pub mod collections;
pub mod core;
pub mod dataclasses;
pub mod r#enum;
mod frozen;
pub mod functools;
mod hashlib;
pub mod heapq;
mod http;
mod importlib;
mod introspect;
pub mod itertools;
pub mod json;
pub mod math;
pub mod numpy;
mod operator;
pub mod os;
pub mod pytest;
pub mod re;
mod scipy;
pub mod string;
mod r#struct;
pub mod subprocess;
pub mod sys;
pub mod time;
pub mod typing;
pub mod unittest;
mod vfs;
mod warnings;
mod zlib;

use super::native::{ModuleDef, ValueKindDef};

/// Collect inline value registrations without teaching the VM about module-owned types.
pub(super) fn value_kinds() -> impl Iterator<Item = &'static ValueKindDef> {
    numpy::value_kinds()
}

/// The registered kind at `index` in [`value_kinds`] order, which is also the order every
/// type registry assigns kind indexes in. Pure protocols such as numeric views use it where
/// no runtime is at hand.
pub(super) fn value_kind(index: u8) -> Option<&'static ValueKindDef> {
    static KINDS: std::sync::OnceLock<Vec<&'static ValueKindDef>> = std::sync::OnceLock::new();
    KINDS
        .get_or_init(|| value_kinds().collect())
        .get(usize::from(index))
        .copied()
}

/// Resolve a capability-free stdlib module implemented in ordinary Python source.
pub(super) fn frozen_module(name: &str) -> Option<&'static str> {
    frozen::module_source(name)
}

/// Resolve a builtin implemented by a frozen module without preloading that module into the VFS.
pub(super) fn frozen_builtin(name: &str) -> Option<(&'static str, &'static str)> {
    match name {
        "open" => Some(("_io", "open")),
        _ => None,
    }
}

/// Resolve one capability-free module through the declarative native registry.
pub(super) fn native_module(name: &str) -> Option<&'static ModuleDef> {
    match name {
        "_asyncio" => Some(&asyncio::MODULE),
        "argparse" => Some(&argparse::MODULE),
        "_base64" => Some(&base64::MODULE),
        "_json" => Some(&json::MODULE),
        "bisect" => Some(&bisect::MODULE),
        "_collections" => Some(&collections::MODULE),
        "dataclasses" => Some(&dataclasses::MODULE),
        "enum" => Some(&r#enum::MODULE),
        "heapq" => Some(&heapq::MODULE),
        "_hashlib" => Some(&hashlib::MODULE),
        "_shellsim_http" => Some(&http::MODULE),
        "_importlib" => Some(&importlib::MODULE),
        "_shellsim_introspect" => Some(&introspect::MODULE),
        "_shellsim_vfs" => Some(&vfs::MODULE),
        "_zlib" => Some(&zlib::MODULE),
        "_functools" => Some(&functools::MODULE),
        "itertools" => Some(&itertools::MODULE),
        "math" => Some(&math::MODULE),
        "_operator" => Some(&operator::MODULE),
        "_os" => Some(&os::MODULE),
        "_pytest" => Some(&pytest::MODULE),
        "re" => Some(&re::MODULE),
        "string" => Some(&string::MODULE),
        "_struct" => Some(&r#struct::MODULE),
        "sys" => Some(&sys::MODULE),
        "_shellsim_subprocess" => Some(&subprocess::MODULE),
        "time" => Some(&time::MODULE),
        "typing" => Some(&typing::MODULE),
        "unittest" => Some(&unittest::MODULE),
        "_shellsim_warnings" => Some(&warnings::MODULE),
        _ if name.starts_with("_numpy") => numpy::native_module(name),
        _ if name.starts_with("_scipy") => scipy::native_module(name),
        _ => None,
    }
}
