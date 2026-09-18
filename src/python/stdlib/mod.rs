//! Deterministic, explicitly capability-scoped Python standard-library slices.
//!
//! The parent Python runtime deliberately wires these modules into its import table separately.
//! Keeping this directory self-contained makes each shim straightforward to review and test.

pub mod argparse;
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
mod importlib;
pub mod itertools;
pub mod json;
pub mod math;
pub mod numpy;
pub mod os;
pub mod pytest;
pub mod re;
pub mod string;
mod r#struct;
pub mod subprocess;
pub mod sys;
pub mod time;
pub mod typing;
pub mod unittest;
mod vfs;
mod zlib;

use super::native::{ModuleDef, ValueKindDef};

/// Collect inline value registrations without teaching the VM about module-owned types.
pub(super) fn value_kinds() -> impl Iterator<Item = &'static ValueKindDef> {
    numpy::value_kinds()
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
        "argparse" => Some(&argparse::MODULE),
        "_base64" => Some(&base64::MODULE),
        "_json" => Some(&json::MODULE),
        "bisect" => Some(&bisect::MODULE),
        "_collections" => Some(&collections::MODULE),
        "dataclasses" => Some(&dataclasses::MODULE),
        "enum" => Some(&r#enum::MODULE),
        "heapq" => Some(&heapq::MODULE),
        "_hashlib" => Some(&hashlib::MODULE),
        "_importlib" => Some(&importlib::MODULE),
        "_shellsim_vfs" => Some(&vfs::MODULE),
        "_zlib" => Some(&zlib::MODULE),
        "functools" => Some(&functools::MODULE),
        "itertools" => Some(&itertools::MODULE),
        "math" => Some(&math::MODULE),
        "numpy" => Some(&numpy::MODULE),
        "_os" => Some(&os::MODULE),
        "pytest" => Some(&pytest::MODULE),
        "re" => Some(&re::MODULE),
        "string" => Some(&string::MODULE),
        "_struct" => Some(&r#struct::MODULE),
        "sys" => Some(&sys::MODULE),
        "_shellsim_subprocess" => Some(&subprocess::MODULE),
        "time" => Some(&time::MODULE),
        "typing" => Some(&typing::MODULE),
        "unittest" => Some(&unittest::MODULE),
        _ => None,
    }
}
