//! Deterministic, explicitly capability-scoped Python standard-library slices.
//!
//! The parent Python runtime deliberately wires these modules into its import table separately.
//! Keeping this directory self-contained makes each shim straightforward to review and test.

pub mod argparse;
pub mod bisect;
pub mod collections;
pub mod core;
pub mod dataclasses;
pub mod r#enum;
mod frozen;
pub mod functools;
mod hashlib;
pub mod heapq;
pub mod itertools;
pub mod json;
pub mod math;
pub mod os;
pub mod pytest;
pub mod re;
pub mod string;
pub mod subprocess;
pub mod sys;
pub mod time;
pub mod typing;
pub mod unittest;
mod vfs;
mod zlib;

use super::native::ModuleDef;

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
        "_json" => Some(&json::MODULE),
        "bisect" => Some(&bisect::MODULE),
        "_collections" => Some(&collections::MODULE),
        "dataclasses" => Some(&dataclasses::MODULE),
        "enum" => Some(&r#enum::MODULE),
        "heapq" => Some(&heapq::MODULE),
        "_hashlib" => Some(&hashlib::MODULE),
        "_shellsim_vfs" => Some(&vfs::MODULE),
        "_zlib" => Some(&zlib::MODULE),
        "functools" => Some(&functools::MODULE),
        "itertools" => Some(&itertools::MODULE),
        "math" => Some(&math::MODULE),
        "_os" => Some(&os::MODULE),
        "pytest" => Some(&pytest::MODULE),
        "re" => Some(&re::MODULE),
        "string" => Some(&string::MODULE),
        "sys" => Some(&sys::MODULE),
        "subprocess" => Some(&subprocess::MODULE),
        "time" => Some(&time::MODULE),
        "typing" => Some(&typing::MODULE),
        "unittest" => Some(&unittest::MODULE),
        _ => None,
    }
}
