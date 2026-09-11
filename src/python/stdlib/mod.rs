//! Deterministic, explicitly capability-scoped Python standard-library slices.
//!
//! The parent Python runtime deliberately wires these modules into its import table separately.
//! Keeping this directory self-contained makes each shim straightforward to review and test.

pub mod argparse;
pub mod bisect;
pub mod collections;
pub mod dataclasses;
pub mod r#enum;
pub mod functools;
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

use super::native::ModuleDef;

/// Resolve one capability-free module through the declarative native registry.
pub(super) fn native_module(name: &str) -> Option<&'static ModuleDef> {
    match name {
        "argparse" => Some(&argparse::MODULE),
        "json" => Some(&json::MODULE),
        "bisect" => Some(&bisect::MODULE),
        "collections" => Some(&collections::MODULE),
        "dataclasses" => Some(&dataclasses::MODULE),
        "enum" => Some(&r#enum::MODULE),
        "heapq" => Some(&heapq::MODULE),
        "functools" => Some(&functools::MODULE),
        "itertools" => Some(&itertools::MODULE),
        "math" => Some(&math::MODULE),
        "os" => Some(&os::MODULE),
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
