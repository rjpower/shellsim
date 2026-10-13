//! Package managers and build tools.
//!
//! Installation happens through the trusted ports catalog before guest execution.
//! Unmounted package managers and external runtimes are unsupported boundaries.

use std::collections::HashMap;

use crate::commands::CommandSpec;

pub fn register(commands: &mut HashMap<&'static str, CommandSpec>) {
    use super::reg_unsupported;
    reg_unsupported(
        commands,
        &[
            "pip",
            "pip3",
            "uv",
            "uvx",
            "uvenv",
            "conda",
            "pipx",
            "apt",
            "apt-get",
            "npm",
            "node",
            "cargo",
            "cmake",
            "gcc",
            "g++",
            "cc",
            "clang",
            "mvn",
            "gradle",
            "javac",
            "java",
            "docker",
            "systemctl",
            "service",
            "uvicorn",
            "gunicorn",
            "flask",
            "ld",
            "rustc",
            "go",
        ],
    );
}
