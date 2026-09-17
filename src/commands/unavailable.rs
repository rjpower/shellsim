//! Known external-program boundaries that shellsim intentionally does not approximate.
//!
//! Registration makes these names distinguishable from typos through the common status 127,
//! diagnostic, and invocation telemetry. Add a name here only when workloads plausibly invoke it
//! and the capability is outside shellsim's modeled process, filesystem, or network boundary.

use std::collections::HashMap;

use crate::commands::{reg_unsupported, CommandSpec};

pub fn register(commands: &mut HashMap<&'static str, CommandSpec>) {
    reg_unsupported(
        commands,
        &[
            // Host administration, kernel, and filesystem tools.
            "debugfs",
            "journalctl",
            "losetup",
            "mkfs.ext2",
            "mount",
            "setfacl",
            "su",
            "umount",
            "useradd",
            // External runtimes, services, and interactive debuggers.
            "gdb",
            "mkdocs",
            "redis-cli",
            "redis-server",
            "rsyslog",
            "wstest",
            // Interfaces whose real semantics require host process or socket capabilities.
            "flock",
            "netstat",
            "nohup",
            "openssl",
            "ss",
        ],
    );
}
