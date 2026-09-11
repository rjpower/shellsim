//! Build-time bundled pure-Python standard-library modules.
//!
//! Frozen modules execute through the same lexer, compiler, object model, and resource accounting
//! as user code. They receive no VFS or host capabilities merely by being part of the stdlib.

/// Return source for a bundled module. Keep this registry closed and explicit so adding a module
/// cannot accidentally expose build-machine files at runtime.
pub(super) fn module_source(name: &str) -> Option<&'static str> {
    match name {
        "abc" => Some(include_str!("source/abc.py")),
        "csv" => Some(include_str!("source/csv.py")),
        "glob" => Some(include_str!("source/glob.py")),
        "hashlib" => Some(include_str!("source/hashlib.py")),
        "_io" => Some(include_str!("source/io.py")),
        "logging" => Some(include_str!("source/logging.py")),
        "pathlib" => Some(include_str!("source/pathlib.py")),
        "uuid" => Some(include_str!("source/uuid.py")),
        _ => None,
    }
}
