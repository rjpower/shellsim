//! Build-time bundled pure-Python standard-library modules.
//!
//! Frozen modules execute through the same lexer, compiler, object model, and resource accounting
//! as user code. They receive no VFS or host capabilities merely by being part of the stdlib.

/// Return source for a bundled module. Keep this registry closed and explicit so adding a module
/// cannot accidentally expose build-machine files at runtime.
pub(super) fn module_source(name: &str) -> Option<&'static str> {
    match name {
        "abc" => Some(include_str!("source/abc.py")),
        "base64" => Some(include_str!("source/base64.py")),
        "codecs" => Some(include_str!("source/codecs.py")),
        "csv" => Some(include_str!("source/csv.py")),
        "datetime" => Some(include_str!("source/datetime.py")),
        "collections" => Some(include_str!("source/collections.py")),
        "glob" => Some(include_str!("source/glob.py")),
        "hashlib" => Some(include_str!("source/hashlib.py")),
        "importlib" | "importlib.util" => Some(include_str!("source/importlib.py")),
        "_io" => Some(include_str!("source/io.py")),
        "io" => Some(include_str!("source/io.py")),
        "json" => Some(include_str!("source/json.py")),
        "logging" => Some(include_str!("source/logging.py")),
        "os" => Some(include_str!("source/os.py")),
        "pathlib" => Some(include_str!("source/pathlib.py")),
        "random" => Some(include_str!("source/random.py")),
        "statistics" => Some(include_str!("source/statistics.py")),
        "struct" => Some(include_str!("source/struct.py")),
        "subprocess" => Some(include_str!("source/subprocess.py")),
        "tempfile" => Some(include_str!("source/tempfile.py")),
        "uuid" => Some(include_str!("source/uuid.py")),
        "zlib" => Some(include_str!("source/zlib.py")),
        _ => None,
    }
}
