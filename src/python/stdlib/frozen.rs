//! Build-time bundled pure-Python standard-library modules.
//!
//! Frozen modules execute through the same lexer, compiler, object model, and resource accounting
//! as user code. They receive no VFS or host capabilities merely by being part of the stdlib.

/// Return source for a bundled module. Keep this registry closed and explicit so adding a module
/// cannot accidentally expose build-machine files at runtime.
pub(super) fn module_source(name: &str) -> Option<&'static str> {
    match name {
        "asyncio" => Some(include_str!("source/asyncio.py")),
        "abc" => Some(include_str!("source/abc.py")),
        "base64" => Some(include_str!("source/base64.py")),
        "codecs" => Some(include_str!("source/codecs.py")),
        "_complex" => Some(include_str!("source/complex.py")),
        "cmath" => Some(include_str!("source/cmath.py")),
        "csv" => Some(include_str!("source/csv.py")),
        "datetime" => Some(include_str!("source/datetime.py")),
        "collections" => Some(include_str!("source/collections.py")),
        "copy" => Some(include_str!("source/copy.py")),
        "glob" => Some(include_str!("source/glob.py")),
        "hashlib" => Some(include_str!("source/hashlib.py")),
        "http" => Some(include_str!("source/http.py")),
        "http.client" => Some(include_str!("source/http_client.py")),
        "importlib" | "importlib.util" => Some(include_str!("source/importlib.py")),
        "_io" => Some(include_str!("source/io.py")),
        "io" => Some(include_str!("source/io.py")),
        "json" => Some(include_str!("source/json.py")),
        "logging" => Some(include_str!("source/logging.py")),
        "numpy.random" => Some(include_str!("source/numpy_random.py")),
        "os" => Some(include_str!("source/os.py")),
        "pathlib" => Some(include_str!("source/pathlib.py")),
        "pytest" => Some(include_str!("source/pytest.py")),
        "random" => Some(include_str!("source/random.py")),
        "shutil" => Some(include_str!("source/shutil.py")),
        "statistics" => Some(include_str!("source/statistics.py")),
        "struct" => Some(include_str!("source/struct.py")),
        "subprocess" => Some(include_str!("source/subprocess.py")),
        "tempfile" => Some(include_str!("source/tempfile.py")),
        "textwrap" => Some(include_str!("source/textwrap.py")),
        "uuid" => Some(include_str!("source/uuid.py")),
        "urllib" => Some(include_str!("source/urllib.py")),
        "urllib.error" => Some(include_str!("source/urllib_error.py")),
        "urllib.request" => Some(include_str!("source/urllib_request.py")),
        "zipfile" => Some(include_str!("source/zipfile.py")),
        "zlib" => Some(include_str!("source/zlib.py")),
        _ => None,
    }
}
