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
        "cmath" => Some(include_str!("source/cmath.py")),
        "csv" => Some(include_str!("source/csv.py")),
        "datetime" => Some(include_str!("source/datetime.py")),
        "collections" => Some(include_str!("source/collections.py")),
        "copy" => Some(include_str!("source/copy.py")),
        "glob" => Some(include_str!("source/glob.py")),
        "functools" => Some(include_str!("source/functools.py")),
        "hashlib" => Some(include_str!("source/hashlib.py")),
        "http" => Some(include_str!("source/http.py")),
        "http.client" => Some(include_str!("source/http_client.py")),
        "importlib" | "importlib.util" => Some(include_str!("source/importlib.py")),
        "_io" => Some(include_str!("source/io.py")),
        "io" => Some(include_str!("source/io.py")),
        "json" => Some(include_str!("source/json.py")),
        "logging" => Some(include_str!("source/logging.py")),
        "numpy" => Some(include_str!("source/numpy/__init__.py")),
        "numpy._arraypad" => Some(include_str!("source/numpy/_arraypad.py")),
        "numpy._arraysetops" => Some(include_str!("source/numpy/_arraysetops.py")),
        "numpy._arrayprint" => Some(include_str!("source/numpy/_arrayprint.py")),
        "numpy._errstate" => Some(include_str!("source/numpy/_errstate.py")),
        "numpy._function_base" => Some(include_str!("source/numpy/_function_base.py")),
        "numpy._getlimits" => Some(include_str!("source/numpy/_getlimits.py")),
        "numpy._histograms" => Some(include_str!("source/numpy/_histograms.py")),
        "numpy._index_tricks" => Some(include_str!("source/numpy/_index_tricks.py")),
        "numpy._methods" => Some(include_str!("source/numpy/_methods.py")),
        "numpy._nanfunctions" => Some(include_str!("source/numpy/_nanfunctions.py")),
        "numpy._numeric" => Some(include_str!("source/numpy/_numeric.py")),
        "numpy._shape_base" => Some(include_str!("source/numpy/_shape_base.py")),
        "numpy._statistics" => Some(include_str!("source/numpy/_statistics.py")),
        "numpy.exceptions" => Some(include_str!("source/numpy/exceptions.py")),
        "numpy.fft" => Some(include_str!("source/numpy/fft.py")),
        "numpy.lib" => Some(include_str!("source/numpy/lib/__init__.py")),
        "numpy.lib._objectpickle" => Some(include_str!("source/numpy/lib/_objectpickle.py")),
        "numpy.lib.format" => Some(include_str!("source/numpy/lib/format.py")),
        "numpy.lib.npyio" => Some(include_str!("source/numpy/lib/npyio.py")),
        "numpy.linalg" => Some(include_str!("source/numpy/linalg.py")),
        "numpy.random" => Some(include_str!("source/numpy/random.py")),
        "numpy.strings" => Some(include_str!("source/numpy/strings.py")),
        "numpy.testing" => Some(include_str!("source/numpy/testing.py")),
        "scipy" => Some(include_str!("source/scipy/__init__.py")),
        "scipy.special" => Some(include_str!("source/scipy/special.py")),
        "scipy.cluster"
        | "scipy.constants"
        | "scipy.datasets"
        | "scipy.differentiate"
        | "scipy.fft"
        | "scipy.fftpack"
        | "scipy.integrate"
        | "scipy.interpolate"
        | "scipy.io"
        | "scipy.linalg"
        | "scipy.ndimage"
        | "scipy.odr"
        | "scipy.optimize"
        | "scipy.signal"
        | "scipy.sparse"
        | "scipy.spatial"
        | "scipy.stats" => Some(include_str!("source/scipy/_unsupported.py")),
        "operator" => Some(include_str!("source/operator.py")),
        "os" => Some(include_str!("source/os.py")),
        "pathlib" => Some(include_str!("source/pathlib.py")),
        "pytest" => Some(include_str!("source/pytest.py")),
        "random" => Some(include_str!("source/random.py")),
        "shutil" => Some(include_str!("source/shutil.py")),
        "signal" => Some(include_str!("source/signal.py")),
        "statistics" => Some(include_str!("source/statistics.py")),
        "struct" => Some(include_str!("source/struct.py")),
        "subprocess" => Some(include_str!("source/subprocess.py")),
        "tempfile" => Some(include_str!("source/tempfile.py")),
        "textwrap" => Some(include_str!("source/textwrap.py")),
        "uuid" => Some(include_str!("source/uuid.py")),
        "warnings" => Some(include_str!("source/warnings.py")),
        "urllib" => Some(include_str!("source/urllib.py")),
        "urllib.error" => Some(include_str!("source/urllib_error.py")),
        "urllib.request" => Some(include_str!("source/urllib_request.py")),
        "zipfile" => Some(include_str!("source/zipfile.py")),
        "zlib" => Some(include_str!("source/zlib.py")),
        _ => None,
    }
}
