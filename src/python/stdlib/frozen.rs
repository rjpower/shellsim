//! Build-time bundled pure-Python standard-library modules.
//!
//! Frozen modules execute through the same lexer, compiler, object model, and resource accounting
//! as user code. They receive no VFS or host capabilities merely by being part of the stdlib.

/// One bundled source file.
#[derive(Clone, Copy)]
pub(in crate::python) struct FrozenModule {
    /// The file's path below `source/`, such as `numpy/__init__.py`.
    pub(in crate::python) file: &'static str,
    pub(in crate::python) source: &'static str,
}

impl FrozenModule {
    /// A package's source is its `__init__.py`. The importer needs this to set `__package__`
    /// for relative imports inside the package's own initializer.
    pub(in crate::python) fn is_package(&self) -> bool {
        self.file.ends_with("__init__.py")
    }
}

macro_rules! frozen {
    ($file:literal) => {
        Some(FrozenModule {
            file: $file,
            source: include_str!(concat!("source/", $file)),
        })
    };
}

/// Return source for a bundled module. Keep this registry closed and explicit so adding a module
/// cannot accidentally expose build-machine files at runtime.
pub(super) fn module_source(name: &str) -> Option<FrozenModule> {
    match name {
        "asyncio" => frozen!("asyncio.py"),
        "abc" => frozen!("abc.py"),
        "base64" => frozen!("base64.py"),
        "codecs" => frozen!("codecs.py"),
        "cmath" => frozen!("cmath.py"),
        "csv" => frozen!("csv.py"),
        "datetime" => frozen!("datetime.py"),
        "collections" => frozen!("collections.py"),
        "contextlib" => frozen!("contextlib.py"),
        "copy" => frozen!("copy.py"),
        "fractions" => frozen!("fractions.py"),
        "glob" => frozen!("glob.py"),
        "functools" => frozen!("functools.py"),
        "hashlib" => frozen!("hashlib.py"),
        "http" => frozen!("http.py"),
        "http.client" => frozen!("http_client.py"),
        "importlib" | "importlib.util" => frozen!("importlib.py"),
        "inspect" => frozen!("inspect.py"),
        "_io" => frozen!("io.py"),
        "io" => frozen!("io.py"),
        "json" => frozen!("json.py"),
        "keyword" => frozen!("keyword.py"),
        "logging" => frozen!("logging.py"),
        "numpy" => frozen!("numpy/__init__.py"),
        "numpy._creation" => frozen!("numpy/_creation.py"),
        "numpy._errors" => frozen!("numpy/_errors.py"),
        "numpy._io" => frozen!("numpy/_io.py"),
        "numpy._math" => frozen!("numpy/_math.py"),
        "numpy._printing" => frozen!("numpy/_printing.py"),
        "numpy._sets" => frozen!("numpy/_sets.py"),
        "numpy._shapes" => frozen!("numpy/_shapes.py"),
        "numpy._stats" => frozen!("numpy/_stats.py"),
        "numpy.exceptions" => frozen!("numpy/exceptions.py"),
        "numpy.fft" => frozen!("numpy/fft.py"),
        "numpy.linalg" => frozen!("numpy/linalg.py"),
        "numpy.random" => frozen!("numpy/random.py"),
        "numpy.testing" => frozen!("numpy/testing.py"),
        "scipy" => frozen!("scipy/__init__.py"),
        "scipy.integrate" => frozen!("scipy/integrate.py"),
        "scipy.interpolate" => frozen!("scipy/interpolate.py"),
        "scipy.linalg" => frozen!("scipy/linalg.py"),
        "scipy.optimize" => frozen!("scipy/optimize.py"),
        "scipy.spatial" => frozen!("scipy/spatial/__init__.py"),
        "scipy.spatial.distance" => frozen!("scipy/spatial/distance.py"),
        "scipy.special" => frozen!("scipy/special.py"),
        "scipy.stats" => frozen!("scipy/stats/__init__.py"),
        "scipy.stats._distributions" => frozen!("scipy/stats/_distributions.py"),
        "scipy.stats._describe" => frozen!("scipy/stats/_describe.py"),
        "scipy.stats._tests" => frozen!("scipy/stats/_tests.py"),
        "operator" => frozen!("operator.py"),
        "os" => frozen!("os.py"),
        "pathlib" => frozen!("pathlib.py"),
        "pytest" => frozen!("pytest.py"),
        "random" => frozen!("random.py"),
        "shutil" => frozen!("shutil.py"),
        "signal" => frozen!("signal.py"),
        "statistics" => frozen!("statistics.py"),
        "struct" => frozen!("struct.py"),
        "subprocess" => frozen!("subprocess.py"),
        "tempfile" => frozen!("tempfile.py"),
        "textwrap" => frozen!("textwrap.py"),
        "uuid" => frozen!("uuid.py"),
        "warnings" => frozen!("warnings.py"),
        "urllib" => frozen!("urllib.py"),
        "urllib.error" => frozen!("urllib_error.py"),
        "urllib.request" => frozen!("urllib_request.py"),
        "zipfile" => frozen!("zipfile.py"),
        "zlib" => frozen!("zlib.py"),
        _ => None,
    }
}
