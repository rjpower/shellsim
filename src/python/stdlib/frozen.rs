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
        "numpy._arraypad" => frozen!("numpy/_arraypad.py"),
        "numpy._arraysetops" => frozen!("numpy/_arraysetops.py"),
        "numpy._arrayprint" => frozen!("numpy/_arrayprint.py"),
        "numpy._errstate" => frozen!("numpy/_errstate.py"),
        "numpy._function_base" => frozen!("numpy/_function_base.py"),
        "numpy._getlimits" => frozen!("numpy/_getlimits.py"),
        "numpy._histograms" => frozen!("numpy/_histograms.py"),
        "numpy._index_tricks" => frozen!("numpy/_index_tricks.py"),
        "numpy._methods" => frozen!("numpy/_methods.py"),
        "numpy._nanfunctions" => frozen!("numpy/_nanfunctions.py"),
        "numpy._numeric" => frozen!("numpy/_numeric.py"),
        "numpy._shape_base" => frozen!("numpy/_shape_base.py"),
        "numpy._statistics" => frozen!("numpy/_statistics.py"),
        "numpy._vectorize" => frozen!("numpy/_vectorize.py"),
        "numpy.exceptions" => frozen!("numpy/exceptions.py"),
        "numpy.fft" => frozen!("numpy/fft.py"),
        "numpy.lib" => frozen!("numpy/lib/__init__.py"),
        "numpy.lib._objectpickle" => frozen!("numpy/lib/_objectpickle.py"),
        "numpy.lib.format" => frozen!("numpy/lib/format.py"),
        "numpy.lib.npyio" => frozen!("numpy/lib/npyio.py"),
        "numpy.linalg" => frozen!("numpy/linalg.py"),
        "numpy.random" => frozen!("numpy/random.py"),
        "numpy.strings" => frozen!("numpy/strings.py"),
        "numpy.testing" => frozen!("numpy/testing.py"),
        "scipy" => frozen!("scipy/__init__.py"),
        "scipy._lib" => frozen!("scipy/_lib/__init__.py"),
        "scipy._lib._util" => frozen!("scipy/_lib/_util.py"),
        "scipy.integrate" => frozen!("scipy/integrate/__init__.py"),
        "scipy.integrate._quadpack" => frozen!("scipy/integrate/_quadpack.py"),
        "scipy.integrate._quadrature" => frozen!("scipy/integrate/_quadrature.py"),
        "scipy.linalg" => frozen!("scipy/linalg.py"),
        "scipy.interpolate" => frozen!("scipy/interpolate/__init__.py"),
        "scipy.spatial" => frozen!("scipy/spatial/__init__.py"),
        "scipy.spatial.distance" => frozen!("scipy/spatial/distance.py"),
        "scipy.special" => frozen!("scipy/special/__init__.py"),
        "scipy.special._ufuncs" => frozen!("scipy/special/_ufuncs.py"),
        "scipy.stats" => frozen!("scipy/stats/__init__.py"),
        "scipy.stats._distributions" => frozen!("scipy/stats/_distributions.py"),
        "scipy.stats._stats" => frozen!("scipy/stats/_stats.py"),
        "scipy.stats._tests" => frozen!("scipy/stats/_tests.py"),
        "scipy.stats.contingency" => frozen!("scipy/stats/contingency.py"),
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
