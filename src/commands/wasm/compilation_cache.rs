//! Host-private persistent Wasmtime compilation artifacts.
//!
//! Only host process configuration selects this cache. Guest paths, environment variables,
//! and native-code bytes never enter it. Wasmtime keys entries by the Wasm bytes, compiler,
//! target ISA, settings, and our patched runtime identity. Cache state does not affect guest
//! admission or resource charges. An unavailable cache leaves compilation enabled.

use std::path::{Path, PathBuf};
use wasmtime::{Cache, CacheConfig, Error};

pub(super) const BUILD_ID: &str = env!("SHELLSIM_WASMTIME_BUILD_ID");
const DEFAULT_CACHE_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Configure the process-wide cache once, before any command engine is created.
/// A host configuration error is reported on the host's stderr, outside guest descriptors.
pub(super) fn from_host() -> Option<Cache> {
    let directory = std::env::var_os("SHELLSIM_WASMTIME_CACHE_DIR");
    if directory.as_deref() == Some(std::ffi::OsStr::new("off")) {
        return None;
    }
    let configured = std::env::var_os("SHELLSIM_WASMTIME_CACHE_CONFIG");
    let result = (|| {
        let mut config = match configured {
            Some(path) => CacheConfig::from_file(Some(Path::new(&path)))?,
            None => {
                let mut config = CacheConfig::new();
                config.with_files_total_size_soft_limit(DEFAULT_CACHE_BYTES);
                config
            }
        };
        let directory = directory
            .map(PathBuf::from)
            .or_else(|| config.directory().cloned())
            .or_else(|| {
                directories_next::BaseDirs::new()
                    .map(|base| base.cache_dir().join("shellsim/wasmtime"))
            })
            .ok_or_else(|| Error::msg("host cache directory is unavailable"))?;
        private_directory(&directory)?;
        config.with_directory(directory);
        Cache::new(config)
    })();
    match result {
        Ok(cache) => Some(cache),
        Err(error) => {
            eprintln!("shellsim: persistent Wasmtime cache unavailable: {error}");
            None
        }
    }
}

/// Native artifacts are trusted host executable code. Require an absolute, private directory;
/// do not repair permissions on an existing path that may belong to another application.
fn private_directory(path: &Path) -> Result<(), Error> {
    if !path.is_absolute() {
        return Err(Error::msg("Wasmtime cache directory must be absolute"));
    }
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(path)?;
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.is_dir() {
        return Err(Error::msg("Wasmtime cache directory must not be a symlink"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o077 != 0 {
            return Err(Error::msg(
                "Wasmtime cache directory must be private (mode 0700)",
            ));
        }
    }
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::MetadataExt;
        // The host's uid is not exposed to any simulated program.
        if metadata.uid() != rustix::process::geteuid().as_raw() {
            return Err(Error::msg(
                "Wasmtime cache directory belongs to another user",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use wasmtime::{Config, Engine, Module, ModuleVersionStrategy};

    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "shellsim-wasmtime-cache-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            private_directory(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            // The worker may still be writing usage statistics to the old path.
            let retired = self.0.with_extension("retired");
            std::fs::rename(&self.0, &retired).unwrap();
            std::fs::remove_dir_all(retired).unwrap();
        }
    }

    fn engine(path: &Path, version: &str) -> (Engine, Cache) {
        let mut cache_config = CacheConfig::new();
        cache_config.with_directory(path);
        let cache = Cache::new(cache_config).unwrap();
        let mut config = Config::new();
        super::super::configure_command_engine(&mut config);
        config.cache(Some(cache.clone()));
        config
            .module_version(ModuleVersionStrategy::Custom(version.into()))
            .unwrap();
        (Engine::new(&config).unwrap(), cache)
    }

    #[test]
    fn fresh_engine_reuses_artifact_and_invalidates_changed_runtime_identity() {
        let scratch = Scratch::new();
        let source = wat::parse_str("(module (func (export \"_start\")))").unwrap();
        let (first, cache) = engine(&scratch.0, BUILD_ID);
        Module::new(&first, &source).unwrap();
        assert_eq!(cache.cache_misses(), 1);
        let (second, cache) = engine(&scratch.0, BUILD_ID);
        Module::new(&second, &source).unwrap();
        assert_eq!(cache.cache_hits(), 1);
        assert_eq!(cache.cache_misses(), 0);
        let (third, cache) = engine(&scratch.0, "changed-vendored-runtime");
        Module::new(&third, &source).unwrap();
        assert_eq!(cache.cache_hits(), 0);
        assert_eq!(cache.cache_misses(), 1);
    }

    #[test]
    fn fresh_process_reuses_cache_with_the_syscall_backstop_installed() {
        let scratch = Scratch::new();
        for mode in ["cold", "warm"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "commands::wasm::compilation_cache::tests::cache_child",
                ])
                .env("SHELLSIM_TEST_WASMTIME_CACHE", mode)
                .env("SHELLSIM_WASMTIME_CACHE_DIR", &scratch.0)
                .env_remove("SHELLSIM_WASMTIME_CACHE_CONFIG")
                .env_remove("SHELLSIM_NO_SANDBOX")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    #[test]
    fn cache_child() {
        let Ok(mode) = std::env::var("SHELLSIM_TEST_WASMTIME_CACHE") else {
            return;
        };
        crate::sandbox::apply();
        let cache = from_host().expect("private child cache");
        let mut config = Config::new();
        super::super::configure_command_engine(&mut config);
        config.cache(Some(cache.clone()));
        let engine = Engine::new(&config).unwrap();
        let source = wat::parse_str("(module (func (export \"_start\")))").unwrap();
        Module::new(&engine, &source).unwrap();
        assert_eq!(cache.cache_hits(), usize::from(mode == "warm"));
        assert_eq!(cache.cache_misses(), usize::from(mode == "cold"));
        let mut environment = crate::Environment::new();
        environment.vfs.write("/", "/app", &source, 0o755).unwrap();
        let (outcome, _, stderr) = environment.run_script_capture("/app");
        assert_eq!(
            outcome.exit_status,
            0,
            "{}",
            String::from_utf8_lossy(&stderr)
        );
    }

    #[test]
    fn rejects_relative_and_nonprivate_cache_directories() {
        assert!(private_directory(Path::new("relative-cache")).is_err());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let scratch = Scratch::new();
            std::fs::set_permissions(&scratch.0, std::fs::Permissions::from_mode(0o755)).unwrap();
            assert!(private_directory(&scratch.0).is_err());
        }
    }
}
