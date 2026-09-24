//! Host-side opt-in builder for external Doomgeneric source and WAD data.
//!
//! The only host import occurs before execution. The guest compiler and game see only the VFS.

use std::path::Path;

use shellsim::{host_ingest::mount_host_tree, Environment, Limits};

const TINYCC: &[u8] = include_bytes!("../../tests/fixtures/tinycc/tcc-shellsim-package.tar.gz");
const SYSROOT: &[u8] = include_bytes!("../../tests/fixtures/wasi-libc/sysroot-34.tar.gz");
const PLATFORM: &[u8] = include_bytes!("../../tests/fixtures/doom_probe/platform.c");
pub const PROGRAM: &str = "/work/doom/doom.wasm";
pub const WAD: &str = "/work/doom/freedoom1.wad";

fn command(environment: &mut Environment, script: &str) -> Result<(), String> {
    let (result, stdout, stderr) = environment.run_script_capture(script);
    if result.exit_status != 0 {
        return Err(format!(
            "{script} failed ({}):\n{}{}",
            result.exit_status,
            String::from_utf8_lossy(&stdout),
            String::from_utf8_lossy(&stderr)
        ));
    }
    Ok(())
}

/// Build external Doomgeneric entirely inside the VFS and install separately supplied WAD data.
pub fn build(source: &Path, wad: &Path) -> Result<Environment, String> {
    let makefile = std::fs::read_to_string(source.join("Makefile.soso"))
        .map_err(|error| format!("read Doomgeneric source list: {error}"))?;
    let objects = makefile
        .lines()
        .find_map(|line| line.strip_prefix("SRC_DOOM = "))
        .ok_or("Doomgeneric Makefile.soso has no SRC_DOOM list")?;
    let sources: Vec<String> = objects
        .split_whitespace()
        .filter(|name| *name != "doomgeneric_soso.o")
        .map(|name| format!("/work/doom/{}", name.replace(".o", ".c")))
        .collect();
    let mut environment = Environment::with_limits(Limits {
        cpu: 400_000_000_000,
        memory: 512 * 1024 * 1024,
        disk: 512 * 1024 * 1024,
        output: 32 * 1024 * 1024,
    });
    for directory in ["/work", "/tcc", "/wasi-sysroot"] {
        environment
            .vfs
            .mkdir_all("/", directory)
            .map_err(|error| error.to_string())?;
    }
    for (path, bytes) in [
        ("/work/tcc.tar.gz", TINYCC),
        ("/work/sysroot.tar.gz", SYSROOT),
    ] {
        environment
            .vfs
            .write("/", path, bytes, 0o644)
            .map_err(|error| error.to_string())?;
    }
    command(&mut environment, "tar -xzf /work/tcc.tar.gz -C /tcc")?;
    command(
        &mut environment,
        "tar -xzf /work/sysroot.tar.gz -C /wasi-sysroot",
    )?;
    environment
        .vfs
        .chmod("/", "/tcc/tcc-shellsim.wasm", 0o755)
        .map_err(|error| error.to_string())?;
    environment
        .vfs
        .symlink("/", "/tcc/tcc-shellsim.wasm", "/usr/bin/cc")
        .map_err(|error| error.to_string())?;
    mount_host_tree(&mut environment, source, "/work/doom")
        .map_err(|error| format!("import Doomgeneric source: {error}"))?;
    // Select Doomgeneric's existing console-error path; its optional Zenity system() call is
    // neither required by the game nor available to a simulated process.
    let system = environment
        .vfs
        .read("/", "/work/doom/i_system.c")
        .map_err(|error| error.to_string())?;
    let system = String::from_utf8(system).map_err(|error| error.to_string())?;
    let system = system
        .replace(
            "!defined(__DJGPP__)",
            "!defined(__DJGPP__) && !defined(SHELLSIM_WASI)",
        )
        .replace(
            "#elif defined(__DJGPP__)",
            "#elif defined(__DJGPP__) || defined(SHELLSIM_WASI)",
        );
    environment
        .vfs
        .write("/", "/work/doom/i_system.c", system.as_bytes(), 0o644)
        .map_err(|error| error.to_string())?;
    environment
        .vfs
        .write("/", "/work/doom/doomgeneric_shellsim.c", PLATFORM, 0o644)
        .map_err(|error| error.to_string())?;
    for (index, source) in sources.iter().enumerate() {
        command(
            &mut environment,
            &format!("cc -DSHELLSIM_WASI -c {source} -o /work/doom/{index}.o"),
        )?;
    }
    let objects = (0..sources.len())
        .map(|index| format!("/work/doom/{index}.o"))
        .collect::<Vec<_>>()
        .join(" ");
    // TinyCC currently preserves the custom import signatures if the adapter is compiled at
    // link time, not if it is passed through its ELF32 object path.
    command(
        &mut environment,
        &format!("cc {objects} /work/doom/doomgeneric_shellsim.c -o {PROGRAM}"),
    )?;
    let data = std::fs::read(wad).map_err(|error| format!("read WAD: {error}"))?;
    if !data.starts_with(b"IWAD") {
        return Err("expected an IWAD file".into());
    }
    environment
        .vfs
        .write("/", WAD, &data, 0o644)
        .map_err(|error| error.to_string())?;
    environment
        .vfs
        .chmod("/", PROGRAM, 0o755)
        .map_err(|error| error.to_string())?;
    Ok(environment)
}
