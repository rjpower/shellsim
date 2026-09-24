//! Opt-in external Doomgeneric build probe; GPL engine source and WAD are never bundled.
//!
//! Set SHELLSIM_DOOM_SOURCE to the external doomgeneric/doomgeneric source directory. This
//! ignored test imports that trusted tree only into the test VFS, then invokes virtual cc.

use std::path::PathBuf;

use shellsim::{display::KeyEvent, host_ingest::mount_host_tree, Environment, Limits};

const TINYCC: &[u8] = include_bytes!("fixtures/tinycc/tcc-shellsim-package.tar.gz");
const SYSROOT: &[u8] = include_bytes!("fixtures/wasi-libc/sysroot-34.tar.gz");
const PLATFORM: &[u8] = include_bytes!("fixtures/doom_probe/platform.c");

fn environment() -> Environment {
    let mut environment = Environment::with_limits(Limits {
        cpu: 400_000_000_000,
        memory: 512 * 1024 * 1024,
        disk: 512 * 1024 * 1024,
        output: 32 * 1024 * 1024,
    });
    for directory in ["/work", "/tcc", "/wasi-sysroot"] {
        environment.vfs.mkdir_all("/", directory).unwrap();
    }
    for (path, bytes) in [
        ("/work/tcc.tar.gz", TINYCC),
        ("/work/sysroot.tar.gz", SYSROOT),
    ] {
        environment.vfs.write("/", path, bytes, 0o644).unwrap();
    }
    for command in [
        "tar -xzf /work/tcc.tar.gz -C /tcc",
        "tar -xzf /work/sysroot.tar.gz -C /wasi-sysroot",
    ] {
        let (result, _, stderr) = environment.run_script_capture(command);
        assert_eq!(
            result.exit_status,
            0,
            "{}",
            String::from_utf8_lossy(&stderr)
        );
    }
    environment
        .vfs
        .chmod("/", "/tcc/tcc-shellsim.wasm", 0o755)
        .unwrap();
    environment
        .vfs
        .symlink("/", "/tcc/tcc-shellsim.wasm", "/usr/bin/cc")
        .unwrap();
    environment
}

#[test]
#[ignore = "requires separately obtained Doomgeneric source and Freedoom data"]
fn external_doomgeneric_builds_and_reacts_to_input_in_virtual_environment() {
    let source =
        PathBuf::from(std::env::var_os("SHELLSIM_DOOM_SOURCE").expect("set SHELLSIM_DOOM_SOURCE"));
    let makefile = std::fs::read_to_string(source.join("Makefile.soso")).unwrap();
    let objects = makefile
        .lines()
        .find_map(|line| line.strip_prefix("SRC_DOOM = "))
        .expect("Doomgeneric Makefile.soso source list");
    let sources: Vec<String> = objects
        .split_whitespace()
        .filter(|name| *name != "doomgeneric_soso.o")
        .map(|name| format!("/work/doom/{}", name.replace(".o", ".c")))
        .collect();

    let mut environment = environment();
    mount_host_tree(&mut environment, &source, "/work/doom").unwrap();
    // Zenity's host `system()` path is only an optional error dialog, not part of Doom's
    // simulation. Select its existing console-error branch for this virtual target.
    let system = environment.vfs.read("/", "/work/doom/i_system.c").unwrap();
    let system = String::from_utf8(system).unwrap();
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
        .unwrap();
    environment
        .vfs
        .write("/", "/work/doom/doomgeneric_shellsim.c", PLATFORM, 0o644)
        .unwrap();
    for (index, source) in sources.iter().enumerate() {
        let command = format!("cc -DSHELLSIM_WASI -c {source} -o /work/doom/{index}.o");
        let (result, _, stderr) = environment.run_script_capture(&command);
        assert_eq!(
            result.exit_status,
            0,
            "{command}: {}",
            String::from_utf8_lossy(&stderr)
        );
    }
    let objects = (0..sources.len())
        .map(|index| format!("/work/doom/{index}.o"))
        .collect::<Vec<_>>()
        .join(" ");
    // TinyCC retains custom import signatures when the platform adapter is compiled at link
    // time; its current ELF32 object path does not carry those import declarations through.
    let command = format!("cc {objects} /work/doom/doomgeneric_shellsim.c -o /work/doom/doom.wasm");
    let (result, _, stderr) = environment.run_script_capture(&command);
    assert_eq!(
        result.exit_status,
        0,
        "link: {}",
        String::from_utf8_lossy(&stderr)
    );
    assert!(environment.vfs.is_file("/", "/work/doom/doom.wasm"));
    let wad = PathBuf::from(std::env::var_os("SHELLSIM_DOOM_WAD").expect("set SHELLSIM_DOOM_WAD"));
    environment
        .vfs
        .write(
            "/",
            "/work/doom/freedoom1.wad",
            &std::fs::read(wad).unwrap(),
            0o644,
        )
        .unwrap();
    assert_eq!(
        &environment
            .vfs
            .read("/", "/work/doom/freedoom1.wad")
            .unwrap()[..4],
        b"IWAD"
    );
    let mut with_input = environment.clone();
    let (result, stdout, stderr) = environment.run_script_capture(
        "chmod +x /work/doom/doom.wasm && /work/doom/doom.wasm -iwad /work/doom/freedoom1.wad",
    );
    assert_eq!(
        result.exit_status,
        0,
        "run:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&stdout),
        String::from_utf8_lossy(&stderr)
    );
    let baseline = environment.display.frame().unwrap();
    assert_eq!((baseline.width, baseline.height), (640, 400));
    assert!(baseline.pixels.iter().any(|byte| *byte != 0));
    with_input
        .inject_key(KeyEvent {
            code: 27, // Doom's KEY_ESCAPE opens the menu.
            pressed: true,
        })
        .unwrap();
    let (result, stdout, stderr) = with_input.run_script_capture(
        "chmod +x /work/doom/doom.wasm && /work/doom/doom.wasm -iwad /work/doom/freedoom1.wad",
    );
    assert_eq!(
        result.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert!(!stdout.is_empty());
    let changed = with_input
        .display
        .frame()
        .unwrap()
        .pixels
        .iter()
        .zip(&baseline.pixels)
        .filter(|(left, right)| left != right)
        .count();
    assert!(changed > 0, "injected key did not change the final frame");
}
