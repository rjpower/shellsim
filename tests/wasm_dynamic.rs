// Synthetic ABI failures run everywhere. Real SDK/CPython artifacts are opt-in and come from
// tests/fixtures/wasm/dynamic/build.py; guest file reads and writes use only the virtual filesystem.
use shellsim::{Environment, Limits};
use std::path::{Path, PathBuf};

fn environment() -> Environment {
    Environment::with_limits(Limits {
        cpu: 2_000_000_000,
        memory: 256 * 1024 * 1024,
        disk: 64 * 1024 * 1024,
        ..Limits::default()
    })
}

fn artifacts() -> PathBuf {
    PathBuf::from(
        std::env::var_os("SHELLSIM_DYNAMIC_ARTIFACTS")
            .expect("set SHELLSIM_DYNAMIC_ARTIFACTS to the built fixture directory"),
    )
}

fn put(environment: &mut Environment, source: &Path, target: &str, executable: bool) {
    let parent = target.rsplit_once('/').unwrap().0;
    environment.vfs.mkdir_all("/", parent).unwrap();
    environment
        .vfs
        .write(
            "/",
            target,
            &std::fs::read(source).unwrap(),
            if executable { 0o755 } else { 0o644 },
        )
        .unwrap();
}

fn run(environment: &mut Environment, command: &str) -> (i32, Vec<u8>, Vec<u8>) {
    let (outcome, stdout, stderr) = environment.run_script_capture(command);
    (outcome.exit_status, stdout, stderr)
}

#[test]
fn dynamic_bridge_rejects_an_unmarked_executable() {
    let mut environment = environment();
    let bytes = wat::parse_str(
        r#"(module
        (import "shellsim_dylink_v1" "open" (func (param i32 i32 i32) (result i32)))
        (memory (export "memory") 1)
        (func (export "_start")))"#,
    )
    .unwrap();
    environment.vfs.write("/", "/app", &bytes, 0o755).unwrap();
    let (status, _, stderr) = run(&mut environment, "/app");
    assert_eq!(status, 126);
    assert!(String::from_utf8(stderr)
        .unwrap()
        .contains("dynamic loading ABI mismatch"));
}

#[test]
#[ignore = "requires built SDK24 dynamic artifacts"]
fn c_library_shares_data_callbacks_constructors_and_handles() {
    let artifacts = artifacts();
    let mut environment = environment();
    put(&mut environment, &artifacts.join("main.wasm"), "/app", true);
    put(
        &mut environment,
        &artifacts.join("library.so"),
        "/lib/libfixture.so",
        false,
    );
    assert_eq!(run(&mut environment, "/app"), (0,
        b"128 14\n230 15\nshared data, callback, constructor, repeat load, missing symbol: ok\n".to_vec(), Vec::new()));
}

#[test]
#[ignore = "requires built SDK24 dynamic artifacts"]
fn c_library_rejects_wrong_abi_and_missing_imports() {
    let artifacts = artifacts();
    let mut environment = environment();
    put(&mut environment, &artifacts.join("main.wasm"), "/app", true);
    put(
        &mut environment,
        &artifacts.join("wrong-abi.so"),
        "/bad.so",
        false,
    );
    let (status, stdout, stderr) = run(&mut environment, "/app /bad.so");
    assert_eq!((status, stderr), (1, Vec::new()));
    assert!(String::from_utf8(stdout)
        .unwrap()
        .contains("dynamic loading ABI mismatch"));
    put(
        &mut environment,
        &artifacts.join("tiny_one.so"),
        "/missing.so",
        false,
    );
    let (status, stdout, stderr) = run(&mut environment, "/app /missing.so");
    assert_eq!((status, stderr), (1, Vec::new()));
    assert!(String::from_utf8(stdout)
        .unwrap()
        .contains("missing dynamic symbol: Py"));
}

#[test]
#[ignore = "requires built SDK24 dynamic artifacts"]
fn dynamic_execution_obeys_cpu_and_memory_limits() {
    let artifacts = artifacts();
    let main = std::fs::read(artifacts.join("main.wasm")).unwrap();
    let mut environment = Environment::with_limits(Limits {
        cpu: 10,
        ..Limits::default()
    });
    environment.vfs.write("/", "/app", &main, 0o755).unwrap();
    assert_eq!(run(&mut environment, "/app").0, 137);
    let mut environment = Environment::with_limits(Limits {
        memory: 1024 * 1024,
        ..Limits::default()
    });
    environment.vfs.write("/", "/app", &main, 0o755).unwrap();
    assert_eq!(run(&mut environment, "/app").0, 137);
}

fn mount_tree(environment: &mut Environment, root: &Path, directory: &Path) {
    let target = format!("/{}", directory.strip_prefix(root).unwrap().display());
    environment.vfs.mkdir_all("/", &target).unwrap();
    let mut entries = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();
    entries.sort();
    for entry in entries {
        if entry.is_dir() {
            mount_tree(environment, root, &entry);
        } else {
            let target = format!("/{}", entry.strip_prefix(root).unwrap().display());
            put(environment, &entry, &target, target.ends_with(".wasm"));
        }
    }
}

#[test]
#[ignore = "requires built SDK24 dynamic artifacts"]
fn cpython_imports_two_independent_extensions_into_one_live_interpreter() {
    let artifacts = artifacts();
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(artifacts.join("manifest.json")).unwrap()).unwrap();
    let root = PathBuf::from(manifest["source_bundle"].as_str().unwrap()).join("rootfs");
    let interpreter = std::fs::read(artifacts.join("python3.wasm")).unwrap();
    let interpreter_hash = <sha2::Sha256 as sha2::Digest>::digest(&interpreter);
    assert_eq!(
        format!("{interpreter_hash:x}"),
        manifest["interpreter_sha256"].as_str().unwrap()
    );
    let mut environment = environment();
    mount_tree(&mut environment, &root, &root);
    put(
        &mut environment,
        &artifacts.join("python3.wasm"),
        "/usr/bin/python3.wasm",
        true,
    );
    put(
        &mut environment,
        &artifacts.join("tiny_one.so"),
        "/usr/lib/python3.13/site-packages/tiny_one.so",
        false,
    );
    put(
        &mut environment,
        &artifacts.join("tiny_two.so"),
        "/tmp/tiny_two.pending",
        false,
    );
    let source = r#"import importlib, sys
import tiny_one
assert tiny_one.value(2) == 19
try:
    tiny_one.value('invalid')
except TypeError:
    pass
else:
    raise AssertionError('expected TypeError')
with open('/tmp/tiny_two.pending', 'rb') as incoming:
    with open('/usr/lib/python3.13/site-packages/tiny_two.so', 'wb') as extension:
        extension.write(incoming.read())
importlib.invalidate_caches()
import tiny_two
assert tiny_two.value(3) == 43
assert tiny_one.value(1) == 20
assert importlib.import_module('tiny_one') is tiny_one
assert tiny_two.value(2) == 45
print('two independent C extensions, late VFS staging, shared Python API and preserved state: ok')
"#;
    environment
        .vfs
        .write("/", "/proof.py", source.as_bytes(), 0o644)
        .unwrap();
    assert_eq!(run(&mut environment, "PYTHONHOME=/usr /usr/bin/python3.wasm /proof.py"), (0,
        b"two independent C extensions, late VFS staging, shared Python API and preserved state: ok\n".to_vec(), Vec::new()));
    assert_eq!(
        environment.vfs.read("/", "/usr/bin/python3.wasm").unwrap(),
        interpreter
    );
}

fn marked(source: &str) -> Vec<u8> {
    wat::parse_str(source).unwrap()
}

fn synthetic_main(body: &str) -> Vec<u8> {
    synthetic_main_profile(false, body, "")
}

fn synthetic_main_profile(v2: bool, body: &str, extra: &str) -> Vec<u8> {
    let (abi, namespace) = if v2 {
        ("shellsim-wasi-sdk34-cpython3137-v2", "shellsim_dylink_v2")
    } else {
        ("shellsim-wasi-sdk24-cpython3137-v1", "shellsim_dylink_v1")
    };
    marked(&format!(
        r#"(module
        (@custom "shellsim.abi" "{abi}")
        (import "{namespace}" "open" (func $open (param i32 i32 i32) (result i32)))
        (import "{namespace}" "symbol" (func $symbol (param i32 i32 i32) (result i32)))
        (memory (export "memory") 1)
        (table (export "__indirect_function_table") 1 funcref)
        (global (export "__stack_pointer") (mut i32) (i32.const 65536))
        (global $heap (mut i32) (i32.const 4096))
        (data (i32.const 32) "/lib.so")
        (data (i32.const 64) "answer")
        {extra}
        (export "fixture_open" (func $open))
        (func (export "malloc") (param $bytes i32) (result i32) (local $base i32)
            (local.set $base (global.get $heap))
            (global.set $heap (i32.add (global.get $heap) (local.get $bytes)))
            (local.get $base))
        (func (export "_start") (local $handle i32) {body}))"#
    ))
}

fn synthetic_library(metadata: &str, extra: &str) -> Vec<u8> {
    synthetic_library_profile(false, metadata, extra)
}

fn synthetic_library_profile(v2: bool, metadata: &str, extra: &str) -> Vec<u8> {
    let abi = if v2 {
        "shellsim-wasi-sdk34-cpython3137-v2"
    } else {
        "shellsim-wasi-sdk24-cpython3137-v1"
    };
    marked(&format!(
        r#"(module
        (@custom "shellsim.abi" "{abi}")
        (@custom "dylink.0" "{metadata}")
        (import "env" "memory" (memory 1))
        {extra}
        (func (export "answer") (result i32) (i32.const 42)))"#
    ))
}

#[test]
fn synthetic_late_load_resolves_exported_function_and_repeated_handle() {
    let mut environment = environment();
    let main = synthetic_main(
        r#"
        (local.set $handle (call $open (i32.const 32) (i32.const 7) (i32.const 2)))
        (if (i32.eqz (local.get $handle)) (then unreachable))
        (if (i32.ne (call $open (i32.const 32) (i32.const 7) (i32.const 2)) (local.get $handle)) (then unreachable))
        (if (i32.ne (call_indirect (result i32)
            (call $symbol (local.get $handle) (i32.const 64) (i32.const 6))) (i32.const 42)) (then unreachable))
    "#,
    );
    let library = synthetic_library(r"\01\04\00\00\00\00", "");
    environment.vfs.write("/", "/app", &main, 0o755).unwrap();
    environment
        .vfs
        .write("/", "/lib.so", &library, 0o644)
        .unwrap();
    assert_eq!(run(&mut environment, "/app"), (0, Vec::new(), Vec::new()));
}

#[test]
fn dynamic_constructor_cpu_is_charged() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 50_000,
        memory: 128 * 1024 * 1024,
        ..Limits::default()
    });
    let main = synthetic_main("(drop (call $open (i32.const 32) (i32.const 7) (i32.const 2)))");
    let library = synthetic_library(
        r"\01\04\00\00\00\00",
        r#"(func (export "__wasm_call_ctors") (loop $again (br $again)))"#,
    );
    environment.vfs.write("/", "/app", &main, 0o755).unwrap();
    environment
        .vfs
        .write("/", "/lib.so", &library, 0o644)
        .unwrap();
    assert_eq!(run(&mut environment, "/app").0, 137);
}

#[test]
fn dynamic_constructor_cannot_recursively_load_a_library() {
    let mut environment = environment();
    let main = synthetic_main(
        "(if (i32.eqz (call $open (i32.const 32) (i32.const 7) (i32.const 2))) (then unreachable))",
    );
    let library = synthetic_library(
        r"\01\04\00\00\00\00",
        r#"
        (import "env" "fixture_open" (func $open (param i32 i32 i32) (result i32)))
        (func (export "__wasm_call_ctors")
            (if (call $open (i32.const 32) (i32.const 7) (i32.const 2)) (then unreachable)))
        "#,
    );
    environment.vfs.write("/", "/app", &main, 0o755).unwrap();
    environment
        .vfs
        .write("/", "/lib.so", &library, 0o644)
        .unwrap();
    assert_eq!(run(&mut environment, "/app"), (0, Vec::new(), Vec::new()));
}

#[test]
fn dynamic_compile_cpu_is_charged_on_cache_hits() {
    let main = synthetic_main("(drop (call $open (i32.const 32) (i32.const 7) (i32.const 2)))");
    let library = synthetic_library(r"\01\04\00\00\00\00", "");
    let mut warm = environment();
    warm.vfs.write("/", "/app", &main, 0o755).unwrap();
    warm.vfs.write("/", "/lib.so", &library, 0o644).unwrap();
    assert_eq!(run(&mut warm, "/app").0, 0);
    let mut limited = Environment::with_limits(Limits {
        cpu: main.len() as u64 * 10 + library.len() as u64 * 5,
        memory: 128 * 1024 * 1024,
        ..Limits::default()
    });
    limited.vfs.write("/", "/app", &main, 0o755).unwrap();
    limited.vfs.write("/", "/lib.so", &library, 0o644).unwrap();
    assert_eq!(run(&mut limited, "/app").0, 137);
}

#[test]
fn dynamic_loader_rejects_extra_memory_needed_libraries_and_overflow() {
    for (metadata, extra) in [
        (r"\01\04\00\00\00\00", "(memory 1)"),
        (r"\01\04\00\00\00\00\02\03\01\01x", ""),
        (r"\01\08\ff\ff\ff\ff\10\00\00\00", ""),
    ] {
        let mut environment = environment();
        let main = synthetic_main(
            r#"(if (call $open (i32.const 32) (i32.const 7) (i32.const 2)) (then unreachable))"#,
        );
        let library = synthetic_library(metadata, extra);
        environment.vfs.write("/", "/app", &main, 0o755).unwrap();
        environment
            .vfs
            .write("/", "/lib.so", &library, 0o644)
            .unwrap();
        assert_eq!(run(&mut environment, "/app"), (0, Vec::new(), Vec::new()));
    }
}

#[test]
fn dynamic_library_symbols_follow_local_and_global_scope() {
    for global in [false, true] {
        let mut environment = environment();
        let flags = if global { 258 } else { 2 };
        let expected = if global { 0 } else { 1 };
        let body = format!(
            r#"
            (if (i32.eqz (call $open (i32.const 32) (i32.const 7) (i32.const {flags}))) (then unreachable))
            (i32.store8 (i32.const 33) (i32.const 117))
            (if (i32.ne (i32.eqz (call $open (i32.const 32) (i32.const 7) (i32.const 2))) (i32.const {expected})) (then unreachable))
        "#
        );
        let main = synthetic_main(&body);
        let provider = synthetic_library(r"\01\04\00\00\00\00", "");
        let consumer = synthetic_library(
            r"\01\04\00\00\00\00",
            r#"(import "env" "answer" (func (result i32)))"#,
        );
        environment.vfs.write("/", "/app", &main, 0o755).unwrap();
        environment
            .vfs
            .write("/", "/lib.so", &provider, 0o644)
            .unwrap();
        environment
            .vfs
            .write("/", "/uib.so", &consumer, 0o644)
            .unwrap();
        assert_eq!(run(&mut environment, "/app"), (0, Vec::new(), Vec::new()));
    }
}

#[test]
fn dynamic_library_count_is_bounded() {
    let mut environment = environment();
    let mut body = String::new();
    let library = synthetic_library(r"\01\04\00\00\00\00", "");
    for index in 0..33 {
        let name = format!("{index:02}");
        let value = u16::from_le_bytes(name.as_bytes().try_into().unwrap());
        body.push_str(&format!("(i32.store16 (i32.const 33) (i32.const {value}))"));
        let expected = if index < 32 { 0 } else { 1 };
        body.push_str(&format!("(if (i32.ne (i32.eqz (call $open (i32.const 32) (i32.const 7) (i32.const 2))) (i32.const {expected})) (then unreachable))"));
        environment
            .vfs
            .write("/", &format!("/{name}b.so"), &library, 0o644)
            .unwrap();
    }
    let main = synthetic_main(&body);
    environment.vfs.write("/", "/app", &main, 0o755).unwrap();
    assert_eq!(run(&mut environment, "/app"), (0, Vec::new(), Vec::new()));
}

#[test]
fn dynamic_constructor_proc_exit_terminates_the_process() {
    let mut environment = environment();
    let main = synthetic_main("(drop (call $open (i32.const 32) (i32.const 7) (i32.const 2)))");
    let library = synthetic_library(
        r"\01\04\00\00\00\00",
        r#"
        (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
        (func (export "__wasm_call_ctors") (call $exit (i32.const 7)))
    "#,
    );
    environment.vfs.write("/", "/app", &main, 0o755).unwrap();
    environment
        .vfs
        .write("/", "/lib.so", &library, 0o644)
        .unwrap();
    assert_eq!(run(&mut environment, "/app"), (7, Vec::new(), Vec::new()));
}

#[test]
fn v2_side_exception_uses_the_main_tag_identity() {
    let mut environment = environment();
    let main = synthetic_main_profile(
        true,
        r#"
        (local.set $handle (call $open (i32.const 32) (i32.const 7) (i32.const 2)))
        (if (i32.eqz (local.get $handle)) (then unreachable))
        (if (i32.ne
          (block $caught (result i32)
            (try_table (catch $cpp $caught)
              (call_indirect
                (call $symbol (local.get $handle) (i32.const 64) (i32.const 13))))
            (i32.const 0))
          (i32.const 42)) (then unreachable))
        "#,
        r#"(tag $cpp (export "__cpp_exception") (param i32))
        (data (i32.const 64) "throw_fixture")"#,
    );
    let library = synthetic_library_profile(
        true,
        r"\01\04\00\00\00\00",
        r#"(import "env" "__cpp_exception" (tag $cpp (param i32)))
        (func (export "throw_fixture") (throw $cpp (i32.const 42)))"#,
    );
    environment.vfs.write("/", "/app", &main, 0o755).unwrap();
    environment
        .vfs
        .write("/", "/lib.so", &library, 0o644)
        .unwrap();
    assert_eq!(run(&mut environment, "/app"), (0, Vec::new(), Vec::new()));
    assert_eq!(environment.resources.memory_mark(), 0);
}

#[test]
fn v2_rejects_private_side_tags_without_names_or_exports() {
    for extra in [
        "(tag (param i32))",
        r#"(tag (export "__cpp_exception") (param i32))"#,
    ] {
        let mut environment = environment();
        let main = synthetic_main_profile(
            true,
            "(if (call $open (i32.const 32) (i32.const 7) (i32.const 2)) (then unreachable))",
            "",
        );
        let library = synthetic_library_profile(true, r"\01\04\00\00\00\00", extra);
        environment.vfs.write("/", "/app", &main, 0o755).unwrap();
        environment
            .vfs
            .write("/", "/lib.so", &library, 0o644)
            .unwrap();
        assert_eq!(run(&mut environment, "/app"), (0, Vec::new(), Vec::new()));
        assert_eq!(environment.resources.memory_mark(), 0);
    }
}

#[test]
fn v2_rejects_a_v1_side_library_and_releases_compile_scratch() {
    let mut environment = environment();
    let main = synthetic_main_profile(
        true,
        "(if (call $open (i32.const 32) (i32.const 7) (i32.const 2)) (then unreachable))",
        "",
    );
    let library = synthetic_library(r"\01\04\00\00\00\00", "");
    environment.vfs.write("/", "/app", &main, 0o755).unwrap();
    environment
        .vfs
        .write("/", "/lib.so", &library, 0o644)
        .unwrap();
    assert_eq!(run(&mut environment, "/app"), (0, Vec::new(), Vec::new()));
    assert_eq!(environment.resources.memory_mark(), 0);
}

#[test]
fn v2_constructor_cancellation_releases_retained_images() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 50_000,
        memory: 128 * 1024 * 1024,
        ..Limits::default()
    });
    let main = synthetic_main_profile(
        true,
        "(drop (call $open (i32.const 32) (i32.const 7) (i32.const 2)))",
        "",
    );
    let library = synthetic_library_profile(
        true,
        r"\01\04\00\00\00\00",
        r#"(func (export "__wasm_call_ctors") (loop $again (br $again)))"#,
    );
    environment.vfs.write("/", "/app", &main, 0o755).unwrap();
    environment
        .vfs
        .write("/", "/lib.so", &library, 0o644)
        .unwrap();
    assert_eq!(run(&mut environment, "/app").0, 137);
    assert_eq!(environment.resources.memory_mark(), 0);
}

#[test]
#[ignore = "requires built SDK34 dynamic artifacts"]
fn sdk34_cross_module_cpp_typed_catch_rethrow_and_destructors() {
    let artifacts = PathBuf::from(
        std::env::var_os("SHELLSIM_DYNAMIC_V2_ARTIFACTS")
            .expect("set SHELLSIM_DYNAMIC_V2_ARTIFACTS to the verified SDK34 fixture directory"),
    );
    let mut environment = Environment::with_limits(Limits {
        cpu: 2_000_000_000,
        memory: 1024 * 1024 * 1024,
        disk: 64 * 1024 * 1024,
        ..Limits::default()
    });
    put(&mut environment, &artifacts.join("main.wasm"), "/app", true);
    put(
        &mut environment,
        &artifacts.join("exception.so"),
        "/lib/exception.so",
        false,
    );
    assert_eq!(
        run(&mut environment, "/app"),
        (
            0,
            b"cross-module typed catch/rethrow/destructors: 3\n".to_vec(),
            Vec::new()
        )
    );
    assert_eq!(environment.resources.memory_mark(), 0);
}

#[test]
#[ignore = "requires built SDK34 dynamic artifacts"]
fn sdk34_cpython_imports_independent_c_and_cpp_extensions() {
    let artifacts = PathBuf::from(std::env::var_os("SHELLSIM_DYNAMIC_V2_ARTIFACTS").unwrap());
    let mut environment = Environment::with_limits(Limits {
        cpu: 2_000_000_000,
        memory: 1024 * 1024 * 1024,
        disk: 64 * 1024 * 1024,
        ..Limits::default()
    });
    let root = artifacts.join("rootfs");
    mount_tree(&mut environment, &root, &root);
    put(
        &mut environment,
        &artifacts.join("python3.wasm"),
        "/usr/bin/python3.wasm",
        true,
    );
    for name in ["tiny_one", "cpp_one"] {
        put(
            &mut environment,
            &artifacts.join(format!("{name}.so")),
            &format!("/usr/lib/python3.13/site-packages/{name}.so"),
            false,
        );
    }
    for name in ["tiny_two", "cpp_two"] {
        put(
            &mut environment,
            &artifacts.join(format!("{name}.so")),
            &format!("/tmp/{name}.pending"),
            false,
        );
    }
    let interpreter_before = environment.vfs.read("/", "/usr/bin/python3.wasm").unwrap();
    let source = r#"import tiny_one, cpp_one, importlib
for name in ('tiny_two', 'cpp_two'):
    with open('/tmp/' + name + '.pending', 'rb') as incoming:
        with open('/usr/lib/python3.13/site-packages/' + name + '.so', 'wb') as extension:
            extension.write(incoming.read())
importlib.invalidate_caches()
import tiny_two, cpp_two
assert tiny_one.value(2) == 19
assert tiny_two.value(3) == 43
assert cpp_one.catch_call(cpp_two.thrower()) == 'independent C++ extension'
assert cpp_two.catch_call(cpp_one.thrower()) == 'independent C++ extension'
assert cpp_one.destroyed() == 2
assert cpp_two.destroyed() == 2
print('independent C and C++ extension imports: ok')
"#;
    environment
        .vfs
        .write("/", "/probe.py", source.as_bytes(), 0o644)
        .unwrap();
    assert_eq!(
        run(&mut environment, "/usr/bin/python3.wasm /probe.py"),
        (
            0,
            b"independent C and C++ extension imports: ok\n".to_vec(),
            Vec::new()
        )
    );
    assert_eq!(environment.resources.memory_mark(), 0);
    assert_eq!(
        environment.vfs.read("/", "/usr/bin/python3.wasm").unwrap(),
        interpreter_before
    );
}

#[test]
#[ignore = "requires built SDK34 dynamic artifacts"]
fn sdk34_c_shared_data_and_callbacks() {
    let artifacts = PathBuf::from(std::env::var_os("SHELLSIM_DYNAMIC_V2_ARTIFACTS").unwrap());
    let mut environment = environment();
    put(
        &mut environment,
        &artifacts.join("c_main.wasm"),
        "/app",
        true,
    );
    put(
        &mut environment,
        &artifacts.join("library.so"),
        "/lib/libfixture.so",
        false,
    );
    assert_eq!(run(&mut environment, "/app"), (0,
        b"128 14\n230 15\nshared data, callback, constructor, repeat load, missing symbol: ok\n".to_vec(), Vec::new()));
    assert_eq!(environment.resources.memory_mark(), 0);
}

#[test]
fn v2_declared_diamond_dependencies_share_state_and_remain_local() {
    let mut environment = environment();
    let main = synthetic_main_profile(
        true,
        r#"
        (local.set $handle (call $open (i32.const 32) (i32.const 7) (i32.const 2)))
        (if (i32.eqz (local.get $handle)) (then unreachable))
        (if (i32.ne (call_indirect (result i32)
            (call $symbol (local.get $handle) (i32.const 64) (i32.const 5))) (i32.const 3)) (then unreachable))
        (if (call $symbol (i32.const 0) (i32.const 80) (i32.const 4)) (then unreachable))
    "#,
        r#"(data (i32.const 64) "probe") (data (i32.const 80) "bump")"#,
    );
    let libraries = [
        (
            "/lib.so",
            synthetic_library_profile(
                true,
                r"\01\04\00\00\00\00\02\12\02\07left.so\08right.so",
                r#"(import "env" "left" (func $left (result i32)))
            (import "env" "right" (func $right (result i32)))
            (func (export "probe") (result i32) (i32.add (call $left) (call $right)))"#,
            ),
        ),
        (
            "/lib/left.so",
            synthetic_library_profile(
                true,
                r"\01\04\00\00\00\00\02\09\01\07leaf.so",
                r#"(import "env" "bump" (func $bump (result i32)))
            (func (export "left") (result i32) (call $bump))"#,
            ),
        ),
        (
            "/lib/right.so",
            synthetic_library_profile(
                true,
                r"\01\04\00\00\00\00\02\09\01\07leaf.so",
                r#"(import "env" "bump" (func $bump (result i32)))
            (func (export "right") (result i32) (call $bump))"#,
            ),
        ),
        (
            "/lib/leaf.so",
            synthetic_library_profile(
                true,
                r"\01\04\00\00\00\00",
                r#"(global $count (mut i32) (i32.const 0))
            (func (export "bump") (result i32)
                (global.set $count (i32.add (global.get $count) (i32.const 1)))
                (global.get $count))"#,
            ),
        ),
    ];
    environment.vfs.mkdir_all("/", "/lib").unwrap();
    environment.vfs.write("/", "/app", &main, 0o755).unwrap();
    for (path, source) in libraries {
        environment.vfs.write("/", path, &source, 0o644).unwrap();
    }
    assert_eq!(run(&mut environment, "/app"), (0, Vec::new(), Vec::new()));
    assert_eq!(environment.resources.memory_mark(), 0);
}

#[test]
fn v2_dependencies_load_from_local_lib_after_the_lib_default() {
    for preferred in [false, true] {
        let mut environment = environment();
        let expected = if preferred { 11 } else { 42 };
        let main = synthetic_main_profile(
            true,
            &format!(
                r#"
            (local.set $handle (call $open (i32.const 32) (i32.const 7) (i32.const 2)))
            (if (i32.eqz (local.get $handle)) (then unreachable))
            (if (i32.ne (call_indirect (result i32)
                (call $symbol (local.get $handle) (i32.const 64) (i32.const 5)))
                (i32.const {expected})) (then unreachable))"#
            ),
            r#"(data (i32.const 64) "probe")"#,
        );
        let root = synthetic_library_profile(
            true,
            r"\01\04\00\00\00\00\02\09\01\07leaf.so",
            r#"(import "env" "value" (func $value (result i32)))
                (func (export "probe") (result i32) (call $value))"#,
        );
        environment.vfs.write("/", "/app", &main, 0o755).unwrap();
        environment.vfs.write("/", "/lib.so", &root, 0o644).unwrap();
        for (directory, value) in [("/usr/local/lib", 42), ("/lib", 11)] {
            if directory == "/lib" && !preferred {
                continue;
            }
            environment.vfs.mkdir_all("/", directory).unwrap();
            let leaf = synthetic_library_profile(
                true,
                r"\01\04\00\00\00\00",
                &format!(r#"(func (export "value") (result i32) (i32.const {value}))"#),
            );
            environment
                .vfs
                .write("/", &format!("{directory}/leaf.so"), &leaf, 0o644)
                .unwrap();
        }
        assert_eq!(run(&mut environment, "/app"), (0, Vec::new(), Vec::new()));
        assert_eq!(environment.resources.memory_mark(), 0);
    }
}

#[test]
fn v2_dependency_failures_release_memory() {
    for failure in ["missing", "mismatch", "cycle", "filename", "directory"] {
        let mut environment = environment();
        let main = synthetic_main_profile(
            true,
            "(if (call $open (i32.const 32) (i32.const 7) (i32.const 2)) (then unreachable))",
            "",
        );
        let metadata = if failure == "filename" {
            r"\01\04\00\00\00\00\02\0b\01\09../bad.so"
        } else {
            r"\01\04\00\00\00\00\02\09\01\07leaf.so"
        };
        let root = synthetic_library_profile(true, metadata, "");
        environment.vfs.mkdir_all("/", "/lib").unwrap();
        environment.vfs.write("/", "/app", &main, 0o755).unwrap();
        environment.vfs.write("/", "/lib.so", &root, 0o644).unwrap();
        let leaf = match failure {
            "cycle" => {
                synthetic_library_profile(true, r"\01\04\00\00\00\00\02\09\01\07leaf.so", "")
            }
            _ => synthetic_library_profile(false, r"\01\04\00\00\00\00", ""),
        };
        if failure == "directory" {
            environment.vfs.mkdir_all("/", "/lib/leaf.so").unwrap();
        } else if failure != "missing" {
            environment
                .vfs
                .write("/", "/lib/leaf.so", &leaf, 0o644)
                .unwrap();
        }
        if matches!(failure, "directory" | "mismatch") {
            // An invalid first candidate must not select a valid later provider.
            environment.vfs.mkdir_all("/", "/usr/local/lib").unwrap();
            let valid = synthetic_library_profile(true, r"\01\04\00\00\00\00", "");
            environment
                .vfs
                .write("/", "/usr/local/lib/leaf.so", &valid, 0o644)
                .unwrap();
        }
        assert_eq!(
            run(&mut environment, "/app"),
            (0, Vec::new(), Vec::new()),
            "{failure}"
        );
        assert_eq!(environment.resources.memory_mark(), 0, "{failure}");
    }
}

#[test]
fn v2_compile_scratch_has_the_same_limit_on_cold_and_cached_modules() {
    let main = synthetic_main_profile(
        true,
        "(if (i32.eqz (call $open (i32.const 32) (i32.const 7) (i32.const 2))) (then unreachable))",
        "",
    );
    let side = synthetic_library_profile(
        true,
        r"\01\04\00\00\00\00",
        &format!("(@custom \"padding\" \"{}\")", "x".repeat(300_000)),
    );
    for memory in [16 * 1024 * 1024, 128 * 1024 * 1024, 16 * 1024 * 1024] {
        let mut environment = Environment::with_limits(Limits {
            memory,
            cpu: 10_000_000,
            ..Limits::default()
        });
        environment.vfs.write("/", "/app", &main, 0o755).unwrap();
        environment.vfs.write("/", "/lib.so", &side, 0o644).unwrap();
        assert_eq!(
            run(&mut environment, "/app").0,
            if memory == 16 * 1024 * 1024 { 137 } else { 0 }
        );
        assert_eq!(environment.resources.memory_mark(), 0);
    }
}

#[test]
#[ignore = "requires built SDK34 dynamic artifacts"]
fn sdk34_cpython_loads_declared_shared_zlib_dependency() {
    let artifacts = PathBuf::from(std::env::var_os("SHELLSIM_DYNAMIC_V2_ARTIFACTS").unwrap());
    let mut environment = Environment::with_limits(Limits {
        cpu: 2_000_000_000,
        memory: 1024 * 1024 * 1024,
        disk: 64 * 1024 * 1024,
        ..Limits::default()
    });
    let root = artifacts.join("rootfs");
    mount_tree(&mut environment, &root, &root);
    put(
        &mut environment,
        &artifacts.join("python3.wasm"),
        "/usr/bin/python3.wasm",
        true,
    );
    put(
        &mut environment,
        &artifacts.join("zlib_consumer.so"),
        "/usr/lib/python3.13/site-packages/zlib_consumer.so",
        false,
    );
    put(
        &mut environment,
        &artifacts.join("libz.so"),
        "/lib/libz.so",
        false,
    );
    let source = r#"import zlib_consumer
payload = bytes(range(256)) * 3
assert zlib_consumer.roundtrip(payload) == payload
assert zlib_consumer.roundtrip(b'') == b''
try:
    zlib_consumer.roundtrip('invalid')
except TypeError:
    pass
else:
    raise AssertionError('expected TypeError')
print('declared shared zlib dependency: ok')
"#;
    environment
        .vfs
        .write("/", "/probe.py", source.as_bytes(), 0o644)
        .unwrap();
    assert_eq!(
        run(&mut environment, "/usr/bin/python3.wasm /probe.py"),
        (
            0,
            b"declared shared zlib dependency: ok\n".to_vec(),
            Vec::new()
        )
    );
    assert_eq!(environment.resources.memory_mark(), 0);
}

#[test]
fn v2_start_initializers_are_metered_and_cannot_load_recursively() {
    for initializer in [
        r#"(func $start (loop $forever (br $forever))) (start $start)"#,
        r#"(import "env" "fixture_open" (func $open (param i32 i32 i32) (result i32)))
        (func $start
            (if (call $open (i32.const 32) (i32.const 7) (i32.const 2)) (then unreachable)))
        (start $start)"#,
    ] {
        let endless = initializer.contains("$forever");
        let mut environment = Environment::with_limits(Limits {
            cpu: 50_000,
            memory: 128 * 1024 * 1024,
            ..Limits::default()
        });
        let main = synthetic_main_profile(true,
            "(if (i32.eqz (call $open (i32.const 32) (i32.const 7) (i32.const 2))) (then unreachable))", "");
        let side = synthetic_library_profile(true, r"\01\04\00\00\00\00", initializer);
        environment.vfs.write("/", "/app", &main, 0o755).unwrap();
        environment.vfs.write("/", "/lib.so", &side, 0o644).unwrap();
        assert_eq!(
            run(&mut environment, "/app").0,
            if endless { 137 } else { 0 }
        );
        assert_eq!(environment.resources.memory_mark(), 0);
    }
}

#[test]
fn v2_start_observes_resolved_external_got_values() {
    let mut environment = environment();
    let main = synthetic_main_profile(
        true,
        "(if (i32.eqz (call $open (i32.const 32) (i32.const 7) (i32.const 2))) (then unreachable))",
        r#"(global (export "external_data") i32 (i32.const 123))"#,
    );
    let side = synthetic_library_profile(
        true,
        r"\01\04\00\00\00\00",
        r#"(import "GOT.mem" "external_data" (global $external (mut i32)))
        (func $start (if (i32.ne (global.get $external) (i32.const 123)) (then unreachable)))
        (start $start)"#,
    );
    environment.vfs.write("/", "/app", &main, 0o755).unwrap();
    environment.vfs.write("/", "/lib.so", &side, 0o644).unwrap();
    assert_eq!(run(&mut environment, "/app"), (0, Vec::new(), Vec::new()));
    assert_eq!(environment.resources.memory_mark(), 0);
}

#[test]
fn dlopen_null_returns_a_nonzero_main_image_handle() {
    for v2 in [false, true] {
        let mut environment = environment();
        let main = synthetic_main_profile(
            v2,
            r#"(local.set $handle (call $open (i32.const 0) (i32.const 0) (i32.const 2)))
            (if (i32.ne (local.get $handle) (i32.const -1)) (then unreachable))
            (if (i32.ne (call_indirect (result i32)
                (call $symbol (local.get $handle) (i32.const 128) (i32.const 11)))
                (i32.const 42)) (then unreachable))
            (if (call $open (i32.const 0) (i32.const 0) (i32.const 0)) (then unreachable))"#,
            r#"(data (i32.const 128) "main_answer")
            (func (export "main_answer") (result i32) (i32.const 42))"#,
        );
        environment.vfs.write("/", "/app", &main, 0o755).unwrap();
        assert_eq!(run(&mut environment, "/app"), (0, Vec::new(), Vec::new()));
        assert_eq!(environment.resources.memory_mark(), 0);
    }
}

#[test]
#[ignore = "requires a built SDK34 Python bundle and FFI fixture extension"]
fn sdk34_python_extension_calls_main_image_c_api() {
    let bundle = PathBuf::from(std::env::var_os("SHELLSIM_FFI_PYTHON_BUNDLE").unwrap());
    let artifacts = PathBuf::from(std::env::var_os("SHELLSIM_FFI_FIXTURE_ARTIFACTS").unwrap());
    let rootfs = bundle.join("rootfs");
    let mut environment = environment();
    mount_tree(&mut environment, &rootfs, &rootfs);
    put(
        &mut environment,
        &artifacts.join("python_main_handle.so"),
        "/work/python_main_handle.so",
        false,
    );
    environment
        .vfs
        .write(
            "/",
            "/work/probe.py",
            b"import python_main_handle\nassert python_main_handle.answer() == 42\nprint('main-image CPython C API: ok')\n",
            0o644,
        )
        .unwrap();
    assert_eq!(
        run(&mut environment, "/usr/bin/python3.wasm /work/probe.py"),
        (0, b"main-image CPython C API: ok\n".to_vec(), Vec::new())
    );
    assert_eq!(environment.resources.memory_mark(), 0);
}

#[test]
fn v2_resolves_own_got_symbols_before_relocations() {
    let mut environment = environment();
    let main = synthetic_main_profile(
        true,
        "(if (i32.eqz (call $open (i32.const 32) (i32.const 7) (i32.const 2))) (then unreachable))",
        "",
    );
    let side = synthetic_library_profile(
        true,
        r"\01\04\00\00\00\00",
        r#"(import "GOT.mem" "own_data" (global $data (mut i32)))
        (import "GOT.func" "own_function" (global $function (mut i32)))
        (global (export "own_data") i32 (i32.const 12))
        (func (export "own_function") (result i32) (i32.const 7))
        (func (export "__wasm_apply_data_relocs")
            (if (i32.eqz (global.get $data)) (then unreachable))
            (if (i32.eqz (global.get $function)) (then unreachable)))"#,
    );
    environment.vfs.write("/", "/app", &main, 0o755).unwrap();
    environment.vfs.write("/", "/lib.so", &side, 0o644).unwrap();
    assert_eq!(run(&mut environment, "/app"), (0, Vec::new(), Vec::new()));
    assert_eq!(environment.resources.memory_mark(), 0);
}

#[test]
fn v2_rejects_own_got_when_a_start_section_could_read_it() {
    let mut environment = environment();
    let main = synthetic_main_profile(
        true,
        "(if (call $open (i32.const 32) (i32.const 7) (i32.const 2)) (then unreachable))",
        "",
    );
    let side = synthetic_library_profile(
        true,
        r"\01\04\00\00\00\00",
        r#"(import "GOT.func" "own_function" (global (mut i32)))
        (func (export "own_function"))
        (func $start) (start $start)"#,
    );
    environment.vfs.write("/", "/app", &main, 0o755).unwrap();
    environment.vfs.write("/", "/lib.so", &side, 0o644).unwrap();
    assert_eq!(run(&mut environment, "/app"), (0, Vec::new(), Vec::new()));
    assert_eq!(environment.resources.memory_mark(), 0);
}

#[test]
fn v2_small_executable_can_grow_above_sixteen_mebibytes() {
    let main = synthetic_main_profile(
        true,
        r#"
        (if (i32.lt_s (memory.grow (i32.const 320)) (i32.const 0)) (then unreachable))
        (i32.store (i32.const 20_971_520) (i32.const 123))
        (if (i32.ne (i32.load (i32.const 20_971_520)) (i32.const 123)) (then unreachable))
    "#,
        "",
    );
    for (budget, status) in [(128 * 1024 * 1024, 0), (8 * 1024 * 1024, 126)] {
        let mut environment = Environment::with_limits(Limits {
            memory: budget,
            ..Limits::default()
        });
        environment.vfs.write("/", "/app", &main, 0o755).unwrap();
        assert_eq!(run(&mut environment, "/app").0, status);
        assert_eq!(environment.resources.memory_mark(), 0);
    }
}

#[test]
fn v2_cancellation_releases_incrementally_grown_memory() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 50_000,
        memory: 128 * 1024 * 1024,
        ..Limits::default()
    });
    let main = synthetic_main_profile(
        true,
        "(drop (memory.grow (i32.const 320))) (loop $forever (br $forever))",
        "",
    );
    environment.vfs.write("/", "/app", &main, 0o755).unwrap();
    let (outcome, _, _) = environment.run_script_capture("/app");
    assert_eq!(outcome.exit_status, 137);
    assert!(outcome.usage.memory_peak > 20 * 1024 * 1024);
    assert_eq!(environment.resources.memory_mark(), 0);
}

#[test]
fn v2_rejects_private_runtime_and_unresolved_pre_start_got() {
    for definition in [
        r#"(func (export "__cxa_throw"))"#,
        r#"(import "GOT.mem" "self_data" (global $self (mut i32)))
        (global (export "self_data") (mut i32) (i32.const 0))
        (func $start unreachable) (start $start)"#,
    ] {
        let mut environment = environment();
        let main = synthetic_main_profile(
            true,
            "(if (call $open (i32.const 32) (i32.const 7) (i32.const 2)) (then unreachable))",
            "",
        );
        let side = synthetic_library_profile(true, r"\01\04\00\00\00\00", definition);
        environment.vfs.write("/", "/app", &main, 0o755).unwrap();
        environment.vfs.write("/", "/lib.so", &side, 0o644).unwrap();
        assert_eq!(run(&mut environment, "/app"), (0, Vec::new(), Vec::new()));
        assert_eq!(environment.resources.memory_mark(), 0);
    }
}

#[test]
#[ignore = "requires built SDK34 dynamic artifacts"]
fn sdk34_cross_module_longjmp_uses_main_runtime_tag() {
    let artifacts = PathBuf::from(std::env::var_os("SHELLSIM_DYNAMIC_V2_ARTIFACTS").unwrap());
    let mut environment = environment();
    put(
        &mut environment,
        &artifacts.join("jump_main.wasm"),
        "/app",
        true,
    );
    put(
        &mut environment,
        &artifacts.join("jump.so"),
        "/lib/jump.so",
        false,
    );
    assert_eq!(
        run(&mut environment, "/app"),
        (0, b"cross-module longjmp: 37\n".to_vec(), Vec::new())
    );
    assert_eq!(environment.resources.memory_mark(), 0);
}

#[test]
fn v2_same_filename_in_distinct_package_paths_has_independent_local_state() {
    let mut environment = environment();
    let main = synthetic_main_profile(
        true,
        r#"
        (local.set $handle (call $open (i32.const 32) (i32.const 12) (i32.const 2)))
        (if (i32.eqz (local.get $handle)) (then unreachable))
        (if (i32.ne (call_indirect (result i32)
            (call $symbol (local.get $handle) (i32.const 64) (i32.const 4))) (i32.const 1)) (then unreachable))
        (local.set $handle (call $open (i32.const 80) (i32.const 12) (i32.const 2)))
        (if (i32.eqz (local.get $handle)) (then unreachable))
        (if (i32.ne (call_indirect (result i32)
            (call $symbol (local.get $handle) (i32.const 64) (i32.const 4))) (i32.const 1)) (then unreachable))
        (if (call $symbol (i32.const 0) (i32.const 64) (i32.const 4)) (then unreachable))
    "#,
        r#"(data (i32.const 32) "/a/_zeros.so") (data (i32.const 80) "/b/_zeros.so")
        (data (i32.const 64) "bump")"#,
    );
    let side = synthetic_library_profile(
        true,
        r"\01\04\00\00\00\00",
        r#"(global $count (mut i32) (i32.const 0))
        (func (export "bump") (result i32)
            (global.set $count (i32.add (global.get $count) (i32.const 1)))
            (global.get $count))"#,
    );
    environment.vfs.write("/", "/app", &main, 0o755).unwrap();
    for directory in ["/a", "/b"] {
        environment.vfs.mkdir_all("/", directory).unwrap();
        environment
            .vfs
            .write("/", &format!("{directory}/_zeros.so"), &side, 0o644)
            .unwrap();
    }
    assert_eq!(run(&mut environment, "/app"), (0, Vec::new(), Vec::new()));
    assert_eq!(environment.resources.memory_mark(), 0);
}

#[test]
fn v2_guest_table_growth_charges_environment_and_releases_on_exit() {
    let main = synthetic_main_profile(
        true,
        r#"
        (if (i32.lt_s (memory.grow (i32.const 26)) (i32.const 0)) (then unreachable))
        (if (i32.ne (table.grow (ref.null func) (i32.const 9999)) (i32.const 1)) (then unreachable))
    "#,
        "",
    );
    // The outer Wasmtime async stack is now charged in addition to linear memory and table growth.
    for (budget, status) in [(6 * 1024 * 1024, 0), (4 * 1024 * 1024, 137)] {
        let mut environment = Environment::with_limits(Limits {
            memory: budget,
            ..Limits::default()
        });
        environment.vfs.write("/", "/app", &main, 0o755).unwrap();
        let (outcome, _, _) = environment.run_script_capture("/app");
        assert_eq!(outcome.exit_status, status);
        assert_eq!(environment.resources.memory_mark(), 0);
        if status == 0 {
            assert!(outcome.usage.memory_peak > 4 * 1024 * 1024);
        }
    }
}

#[test]
fn v2_cancellation_releases_guest_table_growth() {
    let main = synthetic_main_profile(
        true,
        "(drop (table.grow (ref.null func) (i32.const 9999))) (loop $forever (br $forever))",
        "",
    );
    let mut environment = Environment::with_limits(Limits {
        cpu: 50_000,
        memory: 4 * 1024 * 1024,
        ..Limits::default()
    });
    environment.vfs.write("/", "/app", &main, 0o755).unwrap();
    let (outcome, _, _) = environment.run_script_capture("/app");
    assert_eq!(outcome.exit_status, 137);
    assert!(outcome.usage.memory_peak > 400_000);
    assert_eq!(environment.resources.memory_mark(), 0);
}

const DEFERRED_V1: &str = r"\80\18\16shellsim.deferred-init\01";

#[test]
fn deferred_initializers_bind_self_functions_then_run_once_in_order() {
    let mut environment = environment();
    let main = synthetic_main_profile(
        true,
        r#"
        (local.set $handle (call $open (i32.const 32) (i32.const 7) (i32.const 2)))
        (if (i32.eqz (local.get $handle)) (then unreachable))
        (if (i32.ne (call $open (i32.const 32) (i32.const 7) (i32.const 2)) (local.get $handle)) (then unreachable))
    "#,
        "",
    );
    let library = synthetic_library_profile(
        true,
        &format!(r"\01\04\04\00\01\00{DEFERRED_V1}"),
        r#"
        (import "env" "__indirect_function_table" (table 1 funcref))
        (import "env" "__memory_base" (global $base i32))
        (import "env" "__table_base" (global $table i32))
        (import "GOT.func" "self_callback" (global $self (mut i32)))
        (global $phase (mut i32) (i32.const 0))
        (func $self_callback (export "self_callback") (result i32) (i32.const 42))
        (elem (global.get $table) $self_callback)
        (func (export "__wasm_apply_global_relocs")
            (if (global.get $phase) (then unreachable))
            (if (i32.eqz (global.get $self)) (then unreachable))
            (global.set $phase (i32.const 1)))
        (func (export "__wasm_init_memory")
            (if (i32.ne (global.get $phase) (i32.const 1)) (then unreachable))
            (i32.store (global.get $base) (i32.const 17))
            (global.set $phase (i32.const 2)))
        (func (export "__wasm_apply_data_relocs")
            (if (i32.ne (global.get $phase) (i32.const 2)) (then unreachable))
            (if (i32.ne (i32.load (global.get $base)) (i32.const 17)) (then unreachable))
            (if (i32.ne (call_indirect (result i32) (global.get $self)) (i32.const 42)) (then unreachable))
            (global.set $phase (i32.const 3)))
        (func (export "__wasm_call_ctors")
            (if (i32.ne (global.get $phase) (i32.const 3)) (then unreachable))
            (global.set $phase (i32.const 4)))
    "#,
    );
    environment.vfs.write("/", "/app", &main, 0o755).unwrap();
    environment
        .vfs
        .write("/", "/lib.so", &library, 0o644)
        .unwrap();
    assert_eq!(run(&mut environment, "/app"), (0, Vec::new(), Vec::new()));
}

#[test]
fn deferred_initializers_reject_bad_metadata_starts_signatures_and_own_data() {
    for (marker, extra) in [
        (r"\80\18\16shellsim.deferred-init\02".to_owned(), ""),
        (r"\80\18\16shEllsim.deferred-init\01".to_owned(), ""),
        (format!("{DEFERRED_V1}{DEFERRED_V1}"), ""),
        (r"\80\19\16shellsim.deferred-init\01\00".to_owned(), ""),
        (DEFERRED_V1.to_owned(), "(func $start) (start $start)"),
        (DEFERRED_V1.to_owned(), "(func (export \"__wasm_apply_global_relocs\") (param i32))"),
        (DEFERRED_V1.to_owned(), "(import \"GOT.mem\" \"own\" (global (mut i32))) (global (export \"own\") (mut i32) (i32.const 0))"),
    ] {
        let mut environment = environment();
        let main = synthetic_main_profile(true,
            "(if (call $open (i32.const 32) (i32.const 7) (i32.const 2)) (then unreachable))", "");
        let library = synthetic_library_profile(true, &format!(r"\01\04\00\00\00\00{marker}"), extra);
        environment.vfs.write("/", "/app", &main, 0o755).unwrap();
        environment.vfs.write("/", "/lib.so", &library, 0o644).unwrap();
        assert_eq!(run(&mut environment, "/app"), (0, Vec::new(), Vec::new()));
    }
}

#[test]
fn deferred_initializer_trap_releases_images_and_environment_remains_usable() {
    let mut environment = environment();
    let main = synthetic_main_profile(
        true,
        "(if (call $open (i32.const 32) (i32.const 7) (i32.const 2)) (then unreachable))",
        "",
    );
    let library = synthetic_library_profile(
        true,
        &format!(r"\01\04\00\00\00\00{DEFERRED_V1}"),
        r#"(func (export "__wasm_apply_global_relocs") unreachable)"#,
    );
    environment.vfs.write("/", "/app", &main, 0o755).unwrap();
    environment
        .vfs
        .write("/", "/lib.so", &library, 0o644)
        .unwrap();
    assert_eq!(run(&mut environment, "/app").0, 0);
    assert_eq!(environment.resources.memory_mark(), 0);
    let (outcome, stdout, stderr) = environment.run_script_capture("printf recovered");
    assert_eq!(outcome.exit_status, 0);
    assert_eq!(stdout, b"recovered");
    assert!(stderr.is_empty());
}
