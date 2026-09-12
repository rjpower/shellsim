//! Compatibility tests for bounded ripgrep-style search over the virtual filesystem.

use shellsim::{Environment, Limits, StopReason};

fn run(environment: &mut Environment, source: &str) -> (i32, String, String) {
    let (outcome, stdout, stderr) = environment.run_script_capture(source);
    (
        outcome.exit_status,
        String::from_utf8_lossy(&stdout).into_owned(),
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

fn fixture() -> Environment {
    let mut environment = Environment::new();
    environment
        .vfs
        .put_file(
            "/workspace/a.rs",
            b"fn main() {}\n// TODO: rust\n".to_vec(),
            0o644,
        )
        .unwrap();
    environment
        .vfs
        .put_file(
            "/workspace/b.py",
            b"# TODO: python\nprint('ok')\n".to_vec(),
            0o644,
        )
        .unwrap();
    environment
        .vfs
        .put_file("/workspace/.hidden", b"TODO hidden\n".to_vec(), 0o644)
        .unwrap();
    environment
        .vfs
        .put_file("/workspace/binary.dat", b"TODO\0binary\n".to_vec(), 0o644)
        .unwrap();
    environment
}

#[test]
fn recursive_search_filters_globs_types_hidden_and_binary_files() {
    let mut environment = fixture();
    assert_eq!(
        run(&mut environment, "rg -n TODO /workspace"),
        (
            0,
            "workspace/a.rs:2:// TODO: rust\nworkspace/b.py:1:# TODO: python\n".into(),
            String::new(),
        )
    );
    assert_eq!(
        run(&mut environment, "rg -g '*.rs' TODO /workspace").1,
        "workspace/a.rs:// TODO: rust\n"
    );
    assert_eq!(
        run(&mut environment, "rg -t py TODO /workspace").1,
        "workspace/b.py:# TODO: python\n"
    );
    assert_eq!(
        run(&mut environment, "rg --hidden -a -l TODO /workspace").1,
        "workspace/.hidden\nworkspace/a.rs\nworkspace/b.py\nworkspace/binary.dat\n"
    );
}

#[test]
fn fixed_stdin_search_and_exit_codes_match_common_rg_contracts() {
    let mut environment = fixture();
    assert_eq!(
        run(&mut environment, "printf 'a.b\\naxb\\n' | rg -F 'a.b'"),
        (0, "a.b\n".into(), String::new())
    );
    assert_eq!(run(&mut environment, "rg absent /workspace").0, 1);
    let invalid = run(&mut environment, "rg --host-filesystem TODO");
    assert_eq!(invalid.0, 2);
    assert!(invalid.2.contains("unsupported option"), "{}", invalid.2);
}

#[test]
fn files_mode_lists_only_selected_vfs_paths_in_stable_order() {
    let mut environment = fixture();
    assert_eq!(
        run(&mut environment, "rg --files -g '*.rs' /workspace"),
        (0, "workspace/a.rs\n".into(), String::new())
    );
}

#[test]
fn search_materialization_obeys_the_modeled_memory_limit() {
    let mut environment = Environment::with_limits(Limits {
        memory: 24 * 1024,
        ..Limits::unlimited()
    });
    environment
        .vfs
        .put_file("/workspace/large.txt", vec![b'x'; 16 * 1024], 0o644)
        .unwrap();
    let (outcome, stdout, _) = environment.run_script_capture("rg x /workspace/large.txt");
    assert_eq!(outcome.stop_reason, Some(StopReason::MemoryExhausted));
    assert!(stdout.is_empty());
}

#[test]
fn context_output_is_bounded_grouped_and_line_addressable() {
    let mut environment = fixture();
    environment
        .vfs
        .put_file(
            "/workspace/context.txt",
            b"before\nhit one\nafter\ngap\ngap\nbefore two\nhit two\nafter two\n".to_vec(),
            0o644,
        )
        .unwrap();
    assert_eq!(
        run(&mut environment, "rg -B1 -A1 'hit' /workspace/context.txt"),
        (
            0,
            "1-before\n2:hit one\n3-after\n--\n6-before two\n7:hit two\n8-after two\n".into(),
            String::new()
        )
    );
    let invalid = run(&mut environment, "rg -C1001 hit /workspace");
    assert_eq!(invalid.0, 2);
    assert!(invalid.2.contains("context limit"), "{}", invalid.2);
}

#[test]
fn common_language_type_aliases_filter_recursive_search() {
    let mut environment = fixture();
    environment
        .vfs
        .put_file("/workspace/main.go", b"// TODO go\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(
        run(&mut environment, "rg -tgo TODO /workspace").1,
        "workspace/main.go:// TODO go\n"
    );
}
