//! Compatibility strategy for extra filesystem tools: exercise their common build-script forms,
//! error handling, and deterministic resource boundaries through the public shell interface.

use shellsim::interp::{Environment, Interp};
use shellsim::{Limits, StopReason};

fn run(environment: &mut Interp, source: &str) -> (i32, Vec<u8>, String) {
    let (outcome, stdout, stderr) = environment.run_script_capture(source);
    (
        outcome.exit_status,
        stdout,
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

#[test]
fn cp_p_preserves_source_mode_and_mtime() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/source", b"data", 0o640)
        .unwrap();
    environment.vfs.touch("/", "/source", 12_345).unwrap();
    environment
        .vfs
        .write("/", "/target", b"old", 0o600)
        .unwrap();

    let (status, _, stderr) = run(&mut environment, "cp -p /source /target");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(environment.vfs.read("/", "/target").unwrap(), b"data");
    let target = environment.vfs.metadata("/", "/target", true).unwrap();
    assert_eq!(target.mode, 0o640);
    assert_eq!(target.mtime, 12_345);
}

#[test]
fn cp_a_preserves_nested_metadata() {
    let mut environment = Environment::new();
    environment.vfs.mkdir_all("/", "/source/sub").unwrap();
    environment
        .vfs
        .write("/", "/source/sub/file", b"data", 0o640)
        .unwrap();
    environment
        .vfs
        .touch("/", "/source/sub/file", 12_345)
        .unwrap();
    environment.vfs.touch("/", "/source/sub", 23_456).unwrap();

    let (status, _, stderr) = run(&mut environment, "cp -a /source /copy");
    assert_eq!(status, 0, "{stderr}");
    let file = environment
        .vfs
        .metadata("/", "/copy/sub/file", true)
        .unwrap();
    let directory = environment.vfs.metadata("/", "/copy/sub", true).unwrap();
    assert_eq!((file.mode, file.mtime), (0o640, 12_345));
    assert_eq!(directory.mtime, 23_456);
}

#[test]
fn install_copies_files_and_sets_requested_modes() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "printf payload > source; mkdir bin; install source bin/tool; install -m 0640 source configured; stat -c '%a %n' bin/tool configured; cat bin/tool configured",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"755 bin/tool\n640 configured\npayloadpayload");
}

#[test]
fn install_creates_directories_and_leading_components() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "printf '#!/bin/sh' > script; install -d -m 0700 var/lib/app cache; install -Dm755 script usr/local/bin/app; stat -c '%a %F' var/lib/app cache usr/local/bin/app",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"700 directory\n700 directory\n755 regular file\n");
}

#[test]
fn install_reports_invalid_forms_and_quota_failure_is_atomic() {
    let mut environment = Environment::new();
    for source in [
        "install",
        "install -d -D somewhere",
        "install -m nope source target",
        "install one two target",
    ] {
        let (status, _, stderr) = run(&mut environment, source);
        assert_ne!(status, 0, "{source}");
        assert!(!stderr.is_empty(), "{source}");
    }

    let mut bounded = Environment::with_limits(Limits {
        disk: 700,
        ..Limits::unlimited()
    });
    let (status, stdout, stderr) = run(
        &mut bounded,
        "printf payload > source; install -D source new/leading/file; status=$?; test ! -e new; printf '%s' \"$status\"",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"1");
    assert!(stderr.contains("No space left on device"), "{stderr}");
}

#[test]
fn truncate_accepts_absolute_and_relative_sizes() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "printf abcdef > data; truncate -s 3 data; cat data; truncate -s +4 data; stat -c %s data; truncate -s -2 data; stat -c %s data; truncate -s 1K sized; stat -c %s sized",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"abc7\n5\n1024\n");
}

#[test]
fn truncate_preserves_content_when_growth_exceeds_quota() {
    let mut environment = Environment::with_limits(Limits {
        disk: 300,
        ..Limits::unlimited()
    });
    let (status, stdout, stderr) = run(
        &mut environment,
        "printf original > data; truncate -s 100 data || true; cat data; truncate -c -s 4 absent; test ! -e absent",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"original");
    assert!(stderr.contains("No space left on device"), "{stderr}");
}

#[test]
fn truncate_rejects_bad_sizes_and_directory_operands() {
    let mut environment = Environment::new();
    for source in [
        "truncate file",
        "truncate -s bad file",
        "mkdir dir; truncate -s 1 dir",
    ] {
        let (status, _, stderr) = run(&mut environment, source);
        assert_ne!(status, 0, "{source}");
        assert!(!stderr.is_empty(), "{source}");
    }
}

#[test]
fn tree_orders_entries_hides_dotfiles_and_limits_depth() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "mkdir -p root/b/deep root/a; touch root/z root/a/item root/b/deep/end root/.hidden; tree -L 2 root",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(
        stdout,
        concat!(
            "root\n",
            "├── a\n",
            "│   └── item\n",
            "├── b\n",
            "│   └── deep\n",
            "└── z\n",
            "\n",
            "3 directories, 2 files\n"
        )
        .as_bytes()
    );

    let (status, stdout, stderr) = run(&mut environment, "tree -a -L 1 root");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(
        stdout,
        concat!(
            "root\n",
            "├── .hidden\n",
            "├── a\n",
            "├── b\n",
            "└── z\n",
            "\n",
            "2 directories, 2 files\n"
        )
        .as_bytes()
    );
}

#[test]
fn tree_rejects_invalid_options_and_obeys_output_limits() {
    let mut environment = Environment::new();
    for source in ["tree -L 0", "tree --unknown", "tree one two"] {
        let (status, stdout, stderr) = run(&mut environment, source);
        assert_eq!(status, 2, "{source}: {stderr}");
        assert!(stdout.is_empty(), "{source}");
        assert!(!stderr.is_empty(), "{source}");
    }

    let mut bounded = Environment::with_limits(Limits {
        output: 32,
        ..Limits::unlimited()
    });
    let (outcome, _, _) = bounded.run_script_capture(
        "mkdir root; touch root/first root/second root/third root/fourth; tree root",
    );
    assert_eq!(outcome.exit_status, 137);
    assert_eq!(outcome.stop_reason, Some(StopReason::OutputLimitExceeded));
}
