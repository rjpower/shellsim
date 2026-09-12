//! Compatibility and boundary tests for VFS-only compression commands.

use shellsim::Environment;

#[test]
fn gzip_round_trips_named_files_and_standard_streams() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/work/input.txt", b"alpha\nbeta\n", 0o644)
        .unwrap();

    let (status, stdout, stderr) =
        environment.run_script_capture("gzip -k /work/input.txt; gzip -dc /work/input.txt.gz");
    assert_eq!(
        status.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"alpha\nbeta\n");
    assert!(environment.vfs.exists("/", "/work/input.txt"));
    assert!(environment.vfs.exists("/", "/work/input.txt.gz"));

    let (status, stdout, stderr) =
        environment.run_script_capture("printf streamed | gzip -c - | gunzip -c -");
    assert_eq!(
        status.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"streamed");
}

#[test]
fn gunzip_replaces_archives_and_rejects_invalid_inputs() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/work/item", b"contents", 0o644)
        .unwrap();
    let (status, _, stderr) = environment.run_script_capture(
        "gzip /work/item; test ! -e /work/item; gunzip /work/item.gz; cat /work/item",
    );
    assert_eq!(
        status.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );

    environment
        .vfs
        .write("/", "/work/bad.gz", b"not gzip", 0o644)
        .unwrap();
    let (status, _, stderr) = environment.run_script_capture("gunzip /work/bad.gz");
    assert_eq!(status.exit_status, 1);
    assert!(String::from_utf8_lossy(&stderr).contains("gzip"));
    assert!(environment.vfs.exists("/", "/work/bad.gz"));
    assert!(!environment.vfs.exists("/", "/work/bad"));
}

#[test]
fn gzip_rejects_unknown_options_without_host_fallback() {
    let mut environment = Environment::new();
    let (status, _, stderr) = environment.run_script_capture("gzip --rsyncable");
    assert_eq!(status.exit_status, 1);
    assert!(String::from_utf8_lossy(&stderr).contains("unsupported option"));
}
