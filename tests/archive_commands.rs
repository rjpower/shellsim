//! Compatibility and boundary tests for VFS-only compression commands.

use shellsim::{Environment, Limits, StopReason};

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

#[test]
fn tar_creates_lists_and_extracts_gzip_archives() {
    let mut environment = Environment::new();
    environment.vfs.mkdir_all("/", "/work/tree").unwrap();
    environment
        .vfs
        .write("/", "/work/tree/a.txt", b"alpha", 0o644)
        .unwrap();
    environment
        .vfs
        .write("/", "/work/tree/b.bin", &[0, 1, 2, 255], 0o600)
        .unwrap();

    let (status, stdout, stderr) = environment
        .run_script_capture("tar -czf /tmp/tree.tar.gz -C /work tree; tar -tzf /tmp/tree.tar.gz");
    assert_eq!(
        status.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"tree\ntree/a.txt\ntree/b.bin\n");

    environment.vfs.remove_all("/", "/work/tree").unwrap();
    let (status, _, stderr) = environment.run_script_capture("tar -xzf /tmp/tree.tar.gz -C /work");
    assert_eq!(
        status.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(
        environment.vfs.read("/", "/work/tree/a.txt").unwrap(),
        b"alpha"
    );
    assert_eq!(
        environment.vfs.read("/", "/work/tree/b.bin").unwrap(),
        [0, 1, 2, 255]
    );
}

#[test]
fn tar_rejects_traversal_and_rolls_back_extraction() {
    let mut environment = Environment::new();
    let mut archive = vec![0u8; 1024];
    archive[..7].copy_from_slice(b"../bad\0");
    archive[100..108].copy_from_slice(b"0000644\0");
    archive[124..136].copy_from_slice(b"00000000000\0");
    archive[148..156].fill(b' ');
    archive[156] = b'0';
    archive[257..263].copy_from_slice(b"ustar\0");
    archive[263..265].copy_from_slice(b"00");
    let checksum: u64 = archive[..512].iter().map(|byte| u64::from(*byte)).sum();
    archive[148..156].copy_from_slice(format!("{checksum:06o}\0 ").as_bytes());
    environment
        .vfs
        .write("/", "/tmp/unsafe.tar", &archive, 0o644)
        .unwrap();

    let (status, _, stderr) = environment.run_script_capture("tar -xf /tmp/unsafe.tar -C /work");
    assert_eq!(status.exit_status, 2);
    assert!(String::from_utf8_lossy(&stderr).contains("unsafe archive path"));
    assert!(!environment.vfs.exists("/", "/bad"));

    archive[..100].fill(0);
    archive[..9].copy_from_slice(b"link/bad\0");
    archive[148..156].fill(b' ');
    let checksum: u64 = archive[..512].iter().map(|byte| u64::from(*byte)).sum();
    archive[148..156].copy_from_slice(format!("{checksum:06o}\0 ").as_bytes());
    environment.vfs.symlink("/", "/", "/work/link").unwrap();
    environment
        .vfs
        .write("/", "/tmp/symlink.tar", &archive, 0o644)
        .unwrap();
    let (status, _, stderr) = environment.run_script_capture("tar -xf /tmp/symlink.tar -C /work");
    assert_eq!(status.exit_status, 2);
    assert!(String::from_utf8_lossy(&stderr).contains("symbolic-link parent"));
    assert!(!environment.vfs.exists("/", "/bad"));
}

#[test]
fn tar_rejects_archive_materialization_before_exceeding_memory() {
    let mut environment = Environment::with_limits(Limits {
        memory: 20 * 1024,
        ..Limits::unlimited()
    });
    environment
        .vfs
        .write("/", "/work/large", &vec![b'x'; 12 * 1024], 0o644)
        .unwrap();
    let (outcome, _, stderr) =
        environment.run_script_capture("tar -cf /tmp/large.tar -C /work large");
    assert_eq!(outcome.exit_status, 137);
    assert_eq!(outcome.stop_reason, Some(StopReason::MemoryExhausted));
    assert!(String::from_utf8_lossy(&stderr).contains("memory limit exceeded"));
    assert!(!environment.vfs.exists("/", "/tmp/large.tar"));
}
