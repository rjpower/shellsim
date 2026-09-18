//! Compatibility and boundary tests for VFS-only compression and archive commands.

use std::io::Write;

use flate2::write::DeflateEncoder;
use flate2::Compression;
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
fn zip_lists_and_extracts_binary_trees() {
    let mut environment = Environment::new();
    environment.vfs.mkdir_all("/", "/work/tree").unwrap();
    environment
        .vfs
        .write("/", "/work/tree/a.txt", b"alpha", 0o644)
        .unwrap();
    environment
        .vfs
        .write("/", "/work/tree/b.bin", &[0, 1, 255], 0o600)
        .unwrap();

    let (status, stdout, stderr) = environment
        .run_script_capture("cd /work; zip -qr /tmp/tree.zip tree; unzip -Z1 /tmp/tree.zip");
    assert_eq!(
        status.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"tree/\ntree/a.txt\ntree/b.bin\n");

    environment.vfs.remove_all("/", "/work/tree").unwrap();
    let (status, _, stderr) = environment.run_script_capture("unzip -q /tmp/tree.zip -d /work");
    assert_eq!(
        status.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(
        environment.vfs.read("/", "/work/tree/b.bin").unwrap(),
        [0, 1, 255]
    );

    environment
        .vfs
        .write("/", "/work/tree/a.txt", b"stale", 0o644)
        .unwrap();
    let (status, _, stderr) = environment.run_script_capture("unzip -qo /tmp/tree.zip -d /work");
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
}

#[test]
fn unzip_rejects_traversal_without_partial_extraction() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/goodx", b"payload", 0o644)
        .unwrap();
    let (status, _, stderr) = environment.run_script_capture("zip /tmp/unsafe.zip /goodx");
    assert_eq!(
        status.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    let mut archive = environment.vfs.read("/", "/tmp/unsafe.zip").unwrap();
    for start in 0..archive.len().saturating_sub(5) {
        if &archive[start..start + 5] == b"goodx" {
            archive[start..start + 5].copy_from_slice(b"../xx");
        }
    }
    environment
        .vfs
        .write("/", "/tmp/unsafe.zip", &archive, 0o644)
        .unwrap();

    let (status, _, stderr) = environment.run_script_capture("unzip /tmp/unsafe.zip -d /work");
    assert_eq!(status.exit_status, 2);
    assert!(String::from_utf8_lossy(&stderr).contains("unsafe archive path"));
    assert!(!environment.vfs.exists("/", "/xx"));
}

#[test]
fn unzip_accepts_standard_deflated_entries() {
    let mut environment = Environment::new();
    let archive = one_file_deflated_zip("payload.bin", &[0, 1, 2, 255]);
    environment
        .vfs
        .write("/", "/tmp/deflated.zip", &archive, 0o644)
        .unwrap();

    let (status, _, stderr) = environment.run_script_capture("unzip /tmp/deflated.zip -d /work");
    assert_eq!(
        status.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(
        environment.vfs.read("/", "/work/payload.bin").unwrap(),
        [0, 1, 2, 255]
    );
}

fn one_file_deflated_zip(name: &str, data: &[u8]) -> Vec<u8> {
    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data).unwrap();
    let compressed = encoder.finish().unwrap();
    let crc = crc32fast::hash(data);
    let mut archive = Vec::new();
    archive.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
    archive.extend_from_slice(&20u16.to_le_bytes());
    archive.extend_from_slice(&0u16.to_le_bytes());
    archive.extend_from_slice(&8u16.to_le_bytes());
    archive.extend_from_slice(&[0; 4]);
    archive.extend_from_slice(&crc.to_le_bytes());
    archive.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
    archive.extend_from_slice(&(data.len() as u32).to_le_bytes());
    archive.extend_from_slice(&(name.len() as u16).to_le_bytes());
    archive.extend_from_slice(&0u16.to_le_bytes());
    archive.extend_from_slice(name.as_bytes());
    archive.extend_from_slice(&compressed);
    let central_offset = archive.len() as u32;
    archive.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
    archive.extend_from_slice(&0x031eu16.to_le_bytes());
    archive.extend_from_slice(&20u16.to_le_bytes());
    archive.extend_from_slice(&0u16.to_le_bytes());
    archive.extend_from_slice(&8u16.to_le_bytes());
    archive.extend_from_slice(&[0; 4]);
    archive.extend_from_slice(&crc.to_le_bytes());
    archive.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
    archive.extend_from_slice(&(data.len() as u32).to_le_bytes());
    archive.extend_from_slice(&(name.len() as u16).to_le_bytes());
    archive.extend_from_slice(&[0; 8]);
    archive.extend_from_slice(&(0o100644u32 << 16).to_le_bytes());
    archive.extend_from_slice(&0u32.to_le_bytes());
    archive.extend_from_slice(name.as_bytes());
    let central_size = archive.len() as u32 - central_offset;
    archive.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    archive.extend_from_slice(&[0; 4]);
    archive.extend_from_slice(&1u16.to_le_bytes());
    archive.extend_from_slice(&1u16.to_le_bytes());
    archive.extend_from_slice(&central_size.to_le_bytes());
    archive.extend_from_slice(&central_offset.to_le_bytes());
    archive.extend_from_slice(&0u16.to_le_bytes());
    archive
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
