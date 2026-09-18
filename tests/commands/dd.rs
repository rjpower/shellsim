//! Compatibility strategy for `dd`: cover sparse patching and bounded zero-file creation.

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
fn seek_and_notrunc_patch_existing_bytes() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "printf abcdef > image; printf XY | dd of=image bs=1 seek=2 count=2 conv=notrunc; cat image",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"abXYef");
    assert!(stderr.contains("2+0 records in"), "{stderr}");
}

#[test]
fn zero_device_is_available_only_with_a_bounded_count() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "dd if=/dev/zero of=zeros bs=1K count=2 status=none; wc -c < zeros",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"2048\n");

    let (status, stdout, stderr) = run(
        &mut environment,
        "dd if=/dev/zero bs=8 skip=100 count=1 status=none | wc -c",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"8\n");

    let (status, _, stderr) = run(&mut environment, "dd if=/dev/zero of=forever");
    assert_eq!(status, 1);
    assert!(stderr.contains("requires count="), "{stderr}");
}

#[test]
fn skip_count_and_stdout_copy_exact_bytes() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "printf abcdef | dd bs=2 skip=1 count=2 status=noxfer",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"cdef");
    assert_eq!(stderr, "2+0 records in\n2+0 records out\n");
}

#[test]
fn output_growth_obeys_disk_limits() {
    let mut environment = Environment::with_limits(Limits {
        disk: 128,
        ..Limits::unlimited()
    });
    let (outcome, _, stderr) =
        environment.run_script_capture("dd if=/dev/zero of=too-large bs=1K count=1 status=none");
    assert_ne!(outcome.exit_status, 0);
    assert_eq!(outcome.stop_reason, None);
    assert!(String::from_utf8_lossy(&stderr).contains("No space left on device"));

    let mut environment = Environment::with_limits(Limits {
        cpu: 200,
        ..Limits::unlimited()
    });
    let (outcome, _, _) = environment
        .run_script_capture("dd if=/dev/zero of=too-expensive bs=1K count=1 status=none");
    assert_eq!(outcome.exit_status, 137);
    assert_eq!(outcome.stop_reason, Some(StopReason::CpuExhausted));

    let mut environment = Environment::with_limits(Limits {
        memory: 64 * 1024,
        ..Limits::unlimited()
    });
    let (outcome, _, _) =
        environment.run_script_capture("dd if=/dev/zero of=/dev/null bs=1G count=4096 status=none");
    assert_eq!(outcome.exit_status, 137);
    assert_eq!(outcome.stop_reason, Some(StopReason::MemoryExhausted));
}
