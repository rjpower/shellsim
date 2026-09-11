//! Differential probes for the reviewed Python stdlib slice.
//!
//! Each probe is deliberately tiny: the source is installed in shellsim's in-memory VFS, then
//! executed unchanged by the emulator and (when present) CPython 3.14. A frontier probe must be
//! rejected by shellsim even though it is a valid CPython program. This keeps unsupported imports
//! visible instead of silently accepting a partial or host-backed implementation.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use shellsim::Environment;

const CORPUS: &str = "tests/fixtures/python/stdlib_slice";

fn cpython_314_available() -> bool {
    let probe = "import sys; print(sys.implementation.name); print(sys.version_info.major, sys.version_info.minor)";
    match Command::new("python3.14").args(["-c", probe]).output() {
        Ok(output) => {
            output.status.success()
                && output.stdout == b"cpython\n3 14\n"
                && output.stderr.is_empty()
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => panic!("could not inspect python3.14: {error}"),
    }
}

fn reference(path: &Path, available: bool) -> Option<Output> {
    if !available {
        return None;
    }
    match Command::new("python3.14").arg(path).output() {
        Ok(output) => Some(output),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => panic!("could not invoke CPython 3.14: {error}"),
    }
}

#[test]
fn requested_stdlib_slice_matches_cpython_or_rejects_frontier() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(CORPUS);
    let manifest = fs::read_to_string(root.join("manifest.tsv")).expect("stdlib manifest");
    let mut lines = manifest.lines();
    assert_eq!(
        lines.next(),
        Some("id\tstatus\tmodule\tscript\thost_status\tnote")
    );

    let mut supported = 0;
    let mut frontier = 0;
    let reference_available = cpython_314_available();

    for line in lines {
        let fields: Vec<&str> = line.split('\t').collect();
        assert_eq!(fields.len(), 6, "malformed stdlib manifest line: {line}");
        let [id, status, module, script_name, host_status, note] = fields.as_slice() else {
            unreachable!()
        };
        let script_path = root.join(script_name);
        let output_path = script_path.with_extension("out");
        let source = fs::read_to_string(&script_path)
            .unwrap_or_else(|error| panic!("{id} {module} {note}: {error}"));
        let expected = fs::read(&output_path).expect("checked stdlib output");
        let expected_host_status = match *host_status {
            "any" => None,
            value => Some(
                value
                    .parse::<i32>()
                    .unwrap_or_else(|error| panic!("{id} invalid host status {value}: {error}")),
            ),
        };

        let mut environment = Environment::new();
        environment
            .vfs
            .put_file("/case.py", source.into_bytes(), 0o644)
            .expect("install stdlib probe in VFS");
        let (outcome, stdout, stderr) = environment.run_script_capture("python3.14 /case.py");

        let host = reference(&script_path, reference_available);

        match *status {
            "supported" => {
                supported += 1;
                assert_eq!(
                    outcome.exit_status,
                    0,
                    "{id} {module} failed in shellsim: {}",
                    String::from_utf8_lossy(&stderr)
                );
                assert_eq!(stdout, expected, "{id} {module} shellsim stdout");
                assert!(stderr.is_empty(), "{id} {module} wrote stderr: {stderr:?}");
            }
            "frontier" => {
                frontier += 1;
                assert_ne!(
                    outcome.exit_status, 0,
                    "{id} {module} unexpectedly became supported; promote it deliberately"
                );
            }
            other => panic!("{id} {module}: unknown status {other}"),
        }

        if let Some(host) = host {
            let actual_host_status = host.status.code().unwrap_or(-1);
            if let Some(expected_host_status) = expected_host_status {
                assert_eq!(
                    actual_host_status, expected_host_status,
                    "{id} {module}: CPython status changed; update the fixture contract"
                );
            }
            if actual_host_status == 0 {
                assert_eq!(host.stdout, expected, "{id} {module} CPython stdout");
                assert!(
                    host.stderr.is_empty(),
                    "{id} {module} CPython stderr: {:?}",
                    host.stderr
                );
            } else {
                // A host may not have the optional pytest package. The probe has no print
                // statements, so stdout remains deterministic in either host outcome.
                assert!(
                    host.stdout.is_empty(),
                    "{id} {module} unexpected CPython stdout"
                );
            }
        }
    }

    assert_eq!(supported, 20);
    assert_eq!(frontier, 0);
    assert_eq!(supported + frontier, 20);
    eprintln!(
        "stdlib slice differential probes: {supported} supported, {frontier} frontier (CPython 3.14: {})",
        if reference_available { "verified CPython 3.14" } else { "not verified; oracle skipped" }
    );
}
