//! Differential corpus generated from the 100-row TaskTrove/TBLite sample.
//!
//! The checked output files make this test useful on builders without CPython 3.14. When the
//! reference executable is available, every valid case is also run through it and compared.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use shellsim::Environment;

const CORPUS: &str = "tests/python/corpus/tasktrove";

fn cpython_314_available() -> bool {
    let probe = "import sys; print(sys.implementation.name); print(*sys.version_info[:2])";
    match Command::new("python3.14").args(["-c", probe]).output() {
        Ok(output) => output.status.success() && output.stdout == b"cpython\n3 14\n",
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => panic!("could not inspect python3.14: {error}"),
    }
}

fn unescape_stdout(field: &str) -> Vec<u8> {
    let normalized = field.replace("\\\\", "\\");
    let mut result = normalized.replace("\\n", "\n").into_bytes();
    result.push(b'\n');
    result
}

fn reference(path: &Path) -> Option<std::process::Output> {
    match Command::new("python3.14").arg(path).output() {
        Ok(output) => Some(output),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => panic!("could not invoke CPython 3.14: {error}"),
    }
}

#[test]
fn tasktrove_100_cases_match_checked_outputs_and_cpython() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(CORPUS);
    let manifest = fs::read_to_string(root.join("manifest.tsv")).expect("manifest");
    let mut lines = manifest.lines();
    assert_eq!(
        lines.next(),
        Some("id\tstatus\ttask_id\tsource_path\tscript\texpected_stdout")
    );

    let mut supported = 0;
    let mut frontier = 0;
    let reference_available = cpython_314_available();

    for line in lines {
        let fields: Vec<&str> = line.split('\t').collect();
        assert_eq!(fields.len(), 6, "malformed manifest line: {line}");
        let [id, status, task, source_path, script_name, expected_field] = fields.as_slice() else {
            unreachable!()
        };
        let script_path = root.join(script_name);
        let checked_path = script_path.with_extension("out");
        let source = fs::read_to_string(&script_path)
            .unwrap_or_else(|error| panic!("{id} {task} {source_path}: {error}"));
        let expected = unescape_stdout(expected_field);
        let checked = fs::read(&checked_path).expect("checked output");
        assert_eq!(
            checked, expected,
            "{id} checked output disagrees with manifest ({task}:{source_path})"
        );

        let mut environment = Environment::new();
        environment
            .vfs
            .put_file("/case.py", source.into_bytes(), 0o644)
            .expect("install fixture in VFS");
        let (outcome, stdout, stderr) = environment.run_script_capture("python3.14 /case.py");

        let host = reference_available
            .then(|| reference(&script_path))
            .flatten();
        if *status == "supported" {
            supported += 1;
            assert_eq!(
                outcome.exit_status,
                0,
                "{id} {task}:{source_path} failed: {}",
                String::from_utf8_lossy(&stderr)
            );
            assert_eq!(stdout, expected, "{id} {task}:{source_path}");
            assert!(stderr.is_empty(), "{id} wrote stderr: {:?}", stderr);
            if let Some(host) = host {
                assert_eq!(host.status.code().unwrap_or(1), 0, "{id} host status");
                assert_eq!(host.stdout, expected, "{id} host stdout");
                assert!(host.stderr.is_empty(), "{id} host stderr");
            }
        } else {
            assert_eq!(*status, "frontier", "{id}: unknown status {status}");
            frontier += 1;
            assert_ne!(
                outcome.exit_status, 0,
                "{id} {task}:{source_path} unexpectedly became supported"
            );
            if let Some(host) = host {
                assert_eq!(host.status.code().unwrap_or(1), 0, "{id} host status");
                assert_eq!(host.stdout, expected, "{id} host stdout");
            }
        }
    }

    assert_eq!(supported, 100);
    assert_eq!(frontier, 0);
    assert_eq!(supported + frontier, 100);
    eprintln!(
        "TaskTrove differential corpus: {supported} supported, {frontier} frontier (CPython 3.14: {})",
        if reference_available { "available" } else { "not installed; checked outputs authoritative" }
    );
}
