//! Opt-in acceptance tests against complete public TaskTrove reference solutions.
//!
//! Set `TASKTROVE_ROOT` to an extracted OpenThoughts-TBLite checkout. The repository's ordinary
//! test suite remains hermetic; a developer with the pinned corpus can exercise the unmodified
//! full solution rather than only reduced fixtures.

use std::path::Path;
use std::process::Command;

use shellsim::Environment;

fn python_heredoc(script: &str) -> Option<&str> {
    let marker = "<<'PY'\n";
    let start = script.find(marker)? + marker.len();
    let rest = &script[start..];
    let end = rest.rfind("\nPY\n").or_else(|| rest.rfind("\nPY"))?;
    Some(&rest[..end])
}

#[test]
fn complete_build_system_task_ordering_solution_matches_cpython() {
    let Ok(root) = std::env::var("TASKTROVE_ROOT") else {
        return;
    };
    let solve_path = Path::new(&root).join("build-system-task-ordering/solution/solve.sh");
    let wrapper = std::fs::read_to_string(&solve_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", solve_path.display()));
    let mut source = python_heredoc(&wrapper)
        .unwrap_or_else(|| panic!("no Python heredoc found in {}", solve_path.display()))
        .to_string();
    source.push_str(
        r#"
print(solve([
    "TARGET compile\n",
    "Depends = source\n",
    "TARGET source\n",
]))
"#,
    );

    let reference = match Command::new("python3.14").arg("-c").arg(&source).output() {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return,
        Err(error) => panic!("failed to run CPython 3.14: {error}"),
    };

    let mut environment = Environment::new();
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status = shellsim::python::run_python(
        &mut environment,
        &["python3.14".into(), "-c".into(), source],
        Vec::new(),
        &mut stdout,
        &mut stderr,
    );

    assert_eq!(
        status,
        reference.status.code().unwrap_or(1),
        "shellsim stderr: {}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, reference.stdout);
    assert_eq!(stderr, reference.stderr);
}

#[test]
fn heredoc_extraction_is_delimiter_bounded() {
    assert_eq!(
        python_heredoc("cat > /x.py <<'PY'\nprint('PY')\nPY\nafter\n"),
        Some("print('PY')")
    );
}
