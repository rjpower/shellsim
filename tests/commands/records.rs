//! Compatibility tests for buffered record-oriented text transforms.

use shellsim::interp::{Environment, Interp};

fn run(environment: &mut Interp, source: &str) -> (i32, Vec<u8>, String) {
    let (outcome, stdout, stderr) = environment.run_script_capture(source);
    (
        outcome.exit_status,
        stdout,
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

#[test]
fn join_split_and_tsort_cover_common_build_script_usage() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(&mut environment, "printf '1 one\\n2 two\\n' > a; printf '1 uno\\n2 dos\\n' > b; join a b; printf 'a\\nb\\nc\\n' | split -l 2 - part; cat partaa partab; printf 'compile link\\nfetch compile\\n' | tsort");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(
        stdout,
        b"1 one uno\n2 two dos\na\nb\nc\nfetch\ncompile\nlink\n"
    );
}

#[test]
fn split_applies_the_current_umask_to_output_files() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "umask 077; printf 'one\\ntwo\\n' | split -l 1 - part; stat -c '%a' partaa partab",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"600\n600\n");
}

#[test]
fn shuf_is_a_reproducible_permutation() {
    let mut first = Environment::new();
    let mut second = Environment::new();
    let (_, first_stdout, _) = run(&mut first, "printf 'a\\nb\\nc\\nd\\n' | shuf");
    let (_, second_stdout, _) = run(&mut second, "printf 'a\\nb\\nc\\nd\\n' | shuf");
    assert_eq!(first_stdout, second_stdout);
    let mut lines = String::from_utf8(first_stdout)
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect::<Vec<_>>();
    lines.sort();
    assert_eq!(lines, ["a", "b", "c", "d"]);
}
