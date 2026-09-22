//! Compatibility tests for deterministic text transformation and comparison commands.

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
fn diff_uses_lcs_for_insertions_and_supports_unified_output() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "printf 'a\\nb\\nc\\n' > old; printf 'a\\nx\\nb\\nc\\n' > new; diff -u old new",
    );
    assert_eq!(status, 1, "{stderr}");
    assert_eq!(
        stdout,
        b"--- old\n+++ new\n@@ -1,3 +1,4 @@\n a\n+x\n b\n c\n"
    );
}

#[test]
fn recursive_diff_and_brief_mode_report_differences() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "mkdir -p a/sub b/sub; echo one > a/sub/f; echo two > b/sub/f; diff -rq a b",
    );
    assert_eq!(status, 1, "{stderr}");
    assert_eq!(stdout, b"Files a/sub/f and b/sub/f differ\n");
}

#[test]
fn recursive_diff_reports_only_in_directories_at_their_parent() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "mkdir -p a/nested a/left-empty b/nested b/nested/right-empty; diff -r a b",
    );
    assert_eq!(status, 1, "{stderr}");
    assert_eq!(
        stdout,
        b"Only in a: left-empty\nOnly in b/nested: right-empty\n"
    );
}

#[test]
fn diff_whitespace_controls_compare_keys_but_print_original_lines() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "printf 'alpha  beta\\nleft  old\\n' > old; printf 'alpha beta\\nleft new\\n' > new; diff -ub old new",
    );
    assert_eq!(status, 1, "{stderr}");
    assert_eq!(
        stdout,
        b"--- old\n+++ new\n@@ -1,2 +1,2 @@\n alpha  beta\n-left  old\n+left new\n"
    );

    let (status, stdout, stderr) = run(
        &mut environment,
        "printf 'ab c\\n' > old; printf 'a bc\\n' > new; diff --ignore-all-space -q old new; diff --ignore-space-change -q old new",
    );
    assert_eq!(status, 1, "{stderr}");
    assert_eq!(stdout, b"Files old and new differ\n");
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
fn formatting_commands_transform_columns_and_tabs() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(&mut environment, "printf 'a\\tb\\n' | expand -t 4; printf 'a   b\\n' | unexpand -a -t 4; printf 'a:long\\nb:x\\n' | column -t -s :; printf 'one two three four\\n' | fmt -w 9; printf 'abcdef\\n' | fold -w 3");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(
        stdout,
        b"a   b\na\tb\na  long\nb  x\none two\nthree\nfour\nabc\ndef\n"
    );
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

#[test]
fn yes_streams_until_downstream_closes_the_pipe() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(&mut environment, "yes ready | head -n 3");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"ready\nready\nready\n");
}
