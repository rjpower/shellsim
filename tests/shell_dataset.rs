//! Reduced dataset wrapper idioms compared with Bash without accessing host task files.

use shellsim::{Environment, Limits};

fn compare_bash(source: &str) {
    let mut env = Environment::new();
    let (outcome, stdout, stderr) = env.run_script_capture(source);
    let reference = std::process::Command::new("bash")
        .args(["--noprofile", "--norc", "-c", source])
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .output()
        .expect("Bash reference");
    assert_eq!(
        outcome.exit_status,
        reference.status.code().unwrap(),
        "{source}: {stderr:?}"
    );
    assert_eq!(stdout, reference.stdout, "{source}");
    assert_eq!(stderr, reference.stderr, "{source}");
}

#[test]
fn err_traps_preserve_status_and_follow_conditional_suppression() {
    for source in [
        "trap 'printf \"err:%s\\n\" \"$?\"' ERR; false; printf 'status:%s\\n' \"$?\"",
        "set -Eeuo pipefail; trap 'echo err' ERR; false || echo handled; ! true; if false; then echo no; fi; echo done",
        "set -e; trap 'echo err' ERR; false; echo unreachable",
        "trap 'false; echo handled' ERR; false; echo done",
        "trap 'echo no' ERR; trap - ERR; false; echo done",
        "trap 'echo err' ERR; false | true; set -o pipefail; false | true; echo done",
        "set -E; trap 'echo err' ERR; true | false; echo done",
    ] { compare_bash(source); }
}

#[test]
fn errtrace_controls_function_subshell_and_substitution_inheritance() {
    for source in [
        "trap 'echo err' ERR; f() { false; }; f; echo done",
        "set -E; trap 'echo err' ERR; f() { false; }; f; echo done",
        "set -E; trap 'echo err' ERR; f() { false; echo body; }; f || echo no; echo done",
        "set -E; trap 'echo err' ERR; (false); echo done",
        "trap 'echo err' ERR; (false); echo done",
        "set -E; trap 'echo err >&2' ERR; x=$(false); echo done",
        "set -o errtrace; trap 'echo err' ERR; set +E; f() { false; echo body; }; f; echo done",
        "trap 'echo old' ERR; f() { trap 'echo new' ERR; }; f; false; echo done",
    ] {
        compare_bash(source);
    }
}

#[test]
fn read_delimiters_leave_following_records_on_the_descriptor() {
    for source in [
        r#"printf 'one\0two\0' | { IFS= read -r -d '' a; IFS= read -r -d '' b; printf '%s:%s\n' "$a" "$b"; }"#,
        r#"printf 'one:two:tail' | { while IFS= read -rd: x; do printf '[%s]' "$x"; done; printf 'tail=%s\n' "$x"; }"#,
        r#"printf 'one\ntwo\n' | { read -r a; read -r b; printf '%s:%s\n' "$a" "$b"; }"#,
        r#"printf 'a\\:b:c:' | { IFS= read -d : a; IFS= read -d : b; printf '%s:%s\n' "$a" "$b"; }"#,
        r#"printf 'a\0b\n' | { IFS= read -r a; printf '%s\n' "$a"; }"#,
        r#"printf 'a\\\nb\n' | { IFS= read a; printf '%s\n' "$a"; }"#,
        r#"printf 'unfinished' | { IFS= read -r a; printf '%s:%s\n' "$?" "$a"; }"#,
    ] {
        compare_bash(source);
    }
}

#[test]
fn delimiter_read_and_error_traps_remain_resource_bounded() {
    let mut env = Environment::with_limits(Limits {
        cpu: 20_000,
        ..Default::default()
    });
    let (outcome, _, _) =
        env.run_script_capture("set -E; trap 'while true; do :; done' ERR; false");
    assert!(outcome.stop_reason.is_some());
}

#[test]
fn swe_trusted_test_restore_loop_preserves_product_edits() {
    let mut env = Environment::new();
    let source = r#"
set -Eeuo pipefail
cd /work
git init -q
mkdir tests
printf original > 'tests/test spaced.py'
printf original > product.py
printf '*.log\n' > .gitignore
git add tests product.py .gitignore
git commit -qm base
trusted=$(git rev-parse HEAD)
printf edited > 'tests/test spaced.py'
printf edited > product.py
printf extra > tests/test_extra.py
printf ignored > tests/test_hidden.log
while IFS= read -r -d '' path; do
    case "$path" in
        tests/*)
            git clean -ffdx -- "$path"
            rm -rf -- "$path"
            if git cat-file -e "${trusted}:${path}" 2>/dev/null; then
                git archive --format=tar "$trusted" -- "$path" | tar -xf - -C /work
            fi
            ;;
    esac
done < <(
    {
        git diff --name-only -z "$trusted"
        git ls-files --others --exclude-standard -z
        git ls-files --others --ignored --exclude-standard -z
    } | sort -zu
)
test "$(cat 'tests/test spaced.py')" = original
test "$(cat product.py)" = edited
test ! -e tests/test_extra.py
test ! -e tests/test_hidden.log
"#;
    let (outcome, _, stderr) = env.run_script_capture(source);
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
}

#[test]
fn nul_sort_preserves_embedded_newlines_and_terminates_final_record() {
    for source in [
        r#"printf 'b\0a\nx\0b\0a' | sort -zu"#,
        r#"printf 'b\na' | sort"#,
        r#"printf '\0b\0\0a\0' | sort --zero-terminated -u"#,
    ] {
        compare_bash(source);
    }
}

#[test]
fn compound_redirections_expand_substitutions_before_running_body() {
    compare_bash(r#"while IFS= read -r line; do printf '[%s]' "$line"; done < <(printf 'a\nb\n')"#);
    compare_bash(r#"{ read -r line; printf '%s\n' "$line"; } < <(printf 'body\n')"#);
}
