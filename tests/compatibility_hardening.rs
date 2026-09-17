//! Regression tests for command-surface sharp edges and shared simulated-filesystem behavior.
//!
//! These tests favor short agent-shaped shell snippets. They cover both the supported common
//! path and the explicit failure frontier so unsupported syntax cannot silently look successful.

use shellsim::{CommandTrust, Environment};

fn run(source: &str) -> (i32, Vec<u8>, Vec<u8>) {
    let mut environment = Environment::new();
    let (outcome, stdout, stderr) = environment.run_script_capture(source);
    (outcome.exit_status, stdout, stderr)
}

fn text(source: &str) -> (i32, String, String) {
    let (status, stdout, stderr) = run(source);
    (
        status,
        String::from_utf8_lossy(&stdout).into_owned(),
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

#[test]
fn unavailable_binaries_fail_and_remain_queryable() {
    let mut environment = Environment::new();
    let (outcome, stdout, stderr) = environment.run_script_capture("npm install package");

    assert_eq!(outcome.exit_status, 127);
    assert!(stdout.is_empty());
    assert_eq!(stderr, b"npm: not implemented in shellsim\n");
    assert_eq!(environment.unsupported.values(), ["npm"]);

    let invocation = environment.invocations.events().pop().unwrap();
    assert_eq!(invocation.argv, ["npm", "install", "package"]);
    assert_eq!(invocation.trust, CommandTrust::Unsupported);
    assert_eq!(invocation.status, Some(127));
    assert_eq!(
        invocation.unsupported_reason.as_deref(),
        Some("not implemented in shellsim")
    );
}

#[test]
fn pseudo_filesystem_is_visible_to_general_filesystem_commands() {
    assert_eq!(
        text("cd /proc; test -e cpuinfo; printf '%s\n' \"$PWD\"; find . -maxdepth 1 -name cpuinfo; realpath self"),
        (0, "/proc\n./cpuinfo\n/proc/1234\n".into(), String::new())
    );
    let (_, stdout, stderr) = run("find /proc -maxdepth 1 -print0");
    assert!(stderr.is_empty());
    assert!(stdout
        .split(|byte| *byte == 0)
        .any(|path| path == b"/proc/cpuinfo"));
}

#[test]
fn find_rejects_unsupported_and_malformed_predicates() {
    for source in [
        "find /proc -delete",
        "find /proc -type z",
        "find /proc -name",
    ] {
        let (status, _, stderr) = text(source);
        assert_eq!(status, 2, "{source}: {stderr}");
        assert!(!stderr.is_empty(), "{source} had no diagnostic");
    }
    let (status, _, stderr) = text("find /missing");
    assert_eq!(status, 1);
    assert!(stderr.contains("find:"), "{stderr}");
}

#[test]
fn redirected_read_loops_preserve_the_shared_descriptor_cursor() {
    assert_eq!(
        text("printf 'a\nb\n' > /tmp/in; while read item; do echo \"$item\"; done < /tmp/in"),
        (0, "a\nb\n".into(), String::new())
    );
}

#[test]
fn positional_and_local_variables_have_shell_scoping() {
    assert_eq!(
        text("set -- one 'two three'; printf '[%s]\n' \"$@\"; value=outer; f(){ local value=inner; echo \"$value\"; }; f; echo \"$value\""),
        (0, "[one]\n[two three]\ninner\nouter\n".into(), String::new())
    );
    let (status, _, stderr) = text("local value=bad");
    assert_eq!(status, 1);
    assert!(stderr.contains("function"), "{stderr}");
}

#[test]
fn fatal_parameter_expansions_stop_the_script() {
    let (status, stdout, stderr) = text("set -u; echo \"$MISSING\"; echo unreachable");
    assert_eq!(status, 1);
    assert!(stdout.is_empty());
    assert!(stderr.contains("unbound variable"), "{stderr}");

    let (status, stdout, stderr) = text("echo \"${MISSING:?need value}\"; echo unreachable");
    assert_eq!(status, 1);
    assert!(stdout.is_empty());
    assert!(stderr.contains("need value"), "{stderr}");
}

#[test]
fn common_text_options_are_exact_or_explicitly_unsupported() {
    assert_eq!(
        text("printf 'a\nb\nc\n' | sed '2d'; printf 'a\nb\n' | grep -e a -e b; printf 'a b\n' | xargs -n1 echo; printf 'aabb\n' | tr -s a-z"),
        (0, "a\nc\na\nb\na\nb\nab\n".into(), String::new())
    );
    let (status, _, stderr) = text("printf 'a\nb\n' | grep -A1 a");
    assert_eq!(status, 2);
    assert!(stderr.contains("unsupported"), "{stderr}");
    let (status, _, stderr) = text("printf data | fold");
    assert_eq!(status, 127);
    assert!(stderr.contains("not implemented"), "{stderr}");
}

#[test]
fn text_tools_share_option_boundaries_and_cover_common_forms() {
    for command in ["grep", "sed", "sort", "jq"] {
        let (status, stdout, stderr) = text(&format!("{command} --help"));
        assert_eq!(status, 0, "{command}: {stderr}");
        assert!(stdout.starts_with("usage:"), "{command}: {stdout}");
        assert!(stderr.is_empty(), "{command}: {stderr}");
    }
    assert_eq!(
        text("printf 'a+\\naa\\n' | grep '^a+$'; printf 'a+\\naa\\n' | grep -E '^a+$'"),
        (0, "a+\naa\n".into(), String::new())
    );
    assert_eq!(
        text("printf 'a+\\naa\\n' | sed 's/a+/basic/'; printf 'aa\\n' | sed -E 's/a+/extended/'"),
        (0, "basic\naa\nextended\n".into(), String::new())
    );
    assert_eq!(
        text("printf 'x:10\\ny:2\\n' | sort -t: -k2n"),
        (0, "y:2\nx:10\n".into(), String::new())
    );
    assert_eq!(
        text("printf 'aa\\naa extra\\naa\\n' | grep -xm1 aa"),
        (0, "aa\n".into(), String::new())
    );
    assert_eq!(
        text("printf 'aa\\n' | grep -m0 aa"),
        (1, String::new(), String::new())
    );
    let (status, _, stderr) = text("grep -m nope value");
    assert_eq!(status, 2);
    assert!(stderr.contains("invalid max count"), "{stderr}");
    for source in ["grep --wat", "sed --wat", "sort --wat", "jq -S ."] {
        let (status, _, stderr) = text(source);
        assert_eq!(status, 2, "{source}: {stderr}");
        assert!(stderr.contains("unsupported option"), "{source}: {stderr}");
    }
    for source in ["sed 's/a'", "sed 's/a/b/z'"] {
        let (status, _, stderr) = text(source);
        assert_eq!(status, 2, "{source}: {stderr}");
        assert!(
            stderr.contains("invalid substitution"),
            "{source}: {stderr}"
        );
    }
}

#[test]
fn package_tools_only_succeed_for_effects_shellsim_can_supply() {
    assert_eq!(
        text("pip install numpy; python -c 'import numpy; print(numpy.array([1, 2]).sum())'"),
        (0, "3\n".into(), String::new())
    );

    let mut environment = Environment::new();
    let (outcome, _, stderr) = environment.run_script_capture("pip install requests");
    assert_eq!(outcome.exit_status, 1);
    assert!(
        String::from_utf8_lossy(&stderr).contains("not bundled"),
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert!(environment
        .unsupported
        .values()
        .iter()
        .any(|feature| feature.contains("requests")));

    let (status, _, stderr) = text("uv remove numpy");
    assert_eq!(status, 2);
    assert!(stderr.contains("unsupported command"), "{stderr}");

    let (status, stdout, stderr) =
        text("uv run --with numpy python -c 'print(\"should not run\")'");
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(stderr.contains("unsupported launcher option"), "{stderr}");

    let (status, _, stderr) = text("pip install --target /tmp/site numpy");
    assert_eq!(status, 1);
    assert!(stderr.contains("unsupported option"), "{stderr}");
}

#[test]
fn uv_sync_validates_all_dependencies_before_mutating_package_state() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write(
            "/work",
            "pyproject.toml",
            b"[project]\ndependencies = [\"numpy\"]\n",
            0o644,
        )
        .unwrap();
    environment
        .vfs
        .write("/work", "requirements.txt", b"requests\n", 0o644)
        .unwrap();

    let (outcome, _, stderr) = environment.run_script_capture("cd /work; uv sync");
    assert_eq!(outcome.exit_status, 1);
    assert!(String::from_utf8_lossy(&stderr).contains("not bundled"));
    assert!(!environment.packages.contains("numpy"));
}

#[test]
fn byte_oriented_tools_preserve_unterminated_input() {
    for (command, expected) in [
        ("head", b"abc".as_slice()),
        ("sort", b"abc".as_slice()),
        ("uniq", b"abc".as_slice()),
        ("rev", b"cba".as_slice()),
    ] {
        let (status, stdout, stderr) = run(&format!("printf abc | {command}"));
        assert_eq!(status, 0, "{command}: {}", String::from_utf8_lossy(&stderr));
        assert_eq!(stdout, expected, "{command}");
    }
    assert_eq!(run("printf abc | cat -n").1, b"     1\tabc");
}

#[test]
fn file_copy_windows_counts_and_formatting_follow_common_contracts() {
    assert_eq!(
        text("mkdir /a /b; printf x > /a/f; cp /a/f /b; mv /b/f /b/g; cat /b/g"),
        (0, "x".into(), String::new())
    );
    assert_eq!(
        text("printf 'a\nb\n' >/a; printf 'c\nd\n' >/b; head -n1 /a /b"),
        (0, "==> /a <==\na\n\n==> /b <==\nc\n".into(), String::new())
    );
    assert_eq!(run("printf abc | tail -c 2").1, b"bc");
    assert_eq!(run("printf 'a\nb\n' | tail +2").1, b"b\n");
    assert_eq!(run("printf '%05d' -3").1, b"-0003");
    assert_eq!(run("printf '\\101'").1, b"A");
    assert_eq!(run("printf '%bafter' 'before\\cignored'").1, b"before");
}

#[test]
fn system_queries_compose_options_and_reject_unknown_ones() {
    assert_eq!(
        text("uname -m -n; id -u -n; type ls; type echo"),
        (
            0,
            "sandbox x86_64\nroot\nls is /usr/bin/ls\necho is a shell builtin\n".into(),
            String::new()
        )
    );
    for source in ["uname -Q", "df -Q"] {
        let (status, _, stderr) = text(source);
        assert_eq!(status, 2, "{source}: {stderr}");
        assert!(stderr.contains("unimplemented"), "{source}: {stderr}");
    }
    for source in ["yes", "readonly value=one"] {
        let (status, _, stderr) = text(source);
        assert_eq!(status, 127, "{source}: {stderr}");
        assert!(stderr.contains("not implemented"), "{source}: {stderr}");
    }
}
