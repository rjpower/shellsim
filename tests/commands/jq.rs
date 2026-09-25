//! Compatibility strategy for `jq`: run ordinary filters and assert semantic JSON results.
//! The cases cover TaskTrove filter shapes plus explicit syntax and resource boundaries.

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
fn tasktrove_selection_construction_and_sorting_compose() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write(
            "/",
            "users.json",
            br#"[
                {"id":2,"username":"zoe","email":"z@example","status":"active","last_login":"2026-01-02T03:04:05","roles":[]},
                {"id":1,"username":"amy","email":"a@example","status":"active","last_login":"2026-02-03T04:05:06","roles":["admin"]},
                {"id":3,"username":"off","email":"o@example","status":"disabled","last_login":"2026-01-01T00:00:00","roles":[]}
            ]"#,
            0o644,
        )
        .unwrap();
    let filter = r#"[.[] | select(.status == "active") | {user_id: .id, username: .username, last_login: (.last_login | split("T")[0]), role_count: (.roles | length), primary_role: (.roles | if length > 0 then .[0] else null end)}] | sort_by(.username)"#;
    let (status, stdout, stderr) = run(
        &mut environment,
        &format!("jq --indent 2 '{}' users.json", filter),
    );

    assert_eq!(status, 0, "{stderr}");
    let value: serde_json::Value = serde_json::from_slice(&stdout).unwrap();
    assert_eq!(value[0]["username"], "amy");
    assert_eq!(value[0]["last_login"], "2026-02-03");
    assert_eq!(value[0]["primary_role"], "admin");
    assert_eq!(value[1]["username"], "zoe");
    assert!(value[1]["primary_role"].is_null());
}

#[test]
fn jq_reads_large_pipe_and_uses_child_working_directory() {
    let mut environment = Environment::new();
    let payload = format!("{{\"value\":\"{}\"}}", "x".repeat(128 * 1024));
    environment
        .vfs
        .write("/", "/work/input.json", payload.as_bytes(), 0o644)
        .unwrap();
    let (status, stdout, stderr) = run(
        &mut environment,
        "cat /work/input.json | jq -r '.value' | wc -c",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"131073\n");

    let (status, stdout, stderr) = run(
        &mut environment,
        "env -C /work jq -r '.value | length' input.json",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"131072\n");

    let (status, stdout, stderr) = run(&mut environment, "yes | jq -n '1'");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"1\n");
}

#[test]
fn variables_slurp_conditionals_and_summary_filters_compose() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write(
            "/",
            "items.jsonl",
            b"{\"type\":\"ssh_key\",\"path\":\"/z\"}\n{\"type\":\"cron_job\",\"path\":\"/a\"}\n",
            0o644,
        )
        .unwrap();
    let (status, stdout, stderr) = run(
        &mut environment,
        "jq -n -c --arg type cron_job --arg line_num_str 7 '{type: $type, line: (if $line_num_str == \"0\" then null else ($line_num_str | tonumber) end)}'",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"{\"type\":\"cron_job\",\"line\":7}\n");

    let (status, stdout, stderr) = run(
        &mut environment,
        "jq -s -c 'sort_by(.type, .path) | {total: length, cron: ([.[] | select(.type == \"cron_job\")] | length), artifacts: .}' items.jsonl",
    );
    assert_eq!(status, 0, "{stderr}");
    let value: serde_json::Value = serde_json::from_slice(&stdout).unwrap();
    assert_eq!(value["total"], 2);
    assert_eq!(value["cron"], 1);
    assert_eq!(value["artifacts"][0]["type"], "cron_job");
}

#[test]
fn exit_status_slices_and_compound_sort_keys_match_common_jq_behavior() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "printf '[{\"rank\":1,\"score\":2},{\"rank\":1,\"score\":5},{\"rank\":0,\"score\":1}]' | jq -c 'sort_by([.rank, -.score]) | .[0:2]'",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(
        stdout,
        b"[{\"rank\":0,\"score\":1},{\"rank\":1,\"score\":5}]\n"
    );

    assert_eq!(run(&mut environment, "printf 'null' | jq -e '.'").0, 1);
    assert_eq!(run(&mut environment, "printf 'true' | jq -e '.'").0, 0);
    assert_eq!(run(&mut environment, "printf '{}' | jq -e '.missing'").0, 1);

    let (status, stdout, stderr) = run(
        &mut environment,
        "printf '{\"outer\":{\"value\":1}}' | jq --indent 4 '.'",
    );
    assert_eq!(status, 0, "{stderr}");
    assert!(
        String::from_utf8_lossy(&stdout).contains("\n        \"value\""),
        "{}",
        String::from_utf8_lossy(&stdout)
    );

    let (status, stdout, stderr) = run(
        &mut environment,
        "jq -n -c '[1 + 2, [1, 2] | add, -2, (-7 | length), (1.5 + 2)]'",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"[3,3,-2,7,3.5]\n");

    let (status, _, stderr) = run(&mut environment, "jq -n 'true | length'");
    assert_eq!(status, 5);
    assert!(stderr.contains("not defined for boolean"), "{stderr}");
}

#[test]
fn invalid_filters_and_output_growth_fail_explicitly() {
    let mut environment = Environment::new();
    let (status, _, stderr) = run(&mut environment, "printf '{}' | jq 'map(.)'");
    assert_eq!(status, 3);
    assert!(stderr.contains("unsupported function"), "{stderr}");
    assert_eq!(
        environment.unsupported.values(),
        ["jq:unsupported function \"map\""]
    );

    let nested = format!("{}.{})", "(".repeat(129), ")".repeat(128));
    let (status, _, stderr) = run(&mut environment, &format!("jq -n '{nested}'"));
    assert_eq!(status, 3);
    assert!(stderr.contains("nesting exceeds"), "{stderr}");

    let mut environment = Environment::with_limits(Limits {
        output: 64,
        ..Limits::unlimited()
    });
    let (outcome, _, _) = environment.run_script_capture(
        "printf '[\"abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz\"]' | jq '[.[], .[], .[]]'",
    );
    assert_eq!(outcome.exit_status, 137);
    assert_eq!(outcome.stop_reason, Some(StopReason::OutputLimitExceeded));
}
