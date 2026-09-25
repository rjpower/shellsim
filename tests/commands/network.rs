//! Compatibility tests for static HTTP routes and the `curl`/`wget` broker clients.

use shellsim::net::HttpResponse;
use shellsim::Environment;

#[test]
fn curl_sends_typed_request_and_renders_configured_response() {
    let mut environment = Environment::new();
    environment
        .net
        .route(
            "https://api.test/items",
            Some("POST"),
            HttpResponse {
                status: 201,
                headers: vec![("Content-Type".into(), "application/json".into())],
                body: br#"{"id":7}"#.to_vec(),
            },
        )
        .unwrap();

    let (outcome, stdout, stderr) = environment.run_script_capture(
        "curl -i -H 'Accept: application/json' -d 'name=test' https://api.test/items",
    );

    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(
        stdout,
        b"HTTP/1.1 201 Created\r\nContent-Type: application/json\r\n\r\n{\"id\":7}"
    );
    assert!(stderr.is_empty());
    let request = &environment.net.log[0];
    assert_eq!(request.method, "POST");
    assert_eq!(
        request.headers,
        [("Accept".into(), "application/json".into())]
    );
    assert_eq!(request.body_bytes, 9);
    assert_eq!(request.response_status, Some(201));
}

#[test]
fn method_mismatch_http_errors_and_head_have_stable_statuses() {
    let mut environment = Environment::new();
    environment
        .net
        .route(
            "https://api.test/only-post",
            Some("POST"),
            HttpResponse::ok("ok"),
        )
        .unwrap();
    environment
        .net
        .route(
            "https://api.test/missing",
            None,
            HttpResponse {
                status: 404,
                headers: vec![("X-Reason".into(), "fixture".into())],
                body: b"missing".to_vec(),
            },
        )
        .unwrap();

    let (outcome, _, stderr) = environment.run_script_capture("curl https://api.test/only-post");
    assert_eq!(outcome.exit_status, 7);
    assert!(String::from_utf8_lossy(&stderr).contains("no matching virtual HTTP route"));

    let (outcome, stdout, stderr) =
        environment.run_script_capture("curl -sSf https://api.test/missing");
    assert_eq!(outcome.exit_status, 22);
    assert!(stdout.is_empty());
    assert!(String::from_utf8_lossy(&stderr).contains("HTTP response status 404"));

    let (outcome, stdout, stderr) =
        environment.run_script_capture("curl -I https://api.test/missing");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(
        stdout,
        b"HTTP/1.1 404 Not Found\r\nX-Reason: fixture\r\n\r\n"
    );
}

#[test]
fn wget_uses_static_body_and_refuses_http_error_body() {
    let mut environment = Environment::new();
    environment
        .net
        .route_static("https://files.test/data.txt", 200, "payload")
        .unwrap();
    environment
        .net
        .route_static("https://files.test/nope", 503, "do not save")
        .unwrap();

    let (outcome, stdout, stderr) = environment
        .run_script_capture("wget -qO - https://files.test/data.txt; wget https://files.test/nope");

    assert_eq!(outcome.exit_status, 8);
    assert_eq!(stdout, b"payload");
    assert!(String::from_utf8_lossy(&stderr).contains("server returned status 503"));
    assert!(environment.vfs.read("/", "nope").is_err());
}

#[test]
fn route_file_failure_and_invalid_route_configuration_are_explicit() {
    let mut environment = Environment::new();
    let (outcome, _, stderr) = environment.run_script_capture(
        "net route-file https://files.test/missing /missing; curl https://files.test/missing",
    );
    assert_eq!(outcome.exit_status, 23);
    assert!(String::from_utf8_lossy(&stderr).contains("cannot read virtual HTTP body"));

    let (outcome, _, stderr) = environment.run_script_capture("net route https://bad.test 99 body");
    assert_eq!(outcome.exit_status, 2);
    assert!(String::from_utf8_lossy(&stderr).contains("invalid HTTP status 99"));
}

#[test]
fn network_commands_share_virtual_routes_across_child_processes() {
    let mut environment = Environment::new();
    let (outcome, stdout, stderr) = environment.run_script_capture(
        "net route https://files.test/item 200 payload; env -C /work curl -o saved https://files.test/item; cat /work/saved; net log",
    );
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"payloadGET https://files.test/item\n");
    assert!(stderr.is_empty());
    for command in ["net", "curl"] {
        assert!(environment.invocations.events().iter().any(|event| {
            event.pid != 1_234 && event.argv.first().is_some_and(|arg| arg == command)
        }));
    }
}
