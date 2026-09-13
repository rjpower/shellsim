//! Model Context Protocol transport for the persistent harness.
//!
//! This module is a protocol adapter, not a second execution environment. Each tool call is
//! translated into a typed [`HarnessOperation`] and routed through [`HarnessManager`]. Host
//! filesystem ingestion remains a trusted startup concern, and simulated input cannot acquire
//! host process, filesystem, network, environment, or clock capabilities through MCP.

use std::io::{BufRead, Write};

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde_json::{json, Map, Value};

use crate::harness::{HarnessOperation, HarnessRequest, HarnessResponse, HarnessResult};
use crate::harness_manager::HarnessManager;

/// Maximum bytes accepted for one newline-delimited MCP message.
pub const MAX_MCP_MESSAGE_BYTES: usize = 20 * 1024 * 1024;

const LATEST_PROTOCOL_VERSION: &str = "2025-06-18";
const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &["2024-11-05", "2025-03-26", LATEST_PROTOCOL_VERSION];

/// Serve a single MCP stdio connection until input reaches EOF.
///
/// The transport uses the MCP stdio convention of one JSON-RPC message per line. Input is
/// drained after an oversized message so a client can receive the error and continue with the
/// next request. Responses are flushed individually to support interactive agent clients.
pub fn serve(
    input: &mut impl BufRead,
    output: &mut impl Write,
    manager: &mut HarnessManager,
) -> Result<(), String> {
    let mut initialized = false;
    while let Some(line) = read_bounded_line(input).map_err(|error| error.to_string())? {
        let response = match line {
            Ok(line) => match serde_json::from_slice::<Value>(&line) {
                Ok(message) => handle_message(message, manager, &mut initialized),
                Err(error) => Some(rpc_error(
                    Value::Null,
                    -32700,
                    format!("parse error: {error}"),
                )),
            },
            Err(error) => Some(rpc_error(Value::Null, -32600, error)),
        };
        let Some(response) = response else {
            continue;
        };
        serde_json::to_writer(&mut *output, &response).map_err(|error| error.to_string())?;
        output.write_all(b"\n").map_err(|error| error.to_string())?;
        output.flush().map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn handle_message(
    message: Value,
    manager: &mut HarnessManager,
    initialized: &mut bool,
) -> Option<Value> {
    let Some(object) = message.as_object() else {
        return Some(rpc_error(Value::Null, -32600, "request must be an object"));
    };
    if object.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Some(rpc_error(Value::Null, -32600, "jsonrpc must be '2.0'"));
    }
    let id = object.get("id").cloned();
    let Some(method) = object.get("method").and_then(Value::as_str) else {
        return id.map(|id| rpc_error(id, -32600, "request method must be a string"));
    };
    let id = id?;
    if !matches!(id, Value::Null | Value::String(_) | Value::Number(_)) {
        return Some(rpc_error(
            Value::Null,
            -32600,
            "request id has an invalid type",
        ));
    }
    let params = object.get("params").cloned().unwrap_or_else(|| json!({}));
    let result = match method {
        "initialize" => initialize_result(&params).inspect(|_| {
            *initialized = true;
        }),
        "ping" => Ok(json!({})),
        "tools/list" if *initialized => Ok(json!({"tools": tool_definitions()})),
        "tools/call" if *initialized => call_tool(params, manager),
        "tools/list" | "tools/call" => Err((-32002, "server is not initialized".to_string())),
        _ => Err((-32601, format!("method not found: {method}"))),
    };
    Some(match result {
        Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
        Err((code, message)) => rpc_error(id, code, message),
    })
}

fn initialize_result(params: &Value) -> Result<Value, (i64, String)> {
    let object = params
        .as_object()
        .ok_or_else(|| (-32602, "initialize params must be an object".to_string()))?;
    let requested = object
        .get("protocolVersion")
        .and_then(Value::as_str)
        .unwrap_or(LATEST_PROTOCOL_VERSION);
    let version = if SUPPORTED_PROTOCOL_VERSIONS.contains(&requested) {
        requested
    } else {
        LATEST_PROTOCOL_VERSION
    };
    Ok(json!({
        "protocolVersion": version,
        "capabilities": {"tools": {"listChanged": false}},
        "serverInfo": {"name": "shellsim", "version": env!("CARGO_PKG_VERSION")},
        "instructions": "Use these tools to operate a deterministic simulated /work tree. Commands never execute on the host."
    }))
}

fn call_tool(params: Value, manager: &mut HarnessManager) -> Result<Value, (i64, String)> {
    let params = params
        .as_object()
        .ok_or_else(|| (-32602, "tools/call params must be an object".to_string()))?;
    let name = required_string(params, "name")?;
    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let arguments = arguments
        .as_object()
        .ok_or_else(|| (-32602, "tool arguments must be an object".to_string()))?;
    let session_id = optional_u64(arguments, "session_id")?;
    let operation = tool_operation(name, arguments)?;
    let response = manager.handle(HarnessRequest {
        id: None,
        session_id,
        operation,
    });
    let summary = summarize_response(&response);
    let structured = serde_json::to_value(&response)
        .map_err(|error| (-32603, format!("cannot encode harness response: {error}")))?;
    Ok(json!({
        "content": [{"type": "text", "text": summary}],
        "structuredContent": structured,
        "isError": !response.ok
    }))
}

fn tool_operation(
    name: &str,
    arguments: &Map<String, Value>,
) -> Result<HarnessOperation, (i64, String)> {
    let allowed = match name {
        "execute" => &["source", "stdin", "session_id"][..],
        "read_file" | "remove_path" => &["path", "session_id"][..],
        "write_file" => &["path", "content", "mode", "session_id"][..],
        "list_paths" => &["root", "session_id"][..],
        "stat_path" => &["path", "follow_symlinks", "session_id"][..],
        "make_directory" => &["path", "mode", "parents", "session_id"][..],
        "apply_patch" => &["patch", "strip", "session_id"][..],
        "checkpoint" | "workspace_diff" | "reset_workspace" | "inspect" => &["session_id"][..],
        "fork_session" => &["source"][..],
        "drop_session" => &["target"][..],
        _ => return Err((-32602, format!("unknown tool: {name}"))),
    };
    if let Some(field) = arguments
        .keys()
        .find(|field| !allowed.contains(&field.as_str()))
    {
        return Err((-32602, format!("unknown {name} argument: {field}")));
    }
    let invalid = |message: String| (-32602, message);
    match name {
        "execute" => Ok(HarnessOperation::Execute {
            source: required_string(arguments, "source")?.to_string(),
            stdin_base64: STANDARD.encode(optional_string(arguments, "stdin")?.unwrap_or("")),
        }),
        "read_file" => Ok(HarnessOperation::ReadFile {
            path: required_string(arguments, "path")?.to_string(),
        }),
        "write_file" => Ok(HarnessOperation::WriteFile {
            path: required_string(arguments, "path")?.to_string(),
            data_base64: STANDARD.encode(required_string(arguments, "content")?),
            mode: optional_u64(arguments, "mode")?
                .unwrap_or(0o644)
                .try_into()
                .map_err(|_| invalid("mode is outside the u32 range".to_string()))?,
        }),
        "list_paths" => Ok(HarnessOperation::ListPaths {
            root: optional_string(arguments, "root")?
                .unwrap_or("/work")
                .to_string(),
        }),
        "stat_path" => Ok(HarnessOperation::StatPath {
            path: required_string(arguments, "path")?.to_string(),
            follow_symlinks: optional_bool(arguments, "follow_symlinks")?.unwrap_or(true),
        }),
        "make_directory" => Ok(HarnessOperation::MakeDirectory {
            path: required_string(arguments, "path")?.to_string(),
            mode: optional_u64(arguments, "mode")?
                .unwrap_or(0o755)
                .try_into()
                .map_err(|_| invalid("mode is outside the u32 range".to_string()))?,
            parents: optional_bool(arguments, "parents")?.unwrap_or(false),
        }),
        "apply_patch" => Ok(HarnessOperation::ApplyPatch {
            patch: required_string(arguments, "patch")?.to_string(),
            strip: optional_u64(arguments, "strip")?
                .unwrap_or(1)
                .try_into()
                .map_err(|_| invalid("strip is outside the usize range".to_string()))?,
        }),
        "remove_path" => Ok(HarnessOperation::RemovePath {
            path: required_string(arguments, "path")?.to_string(),
        }),
        "checkpoint" => Ok(HarnessOperation::Checkpoint),
        "workspace_diff" => Ok(HarnessOperation::WorkspaceDiff),
        "reset_workspace" => Ok(HarnessOperation::ResetWorkspace),
        "inspect" => Ok(HarnessOperation::Inspect),
        "fork_session" => Ok(HarnessOperation::ForkSession {
            source: optional_u64(arguments, "source")?.unwrap_or(0),
        }),
        "drop_session" => Ok(HarnessOperation::DropSession {
            target: required_u64(arguments, "target")?,
        }),
        _ => unreachable!("tool name was validated above"),
    }
}

fn summarize_response(response: &HarnessResponse) -> String {
    if let Some(error) = response.error.as_ref() {
        return error.clone();
    }
    match response.result.as_ref() {
        Some(HarnessResult::Execute(result)) => {
            let stdout = STANDARD
                .decode(&result.stdout_base64)
                .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                .unwrap_or_else(|_| "<invalid encoded stdout>".to_string());
            let stderr = STANDARD
                .decode(&result.stderr_base64)
                .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                .unwrap_or_else(|_| "<invalid encoded stderr>".to_string());
            format!(
                "exit status: {}\nstdout:\n{}\nstderr:\n{}",
                result.outcome.exit_status, stdout, stderr
            )
        }
        Some(HarnessResult::File(file)) => match STANDARD.decode(&file.data_base64) {
            Ok(bytes) => match String::from_utf8(bytes) {
                Ok(text) => text,
                Err(error) => format!(
                    "binary file: {} bytes (data is available in structuredContent)",
                    error.as_bytes().len()
                ),
            },
            Err(_) => "invalid encoded file response".to_string(),
        },
        Some(HarnessResult::Paths { paths }) => paths.join("\n"),
        Some(HarnessResult::Session { session_id }) => format!("session_id: {session_id}"),
        Some(HarnessResult::Acknowledged) => "ok".to_string(),
        Some(result) => serde_json::to_string_pretty(result)
            .unwrap_or_else(|_| "harness response could not be summarized".to_string()),
        None => "empty harness response".to_string(),
    }
}

fn tool_definitions() -> Vec<Value> {
    vec![
        tool(
            "execute",
            "Run shell source in the persistent simulated environment.",
            object_schema(&[
                ("source", "string", true),
                ("stdin", "string", false),
                ("session_id", "integer", false),
            ]),
            false,
            false,
        ),
        tool(
            "read_file",
            "Read a file from the simulated /work tree.",
            object_schema(&[("path", "string", true), ("session_id", "integer", false)]),
            true,
            false,
        ),
        tool(
            "write_file",
            "Write UTF-8 text to a file in the simulated /work tree.",
            object_schema(&[
                ("path", "string", true),
                ("content", "string", true),
                ("mode", "integer", false),
                ("session_id", "integer", false),
            ]),
            false,
            false,
        ),
        tool(
            "list_paths",
            "List paths below a simulated workspace directory.",
            object_schema(&[("root", "string", false), ("session_id", "integer", false)]),
            true,
            false,
        ),
        tool(
            "stat_path",
            "Inspect metadata for a simulated workspace path.",
            object_schema(&[
                ("path", "string", true),
                ("follow_symlinks", "boolean", false),
                ("session_id", "integer", false),
            ]),
            true,
            false,
        ),
        tool(
            "make_directory",
            "Create a directory in the simulated workspace.",
            object_schema(&[
                ("path", "string", true),
                ("mode", "integer", false),
                ("parents", "boolean", false),
                ("session_id", "integer", false),
            ]),
            false,
            false,
        ),
        tool(
            "apply_patch",
            "Apply a unified patch atomically below simulated /work.",
            object_schema(&[
                ("patch", "string", true),
                ("strip", "integer", false),
                ("session_id", "integer", false),
            ]),
            false,
            false,
        ),
        tool(
            "remove_path",
            "Remove a path recursively from the simulated workspace.",
            object_schema(&[("path", "string", true), ("session_id", "integer", false)]),
            false,
            true,
        ),
        tool(
            "checkpoint",
            "Set the current simulated workspace as the diff/reset baseline.",
            object_schema(&[("session_id", "integer", false)]),
            false,
            false,
        ),
        tool(
            "workspace_diff",
            "Return typed changes from the workspace checkpoint.",
            object_schema(&[("session_id", "integer", false)]),
            true,
            false,
        ),
        tool(
            "reset_workspace",
            "Restore the simulated workspace checkpoint without rewinding machine state.",
            object_schema(&[("session_id", "integer", false)]),
            false,
            true,
        ),
        tool(
            "inspect",
            "Inspect modeled processes, resources, terminal state, and telemetry.",
            object_schema(&[("session_id", "integer", false)]),
            true,
            false,
        ),
        tool(
            "fork_session",
            "Clone a complete bounded simulated machine state.",
            object_schema(&[("source", "integer", false)]),
            false,
            false,
        ),
        tool(
            "drop_session",
            "Release a cloned simulated session.",
            object_schema(&[("target", "integer", true)]),
            false,
            true,
        ),
    ]
}

fn tool(
    name: &str,
    description: &str,
    input_schema: Value,
    read_only: bool,
    destructive: bool,
) -> Value {
    json!({
        "name": name,
        "description": description,
        "inputSchema": input_schema,
        "annotations": {
            "readOnlyHint": read_only,
            "destructiveHint": destructive,
            "idempotentHint": read_only,
            "openWorldHint": false
        }
    })
}

fn object_schema(fields: &[(&str, &str, bool)]) -> Value {
    let properties = fields
        .iter()
        .map(|(name, kind, _)| ((*name).to_string(), json!({"type": kind})))
        .collect::<Map<_, _>>();
    let required = fields
        .iter()
        .filter(|(_, _, required)| *required)
        .map(|(name, _, _)| Value::String((*name).to_string()))
        .collect::<Vec<_>>();
    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false
    })
}

fn required_string<'a>(
    object: &'a Map<String, Value>,
    name: &str,
) -> Result<&'a str, (i64, String)> {
    object
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| (-32602, format!("{name} must be a string")))
}

fn optional_string<'a>(
    object: &'a Map<String, Value>,
    name: &str,
) -> Result<Option<&'a str>, (i64, String)> {
    match object.get(name) {
        None => Ok(None),
        Some(value) => value
            .as_str()
            .map(Some)
            .ok_or_else(|| (-32602, format!("{name} must be a string"))),
    }
}

fn required_u64(object: &Map<String, Value>, name: &str) -> Result<u64, (i64, String)> {
    optional_u64(object, name)?.ok_or_else(|| (-32602, format!("{name} must be an integer")))
}

fn optional_u64(object: &Map<String, Value>, name: &str) -> Result<Option<u64>, (i64, String)> {
    match object.get(name) {
        None => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| (-32602, format!("{name} must be a non-negative integer"))),
    }
}

fn optional_bool(object: &Map<String, Value>, name: &str) -> Result<Option<bool>, (i64, String)> {
    match object.get(name) {
        None => Ok(None),
        Some(value) => value
            .as_bool()
            .map(Some)
            .ok_or_else(|| (-32602, format!("{name} must be a boolean"))),
    }
}

fn rpc_error(id: Value, code: i64, message: impl Into<String>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {"code": code, "message": message.into()}
    })
}

fn read_bounded_line(input: &mut impl BufRead) -> std::io::Result<Option<Result<Vec<u8>, String>>> {
    let mut line = Vec::new();
    let mut too_large = false;
    let mut saw_input = false;
    loop {
        let available = input.fill_buf()?;
        if available.is_empty() {
            return if saw_input {
                Ok(Some(if too_large {
                    Err(format!(
                        "message exceeds the {MAX_MCP_MESSAGE_BYTES}-byte limit"
                    ))
                } else {
                    Ok(line)
                }))
            } else {
                Ok(None)
            };
        }
        saw_input = true;
        let consumed = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |position| position + 1);
        if !too_large {
            let remaining = MAX_MCP_MESSAGE_BYTES
                .saturating_add(1)
                .saturating_sub(line.len());
            line.extend_from_slice(&available[..consumed.min(remaining)]);
            too_large = line.len() > MAX_MCP_MESSAGE_BYTES;
        }
        let complete = available[..consumed].ends_with(b"\n");
        input.consume(consumed);
        if complete {
            if !too_large {
                line.pop();
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
            }
            return Ok(Some(if too_large {
                Err(format!(
                    "message exceeds the {MAX_MCP_MESSAGE_BYTES}-byte limit"
                ))
            } else {
                Ok(line)
            }));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::HarnessSession;
    use crate::Limits;
    use std::io::Cursor;

    fn exchange(lines: &[Value]) -> Vec<Value> {
        let mut input = lines
            .iter()
            .map(|line| serde_json::to_string(line).unwrap())
            .collect::<Vec<_>>()
            .join("\n")
            .into_bytes();
        input.push(b'\n');
        let mut output = Vec::new();
        let mut manager = HarnessManager::new(HarnessSession::new(Limits::default()));
        serve(&mut Cursor::new(input), &mut output, &mut manager).unwrap();
        String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[test]
    fn initializes_lists_tools_and_runs_in_one_persistent_workspace() {
        let responses = exchange(&[
            json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","clientInfo":{"name":"test","version":"1"},"capabilities":{}}}),
            json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
            json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}),
            json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"execute","arguments":{"source":"printf hello > note"}}}),
            json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"read_file","arguments":{"path":"note"}}}),
        ]);
        assert_eq!(responses.len(), 4);
        assert_eq!(responses[0]["result"]["protocolVersion"], "2025-06-18");
        assert!(responses[1]["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tool| tool["name"] == "execute"));
        assert_eq!(responses[2]["result"]["isError"], false);
        assert_eq!(responses[3]["result"]["content"][0]["text"], "hello");
    }

    #[test]
    fn reports_protocol_and_tool_errors_without_ending_the_connection() {
        let responses = exchange(&[
            json!({"jsonrpc":"2.0","id":1,"method":"tools/list"}),
            json!({"jsonrpc":"2.0","id":2,"method":"initialize","params":{}}),
            json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"read_file","arguments":{"path":"/etc/passwd"}}}),
            json!({"jsonrpc":"2.0","id":4,"method":"ping"}),
        ]);
        assert_eq!(responses[0]["error"]["code"], -32002);
        assert_eq!(responses[2]["result"]["isError"], true);
        assert!(responses[2]["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("/work"));
        assert_eq!(responses[3]["result"], json!({}));
    }
}
