//! A small `jq` subset over serde_json. Covers the filters that appear most in the corpus:
//! identity `.`, field access `.a.b`, optional `.a?`, array/object iteration `.[]`,
//! indexing `.[n]`, pipes `f | g`, and the builtins `keys`, `length`, `values`, `type`,
//! `has(k)`, `to_entries`, `add`. Flags: `-r/--raw-output`, `-c/--compact-output`,
//! `-n/--null-input`. Anything outside this set is reported as unsupported.

use crate::commands::util::{parse_options, OptionSpec};
use crate::interp::Interp;
use serde_json::Value;

type Out<'a> = &'a mut Vec<u8>;

pub fn jq(interp: &mut Interp, args: &[String], stdin: Vec<u8>, out: Out, err: Out) -> i32 {
    const OPTIONS: &[OptionSpec] = &[
        OptionSpec::flag("raw", Some('r'), Some("raw-output")),
        OptionSpec::flag("compact", Some('c'), Some("compact-output")),
        OptionSpec::flag("null_input", Some('n'), Some("null-input")),
        OptionSpec::flag("help", None, Some("help")),
    ];
    let parsed = match parse_options(args, OPTIONS) {
        Ok(parsed) => parsed,
        Err(error) => {
            ewln(err, &format!("jq: {error}"));
            return 2;
        }
    };
    let mut raw = false;
    let mut compact = false;
    let mut null_input = false;
    for option in parsed.options {
        match option.key {
            "raw" => raw = true,
            "compact" => compact = true,
            "null_input" => null_input = true,
            "help" => {
                out.extend_from_slice(b"usage: jq [-rcn] FILTER [FILE...]\n");
                return 0;
            }
            _ => unreachable!("option keys come from OPTIONS"),
        }
    }
    let mut operands = parsed.operands.into_iter();
    let filter = operands.next().unwrap_or_else(|| ".".to_string());
    let files = operands.collect::<Vec<_>>();
    let input_bytes = if null_input {
        b"null".to_vec()
    } else if files.is_empty() {
        stdin
    } else {
        let mut d = Vec::new();
        for f in &files {
            match interp.vfs.read(&interp.cwd, f) {
                Ok(b) => d.extend(b),
                Err(e) => {
                    ewln(err, &format!("jq: error: {e}"));
                    return 2;
                }
            }
        }
        d
    };
    let root: Value = match serde_json::from_slice(&input_bytes) {
        Ok(v) => v,
        Err(e) => {
            ewln(err, &format!("jq: parse error: {e}"));
            return 2;
        }
    };
    let prog = match parse_filter(&filter) {
        Ok(p) => p,
        Err(e) => {
            interp.note_unsupported(&format!("jq:{e}"));
            ewln(err, &format!("jq: unsupported filter: {e}"));
            return 3;
        }
    };
    let mut results = Vec::new();
    if let Err(e) = eval(&prog, &root, &mut results) {
        ewln(err, &format!("jq: error: {e}"));
        return 5;
    }
    for v in results {
        emit(&v, raw, compact, out);
    }
    0
}

fn emit(v: &Value, raw: bool, compact: bool, out: Out) {
    if raw {
        if let Value::String(s) = v {
            out.extend_from_slice(s.as_bytes());
            out.push(b'\n');
            return;
        }
    }
    let s = if compact {
        serde_json::to_string(v).unwrap_or_default()
    } else {
        serde_json::to_string_pretty(v).unwrap_or_default()
    };
    out.extend_from_slice(s.as_bytes());
    out.push(b'\n');
}

// ---- filter AST ----

#[derive(Debug, Clone)]
enum Filter {
    Identity,
    Field(String),
    OptField(String),
    Index(i64),
    Iterate,
    Pipe(Vec<Filter>),
    Keys,
    Length,
    Values,
    Type,
    Add,
    ToEntries,
    Has(String),
}

fn parse_filter(s: &str) -> Result<Filter, String> {
    let stages: Vec<&str> = s.split('|').map(|x| x.trim()).collect();
    let mut parsed = Vec::new();
    for st in stages {
        parsed.push(parse_stage(st)?);
    }
    if parsed.len() == 1 {
        Ok(parsed.pop().unwrap())
    } else {
        Ok(Filter::Pipe(parsed))
    }
}

fn parse_stage(s: &str) -> Result<Filter, String> {
    let s = s.trim();
    match s {
        "." => return Ok(Filter::Identity),
        "keys" | "keys_unsorted" => return Ok(Filter::Keys),
        "length" => return Ok(Filter::Length),
        "values" | ".[]?" => {}
        "type" => return Ok(Filter::Type),
        "add" => return Ok(Filter::Add),
        "to_entries" => return Ok(Filter::ToEntries),
        ".[]" => return Ok(Filter::Iterate),
        _ => {}
    }
    if s == "values" {
        return Ok(Filter::Values);
    }
    if let Some(rest) = s.strip_prefix("has(") {
        let key = rest.trim_end_matches(')').trim().trim_matches('"');
        return Ok(Filter::Has(key.to_string()));
    }
    // path expression like .a.b[0].c[]
    if s.starts_with('.') {
        return parse_path(s);
    }
    Err(format!("'{s}'"))
}

fn parse_path(s: &str) -> Result<Filter, String> {
    // build a pipe of field/index/iterate ops
    let mut ops = Vec::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '.' => {
                i += 1;
                // .[ ... ] or .field
                if i < chars.len() && chars[i] == '[' {
                    // handled in '[' branch below by not advancing
                    continue;
                }
                let mut name = String::new();
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    name.push(chars[i]);
                    i += 1;
                }
                if name.is_empty() {
                    if i >= chars.len() {
                        ops.push(Filter::Identity);
                    }
                    continue;
                }
                let optional = i < chars.len() && chars[i] == '?';
                if optional {
                    i += 1;
                    ops.push(Filter::OptField(name));
                } else {
                    ops.push(Filter::Field(name));
                }
            }
            '[' => {
                i += 1;
                let mut inner = String::new();
                while i < chars.len() && chars[i] != ']' {
                    inner.push(chars[i]);
                    i += 1;
                }
                i += 1; // ]
                let inner = inner.trim();
                if inner.is_empty() {
                    ops.push(Filter::Iterate);
                } else if let Ok(n) = inner.parse::<i64>() {
                    ops.push(Filter::Index(n));
                } else {
                    let key = inner.trim_matches('"');
                    ops.push(Filter::Field(key.to_string()));
                }
                if i < chars.len() && chars[i] == '?' {
                    i += 1;
                }
            }
            '?' => {
                i += 1;
            }
            c if c.is_whitespace() => {
                i += 1;
            }
            _ => return Err(format!("path '{s}'")),
        }
    }
    if ops.len() == 1 {
        Ok(ops.pop().unwrap())
    } else if ops.is_empty() {
        Ok(Filter::Identity)
    } else {
        Ok(Filter::Pipe(ops))
    }
}

fn eval(f: &Filter, input: &Value, out: &mut Vec<Value>) -> Result<(), String> {
    match f {
        Filter::Identity => out.push(input.clone()),
        Filter::Field(name) | Filter::OptField(name) => match input {
            Value::Object(m) => out.push(m.get(name).cloned().unwrap_or(Value::Null)),
            Value::Null if matches!(f, Filter::OptField(_)) => {}
            Value::Null => out.push(Value::Null),
            _ => {
                if matches!(f, Filter::OptField(_)) {
                } else {
                    return Err(format!("cannot index {} with \"{name}\"", type_of(input)));
                }
            }
        },
        Filter::Index(n) => {
            if let Value::Array(a) = input {
                let idx = if *n < 0 { a.len() as i64 + n } else { *n };
                out.push(a.get(idx as usize).cloned().unwrap_or(Value::Null));
            } else {
                out.push(Value::Null);
            }
        }
        Filter::Iterate => match input {
            Value::Array(a) => {
                for v in a {
                    out.push(v.clone());
                }
            }
            Value::Object(m) => {
                for v in m.values() {
                    out.push(v.clone());
                }
            }
            _ => return Err(format!("cannot iterate over {}", type_of(input))),
        },
        Filter::Pipe(stages) => {
            let mut current = vec![input.clone()];
            for st in stages {
                let mut next = Vec::new();
                for v in &current {
                    eval(st, v, &mut next)?;
                }
                current = next;
            }
            out.extend(current);
        }
        Filter::Keys => {
            if let Value::Object(m) = input {
                let mut keys: Vec<String> = m.keys().cloned().collect();
                keys.sort();
                out.push(Value::Array(keys.into_iter().map(Value::String).collect()));
            } else if let Value::Array(a) = input {
                out.push(Value::Array((0..a.len()).map(Value::from).collect()));
            } else {
                return Err("keys on non-object".into());
            }
        }
        Filter::Length => {
            let n = match input {
                Value::Array(a) => a.len(),
                Value::Object(m) => m.len(),
                Value::String(s) => s.chars().count(),
                Value::Null => 0,
                _ => 1,
            };
            out.push(Value::from(n));
        }
        Filter::Values => {
            if let Value::Object(m) = input {
                for v in m.values() {
                    out.push(v.clone());
                }
            }
        }
        Filter::Type => out.push(Value::String(type_of(input).to_string())),
        Filter::Add => {
            if let Value::Array(a) = input {
                if a.iter().all(|v| v.is_number()) {
                    let sum: f64 = a.iter().filter_map(|v| v.as_f64()).sum();
                    out.push(Value::from(sum));
                } else if a.iter().all(|v| v.is_string()) {
                    let s: String = a.iter().filter_map(|v| v.as_str()).collect();
                    out.push(Value::String(s));
                } else {
                    out.push(Value::Null);
                }
            }
        }
        Filter::ToEntries => {
            if let Value::Object(m) = input {
                let arr: Vec<Value> = m
                    .iter()
                    .map(|(k, v)| serde_json::json!({"key": k, "value": v}))
                    .collect();
                out.push(Value::Array(arr));
            }
        }
        Filter::Has(k) => {
            let b = matches!(input, Value::Object(m) if m.contains_key(k));
            out.push(Value::Bool(b));
        }
    }
    Ok(())
}

fn type_of(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn ewln(err: Out, s: &str) {
    err.extend_from_slice(s.as_bytes());
    err.push(b'\n');
}
