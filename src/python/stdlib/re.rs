//! Bounded, capability-free regular-expression module over Rust's linear-time regex engine.
//!
//! Inputs, matches, and substitution output are metered before host allocation. Unsupported
//! Python flags and regex constructs fail explicitly.

use regex::{Regex, RegexBuilder};

use super::super::native::{
    CallArgs, FunctionDef, MethodDef, ModuleDef, NativeTypeDef, PyConstant, PyError, PyIndex,
    PyMatch, PyRegex, PyResult, PyRuntime, PyString, PyValue, PyValueCast, ValueDef,
};
use super::super::Value;

pub(super) const IGNORECASE: u32 = 2;
pub(super) const MULTILINE: u32 = 8;
const MAX_PATTERN: usize = 4096;
const MAX_INPUT: usize = 1_048_576;
const MAX_MATCHES: usize = 100_000;

pub(in crate::python) static PATTERN_TYPE: NativeTypeDef = NativeTypeDef {
    name: "re.Pattern",
    methods: &[
        MethodDef {
            type_name: "re.Pattern",
            name: "search",
            call: pattern_search,
        },
        MethodDef {
            type_name: "re.Pattern",
            name: "match",
            call: pattern_match,
        },
        MethodDef {
            type_name: "re.Pattern",
            name: "fullmatch",
            call: pattern_fullmatch,
        },
        MethodDef {
            type_name: "re.Pattern",
            name: "findall",
            call: pattern_findall,
        },
        MethodDef {
            type_name: "re.Pattern",
            name: "finditer",
            call: pattern_finditer,
        },
        MethodDef {
            type_name: "re.Pattern",
            name: "sub",
            call: pattern_sub,
        },
    ],
};

pub(in crate::python) static MATCH_TYPE: NativeTypeDef = NativeTypeDef {
    name: "re.Match",
    methods: &[
        MethodDef {
            type_name: "re.Match",
            name: "group",
            call: match_group,
        },
        MethodDef {
            type_name: "re.Match",
            name: "groups",
            call: match_groups,
        },
        MethodDef {
            type_name: "re.Match",
            name: "start",
            call: match_start,
        },
        MethodDef {
            type_name: "re.Match",
            name: "end",
            call: match_end,
        },
    ],
};

fn pattern_search(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    compiled_capture(runtime, receiver, args, CaptureMode::Search)
}

fn pattern_match(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    compiled_capture(runtime, receiver, args, CaptureMode::Match)
}

fn pattern_fullmatch(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    compiled_capture(runtime, receiver, args, CaptureMode::FullMatch)
}

fn compiled_capture(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
    mode: CaptureMode,
) -> PyResult {
    args.expect_positional("compiled regex match", 1, 1)?;
    args.reject_keywords("compiled regex match")?;
    let regex = receiver.cast::<PyRegex>(runtime)?;
    let (pattern, flags) = runtime.regex_parts(regex)?;
    capture(
        runtime,
        CallArgs::new(
            vec![
                Value::String(pattern),
                args.positional()[0].clone(),
                Value::Int(i64::from(flags)),
            ],
            Vec::new(),
        ),
        mode,
    )
}

fn pattern_findall(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    compiled_find(runtime, receiver, args, false)
}

fn pattern_finditer(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    compiled_find(runtime, receiver, args, true)
}

fn compiled_find(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
    return_matches: bool,
) -> PyResult {
    args.expect_positional("compiled regex find", 1, 1)?;
    args.reject_keywords("compiled regex find")?;
    let regex = receiver.cast::<PyRegex>(runtime)?;
    let (pattern, flags) = runtime.regex_parts(regex)?;
    find(
        runtime,
        CallArgs::new(
            vec![
                Value::String(pattern),
                args.positional()[0].clone(),
                Value::Int(i64::from(flags)),
            ],
            Vec::new(),
        ),
        return_matches,
    )
}

fn pattern_sub(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("compiled regex sub", 2, 3)?;
    args.reject_keywords("compiled regex sub")?;
    let regex = receiver.cast::<PyRegex>(runtime)?;
    let (pattern, flags) = runtime.regex_parts(regex)?;
    let count = args.positional().get(2).cloned().unwrap_or(Value::Int(0));
    sub(
        runtime,
        CallArgs::new(
            vec![
                Value::String(pattern),
                args.positional()[0].clone(),
                args.positional()[1].clone(),
                count,
                Value::Int(i64::from(flags)),
            ],
            Vec::new(),
        ),
    )
}

fn match_group(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("re.Match.group", 0, 1)?;
    args.reject_keywords("re.Match.group")?;
    let index = args
        .positional()
        .first()
        .cloned()
        .map(|value| value.cast::<PyIndex>(runtime).map(|index| index.0))
        .transpose()?
        .unwrap_or(0);
    let index = usize::try_from(index).map_err(|_| PyError::value_error("no such group"))?;
    let data = runtime.match_data(receiver.cast::<PyMatch>(runtime)?)?;
    Ok(data
        .groups
        .get(index)
        .cloned()
        .flatten()
        .map_or(Value::None, Value::String))
}

fn match_groups(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    args.expect_positional("re.Match.groups", 0, 0)?;
    args.reject_keywords("re.Match.groups")?;
    let data = runtime.match_data(receiver.cast::<PyMatch>(runtime)?)?;
    runtime.new_tuple(
        data.groups
            .into_iter()
            .skip(1)
            .map(|group| group.map_or(Value::None, Value::String))
            .collect(),
    )
}

fn match_start(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    match_position(runtime, receiver, args, true)
}

fn match_end(runtime: &mut dyn PyRuntime, receiver: PyValue, args: CallArgs) -> PyResult {
    match_position(runtime, receiver, args, false)
}

fn match_position(
    runtime: &mut dyn PyRuntime,
    receiver: PyValue,
    args: CallArgs,
    start: bool,
) -> PyResult {
    args.expect_positional("regex match position", 0, 1)?;
    args.reject_keywords("regex match position")?;
    if let Some(group) = args.positional().first().cloned() {
        let index = group.cast::<PyIndex>(runtime)?.0;
        if index != 0 {
            return Err(PyError::value_error(
                "group-specific match positions are not implemented",
            ));
        }
    }
    let data = runtime.match_data(receiver.cast::<PyMatch>(runtime)?)?;
    let position = if start { data.start } else { data.end };
    i64::try_from(position)
        .map(Value::Int)
        .map_err(|_| PyError::overflow_error("match position is outside Python int range"))
}

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "re",
    functions: &[
        FunctionDef {
            module: "re",
            name: "compile",
            call: compile,
        },
        FunctionDef {
            module: "re",
            name: "search",
            call: search,
        },
        FunctionDef {
            module: "re",
            name: "match",
            call: match_,
        },
        FunctionDef {
            module: "re",
            name: "fullmatch",
            call: fullmatch,
        },
        FunctionDef {
            module: "re",
            name: "findall",
            call: findall,
        },
        FunctionDef {
            module: "re",
            name: "finditer",
            call: finditer,
        },
        FunctionDef {
            module: "re",
            name: "sub",
            call: sub,
        },
        FunctionDef {
            module: "re",
            name: "escape",
            call: escape,
        },
    ],
    values: &[
        ValueDef::Constant {
            name: "IGNORECASE",
            value: PyConstant::Int(IGNORECASE as i64),
        },
        ValueDef::Constant {
            name: "MULTILINE",
            value: PyConstant::Int(MULTILINE as i64),
        },
    ],
};

fn compile(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("re.compile", 1, 2)?;
    args.reject_unknown_keywords("re.compile", &["flags"])?;
    let pattern = string_arg(runtime, &args.positional()[0], "regex pattern", MAX_PATTERN)?;
    let flags = flags_arg(runtime, &args, 1, "re.compile")?;
    build_regex(&pattern, flags)?;
    runtime.new_regex(pattern, flags)
}

fn search(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    capture(runtime, args, CaptureMode::Search)
}
fn match_(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    capture(runtime, args, CaptureMode::Match)
}
fn fullmatch(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    capture(runtime, args, CaptureMode::FullMatch)
}

#[derive(Clone, Copy)]
enum CaptureMode {
    Search,
    Match,
    FullMatch,
}

fn capture(runtime: &mut dyn PyRuntime, args: CallArgs, mode: CaptureMode) -> PyResult {
    let name = match mode {
        CaptureMode::Search => "re.search",
        CaptureMode::Match => "re.match",
        CaptureMode::FullMatch => "re.fullmatch",
    };
    args.expect_positional(name, 2, 3)?;
    args.reject_unknown_keywords(name, &["flags"])?;
    let pattern = string_arg(runtime, &args.positional()[0], "regex pattern", MAX_PATTERN)?;
    let text = string_arg(runtime, &args.positional()[1], "regex input", MAX_INPUT)?;
    let flags = flags_arg(runtime, &args, 2, name)?;
    runtime.charge_cpu(u64::try_from(text.len()).unwrap_or(u64::MAX))?;
    let regex = build_regex(&pattern, flags)?;
    let captures = match mode {
        CaptureMode::Search => regex.captures(&text),
        CaptureMode::Match => regex.captures_at(&text, 0),
        CaptureMode::FullMatch => regex.captures(&text).filter(|captures| {
            captures
                .get(0)
                .is_some_and(|matched| matched.start() == 0 && matched.end() == text.len())
        }),
    };
    match captures {
        Some(captures) => allocate_match(runtime, &text, &captures),
        None => Ok(Value::None),
    }
}

fn findall(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    find(runtime, args, false)
}
fn finditer(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    find(runtime, args, true)
}

fn find(runtime: &mut dyn PyRuntime, args: CallArgs, return_matches: bool) -> PyResult {
    let name = if return_matches {
        "re.finditer"
    } else {
        "re.findall"
    };
    args.expect_positional(name, 2, 3)?;
    args.reject_unknown_keywords(name, &["flags"])?;
    let pattern = string_arg(runtime, &args.positional()[0], "regex pattern", MAX_PATTERN)?;
    let text = string_arg(runtime, &args.positional()[1], "regex input", MAX_INPUT)?;
    let flags = flags_arg(runtime, &args, 2, name)?;
    runtime.charge_cpu(u64::try_from(text.len()).unwrap_or(u64::MAX))?;
    let regex = build_regex(&pattern, flags)?;
    let capture_count = regex.captures_len().saturating_sub(1);
    let mut values = Vec::new();
    for captures in regex.captures_iter(&text) {
        runtime.charge_cpu(1)?;
        if values.len() == MAX_MATCHES {
            return Err(PyError::resource_error(
                "regex result exceeds the bounded match limit",
            ));
        }
        runtime.reserve_memory(64)?;
        let value = if return_matches {
            allocate_match(runtime, &text, &captures)?
        } else {
            match capture_count {
                0 => Value::String(captures[0].to_string()),
                1 => Value::String(
                    captures
                        .get(1)
                        .map_or_else(String::new, |matched| matched.as_str().to_string()),
                ),
                _ => runtime.new_tuple(
                    (1..=capture_count)
                        .map(|index| {
                            Value::String(
                                captures.get(index).map_or_else(String::new, |matched| {
                                    matched.as_str().to_string()
                                }),
                            )
                        })
                        .collect(),
                )?,
            }
        };
        values.push(value);
    }
    if return_matches {
        runtime.new_iterator(values)
    } else {
        runtime.new_list(values)
    }
}

fn sub(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("re.sub", 3, 5)?;
    args.reject_unknown_keywords("re.sub", &["flags"])?;
    let pattern = string_arg(runtime, &args.positional()[0], "regex pattern", MAX_PATTERN)?;
    let replacement = string_arg(runtime, &args.positional()[1], "replacement", MAX_INPUT)?;
    let text = string_arg(runtime, &args.positional()[2], "regex input", MAX_INPUT)?;
    let count = args
        .positional()
        .get(3)
        .cloned()
        .map(|value| value.cast::<PyIndex>(runtime).map(|value| value.0))
        .transpose()?
        .unwrap_or(0);
    if count < 0 {
        return Err(PyError::value_error("re.sub() count must be non-negative"));
    }
    let flags = flags_arg(runtime, &args, 4, "re.sub")?;
    let regex = build_regex(&pattern, flags)?;
    let replacement = normalize_replacement(&replacement)?;
    let potential_matches = text.len().saturating_add(1);
    let matches = if count == 0 {
        potential_matches
    } else {
        potential_matches.min(
            usize::try_from(count)
                .map_err(|_| PyError::overflow_error("regex count is too large"))?,
        )
    };
    let bound = text
        .len()
        .checked_add(
            matches
                .checked_mul(replacement.len())
                .ok_or_else(|| PyError::resource_error("regex substitution result is too large"))?,
        )
        .and_then(|bytes| bytes.checked_add(64))
        .ok_or_else(|| PyError::resource_error("regex substitution result is too large"))?;
    runtime.reserve_memory(bound)?;
    runtime.charge_cpu(
        u64::try_from(text.len().saturating_add(replacement.len())).unwrap_or(u64::MAX),
    )?;
    let rendered = if count == 0 {
        regex.replace_all(&text, replacement.as_str()).into_owned()
    } else {
        regex
            .replacen(&text, count as usize, replacement.as_str())
            .into_owned()
    };
    Ok(Value::String(rendered))
}

fn escape(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("re.escape", 1, 1)?;
    args.reject_keywords("re.escape")?;
    let text = string_arg(runtime, &args.positional()[0], "escape input", MAX_INPUT)?;
    runtime.charge_cpu(u64::try_from(text.len()).unwrap_or(u64::MAX))?;
    Ok(Value::String(python_escape(&text)))
}

fn flags_arg(
    runtime: &dyn PyRuntime,
    args: &CallArgs,
    positional_index: usize,
    function: &str,
) -> PyResult<u32> {
    let positional = args.positional().get(positional_index).cloned();
    let keyword = args.keyword(function, "flags")?.cloned();
    if positional.is_some() && keyword.is_some() {
        return Err(PyError::type_error("regex flags passed more than once"));
    }
    let value = positional
        .or(keyword)
        .map(|value| value.cast::<PyIndex>(runtime).map(|value| value.0))
        .transpose()?
        .unwrap_or(0);
    let flags =
        u32::try_from(value).map_err(|_| PyError::value_error("regex flags are out of range"))?;
    if flags & !(IGNORECASE | MULTILINE) != 0 {
        return Err(PyError::value_error(
            "regex flags are limited to IGNORECASE and MULTILINE",
        ));
    }
    Ok(flags)
}

fn string_arg(
    runtime: &dyn PyRuntime,
    value: &Value,
    label: &str,
    maximum: usize,
) -> PyResult<String> {
    let PyString(value) = value.clone().cast(runtime)?;
    if value.len() > maximum {
        Err(PyError::value_error(format!(
            "{label} exceeds the bounded regex input limit"
        )))
    } else {
        Ok(value)
    }
}

pub(in crate::python) fn build_regex(pattern: &str, flags: u32) -> PyResult<Regex> {
    if pattern.len() > MAX_PATTERN {
        return Err(PyError::value_error(
            "regex pattern exceeds the bounded pattern limit",
        ));
    }
    RegexBuilder::new(pattern)
        .case_insensitive(flags & IGNORECASE != 0)
        .multi_line(flags & MULTILINE != 0)
        .build()
        .map_err(|error| PyError::value_error(format!("unsupported regex pattern: {error}")))
}

fn allocate_match(
    runtime: &mut dyn PyRuntime,
    text: &str,
    captures: &regex::Captures<'_>,
) -> PyResult {
    let whole = captures
        .get(0)
        .ok_or_else(|| PyError::runtime_error("regex engine returned no whole match"))?;
    let groups = captures
        .iter()
        .map(|capture| capture.map(|matched| matched.as_str().to_string()))
        .collect();
    let start = text[..whole.start()].chars().count();
    let end = text[..whole.end()].chars().count();
    runtime.new_match(whole.as_str().to_string(), groups, start, end)
}

pub(in crate::python) fn normalize_replacement(replacement: &str) -> PyResult<String> {
    let mut output = String::with_capacity(replacement.len());
    let mut chars = replacement.chars();
    while let Some(character) = chars.next() {
        if character != '\\' {
            output.push(character);
            continue;
        }
        let Some(next) = chars.next() else {
            return Err(PyError::value_error("bad escape in replacement"));
        };
        if next.is_ascii_digit() {
            output.push('$');
            output.push(next);
        } else if next == '\\' {
            output.push('\\');
            output.push('\\');
        } else {
            output.push('\\');
            output.push(next);
        }
    }
    Ok(output)
}

fn python_escape(text: &str) -> String {
    text.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' || ",:;!/".contains(character)
            {
                character.to_string()
            } else {
                format!("\\{character}")
            }
        })
        .collect()
}
