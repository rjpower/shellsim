//! Bounded, capability-free regular-expression core (`_re`) over Rust's linear-time engine.
//!
//! Python pattern syntax is translated to the `regex` crate's dialect before compilation:
//! ASCII-mode classes, `\Z`, literal braces, `(?#...)` comments and the `a`/`u` inline flags.
//! Constructs the engine lacks (backreferences, lookaround) and the LOCALE and DEBUG flags fail
//! explicitly. Compiled patterns are memoized host-side in a small cache keyed by source and
//! flags, which changes host time only.
//!
//! Inputs, matches and substitution output are metered before host allocation. A match keeps
//! the subject string and its pattern by reference, so `finditer` over a large string does not
//! copy the string per match.

use std::cell::RefCell;

use regex::{Captures, Regex, RegexBuilder};

use super::super::heap::{NativeObject, Ref};
use super::super::native::PyValue as Value;
use super::super::native::{
    CallArgs, FunctionDef, GetterDef, MethodDef, ModuleDef, NativeTypeDef, OwnedPyString,
    PyConstant, PyError, PyIndex, PyMatch, PyMatchData, PyRegex, PyResult, PyRuntime, PyValue,
    PyValueCast, ValueDef,
};
use super::super::object_model::{BuiltinType, TypeId};
use super::super::protocol::quote_string;

pub(super) const IGNORECASE: u32 = 2;
pub(super) const LOCALE: u32 = 4;
pub(super) const MULTILINE: u32 = 8;
pub(super) const DOTALL: u32 = 16;
pub(super) const UNICODE: u32 = 32;
pub(super) const VERBOSE: u32 = 64;
pub(super) const DEBUG: u32 = 128;
pub(super) const ASCII: u32 = 256;
/// The removed TEMPLATE flag, which CPython still accepts as a no-op bit.
const TEMPLATE: u32 = 1;
const SUPPORTED_FLAGS: u32 = TEMPLATE | IGNORECASE | MULTILINE | DOTALL | UNICODE | VERBOSE | ASCII;
const MAX_PATTERN: usize = 4096;
const MAX_INPUT: usize = 1_048_576;
const MAX_MATCHES: usize = 100_000;
/// Compiled patterns retained host-side; each is bounded by `COMPILED_SIZE_LIMIT`.
const CACHE_SIZE: usize = 16;
const COMPILED_SIZE_LIMIT: usize = 4 * 1024 * 1024;

pub(in crate::python) static PATTERN_TYPE: NativeTypeDef = NativeTypeDef {
    name: "re.Pattern",
    methods: &[
        pattern_method("search", pattern_search),
        pattern_method("match", pattern_match),
        pattern_method("fullmatch", pattern_fullmatch),
        pattern_method("findall", pattern_findall),
        pattern_method("finditer", pattern_finditer),
        pattern_method("sub", pattern_sub),
        pattern_method("subn", pattern_subn),
        pattern_method("split", pattern_split),
    ],
    getters: &[
        pattern_getter("pattern", pattern_pattern),
        pattern_getter("flags", pattern_flags),
        pattern_getter("groups", pattern_groups),
        pattern_getter("groupindex", pattern_groupindex),
    ],
};

pub(in crate::python) static MATCH_TYPE: NativeTypeDef = NativeTypeDef {
    name: "re.Match",
    methods: &[
        match_method("group", match_group),
        match_method("__getitem__", match_getitem),
        match_method("groups", match_groups),
        match_method("groupdict", match_groupdict),
        match_method("start", match_start),
        match_method("end", match_end),
        match_method("span", match_span),
        match_method("expand", match_expand),
    ],
    getters: &[
        match_getter("string", match_string),
        match_getter("re", match_re),
        match_getter("pos", match_pos),
        match_getter("endpos", match_endpos),
        match_getter("lastindex", match_lastindex),
        match_getter("lastgroup", match_lastgroup),
        match_getter("regs", match_regs),
    ],
};

type Method = for<'s> fn(&mut dyn PyRuntime<'s>, PyValue<'s>, CallArgs<'s>) -> PyResult<'s>;
type Getter = for<'s> fn(&mut dyn PyRuntime<'s>, PyValue<'s>) -> PyResult<'s>;

const fn pattern_method(name: &'static str, call: Method) -> MethodDef {
    MethodDef {
        type_name: "re.Pattern",
        name,
        call,
    }
}

const fn pattern_getter(name: &'static str, get: Getter) -> GetterDef {
    GetterDef {
        owner: "re.Pattern",
        name,
        get,
    }
}

const fn match_method(name: &'static str, call: Method) -> MethodDef {
    MethodDef {
        type_name: "re.Match",
        name,
        call,
    }
}

const fn match_getter(name: &'static str, get: Getter) -> GetterDef {
    GetterDef {
        owner: "re.Match",
        name,
        get,
    }
}

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "_re",
    functions: &[
        FunctionDef {
            module: "re",
            name: "compile",
            call: compile,
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
            name: "LOCALE",
            value: PyConstant::Int(LOCALE as i64),
        },
        ValueDef::Constant {
            name: "MULTILINE",
            value: PyConstant::Int(MULTILINE as i64),
        },
        ValueDef::Constant {
            name: "DOTALL",
            value: PyConstant::Int(DOTALL as i64),
        },
        ValueDef::Constant {
            name: "UNICODE",
            value: PyConstant::Int(UNICODE as i64),
        },
        ValueDef::Constant {
            name: "VERBOSE",
            value: PyConstant::Int(VERBOSE as i64),
        },
        ValueDef::Constant {
            name: "DEBUG",
            value: PyConstant::Int(DEBUG as i64),
        },
        ValueDef::Constant {
            name: "ASCII",
            value: PyConstant::Int(ASCII as i64),
        },
    ],
};

/// A compiled regular expression as the heap stores it. The pattern is compiled at the operation
/// boundary so regex execution never gets a host capability; keeping the source and flags here
/// also makes the object cheap to copy and deterministic to inspect.
#[derive(Debug)]
pub(crate) struct RegexObject {
    pub pattern: String,
    pub flags: u32,
}

impl NativeObject for RegexObject {
    fn python_type(&self) -> TypeId {
        BuiltinType::Regex.id()
    }

    fn modeled_bytes(&self) -> Result<u64, String> {
        u64::try_from(self.pattern.len()).map_err(|_| "modeled object size overflow".into())
    }

    fn dup(&self) -> Box<dyn NativeObject> {
        Box::new(Self {
            pattern: self.pattern.clone(),
            flags: self.flags,
        })
    }

    fn repr(&self, _: &mut dyn FnMut(&Ref) -> Result<String, String>) -> Result<String, String> {
        let flags = flag_repr(self.flags);
        let pattern = quote_string(&self.pattern);
        Ok(if flags.is_empty() {
            format!("re.compile({pattern})")
        } else {
            format!("re.compile({pattern}, {flags})")
        })
    }
}

/// A match result as the heap stores it. The subject string and the `re.Pattern` are held by
/// reference so a match costs its own group text, not a copy of the string it was found in.
/// `spans` are character offsets into the subject per group, `None` for a group that did not
/// participate.
#[derive(Debug)]
pub(crate) struct MatchObject {
    pub subject: Ref,
    pub regex: Ref,
    pub text: String,
    pub groups: Vec<Option<String>>,
    pub group_names: Vec<Option<String>>,
    pub spans: Vec<Option<(usize, usize)>>,
    pub pos: usize,
    pub endpos: usize,
}

impl NativeObject for MatchObject {
    fn python_type(&self) -> TypeId {
        BuiltinType::Match.id()
    }

    fn modeled_bytes(&self) -> Result<u64, String> {
        let text = |values: &[Option<String>]| {
            values
                .iter()
                .map(|value| value.as_ref().map_or(0, String::len))
                .sum::<usize>()
        };
        self.text
            .len()
            .checked_add(self.spans.len().saturating_mul(16))
            .and_then(|size| size.checked_add(text(&self.groups)))
            .and_then(|size| size.checked_add(text(&self.group_names)))
            .and_then(|size| u64::try_from(size).ok())
            .ok_or_else(|| "modeled object size overflow".into())
    }

    fn visit_refs(&mut self, visit: &mut dyn FnMut(&mut Ref)) {
        visit(&mut self.subject);
        visit(&mut self.regex);
    }

    fn dup(&self) -> Box<dyn NativeObject> {
        Box::new(Self {
            subject: self.subject.dup(),
            regex: self.regex.dup(),
            text: self.text.clone(),
            groups: self.groups.clone(),
            group_names: self.group_names.clone(),
            spans: self.spans.clone(),
            pos: self.pos,
            endpos: self.endpos,
        })
    }

    fn repr(&self, _: &mut dyn FnMut(&Ref) -> Result<String, String>) -> Result<String, String> {
        let (start, end) = self.spans.first().copied().flatten().unwrap_or((0, 0));
        Ok(format!(
            "<re.Match object; span=({start}, {end}), match={}>",
            quote_string(&self.text)
        ))
    }
}

/// `re.compile('...', re.IGNORECASE|re.VERBOSE)`: the flag part of a pattern's repr, empty
/// when only the implied UNICODE flag is set.
fn flag_repr(flags: u32) -> String {
    const NAMES: [(u32, &str); 6] = [
        (ASCII, "re.ASCII"),
        (IGNORECASE, "re.IGNORECASE"),
        (MULTILINE, "re.MULTILINE"),
        (DOTALL, "re.DOTALL"),
        (VERBOSE, "re.VERBOSE"),
        (DEBUG, "re.DEBUG"),
    ];
    NAMES
        .iter()
        .filter(|(flag, _)| flags & flag != 0)
        .map(|(_, name)| *name)
        .collect::<Vec<_>>()
        .join("|")
}

/// `_re.compile(pattern, flags)`: validate and compile a pattern string into a `re.Pattern`.
fn compile<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("re.compile", 1, 2)?;
    args.reject_unknown_keywords("re.compile", &["flags"])?;
    let pattern = string_arg(runtime, &args.positional()[0], "regex pattern", MAX_PATTERN)?;
    let flags = flags_arg(runtime, &args, 1, "re.compile")?;
    build_regex(&pattern, flags)?;
    runtime.new_regex(pattern, flags)
}

fn escape<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("re.escape", 1, 1)?;
    args.reject_keywords("re.escape")?;
    let text = string_arg(runtime, &args.positional()[0], "escape input", MAX_INPUT)?;
    runtime.charge_cpu(u64::try_from(text.len()).unwrap_or(u64::MAX))?;
    runtime.new_string(python_escape(&text))
}

/// A `re.Pattern` receiver with its compiled engine.
struct Compiled<'s> {
    value: Value<'s>,
    pattern: String,
    flags: u32,
    regex: Regex,
}

fn compiled<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: Value<'s>,
) -> PyResult<'s, Compiled<'s>> {
    let (pattern, flags) = runtime.regex_parts(receiver.cast::<PyRegex<'s>>(runtime)?)?;
    let regex = build_regex(&pattern, flags)?;
    Ok(Compiled {
        value: receiver,
        pattern,
        flags,
        regex,
    })
}

fn pattern_pattern<'s>(runtime: &mut dyn PyRuntime<'s>, receiver: PyValue<'s>) -> PyResult<'s> {
    let (pattern, _) = runtime.regex_parts(receiver.cast::<PyRegex<'s>>(runtime)?)?;
    runtime.new_string(pattern)
}

/// `Pattern.flags` includes the implied UNICODE flag unless ASCII was requested, as CPython
/// reports for str patterns.
fn pattern_flags<'s>(runtime: &mut dyn PyRuntime<'s>, receiver: PyValue<'s>) -> PyResult<'s> {
    let (_, flags) = runtime.regex_parts(receiver.cast::<PyRegex<'s>>(runtime)?)?;
    let implied = if flags & ASCII != 0 { 0 } else { UNICODE };
    Ok(Value::Int(i64::from(flags | implied)))
}

fn pattern_groups<'s>(runtime: &mut dyn PyRuntime<'s>, receiver: PyValue<'s>) -> PyResult<'s> {
    let compiled = compiled(runtime, receiver)?;
    int_value(compiled.regex.captures_len().saturating_sub(1))
}

fn pattern_groupindex<'s>(runtime: &mut dyn PyRuntime<'s>, receiver: PyValue<'s>) -> PyResult<'s> {
    let compiled = compiled(runtime, receiver)?;
    let mut items = Vec::new();
    for (index, name) in compiled.regex.capture_names().enumerate() {
        if let Some(name) = name {
            items.push((runtime.new_string(name.to_string())?, int_value(index)?));
        }
    }
    runtime.new_dict(items)
}

#[derive(Clone, Copy)]
enum CaptureMode {
    Search,
    Match,
    FullMatch,
}

fn pattern_search<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    pattern_capture(runtime, receiver, args, CaptureMode::Search)
}

fn pattern_match<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    pattern_capture(runtime, receiver, args, CaptureMode::Match)
}

fn pattern_fullmatch<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    pattern_capture(runtime, receiver, args, CaptureMode::FullMatch)
}

/// The `string, pos=0, endpos=len(string)` arguments shared by the capture and find methods.
fn subject_args<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    args: &CallArgs<'s>,
    name: &str,
) -> PyResult<'s, (Value<'s>, String, Window)> {
    args.expect_positional(name, 1, 3)?;
    args.reject_unknown_keywords(name, &["pos", "endpos"])?;
    let subject = args.positional()[0];
    let text = string_arg(runtime, &subject, "regex input", MAX_INPUT)?;
    let pos = args
        .positional()
        .get(1)
        .or(args.keyword(name, "pos")?)
        .copied();
    let endpos = args
        .positional()
        .get(2)
        .or(args.keyword(name, "endpos")?)
        .copied();
    runtime.charge_cpu(u64::try_from(text.len()).unwrap_or(u64::MAX))?;
    let window = window(runtime, &text, pos, endpos)?;
    Ok((subject, text, window))
}

fn pattern_capture<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
    args: CallArgs<'s>,
    mode: CaptureMode,
) -> PyResult<'s> {
    let name = match mode {
        CaptureMode::Search => "re.Pattern.search",
        CaptureMode::Match => "re.Pattern.match",
        CaptureMode::FullMatch => "re.Pattern.fullmatch",
    };
    let compiled = compiled(runtime, receiver)?;
    let (subject, text, window) = subject_args(runtime, &args, name)?;
    let haystack = &text[..window.end_byte];
    let captures = match mode {
        CaptureMode::Search => compiled.regex.captures_at(haystack, window.start_byte),
        CaptureMode::Match => compiled
            .regex
            .captures_at(haystack, window.start_byte)
            .filter(|captures| {
                captures
                    .get(0)
                    .is_some_and(|m| m.start() == window.start_byte)
            }),
        CaptureMode::FullMatch => {
            // Leftmost-first search may pick an alternative that stops short of the end even
            // when another reaches it, so anchoring the pattern itself makes the engine look
            // for an alternative that spans the whole window.
            let separator = if compiled.flags & VERBOSE != 0 {
                "\n"
            } else {
                ""
            };
            let anchored = build_regex(
                &format!("(?:{}{separator})\\z", compiled.pattern),
                compiled.flags,
            )?;
            anchored
                .captures_at(haystack, window.start_byte)
                .filter(|captures| {
                    captures
                        .get(0)
                        .is_some_and(|m| m.start() == window.start_byte)
                })
        }
    };
    let Some(captures) = captures else {
        return Ok(Value::None);
    };
    let context = MatchContext {
        regex: compiled.value,
        engine: &compiled.regex,
        subject,
        text: &text,
        offset: 0,
        window: &window,
    };
    let mut cursor = CharCursor::new(window.start_byte, window.pos);
    allocate_match(runtime, &context, &captures, &mut cursor)
}

fn pattern_findall<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    pattern_find(runtime, receiver, args, false)
}

fn pattern_finditer<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    pattern_find(runtime, receiver, args, true)
}

fn pattern_find<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
    args: CallArgs<'s>,
    return_matches: bool,
) -> PyResult<'s> {
    let name = if return_matches {
        "re.Pattern.finditer"
    } else {
        "re.Pattern.findall"
    };
    let compiled = compiled(runtime, receiver)?;
    let (subject, text, window) = subject_args(runtime, &args, name)?;
    let haystack = &text[window.start_byte..window.end_byte];
    let context = MatchContext {
        regex: compiled.value,
        engine: &compiled.regex,
        subject,
        text: &text,
        offset: window.start_byte,
        window: &window,
    };
    let mut cursor = CharCursor::new(window.start_byte, window.pos);
    let capture_count = compiled.regex.captures_len().saturating_sub(1);
    let mut values = Vec::new();
    for captures in compiled.regex.captures_iter(haystack) {
        runtime.charge_cpu(1)?;
        if values.len() == MAX_MATCHES {
            return Err(PyError::resource_error(
                "regex result exceeds the bounded match limit",
            ));
        }
        runtime.reserve_memory(64)?;
        let value = if return_matches {
            allocate_match(runtime, &context, &captures, &mut cursor)?
        } else {
            let group = |index: usize| {
                captures
                    .get(index)
                    .map_or_else(String::new, |matched| matched.as_str().to_string())
            };
            match capture_count {
                0 => runtime.new_string(group(0))?,
                1 => runtime.new_string(group(1))?,
                _ => {
                    let mut groups = Vec::with_capacity(capture_count);
                    for index in 1..=capture_count {
                        groups.push(runtime.new_string(group(index))?);
                    }
                    runtime.new_tuple(groups)?
                }
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

fn pattern_sub<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    let (rendered, _) = pattern_substitute(runtime, receiver, args, "re.Pattern.sub")?;
    runtime.new_string(rendered)
}

fn pattern_subn<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    let (rendered, count) = pattern_substitute(runtime, receiver, args, "re.Pattern.subn")?;
    let rendered = runtime.new_string(rendered)?;
    let count = int_value(count)?;
    runtime.new_tuple(vec![rendered, count])
}

/// `Pattern.sub(repl, string, count=0)` and `subn`: the replaced text and the number of
/// replacements. `repl` is a template string or a callable receiving each match.
fn pattern_substitute<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
    args: CallArgs<'s>,
    name: &str,
) -> PyResult<'s, (String, usize)> {
    args.expect_positional(name, 2, 3)?;
    args.reject_unknown_keywords(name, &["count"])?;
    let compiled = compiled(runtime, receiver)?;
    let replacement = args.positional()[0];
    let subject = args.positional()[1];
    let text = string_arg(runtime, &subject, "regex input", MAX_INPUT)?;
    let count = args
        .positional()
        .get(2)
        .or(args.keyword(name, "count")?)
        .copied();
    let count = count_arg(runtime, count, "count")?;
    let replacement = match runtime.string_value(&replacement)? {
        Some(template) => {
            if template.len() > MAX_INPUT {
                return Err(PyError::value_error(
                    "replacement exceeds the bounded regex input limit",
                ));
            }
            Replacement::Template(parse_template(&template, &compiled.regex)?)
        }
        None if runtime.is_callable(&replacement)? => Replacement::Callable(replacement),
        None => {
            return Err(PyError::type_error(
                "re.sub() replacement must be a string or a callable",
            ))
        }
    };
    runtime.charge_cpu(u64::try_from(text.len()).unwrap_or(u64::MAX))?;
    runtime.reserve_memory(text.len().saturating_add(64))?;
    let window = window(runtime, &text, None, None)?;
    let context = MatchContext {
        regex: compiled.value,
        engine: &compiled.regex,
        subject,
        text: &text,
        offset: 0,
        window: &window,
    };
    let mut cursor = CharCursor::new(0, 0);
    let mut output = String::with_capacity(text.len());
    let mut last = 0;
    let mut replaced = 0;
    for captures in compiled.regex.captures_iter(&text) {
        if count != 0 && replaced == count {
            break;
        }
        runtime.charge_cpu(1)?;
        if replaced == MAX_MATCHES {
            return Err(PyError::resource_error(
                "regex substitution exceeds the bounded match limit",
            ));
        }
        let whole = captures
            .get(0)
            .ok_or_else(|| PyError::runtime_error("regex engine returned no whole match"))?;
        output.push_str(&text[last..whole.start()]);
        let before = output.len();
        match &replacement {
            Replacement::Template(pieces) => {
                expand_pieces(pieces, &mut output, |index| {
                    captures.get(index).map(|matched| matched.as_str())
                });
            }
            Replacement::Callable(function) => {
                let matched = allocate_match(runtime, &context, &captures, &mut cursor)?;
                let result =
                    runtime.call_value(*function, CallArgs::new(vec![matched], Vec::new()))?;
                let piece = runtime.string_value(&result)?.ok_or_else(|| {
                    PyError::type_error("re.sub() replacement function must return a str")
                })?;
                output.push_str(&piece);
            }
        }
        runtime.reserve_memory(output.len() - before)?;
        last = whole.end();
        replaced += 1;
    }
    output.push_str(&text[last..]);
    Ok((output, replaced))
}

/// `Pattern.split(string, maxsplit=0)`: the text between matches, with each capturing group's
/// text (or `None`) inserted between pieces as CPython does.
fn pattern_split<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    const NAME: &str = "re.Pattern.split";
    args.expect_positional(NAME, 1, 2)?;
    args.reject_unknown_keywords(NAME, &["maxsplit"])?;
    let compiled = compiled(runtime, receiver)?;
    let text = string_arg(runtime, &args.positional()[0], "regex input", MAX_INPUT)?;
    let maxsplit = args
        .positional()
        .get(1)
        .or(args.keyword(NAME, "maxsplit")?)
        .copied();
    let maxsplit = count_arg(runtime, maxsplit, "maxsplit")?;
    runtime.charge_cpu(u64::try_from(text.len()).unwrap_or(u64::MAX))?;
    let mut pieces = Vec::new();
    let mut last = 0;
    for (splits, captures) in compiled.regex.captures_iter(&text).enumerate() {
        if maxsplit != 0 && splits == maxsplit {
            break;
        }
        runtime.charge_cpu(1)?;
        if splits == MAX_MATCHES {
            return Err(PyError::resource_error(
                "regex split exceeds the bounded match limit",
            ));
        }
        let whole = captures
            .get(0)
            .ok_or_else(|| PyError::runtime_error("regex engine returned no whole match"))?;
        runtime.reserve_memory(64)?;
        pieces.push(runtime.new_string(text[last..whole.start()].to_string())?);
        for index in 1..captures.len() {
            pieces.push(match captures.get(index) {
                Some(group) => runtime.new_string(group.as_str().to_string())?,
                None => Value::None,
            });
        }
        last = whole.end();
    }
    pieces.push(runtime.new_string(text[last..].to_string())?);
    runtime.new_list(pieces)
}

enum Replacement<'s> {
    Template(Vec<Piece>),
    Callable(Value<'s>),
}

/// One element of a parsed replacement template.
enum Piece {
    Literal(String),
    Group(usize),
}

/// Parse a `re.sub` template: `\1`, `\g<1>` and `\g<name>` group references, the C escapes
/// CPython accepts, and literal text. Unknown alphabetic escapes and unknown groups are errors.
fn parse_template<'s>(template: &str, regex: &Regex) -> PyResult<'s, Vec<Piece>> {
    let group_count = regex.captures_len();
    let mut pieces = Vec::new();
    let mut literal = String::new();
    let mut chars = template.chars().peekable();
    let flush = |literal: &mut String, pieces: &mut Vec<Piece>| {
        if !literal.is_empty() {
            pieces.push(Piece::Literal(std::mem::take(literal)));
        }
    };
    let group = |index: usize| -> PyResult<'s, Piece> {
        if index < group_count {
            Ok(Piece::Group(index))
        } else {
            Err(PyError::exception(
                "IndexError",
                format!("invalid group reference {index}"),
            ))
        }
    };
    while let Some(character) = chars.next() {
        if character != '\\' {
            literal.push(character);
            continue;
        }
        let Some(next) = chars.next() else {
            return Err(PyError::value_error("bad escape (end of pattern)"));
        };
        match next {
            'n' => literal.push('\n'),
            't' => literal.push('\t'),
            'r' => literal.push('\r'),
            'f' => literal.push('\x0c'),
            'v' => literal.push('\x0b'),
            'a' => literal.push('\x07'),
            'b' => literal.push('\x08'),
            '\\' => literal.push('\\'),
            '0'..='9' => {
                let mut index = next.to_digit(10).unwrap_or(0) as usize;
                if let Some(digit) = chars.peek().and_then(|c| c.to_digit(10)) {
                    index = index * 10 + digit as usize;
                    chars.next();
                }
                flush(&mut literal, &mut pieces);
                pieces.push(group(index)?);
            }
            'g' => {
                if chars.next() != Some('<') {
                    return Err(PyError::value_error("missing <"));
                }
                let mut name = String::new();
                loop {
                    match chars.next() {
                        Some('>') => break,
                        Some(c) => name.push(c),
                        None => return Err(PyError::value_error("missing >, unterminated name")),
                    }
                }
                if name.is_empty() {
                    return Err(PyError::value_error("missing group name"));
                }
                flush(&mut literal, &mut pieces);
                let index = if name.chars().all(|c| c.is_ascii_digit()) {
                    name.parse::<usize>()
                        .map_err(|_| PyError::exception("IndexError", "invalid group reference"))?
                } else {
                    regex
                        .capture_names()
                        .position(|candidate| candidate == Some(name.as_str()))
                        .ok_or_else(|| {
                            PyError::exception("IndexError", format!("unknown group name '{name}'"))
                        })?
                };
                pieces.push(group(index)?);
            }
            c if c.is_ascii_alphabetic() => {
                return Err(PyError::value_error(format!("bad escape \\{c}")));
            }
            other => {
                literal.push('\\');
                literal.push(other);
            }
        }
    }
    flush(&mut literal, &mut pieces);
    Ok(pieces)
}

/// Append the expansion of `pieces` to `output`; an unmatched group expands to nothing.
fn expand_pieces<'t>(
    pieces: &[Piece],
    output: &mut String,
    group: impl Fn(usize) -> Option<&'t str>,
) {
    for piece in pieces {
        match piece {
            Piece::Literal(text) => output.push_str(text),
            Piece::Group(index) => output.push_str(group(*index).unwrap_or("")),
        }
    }
}

/// The searched region of a subject string, in bytes for the engine and in characters for the
/// Python-visible `pos` and `endpos`.
struct Window {
    start_byte: usize,
    end_byte: usize,
    pos: usize,
    endpos: usize,
}

fn window<'s>(
    runtime: &dyn PyRuntime<'s>,
    text: &str,
    pos: Option<Value<'s>>,
    endpos: Option<Value<'s>>,
) -> PyResult<'s, Window> {
    let length = text.chars().count();
    let clamp = |value: Option<Value<'s>>| -> PyResult<'s, Option<usize>> {
        let Some(value) = value.filter(|value| !value.is_none()) else {
            return Ok(None);
        };
        let index = value.cast::<PyIndex>(runtime)?.0;
        Ok(Some(usize::try_from(index).unwrap_or(0).min(length)))
    };
    let pos = clamp(pos)?.unwrap_or(0);
    let endpos = clamp(endpos)?.unwrap_or(length).max(pos);
    let byte_at = |target: usize| {
        text.char_indices()
            .nth(target)
            .map_or(text.len(), |(byte, _)| byte)
    };
    let start_byte = if pos == 0 { 0 } else { byte_at(pos) };
    let end_byte = if endpos == length {
        text.len()
    } else {
        byte_at(endpos)
    };
    Ok(Window {
        start_byte,
        end_byte,
        pos,
        endpos,
    })
}

/// Everything a match object records about where it came from.
struct MatchContext<'c, 's> {
    regex: Value<'s>,
    engine: &'c Regex,
    subject: Value<'s>,
    /// The whole subject string.
    text: &'c str,
    /// Byte offset of the haystack the engine searched within `text`.
    offset: usize,
    window: &'c Window,
}

/// Converts increasing byte offsets to character offsets in one forward pass.
struct CharCursor {
    byte: usize,
    chars: usize,
}

impl CharCursor {
    fn new(byte: usize, chars: usize) -> Self {
        Self { byte, chars }
    }

    fn at(&mut self, text: &str, byte: usize) -> usize {
        if byte < self.byte {
            self.byte = 0;
            self.chars = 0;
        }
        self.chars += text[self.byte..byte].chars().count();
        self.byte = byte;
        self.chars
    }
}

fn allocate_match<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    context: &MatchContext<'_, 's>,
    captures: &Captures<'_>,
    cursor: &mut CharCursor,
) -> PyResult<'s> {
    let whole = captures
        .get(0)
        .ok_or_else(|| PyError::runtime_error("regex engine returned no whole match"))?;
    let text = context.text;
    let whole_start = context.offset + whole.start();
    let base = cursor.at(text, whole_start);
    let mut groups = Vec::with_capacity(captures.len());
    let mut spans = Vec::with_capacity(captures.len());
    for group in captures.iter() {
        match group {
            Some(matched) => {
                let start_byte = context.offset + matched.start();
                let end_byte = context.offset + matched.end();
                let start = base + text[whole_start..start_byte].chars().count();
                let end = start + text[start_byte..end_byte].chars().count();
                groups.push(Some(matched.as_str().to_string()));
                spans.push(Some((start, end)));
            }
            None => {
                groups.push(None);
                spans.push(None);
            }
        }
    }
    let group_names = context
        .engine
        .capture_names()
        .map(|name| name.map(str::to_string))
        .collect();
    runtime.new_match(PyMatchData {
        subject: context.subject,
        regex: context.regex,
        text: whole.as_str().to_string(),
        groups,
        group_names,
        spans,
        pos: context.window.pos,
        endpos: context.window.endpos,
    })
}

fn match_data<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
) -> PyResult<'s, PyMatchData<'s>> {
    runtime.match_data(receiver.cast::<PyMatch<'s>>(runtime)?)
}

/// Resolve a group argument (an index or a name) to its position.
fn group_index<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    data: &PyMatchData<'s>,
    value: &Value<'s>,
) -> PyResult<'s, usize> {
    let no_such_group = || PyError::exception("IndexError", "no such group");
    if let Some(name) = runtime.string_value(value)? {
        return data
            .group_names
            .iter()
            .position(|candidate| candidate.as_deref() == Some(name.as_str()))
            .ok_or_else(no_such_group);
    }
    let index = (*value).cast::<PyIndex>(runtime)?.0;
    usize::try_from(index)
        .ok()
        .filter(|index| *index < data.groups.len())
        .ok_or_else(no_such_group)
}

fn group_value<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    data: &PyMatchData<'s>,
    index: usize,
    default: Value<'s>,
) -> PyResult<'s> {
    match data.groups.get(index) {
        Some(Some(text)) => runtime.new_string(text.clone()),
        _ => Ok(default),
    }
}

/// `Match.group(*groups)`: one group's text, or a tuple when several are asked for.
fn match_group<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    args.reject_keywords("re.Match.group")?;
    let data = match_data(runtime, receiver)?;
    match args.positional() {
        [] => group_value(runtime, &data, 0, Value::None),
        [only] => {
            let index = group_index(runtime, &data, only)?;
            group_value(runtime, &data, index, Value::None)
        }
        many => {
            let mut values = Vec::with_capacity(many.len());
            for value in many {
                let index = group_index(runtime, &data, value)?;
                values.push(group_value(runtime, &data, index, Value::None)?);
            }
            runtime.new_tuple(values)
        }
    }
}

fn match_getitem<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    args.expect_positional("re.Match.__getitem__", 1, 1)?;
    args.reject_keywords("re.Match.__getitem__")?;
    let data = match_data(runtime, receiver)?;
    let index = group_index(runtime, &data, &args.positional()[0])?;
    group_value(runtime, &data, index, Value::None)
}

fn default_arg<'s>(args: &CallArgs<'s>, name: &str) -> PyResult<'s, Value<'s>> {
    args.expect_positional(name, 0, 1)?;
    args.reject_unknown_keywords(name, &["default"])?;
    Ok(args
        .positional()
        .first()
        .or(args.keyword(name, "default")?)
        .copied()
        .unwrap_or(Value::None))
}

fn match_groups<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    let default = default_arg(&args, "re.Match.groups")?;
    let data = match_data(runtime, receiver)?;
    let mut groups = Vec::with_capacity(data.groups.len().saturating_sub(1));
    for index in 1..data.groups.len() {
        groups.push(group_value(runtime, &data, index, default)?);
    }
    runtime.new_tuple(groups)
}

fn match_groupdict<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    let default = default_arg(&args, "re.Match.groupdict")?;
    let data = match_data(runtime, receiver)?;
    let mut items = Vec::new();
    for (index, name) in data.group_names.iter().enumerate() {
        if let Some(name) = name {
            let key = runtime.new_string(name.clone())?;
            items.push((key, group_value(runtime, &data, index, default)?));
        }
    }
    runtime.new_dict(items)
}

fn match_start<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    let (start, _) = match_span_of(runtime, receiver, args, "re.Match.start")?;
    Ok(Value::Int(start))
}

fn match_end<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    let (_, end) = match_span_of(runtime, receiver, args, "re.Match.end")?;
    Ok(Value::Int(end))
}

fn match_span<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    let (start, end) = match_span_of(runtime, receiver, args, "re.Match.span")?;
    runtime.new_tuple(vec![Value::Int(start), Value::Int(end)])
}

/// The `(start, end)` of the requested group, `(-1, -1)` when it did not participate.
fn match_span_of<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
    args: CallArgs<'s>,
    name: &str,
) -> PyResult<'s, (i64, i64)> {
    args.expect_positional(name, 0, 1)?;
    args.reject_keywords(name)?;
    let data = match_data(runtime, receiver)?;
    let index = match args.positional().first() {
        Some(value) => group_index(runtime, &data, value)?,
        None => 0,
    };
    span_pair(data.spans.get(index).copied().flatten())
}

fn span_pair<'s>(span: Option<(usize, usize)>) -> PyResult<'s, (i64, i64)> {
    let Some((start, end)) = span else {
        return Ok((-1, -1));
    };
    let convert = |value: usize| {
        i64::try_from(value)
            .map_err(|_| PyError::overflow_error("match position is outside Python int range"))
    };
    Ok((convert(start)?, convert(end)?))
}

fn match_expand<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    receiver: PyValue<'s>,
    args: CallArgs<'s>,
) -> PyResult<'s> {
    args.expect_positional("re.Match.expand", 1, 1)?;
    args.reject_keywords("re.Match.expand")?;
    let data = match_data(runtime, receiver)?;
    let template = string_arg(runtime, &args.positional()[0], "replacement", MAX_INPUT)?;
    let compiled = compiled(runtime, data.regex)?;
    let pieces = parse_template(&template, &compiled.regex)?;
    let mut output = String::new();
    expand_pieces(&pieces, &mut output, |index| {
        data.groups.get(index).and_then(|group| group.as_deref())
    });
    runtime.reserve_memory(output.len())?;
    runtime.new_string(output)
}

fn match_string<'s>(runtime: &mut dyn PyRuntime<'s>, receiver: PyValue<'s>) -> PyResult<'s> {
    Ok(match_data(runtime, receiver)?.subject)
}

fn match_re<'s>(runtime: &mut dyn PyRuntime<'s>, receiver: PyValue<'s>) -> PyResult<'s> {
    Ok(match_data(runtime, receiver)?.regex)
}

fn match_pos<'s>(runtime: &mut dyn PyRuntime<'s>, receiver: PyValue<'s>) -> PyResult<'s> {
    int_value(match_data(runtime, receiver)?.pos)
}

fn match_endpos<'s>(runtime: &mut dyn PyRuntime<'s>, receiver: PyValue<'s>) -> PyResult<'s> {
    int_value(match_data(runtime, receiver)?.endpos)
}

/// The group CPython reports as last matched: the participating group that ends last, with the
/// outermost (lowest-numbered) group winning ties, since it closes after the groups it contains.
fn last_group(data: &PyMatchData<'_>) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None;
    for (index, span) in data.spans.iter().enumerate().skip(1) {
        let Some((_, end)) = span else {
            continue;
        };
        if best.is_none_or(|(_, best_end)| *end > best_end) {
            best = Some((index, *end));
        }
    }
    best.map(|(index, _)| index)
}

fn match_lastindex<'s>(runtime: &mut dyn PyRuntime<'s>, receiver: PyValue<'s>) -> PyResult<'s> {
    let data = match_data(runtime, receiver)?;
    match last_group(&data) {
        Some(index) => int_value(index),
        None => Ok(Value::None),
    }
}

fn match_lastgroup<'s>(runtime: &mut dyn PyRuntime<'s>, receiver: PyValue<'s>) -> PyResult<'s> {
    let data = match_data(runtime, receiver)?;
    match last_group(&data).and_then(|index| data.group_names.get(index).cloned().flatten()) {
        Some(name) => runtime.new_string(name),
        None => Ok(Value::None),
    }
}

fn match_regs<'s>(runtime: &mut dyn PyRuntime<'s>, receiver: PyValue<'s>) -> PyResult<'s> {
    let data = match_data(runtime, receiver)?;
    let mut spans = Vec::with_capacity(data.spans.len());
    for span in &data.spans {
        let (start, end) = span_pair(*span)?;
        spans.push(runtime.new_tuple(vec![Value::Int(start), Value::Int(end)])?);
    }
    runtime.new_tuple(spans)
}

fn int_value<'s>(value: usize) -> PyResult<'s> {
    i64::try_from(value)
        .map(Value::Int)
        .map_err(|_| PyError::overflow_error("value is outside Python int range"))
}

/// A `count`/`maxsplit` argument: zero means unlimited, negative values are rejected.
fn count_arg<'s>(
    runtime: &dyn PyRuntime<'s>,
    value: Option<Value<'s>>,
    name: &str,
) -> PyResult<'s, usize> {
    let Some(value) = value else {
        return Ok(0);
    };
    let count = value.cast::<PyIndex>(runtime)?.0;
    usize::try_from(count).map_err(|_| PyError::value_error(format!("{name} must be non-negative")))
}

fn flags_arg<'s>(
    runtime: &dyn PyRuntime<'s>,
    args: &CallArgs<'s>,
    positional_index: usize,
    function: &str,
) -> PyResult<'s, u32> {
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
    u32::try_from(value).map_err(|_| PyError::value_error("regex flags are out of range"))
}

fn string_arg<'s>(
    runtime: &dyn PyRuntime<'s>,
    value: &Value<'s>,
    label: &str,
    maximum: usize,
) -> PyResult<'s, String> {
    let OwnedPyString(value) = (*value).cast(runtime)?;
    if value.len() > maximum {
        Err(PyError::value_error(format!(
            "{label} exceeds the bounded regex input limit"
        )))
    } else {
        Ok(value)
    }
}

thread_local! {
    /// Compiled patterns by source and flags, most recently used first.
    static CACHE: RefCell<Vec<(String, u32, Regex)>> = const { RefCell::new(Vec::new()) };
}

/// Compile a Python pattern with `flags`, through the host-side cache.
pub(in crate::python) fn build_regex<'s>(pattern: &str, flags: u32) -> PyResult<'s, Regex> {
    if pattern.len() > MAX_PATTERN {
        return Err(PyError::value_error(
            "regex pattern exceeds the bounded pattern limit",
        ));
    }
    if flags & (LOCALE | DEBUG) != 0 {
        return Err(PyError::value_error(
            "regex flags LOCALE and DEBUG are not supported",
        ));
    }
    if flags & !SUPPORTED_FLAGS != 0 {
        return Err(PyError::value_error("regex flags are out of range"));
    }
    let cached = CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let position = cache
            .iter()
            .position(|(source, cached_flags, _)| source == pattern && *cached_flags == flags)?;
        let entry = cache.remove(position);
        let regex = entry.2.clone();
        cache.insert(0, entry);
        Some(regex)
    });
    if let Some(regex) = cached {
        return Ok(regex);
    }
    let translated = translate_pattern(pattern, flags)?;
    let regex = RegexBuilder::new(&translated)
        .case_insensitive(flags & IGNORECASE != 0)
        .multi_line(flags & MULTILINE != 0)
        .dot_matches_new_line(flags & DOTALL != 0)
        .ignore_whitespace(flags & VERBOSE != 0)
        .size_limit(COMPILED_SIZE_LIMIT)
        .build()
        .map_err(|error| PyError::value_error(format!("unsupported regex pattern: {error}")))?;
    CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.len() == CACHE_SIZE {
            cache.pop();
        }
        cache.insert(0, (pattern.to_string(), flags, regex.clone()));
    });
    Ok(regex)
}

/// Rewrite Python regex syntax into the `regex` crate's dialect.
///
/// Handles `\Z`, ASCII-mode `\w \d \s \b` classes, `{` and `}` that do not form a repetition,
/// `[` inside a class, `(?#...)` comments and the `a`, `u` and `L` inline flags. In verbose
/// mode, whitespace and `#` inside a class are escaped because the engine would otherwise drop
/// them where Python keeps them literal.
fn translate_pattern<'s>(pattern: &str, flags: u32) -> PyResult<'s, String> {
    let mut ascii = flags & ASCII != 0;
    let verbose = flags & VERBOSE != 0;
    let chars: Vec<char> = pattern.chars().collect();
    let mut out = String::with_capacity(pattern.len() + 16);
    let mut index = 0;
    let mut in_class = false;
    let mut class_start = false;
    while index < chars.len() {
        let character = chars[index];
        match character {
            '\\' => {
                let Some(&next) = chars.get(index + 1) else {
                    return Err(PyError::value_error("bad escape (end of pattern)"));
                };
                index += 2;
                match next {
                    'Z' if !in_class => out.push_str("\\z"),
                    'd' if ascii => out.push_str("[0-9]"),
                    'D' if ascii => out.push_str("[^0-9]"),
                    'w' if ascii => out.push_str("[0-9A-Za-z_]"),
                    'W' if ascii => out.push_str("[^0-9A-Za-z_]"),
                    's' if ascii => out.push_str("[\\t\\n\\x0b\\x0c\\r ]"),
                    'S' if ascii => out.push_str("[^\\t\\n\\x0b\\x0c\\r ]"),
                    'b' if ascii && !in_class => out.push_str("(?-u:\\b)"),
                    'B' if ascii && !in_class => out.push_str("(?-u:\\B)"),
                    _ => {
                        out.push('\\');
                        out.push(next);
                    }
                }
                class_start = false;
                continue;
            }
            '[' if !in_class => {
                in_class = true;
                class_start = true;
                out.push('[');
                index += 1;
                if chars.get(index) == Some(&'^') {
                    out.push('^');
                    index += 1;
                }
                continue;
            }
            '[' => out.push_str("\\["),
            ']' if in_class && !class_start => {
                in_class = false;
                out.push(']');
            }
            '(' if !in_class && chars.get(index + 1) == Some(&'?') => {
                if chars.get(index + 2) == Some(&'#') {
                    let close = chars[index..]
                        .iter()
                        .position(|&c| c == ')')
                        .ok_or_else(|| PyError::value_error("missing ), unterminated comment"))?;
                    index += close + 1;
                    continue;
                }
                let mut end = index + 2;
                let mut letters = String::new();
                while end < chars.len() && "aiLmsux".contains(chars[end]) {
                    letters.push(chars[end]);
                    end += 1;
                }
                let terminator = chars.get(end).copied();
                if !letters.is_empty() && matches!(terminator, Some(')' | ':' | '-')) {
                    if letters.contains('L') {
                        return Err(PyError::value_error("regex flag LOCALE is not supported"));
                    }
                    if letters.contains('a') {
                        ascii = true;
                    }
                    let kept: String = letters.chars().filter(|c| !"au".contains(*c)).collect();
                    match terminator {
                        Some(')') if kept.is_empty() => {}
                        Some(')') => {
                            out.push_str("(?");
                            out.push_str(&kept);
                            out.push(')');
                        }
                        _ => {
                            out.push_str("(?");
                            out.push_str(&kept);
                            out.push(terminator.unwrap_or(':'));
                        }
                    }
                    index = end + 1;
                    continue;
                }
                out.push('(');
            }
            '{' if !in_class => {
                if let Some(end) = repetition_end(&chars, index) {
                    let mut body: String = chars[index + 1..end].iter().collect();
                    if body.starts_with(',') {
                        body.insert(0, '0');
                    }
                    out.push('{');
                    out.push_str(&body);
                    out.push('}');
                    index = end + 1;
                    continue;
                }
                out.push_str("\\{");
            }
            '}' if !in_class => out.push_str("\\}"),
            c if verbose && in_class && (c.is_whitespace() || c == '#') => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
        class_start = false;
        index += 1;
    }
    Ok(out)
}

/// The index of the `}` closing a `{m}`, `{m,}`, `{,n}` or `{m,n}` repetition starting at
/// `open`, or `None` when the brace is a literal as Python treats it.
fn repetition_end(chars: &[char], open: usize) -> Option<usize> {
    let mut index = open + 1;
    let mut digits = 0;
    let mut comma = false;
    while let Some(&c) = chars.get(index) {
        match c {
            '0'..='9' => digits += 1,
            ',' if !comma => comma = true,
            '}' => return (digits > 0).then_some(index),
            _ => return None,
        }
        index += 1;
    }
    None
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_python_only_syntax() {
        assert_eq!(translate_pattern(r"a\Z", 0).unwrap(), r"a\z");
        assert_eq!(
            translate_pattern(r"{(?P<x>a)}", 0).unwrap(),
            r"\{(?P<x>a)\}"
        );
        assert_eq!(
            translate_pattern(r"a{2,3}b{,2}c{4}", 0).unwrap(),
            r"a{2,3}b{0,2}c{4}"
        );
        assert_eq!(translate_pattern(r"(?#note)a(?i)b", 0).unwrap(), r"a(?i)b");
        assert_eq!(
            translate_pattern(r"(?a:\w+)", 0).unwrap(),
            r"(?:[0-9A-Za-z_]+)"
        );
        assert_eq!(
            translate_pattern(r"\d[\d-]", ASCII).unwrap(),
            r"[0-9][[0-9]-]"
        );
        assert_eq!(translate_pattern(r"[a b#]", VERBOSE).unwrap(), r"[a\ b\#]");
        assert_eq!(translate_pattern(r"[[]]", 0).unwrap(), r"[\[]]");
    }

    #[test]
    fn repetition_detection() {
        let chars: Vec<char> = "{3}".chars().collect();
        assert_eq!(repetition_end(&chars, 0), Some(2));
        let chars: Vec<char> = "{x}".chars().collect();
        assert_eq!(repetition_end(&chars, 0), None);
        let chars: Vec<char> = "{,}".chars().collect();
        assert_eq!(repetition_end(&chars, 0), None);
    }
}
