//! Deterministic text-only sed over simulated files and standard input.
//!
//! Scripts are parsed completely before input or in-place mutations. The supported slice covers
//! ordinary addresses, ranges, substitution, selection, early exit, text insertion/replacement,
//! transliteration, and line numbering. Binary input and the larger hold-space/branch language
//! are explicit boundaries.

use std::collections::HashMap;

use super::options::{parse_options_or_report, OptionSpec};
use super::regex_compat::basic_regex_to_rust;
use super::util::{ewln, read_inputs_system, uses_standard_input, w};
use super::{CommandSpec, Io, Trust};
use crate::exec::ShellPoll;
use crate::program::ProcessContext;
use crate::syscalls::System;

pub fn register(commands: &mut HashMap<&'static str, CommandSpec>) {
    super::reg_system_poll(commands, "/usr/bin/sed", Trust::Partial, run);
}

fn run(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    #[derive(Clone, Copy, PartialEq)]
    enum Key {
        InPlace,
        Quiet,
        Extended,
        Expression,
        File,
        Help,
    }
    const OPTIONS: &[OptionSpec<Key>] = &[
        OptionSpec::optional_attached(Key::InPlace, Some('i'), Some("in-place")),
        OptionSpec::flag(Key::Quiet, Some('n'), Some("quiet")),
        OptionSpec::flag(Key::Quiet, None, Some("silent")),
        OptionSpec::flag(Key::Extended, Some('r'), Some("regexp-extended")),
        OptionSpec::flag(Key::Extended, Some('E'), None),
        OptionSpec::required(Key::Expression, Some('e'), Some("expression")),
        OptionSpec::required(Key::File, Some('f'), Some("file")),
        OptionSpec::flag(Key::Help, None, Some("help")),
    ];
    let parsed = match parse_options_or_report(
        "sed",
        context.args,
        OPTIONS,
        (
            Key::Help,
            "usage: sed [OPTIONS] SCRIPT [FILE...]\nsupported: -n -E -r -e SCRIPT -f FILE -i\n",
        ),
        io.out,
        io.err,
    ) {
        Ok(parsed) => parsed,
        Err(status) => return ShellPoll::Ready(status),
    };

    let mut in_place = false;
    let mut quiet = false;
    let mut extended = false;
    let mut scripts = Vec::new();
    for option in parsed.options {
        match option.key {
            Key::InPlace => {
                in_place = true;
                if option.value.is_some_and(|suffix| !suffix.is_empty()) {
                    ewln(io.err, "sed: unsupported in-place backup suffix");
                    return ShellPoll::Ready(2);
                }
            }
            Key::Quiet => quiet = true,
            Key::Extended => extended = true,
            Key::Expression => scripts.push(option.value.expect("required option value")),
            Key::File => {
                let file = option.value.expect("required option value");
                let cwd = context.system.cwd().to_string();
                let maximum = usize::try_from(context.system.limits().memory).unwrap_or(usize::MAX);
                match context.system.read_file_limited(&cwd, &file, maximum) {
                    Ok(bytes) => match String::from_utf8(bytes) {
                        Ok(script) => scripts.push(script),
                        Err(_) => {
                            ewln(io.err, &format!("sed: {file}: script is not valid UTF-8"));
                            return ShellPoll::Ready(2);
                        }
                    },
                    Err(error) => {
                        ewln(io.err, &format!("sed: {file}: {error}"));
                        return ShellPoll::Ready(2);
                    }
                }
            }
            Key::Help => unreachable!("help is handled by the shared option parser"),
        }
    }
    let mut operands = parsed.operands;
    if scripts.is_empty() {
        if operands.is_empty() {
            ewln(io.err, "sed: missing command");
            return ShellPoll::Ready(1);
        }
        scripts.push(operands.remove(0));
    }
    if in_place && operands.is_empty() {
        ewln(io.err, "sed: -i requires at least one file operand");
        return ShellPoll::Ready(2);
    }

    let mut commands = Vec::new();
    for script in scripts {
        match ScriptParser::new(&script, extended).parse() {
            Ok(parsed) => commands.extend(parsed),
            Err(error) => {
                context.system.note_unsupported(&format!("sed:{error}"));
                ewln(io.err, &format!("sed: {error}"));
                return ShellPoll::Ready(2);
            }
        }
    }
    if commands.is_empty() {
        ewln(io.err, "sed: empty command");
        return ShellPoll::Ready(2);
    }

    if in_place {
        let cwd = context.system.cwd().to_string();
        for file in operands {
            let maximum = usize::try_from(context.system.limits().memory).unwrap_or(usize::MAX);
            let data = match context.system.read_file_limited(&cwd, &file, maximum) {
                Ok(data) => data,
                Err(error) => {
                    ewln(io.err, &format!("sed: can't read {file}: {error}"));
                    return ShellPoll::Ready(1);
                }
            };
            let text = match std::str::from_utf8(&data) {
                Ok(text) => text,
                Err(_) => {
                    ewln(io.err, &format!("sed: {file}: input is not valid UTF-8"));
                    return ShellPoll::Ready(2);
                }
            };
            let mut file_commands = commands.clone();
            let output = match process(context.system, text, &mut file_commands, quiet) {
                Ok(output) => output,
                Err(status) => return ShellPoll::Ready(status),
            };
            let mode = match context.system.metadata(&cwd, &file, true) {
                Ok(info) => info.mode,
                Err(error) => {
                    ewln(io.err, &format!("sed: can't stat {file}: {error}"));
                    return ShellPoll::Ready(1);
                }
            };
            if let Err(error) = context
                .system
                .write_file(&cwd, &file, output.as_bytes(), mode)
            {
                ewln(io.err, &format!("sed: can't write {file}: {error}"));
                return ShellPoll::Ready(1);
            }
        }
        return ShellPoll::Ready(0);
    }

    let files = operands.iter().collect::<Vec<_>>();
    if uses_standard_input(&files) {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let (data, errors) = read_inputs_system(context.system, &files, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("sed: {error}"));
        return ShellPoll::Ready(1);
    }
    let text = match std::str::from_utf8(&data) {
        Ok(text) => text,
        Err(_) => {
            ewln(io.err, "sed: input is not valid UTF-8");
            return ShellPoll::Ready(2);
        }
    };
    match process(context.system, text, &mut commands, quiet) {
        Ok(output) => {
            w(io.out, &output);
            ShellPoll::Ready(0)
        }
        Err(status) => ShellPoll::Ready(status),
    }
}

#[derive(Clone)]
enum Address {
    Line(usize),
    Last,
    Regex(regex::Regex),
}

impl Address {
    fn matches(&self, line: usize, total: usize, text: &str) -> bool {
        match self {
            Self::Line(expected) => line == *expected,
            Self::Last => line == total,
            Self::Regex(regex) => regex.is_match(text),
        }
    }
}

#[derive(Clone)]
struct Command {
    first: Option<Address>,
    second: Option<Address>,
    negate: bool,
    range_active: bool,
    kind: CommandKind,
}

impl Command {
    fn selection_for(&mut self, line: usize, total: usize, text: &str) -> Selection {
        let selected = match (&self.first, &self.second) {
            (None, _) => Selection::Single,
            (Some(first), None) => {
                if first.matches(line, total, text) {
                    Selection::Single
                } else {
                    Selection::No
                }
            }
            (Some(first), Some(second)) if self.range_active => {
                if range_ends(second, line, total, text, false) {
                    self.range_active = false;
                    Selection::RangeEnd
                } else {
                    Selection::RangeBody
                }
            }
            (Some(first), Some(second)) if first.matches(line, total, text) => {
                if range_ends(second, line, total, text, true) {
                    Selection::RangeEnd
                } else {
                    self.range_active = true;
                    Selection::RangeBody
                }
            }
            (Some(_), Some(_)) => Selection::No,
        };
        if self.negate {
            match selected {
                Selection::No => Selection::Single,
                Selection::Single | Selection::RangeBody | Selection::RangeEnd => Selection::No,
            }
        } else {
            selected
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Selection {
    No,
    Single,
    RangeBody,
    RangeEnd,
}

fn range_ends(address: &Address, line: usize, total: usize, text: &str, starting: bool) -> bool {
    match address {
        Address::Line(expected) => line >= *expected,
        Address::Last => line == total,
        Address::Regex(regex) => !starting && regex.is_match(text),
    }
}

#[derive(Clone)]
enum CommandKind {
    Substitute {
        regex: regex::Regex,
        replacement: String,
        global: bool,
        nth: usize,
        print: bool,
    },
    Delete,
    Print,
    Quit,
    Append(String),
    Insert(String),
    Change(String),
    Transliterate(Vec<(char, char)>),
    LineNumber,
}

fn process(
    system: &mut dyn System,
    text: &str,
    commands: &mut [Command],
    quiet: bool,
) -> Result<String, i32> {
    let memory_mark = system.memory_used();
    let scratch = (text.len() as u64)
        .saturating_mul(2)
        .saturating_add((commands.len() as u64).saturating_mul(64));
    if !system.reserve_memory(scratch) {
        return Err(system.stop_status());
    }
    let result = process_reserved(system, text, commands, quiet);
    system.release_memory(system.memory_used().saturating_sub(memory_mark));
    result
}

fn process_reserved(
    system: &mut dyn System,
    text: &str,
    commands: &mut [Command],
    quiet: bool,
) -> Result<String, i32> {
    let lines = text.split_inclusive('\n').collect::<Vec<_>>();
    let mut output = String::new();
    for (index, raw_line) in lines.iter().enumerate() {
        if !system.charge_cpu(raw_line.len() as u64 + commands.len() as u64) {
            return Err(system.stop_status());
        }
        let had_newline = raw_line.ends_with('\n');
        let mut pattern = raw_line.strip_suffix('\n').unwrap_or(raw_line).to_string();
        let mut deleted = false;
        let mut changed = false;
        let mut quit = false;
        let mut after = Vec::new();
        for command in commands.iter_mut() {
            let selection = command.selection_for(index + 1, lines.len(), &pattern);
            if selection == Selection::No {
                continue;
            }
            match &command.kind {
                CommandKind::Substitute {
                    regex,
                    replacement,
                    global,
                    nth,
                    print,
                } => {
                    let bound = (pattern.len() as u64)
                        .saturating_mul((replacement.len() as u64).saturating_add(1))
                        .saturating_add(replacement.len() as u64);
                    if !system.reserve_memory(bound) {
                        return Err(system.stop_status());
                    }
                    if !system
                        .charge_cpu((pattern.len() as u64).saturating_add(replacement.len() as u64))
                    {
                        system.release_memory(bound);
                        return Err(system.stop_status());
                    }
                    let (result, substitutions) =
                        substitute(regex, replacement, &pattern, *global, *nth);
                    system.release_memory(bound);
                    pattern = result;
                    if *print && substitutions != 0 {
                        push_line(system, &mut output, &pattern, true)?;
                    }
                }
                CommandKind::Delete => {
                    deleted = true;
                    break;
                }
                CommandKind::Print => push_line(system, &mut output, &pattern, true)?,
                CommandKind::Quit => {
                    quit = true;
                    break;
                }
                CommandKind::Append(text) => after.push(text.clone()),
                CommandKind::Insert(text) => push_line(system, &mut output, text, true)?,
                CommandKind::Change(text) => {
                    if selection != Selection::RangeBody {
                        pattern = text.clone();
                        changed = true;
                    } else {
                        deleted = true;
                    }
                    break;
                }
                CommandKind::Transliterate(mapping) => {
                    pattern = pattern
                        .chars()
                        .map(|character| {
                            mapping
                                .iter()
                                .find_map(|(from, to)| (*from == character).then_some(*to))
                                .unwrap_or(character)
                        })
                        .collect();
                }
                CommandKind::LineNumber => {
                    push_line(system, &mut output, &(index + 1).to_string(), true)?
                }
            }
        }
        if changed {
            push_line(system, &mut output, &pattern, true)?;
        } else if !quiet && !deleted {
            push_line(system, &mut output, &pattern, had_newline)?;
        }
        for text in after {
            push_line(system, &mut output, &text, true)?;
        }
        if quit {
            break;
        }
    }
    Ok(output)
}

fn push_line(
    system: &mut dyn System,
    output: &mut String,
    text: &str,
    newline: bool,
) -> Result<(), i32> {
    let bytes = (text.len() as u64).saturating_add(u64::from(newline));
    let projected = (output.len() as u64).saturating_add(bytes);
    if projected > system.output_remaining() {
        let request = system.output_remaining().saturating_add(1);
        let _ = system.charge_output(request);
        return Err(system.stop_status());
    }
    if !system.reserve_memory(bytes) || !system.charge_cpu(bytes) {
        return Err(system.stop_status());
    }
    output.push_str(text);
    if newline {
        output.push('\n');
    }
    Ok(())
}

fn substitute(
    regex: &regex::Regex,
    replacement: &str,
    text: &str,
    global: bool,
    nth: usize,
) -> (String, usize) {
    let count = regex.find_iter(text).count();
    if global && nth == 0 {
        return (regex.replace_all(text, replacement).into_owned(), count);
    }
    if nth > 0 {
        let mut seen = 0usize;
        let replaced = regex
            .replace_all(text, |captures: &regex::Captures| {
                seen += 1;
                if seen == nth || global && seen >= nth {
                    let mut output = String::new();
                    captures.expand(replacement, &mut output);
                    output
                } else {
                    captures[0].to_string()
                }
            })
            .into_owned();
        return (replaced, usize::from(count >= nth));
    }
    (
        regex.replace(text, replacement).into_owned(),
        usize::from(count != 0),
    )
}

struct ScriptParser<'a> {
    source: &'a str,
    cursor: usize,
    extended: bool,
}

impl<'a> ScriptParser<'a> {
    fn new(source: &'a str, extended: bool) -> Self {
        Self {
            source,
            cursor: 0,
            extended,
        }
    }

    fn parse(mut self) -> Result<Vec<Command>, String> {
        let mut commands = Vec::new();
        self.separators();
        while self.peek().is_some() {
            let first = self.address()?;
            let second = if first.is_some() && self.take(',') {
                Some(
                    self.address()?
                        .ok_or_else(|| "missing second range address".to_string())?,
                )
            } else {
                None
            };
            self.spaces();
            let negate = self.take('!');
            self.spaces();
            let command = self
                .bump()
                .ok_or_else(|| "missing command after address".to_string())?;
            let kind = match command {
                's' => self
                    .substitution()
                    .map_err(|error| format!("invalid substitution: {error}"))?,
                'd' => CommandKind::Delete,
                'p' => CommandKind::Print,
                'q' => CommandKind::Quit,
                '=' => CommandKind::LineNumber,
                'a' => CommandKind::Append(self.text_argument()?),
                'i' => CommandKind::Insert(self.text_argument()?),
                'c' => CommandKind::Change(self.text_argument()?),
                'y' => self.transliteration()?,
                other => return Err(format!("unimplemented command '{other}'")),
            };
            commands.push(Command {
                first,
                second,
                negate,
                range_active: false,
                kind,
            });
            if !matches!(command, 'a' | 'i' | 'c') {
                self.spaces();
                if self
                    .peek()
                    .is_some_and(|value| !matches!(value, ';' | '\n'))
                {
                    return Err(format!("unexpected text after '{command}' command"));
                }
            }
            self.separators();
        }
        Ok(commands)
    }

    fn address(&mut self) -> Result<Option<Address>, String> {
        self.spaces();
        if self
            .peek()
            .is_some_and(|character| character.is_ascii_digit())
        {
            let digits = self.take_while(|character| character.is_ascii_digit());
            let line = digits
                .parse()
                .map_err(|_| format!("invalid line address '{digits}'"))?;
            if line == 0 {
                return Err("line address must be at least 1".to_string());
            }
            return Ok(Some(Address::Line(line)));
        }
        if self.take('$') {
            return Ok(Some(Address::Last));
        }
        if self.peek() == Some('/') {
            let pattern = self.delimited()?;
            if pattern.is_empty() {
                return Err("empty address regex reuse is not supported".to_string());
            }
            return self.compile_regex(&pattern).map(Address::Regex).map(Some);
        }
        Ok(None)
    }

    fn substitution(&mut self) -> Result<CommandKind, String> {
        let pattern = self.delimited()?;
        if pattern.is_empty() {
            return Err("empty regex reuse is not supported".to_string());
        }
        let replacement = self.delimited_body(self.last_delimiter()?)?;
        let flags = self.take_until_separator();
        if let Some(flag) = flags
            .chars()
            .find(|flag| !matches!(flag, 'g' | 'p' | 'i' | 'I' | '0'..='9'))
        {
            return Err(format!("unsupported substitution flag '{flag}'"));
        }
        let ignore_case = flags.contains('i') || flags.contains('I');
        let global = flags.contains('g');
        let print = flags.contains('p');
        let digits = flags
            .chars()
            .filter(char::is_ascii_digit)
            .collect::<String>();
        let nth = if digits.is_empty() {
            0
        } else {
            digits
                .parse()
                .map_err(|_| format!("invalid substitution occurrence '{digits}'"))?
        };
        let pattern = if self.extended {
            pattern
        } else {
            basic_regex_to_rust(&pattern)?
        };
        let regex = regex::RegexBuilder::new(&pattern)
            .case_insensitive(ignore_case)
            .build()
            .map_err(|error| format!("invalid substitution regex: {error}"))?;
        Ok(CommandKind::Substitute {
            regex,
            replacement: sed_replacement(&replacement),
            global,
            nth,
            print,
        })
    }

    fn transliteration(&mut self) -> Result<CommandKind, String> {
        let from = self.delimited()?;
        let to = self.delimited_body(self.last_delimiter()?)?;
        let from = unescape_chars(&from);
        let to = unescape_chars(&to);
        if from.len() != to.len() {
            return Err("transliteration strings have different lengths".to_string());
        }
        Ok(CommandKind::Transliterate(
            from.into_iter().zip(to).collect(),
        ))
    }

    fn text_argument(&mut self) -> Result<String, String> {
        self.spaces();
        if self.take('\\') && self.take('\n') {
            // Traditional sed writes text on the line following `a\`, `i\`, or `c\`.
        } else {
            self.spaces();
        }
        let text = self.take_until_separator();
        if text.is_empty() {
            return Err("text command requires an argument".to_string());
        }
        Ok(text.replace("\\n", "\n").replace("\\t", "\t"))
    }

    fn compile_regex(&self, pattern: &str) -> Result<regex::Regex, String> {
        let pattern = if self.extended {
            pattern.to_string()
        } else {
            basic_regex_to_rust(pattern)?
        };
        regex::Regex::new(&pattern).map_err(|error| format!("invalid address regex: {error}"))
    }

    fn delimited(&mut self) -> Result<String, String> {
        let delimiter = self.bump().ok_or_else(|| "missing delimiter".to_string())?;
        if delimiter == '\n' {
            return Err("newline cannot be a delimiter".to_string());
        }
        self.delimited_body(delimiter)
    }

    fn delimited_body(&mut self, delimiter: char) -> Result<String, String> {
        let mut output = String::new();
        let mut escaped = false;
        while let Some(character) = self.bump() {
            if escaped {
                if character == delimiter {
                    output.push(character);
                } else {
                    output.push('\\');
                    output.push(character);
                }
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == delimiter {
                return Ok(output);
            } else if character == '\n' {
                return Err("unterminated delimited expression".to_string());
            } else {
                output.push(character);
            }
        }
        Err("unterminated delimited expression".to_string())
    }

    fn last_delimiter(&self) -> Result<char, String> {
        self.source[..self.cursor]
            .chars()
            .next_back()
            .ok_or_else(|| "missing delimiter".to_string())
    }

    fn take_until_separator(&mut self) -> String {
        let start = self.cursor;
        while self
            .peek()
            .is_some_and(|value| !matches!(value, ';' | '\n'))
        {
            self.bump();
        }
        self.source[start..self.cursor].trim().to_string()
    }

    fn separators(&mut self) {
        loop {
            while self
                .peek()
                .is_some_and(|value| value.is_whitespace() || value == ';')
            {
                self.bump();
            }
            if self.peek() != Some('#') {
                break;
            }
            while self.peek().is_some_and(|value| value != '\n') {
                self.bump();
            }
        }
    }

    fn spaces(&mut self) {
        while self
            .peek()
            .is_some_and(|value| matches!(value, ' ' | '\t' | '\r'))
        {
            self.bump();
        }
    }

    fn take_while(&mut self, predicate: impl Fn(char) -> bool) -> &'a str {
        let start = self.cursor;
        while self.peek().is_some_and(&predicate) {
            self.bump();
        }
        &self.source[start..self.cursor]
    }

    fn take(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn peek(&self) -> Option<char> {
        self.source[self.cursor..].chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let character = self.peek()?;
        self.cursor += character.len_utf8();
        Some(character)
    }
}

fn sed_replacement(value: &str) -> String {
    let mut output = String::new();
    let mut characters = value.chars().peekable();
    while let Some(character) = characters.next() {
        match character {
            '&' => output.push_str("${0}"),
            '$' => output.push_str("$$"),
            '\\' => match characters.next() {
                Some(next) if next.is_ascii_digit() => output.push_str(&format!("${{{next}}}")),
                Some('n') => output.push('\n'),
                Some('t') => output.push('\t'),
                Some(next) => output.push(next),
                None => output.push('\\'),
            },
            other => output.push(other),
        }
    }
    output
}

fn unescape_chars(value: &str) -> Vec<char> {
    let mut output = Vec::new();
    let mut characters = value.chars();
    while let Some(character) = characters.next() {
        if character == '\\' {
            output.push(match characters.next() {
                Some('n') => '\n',
                Some('t') => '\t',
                Some(next) => next,
                None => '\\',
            });
        } else {
            output.push(character);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::ScriptParser;

    #[test]
    fn parses_delimiters_ranges_and_common_commands() {
        let commands =
            ScriptParser::new(r#"/start/,/end/p; s/;/x/g; 3q; y/ab/AB/; /skip/!p"#, false)
                .parse()
                .unwrap();
        assert_eq!(commands.len(), 5);
    }

    #[test]
    fn rejects_incomplete_or_unknown_commands() {
        assert!(ScriptParser::new("s/a", false).parse().is_err());
        assert!(ScriptParser::new("h", false).parse().is_err());
        assert!(ScriptParser::new("y/ab/a/", false).parse().is_err());
    }
}
