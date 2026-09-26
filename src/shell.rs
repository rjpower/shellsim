//! A pragmatic bash-subset parser and executor.
//!
//! Scope includes pipelines, `&& || ; &` lists, redirects + heredocs, `if/for/while/until/case`,
//! function definitions, subshells/groups, and word expansion (quoting, `$VAR`/`${...}`
//! parameter expansion, `$(...)`/backtick command substitution, `$((...))` arithmetic,
//! tilde and globbing). It is not a complete bash, but it is faithful where it matters.

use crate::interp::Interp;
use std::fmt;

/// A syntax error found before shell execution begins.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShellError {
    /// Input ended before a required token or delimiter was found.
    UnexpectedEof { expected: String },
    /// A token appeared where the grammar required something else.
    UnexpectedToken { found: String, expected: String },
    /// A quoted word was not terminated.
    UnclosedQuote(char),
}

impl fmt::Display for ShellError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof { expected } => {
                write!(f, "unexpected end of file (expected {expected})")
            }
            Self::UnexpectedToken { found, expected } => {
                write!(f, "unexpected token {found} (expected {expected})")
            }
            Self::UnclosedQuote(quote) => write!(f, "unclosed {quote} quote"),
        }
    }
}

impl std::error::Error for ShellError {}

// ===================== AST =====================

#[derive(Clone, Debug)]
pub enum Node {
    Command {
        assigns: Vec<(String, String)>,
        words: Vec<String>,
        redirects: Vec<Redirect>,
    },
    /// Pre-expanded argv supplied by an internal modeled process launcher.
    ///
    /// The parser never constructs this variant. It prevents Python subprocess arguments from
    /// being interpreted a second time as shell source.
    ArgvCommand(Vec<String>),
    Pipeline(Vec<Node>),
    And(Box<Node>, Box<Node>),
    Or(Box<Node>, Box<Node>),
    Seq(Vec<Node>),
    Background(Box<Node>),
    Subshell(Box<Node>),
    Group(Box<Node>),
    If {
        cond: Box<Node>,
        then: Box<Node>,
        elifs: Vec<(Node, Node)>,
        els: Option<Box<Node>>,
    },
    For {
        var: String,
        words: Vec<String>,
        body: Box<Node>,
    },
    CFor {
        init: String,
        cond: String,
        update: String,
        body: Box<Node>,
    },
    While {
        cond: Box<Node>,
        body: Box<Node>,
        until: bool,
    },
    Case {
        word: String,
        arms: Vec<(Vec<String>, Node)>,
    },
    FuncDef {
        name: String,
        body: Box<Node>,
    },
    Arithmetic(String),
    Not(Box<Node>),
    Redirected(Box<Node>, Vec<Redirect>),
    Empty,
}

impl Node {
    /// Estimate owned bytes copied when a shell process forks its function table.
    ///
    /// The estimate is deliberately conservative and saturating. It is used only to reject or
    /// meter a child-state clone before performing host allocations.
    pub(crate) fn estimated_bytes(&self) -> u64 {
        const NODE_OVERHEAD: u64 = std::mem::size_of::<Node>() as u64;
        let strings = |values: &[String]| {
            values.iter().fold(0_u64, |total, value| {
                total.saturating_add(value.len() as u64).saturating_add(24)
            })
        };
        let redirects = |values: &[Redirect]| {
            values.iter().fold(0_u64, |total, value| {
                total
                    .saturating_add(value.target.len() as u64)
                    .saturating_add(std::mem::size_of::<Redirect>() as u64)
            })
        };
        NODE_OVERHEAD.saturating_add(match self {
            Self::Command {
                assigns,
                words,
                redirects: redirections,
            } => assigns
                .iter()
                .fold(strings(words), |total, (name, value)| {
                    total
                        .saturating_add(name.len() as u64)
                        .saturating_add(value.len() as u64)
                        .saturating_add(48)
                })
                .saturating_add(redirects(redirections)),
            Self::ArgvCommand(words) => strings(words),
            Self::Pipeline(nodes) | Self::Seq(nodes) => nodes.iter().fold(0, |total, node| {
                total.saturating_add(node.estimated_bytes())
            }),
            Self::And(left, right) | Self::Or(left, right) => left
                .estimated_bytes()
                .saturating_add(right.estimated_bytes()),
            Self::Background(node) | Self::Subshell(node) | Self::Group(node) | Self::Not(node) => {
                node.estimated_bytes()
            }
            Self::If {
                cond,
                then,
                elifs,
                els,
            } => elifs
                .iter()
                .fold(
                    cond.estimated_bytes()
                        .saturating_add(then.estimated_bytes()),
                    |total, (condition, body)| {
                        total
                            .saturating_add(condition.estimated_bytes())
                            .saturating_add(body.estimated_bytes())
                    },
                )
                .saturating_add(els.as_deref().map_or(0, Self::estimated_bytes)),
            Self::For { var, words, body } => (var.len() as u64)
                .saturating_add(strings(words))
                .saturating_add(body.estimated_bytes()),
            Self::CFor {
                init,
                cond,
                update,
                body,
            } => (init.len() as u64)
                .saturating_add(cond.len() as u64)
                .saturating_add(update.len() as u64)
                .saturating_add(body.estimated_bytes()),
            Self::While { cond, body, .. } => cond
                .estimated_bytes()
                .saturating_add(body.estimated_bytes()),
            Self::Case { word, arms } => {
                arms.iter()
                    .fold(word.len() as u64, |total, (patterns, body)| {
                        total
                            .saturating_add(strings(patterns))
                            .saturating_add(body.estimated_bytes())
                    })
            }
            Self::FuncDef { name, body } => {
                (name.len() as u64).saturating_add(body.estimated_bytes())
            }
            Self::Arithmetic(expression) => expression.len() as u64,
            Self::Redirected(node, redirections) => node
                .estimated_bytes()
                .saturating_add(redirects(redirections)),
            Self::Empty => 0,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Redirect {
    pub fd: i32, // 0 stdin, 1 stdout, 2 stderr
    pub op: RedirOp,
    pub target: String, // filename word (unexpanded), or heredoc body, or "&N"
}

#[derive(Clone, Debug, PartialEq)]
pub enum RedirOp {
    Read,       // <
    Write,      // >
    Append,     // >>
    ReadWrite,  // <>
    DupOut,     // >&N  / N>&M
    Close,      // >&-
    Heredoc,    // << (target carries the already-captured body; quoted flag in op variant below)
    HeredocRaw, // << with quoted delimiter (no expansion of body)
    HereString, // <<< word
}

// ===================== Lexer =====================

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Word(String),
    Op(String),            // ; ;; & && | || ( ) newline
    Less,                  // <
    Great,                 // >
    DGreat,                // >>
    Heredoc(String, bool), // body, quoted-delim
    Arithmetic(String),    // (( expression ))
    HereString(String),    // <<< word
    GreatAmp(i32),         // >&N captured fd source default 1; store dest in word? we encode as op
    CloseOut,              // >&-
    RedirFd(i32, String),  // e.g. 2> with op ; we keep simple
    Eof,
}

struct Lexer {
    chars: Vec<char>,
    i: usize,
    toks: Vec<Tok>,
    in_double_bracket: bool,
    heredocs_complete: bool,
    error: Option<ShellError>,
}

impl Lexer {
    fn new(src: &str) -> Self {
        Lexer {
            chars: src.chars().collect(),
            i: 0,
            toks: Vec::new(),
            in_double_bracket: false,
            heredocs_complete: true,
            error: None,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.i).copied()
    }
    fn at(&self, o: usize) -> Option<char> {
        self.chars.get(self.i + o).copied()
    }

    fn fd_redirect_len(&self) -> Option<usize> {
        let digits = self.chars[self.i..]
            .iter()
            .take_while(|character| character.is_ascii_digit())
            .count();
        (digits > 0 && matches!(self.at(digits), Some('<' | '>'))).then_some(digits)
    }

    fn tokenize(mut self) -> (Vec<Tok>, bool, Option<ShellError>) {
        // pending heredocs: (delim, quoted, token-index placeholder)
        let mut pending: Vec<(String, bool, usize)> = Vec::new();
        while let Some(c) = self.peek() {
            if self.in_double_bracket
                && matches!(self.toks.last(), Some(Tok::Word(operator)) if operator == "=~")
                && !matches!(c, ' ' | '\t' | '\n')
            {
                let pattern = self.read_double_bracket_regex();
                let pattern = if matches!(pattern.as_bytes().first(), Some(b'\'' | b'"')) {
                    pattern
                } else {
                    format!("\"{pattern}\"")
                };
                self.toks.push(Tok::Word(pattern));
                continue;
            }
            match c {
                ' ' | '\t' => {
                    self.i += 1;
                }
                '\\' if self.at(1) == Some('\n') => {
                    self.i += 2; // line continuation
                }
                '\n' => {
                    self.i += 1;
                    // resolve heredocs queued on this line
                    if !pending.is_empty() {
                        let queued = std::mem::take(&mut pending);
                        for (delim, quoted, idx) in queued {
                            let body = self.read_heredoc_body(&delim);
                            self.toks[idx] = Tok::Heredoc(body, quoted);
                        }
                    }
                    self.toks.push(Tok::Op("\n".into()));
                }
                '#' if self.prev_is_boundary() => {
                    while let Some(c) = self.peek() {
                        if c == '\n' {
                            break;
                        }
                        self.i += 1;
                    }
                }
                ';' => {
                    if self.at(1) == Some(';') {
                        self.toks.push(Tok::Op(";;".into()));
                        self.i += 2;
                    } else {
                        self.toks.push(Tok::Op(";".into()));
                        self.i += 1;
                    }
                }
                '&' => {
                    if self.in_double_bracket {
                        let word = if self.at(1) == Some('&') { "&&" } else { "&" };
                        self.i += word.len();
                        self.toks.push(Tok::Word(word.into()));
                    } else if self.at(1) == Some('&') {
                        self.toks.push(Tok::Op("&&".into()));
                        self.i += 2;
                    } else if self.at(1) == Some('>') {
                        // &> file  → redirect both
                        let append = self.at(2) == Some('>');
                        self.i += if append { 3 } else { 2 };
                        self.toks
                            .push(Tok::RedirFd(1, if append { "&>>" } else { "&>" }.into()));
                    } else {
                        self.toks.push(Tok::Op("&".into()));
                        self.i += 1;
                    }
                }
                '|' => {
                    if self.in_double_bracket {
                        let word = if self.at(1) == Some('|') { "||" } else { "|" };
                        self.i += word.len();
                        self.toks.push(Tok::Word(word.into()));
                    } else if self.at(1) == Some('|') {
                        self.toks.push(Tok::Op("||".into()));
                        self.i += 2;
                    } else if self.at(1) == Some('&') {
                        self.toks.push(Tok::Op("|&".into()));
                        self.i += 2;
                    } else {
                        self.toks.push(Tok::Op("|".into()));
                        self.i += 1;
                    }
                }
                '(' => {
                    if self.in_double_bracket {
                        self.toks.push(Tok::Word("(".into()));
                        self.i += 1;
                    } else if self.at(1) == Some('(') {
                        let expression = self.read_arithmetic_command();
                        self.toks.push(Tok::Arithmetic(expression));
                    } else {
                        self.toks.push(Tok::Op("(".into()));
                        self.i += 1;
                    }
                }
                ')' => {
                    self.toks.push(if self.in_double_bracket {
                        Tok::Word(")".into())
                    } else {
                        Tok::Op(")".into())
                    });
                    self.i += 1;
                }
                '<' => {
                    if self.in_double_bracket {
                        self.toks.push(Tok::Word("<".into()));
                        self.i += 1;
                    } else if self.at(1) == Some('(') {
                        self.i += 1;
                        let group = self.read_balanced_paren();
                        self.toks.push(Tok::Word(format!("<{group}")));
                    } else if self.at(1) == Some('&') {
                        self.i += 2;
                        let target = self.read_descriptor_target();
                        self.toks.push(Tok::RedirFd(0, format!("<&{target}")));
                    } else if self.at(1) == Some('>') {
                        self.i += 2;
                        self.toks.push(Tok::RedirFd(0, "<>".into()));
                    } else if self.at(1) == Some('<') && self.at(2) == Some('<') {
                        self.i += 3;
                        while matches!(self.peek(), Some(' ' | '\t')) {
                            self.i += 1;
                        }
                        let word = self.read_word();
                        self.toks.push(Tok::HereString(word));
                    } else if self.at(1) == Some('<') {
                        // heredoc << or <<-
                        let dashed = self.at(2) == Some('-');
                        self.i += if dashed { 3 } else { 2 };
                        // skip spaces
                        while self.peek() == Some(' ') || self.peek() == Some('\t') {
                            self.i += 1;
                        }
                        let (delim, quoted) = self.read_heredoc_delim();
                        let idx = self.toks.len();
                        self.toks.push(Tok::Heredoc(String::new(), quoted));
                        pending.push((delim, quoted || dashed && false, idx));
                        // note: dashed strips leading tabs; handled in read_heredoc_body via delim trim
                        if dashed {
                            // mark via a sentinel: store delim with leading marker
                            if let Some(last) = pending.last_mut() {
                                last.0 = format!("\t-{}", last.0); // encode dashed
                            }
                        }
                    } else {
                        self.toks.push(Tok::Less);
                        self.i += 1;
                    }
                }
                '>' => {
                    if self.in_double_bracket {
                        self.toks.push(Tok::Word(">".into()));
                        self.i += 1;
                    } else if self.at(1) == Some('(') {
                        self.i += 1;
                        let group = self.read_balanced_paren();
                        self.toks.push(Tok::Word(format!(">{group}")));
                    } else if self.at(1) == Some('>') {
                        self.toks.push(Tok::DGreat);
                        self.i += 2;
                    } else if self.at(1) == Some('|') {
                        self.toks.push(Tok::Great);
                        self.i += 2;
                    } else if self.at(1) == Some('&') {
                        // >&N
                        self.i += 2;
                        if self.peek() == Some('-') {
                            self.i += 1;
                            self.toks.push(Tok::CloseOut);
                            continue;
                        }
                        let mut n = String::new();
                        while let Some(d) = self.peek() {
                            if d.is_ascii_digit() {
                                n.push(d);
                                self.i += 1;
                            } else {
                                break;
                            }
                        }
                        self.toks.push(Tok::GreatAmp(n.parse().unwrap_or(1)));
                    } else {
                        self.toks.push(Tok::Great);
                        self.i += 1;
                    }
                }
                c if c.is_ascii_digit() && self.fd_redirect_len().is_some() => {
                    // fd-prefixed redirect like 2>, 200>>, or 10>&1
                    let digits = self.fd_redirect_len().expect("guard checked redirect");
                    let fd = self.chars[self.i..self.i + digits]
                        .iter()
                        .collect::<String>()
                        .parse::<i32>()
                        .unwrap_or(-1);
                    self.i += digits;
                    if self.peek() == Some('>') {
                        if self.at(1) == Some('>') {
                            self.i += 2;
                            self.toks.push(Tok::RedirFd(fd, ">>".into()));
                        } else if self.at(1) == Some('&') {
                            self.i += 2;
                            if self.peek() == Some('-') {
                                self.i += 1;
                                self.toks.push(Tok::RedirFd(fd, ">&-".into()));
                                continue;
                            }
                            let mut n = String::new();
                            while let Some(d) = self.peek() {
                                if d.is_ascii_digit() {
                                    n.push(d);
                                    self.i += 1;
                                } else {
                                    break;
                                }
                            }
                            self.toks.push(Tok::RedirFd(fd, format!(">&{n}")));
                        } else {
                            self.i += 1;
                            self.toks.push(Tok::RedirFd(fd, ">".into()));
                        }
                    } else {
                        self.i += 1;
                        if self.peek() == Some('&') {
                            self.i += 1;
                            let target = self.read_descriptor_target();
                            self.toks.push(Tok::RedirFd(fd, format!("<&{target}")));
                        } else if self.peek() == Some('>') {
                            self.i += 1;
                            self.toks.push(Tok::RedirFd(fd, "<>".into()));
                        } else {
                            self.toks.push(Tok::RedirFd(fd, "<".into()));
                        }
                    }
                }
                _ => {
                    let w = self.read_word();
                    if w == "[[" {
                        self.in_double_bracket = true;
                    } else if w == "]]" && self.in_double_bracket {
                        self.in_double_bracket = false;
                    }
                    self.toks.push(Tok::Word(w));
                }
            }
        }
        // any heredocs not yet resolved (EOF without newline)
        if !pending.is_empty() {
            let queued = std::mem::take(&mut pending);
            for (delim, quoted, idx) in queued {
                let body = self.read_heredoc_body(&delim);
                self.toks[idx] = Tok::Heredoc(body, quoted);
            }
        }
        if self.in_double_bracket {
            self.error.get_or_insert(ShellError::UnexpectedEof {
                expected: "`]]`".to_string(),
            });
        }
        self.toks.push(Tok::Eof);
        (self.toks, self.heredocs_complete, self.error)
    }

    fn read_descriptor_target(&mut self) -> String {
        if self.peek() == Some('-') {
            self.i += 1;
            return "-".into();
        }
        let start = self.i;
        while self.peek().is_some_and(|value| value.is_ascii_digit()) {
            self.i += 1;
        }
        self.chars[start..self.i].iter().collect()
    }

    fn prev_is_boundary(&self) -> bool {
        // a '#' starts a comment only at the start of a word
        if self.i == 0 {
            return true;
        }
        matches!(
            self.chars.get(self.i - 1),
            Some(' ') | Some('\t') | Some('\n') | Some(';') | Some('&') | Some('|') | Some('(')
        )
    }

    /// Consume a top-level bash arithmetic command, preserving its expression as source.
    fn read_arithmetic_command(&mut self) -> String {
        self.i += 2; // opening ((
        let start = self.i;
        let mut depth = 1usize;
        while self.i < self.chars.len() {
            if self.peek() == Some('(') && self.at(1) == Some('(') {
                depth += 1;
                self.i += 2;
            } else if self.peek() == Some(')') && self.at(1) == Some(')') {
                depth -= 1;
                if depth == 0 {
                    let expression: String = self.chars[start..self.i].iter().collect();
                    self.i += 2;
                    return expression;
                }
                self.i += 2;
            } else {
                self.i += 1;
            }
        }
        self.error.get_or_insert(ShellError::UnexpectedEof {
            expected: "`))`".to_string(),
        });
        self.chars[start..].iter().collect()
    }

    fn read_heredoc_delim(&mut self) -> (String, bool) {
        let mut delim = String::new();
        let mut quoted = false;
        while let Some(c) = self.peek() {
            match c {
                '\'' | '"' => {
                    quoted = true;
                    self.i += 1;
                    while let Some(d) = self.peek() {
                        if d == c {
                            self.i += 1;
                            break;
                        }
                        delim.push(d);
                        self.i += 1;
                    }
                }
                ' ' | '\t' | '\n' | ';' | '&' | '|' | '<' | '>' => break,
                _ => {
                    delim.push(c);
                    self.i += 1;
                }
            }
        }
        (delim, quoted)
    }

    fn read_heredoc_body(&mut self, delim_enc: &str) -> String {
        // decode dashed marker
        let (dashed, delim) = if let Some(rest) = delim_enc.strip_prefix("\t-") {
            (true, rest.to_string())
        } else {
            (false, delim_enc.to_string())
        };
        let mut body = String::new();
        loop {
            // read one line
            let start = self.i;
            while let Some(c) = self.peek() {
                if c == '\n' {
                    break;
                }
                self.i += 1;
            }
            let line: String = self.chars[start..self.i].iter().collect();
            // consume newline if present
            let had_nl = self.peek() == Some('\n');
            if had_nl {
                self.i += 1;
            }
            let check = if dashed {
                line.trim_start_matches('\t')
            } else {
                line.as_str()
            };
            if check == delim {
                break;
            }
            let line = if dashed {
                line.trim_start_matches('\t').to_string()
            } else {
                line
            };
            body.push_str(&line);
            body.push('\n');
            if !had_nl && self.i >= self.chars.len() {
                self.heredocs_complete = false;
                break;
            }
        }
        body
    }

    fn read_word(&mut self) -> String {
        let mut w = String::new();
        // Array assignment literal: `name=( … )` or `name+=( … )` (optionally a `name[idx]=…`
        // form is handled later as an ordinary word). Absorb the parenthesized element list so
        // the parser sees it as one assignment word rather than an empty assign + a subshell.
        if let Some(consumed) = self.try_array_assign_prefix() {
            w.push_str(&consumed);
        }
        while let Some(c) = self.peek() {
            match c {
                ' ' | '\t' | '\n' | ';' | '&' | '|' | '(' | ')' | '<' | '>' => break,
                '\\' => {
                    w.push(c);
                    self.i += 1;
                    if let Some(n) = self.peek() {
                        w.push(n);
                        self.i += 1;
                    }
                }
                '\'' => {
                    w.push(c);
                    self.i += 1;
                    let mut closed = false;
                    while let Some(n) = self.peek() {
                        w.push(n);
                        self.i += 1;
                        if n == '\'' {
                            closed = true;
                            break;
                        }
                    }
                    if !closed {
                        self.error.get_or_insert(ShellError::UnclosedQuote('\''));
                    }
                }
                '"' => {
                    w.push(c);
                    self.i += 1;
                    let mut closed = false;
                    while let Some(n) = self.peek() {
                        w.push(n);
                        self.i += 1;
                        if n == '\\' {
                            if let Some(m) = self.peek() {
                                w.push(m);
                                self.i += 1;
                            }
                            continue;
                        }
                        if n == '"' {
                            closed = true;
                            break;
                        }
                    }
                    if !closed {
                        self.error.get_or_insert(ShellError::UnclosedQuote('"'));
                    }
                }
                '$' if self.at(1) == Some('(') => {
                    // command substitution or arithmetic: copy balanced
                    w.push(c);
                    self.i += 1;
                    let arith = self.at(1) == Some('(');
                    let _ = arith;
                    w.push_str(&self.read_balanced_paren());
                }
                '`' => {
                    w.push(c);
                    self.i += 1;
                    let mut closed = false;
                    while let Some(n) = self.peek() {
                        w.push(n);
                        self.i += 1;
                        if n == '`' {
                            closed = true;
                            break;
                        }
                    }
                    if !closed {
                        self.error.get_or_insert(ShellError::UnclosedQuote('`'));
                    }
                }
                '{' | '}' => {
                    // brace is part of words like ${..} (handled by $) or literal; treat as literal
                    w.push(c);
                    self.i += 1;
                }
                _ => {
                    w.push(c);
                    self.i += 1;
                }
            }
        }
        w
    }

    /// Read the unquoted right-hand side of `[[ value =~ regex ]]` as one expression token.
    /// Shell metacharacters such as `(` and `|` are regex syntax here, while escaped whitespace
    /// remains part of the pattern.
    fn read_double_bracket_regex(&mut self) -> String {
        let mut word = String::new();
        while let Some(c) = self.peek() {
            if matches!(c, ' ' | '\t' | '\n') {
                break;
            }
            if c == ']' && self.at(1) == Some(']') {
                break;
            }
            word.push(c);
            self.i += 1;
            if c == '\\' {
                if let Some(next) = self.peek() {
                    word.push(next);
                    self.i += 1;
                }
            }
        }
        word
    }

    /// If the word at the cursor is an array-assignment literal `name=( … )` or `name+=( … )`,
    /// consume `name`, the `=`/`+=`, and the whole balanced `( … )` (honoring quotes), and return
    /// the consumed text. Otherwise consume nothing and return `None`.
    fn try_array_assign_prefix(&mut self) -> Option<String> {
        let start = self.i;
        // identifier
        let mut j = self.i;
        if !matches!(self.chars.get(j), Some(c) if c.is_ascii_alphabetic() || *c == '_') {
            return None;
        }
        while matches!(self.chars.get(j), Some(c) if c.is_ascii_alphanumeric() || *c == '_') {
            j += 1;
        }
        // optional += / =
        if self.chars.get(j) == Some(&'+') && self.chars.get(j + 1) == Some(&'=') {
            j += 2;
        } else if self.chars.get(j) == Some(&'=') {
            j += 1;
        } else {
            return None;
        }
        if self.chars.get(j) != Some(&'(') {
            return None;
        }
        // commit: copy name..='(' then read balanced parens
        let prefix: String = self.chars[start..j].iter().collect();
        self.i = j;
        let parens = self.read_balanced_paren_quoted();
        Some(format!("{prefix}{parens}"))
    }

    /// Read a balanced `( … )` group starting at the current `(`, copying quoted regions
    /// verbatim (so `arr=("$x" 'a b')` keeps the spaces and `)` inside quotes is ignored).
    fn read_balanced_paren_quoted(&mut self) -> String {
        let mut out = String::new();
        let mut depth = 0;
        while let Some(c) = self.peek() {
            match c {
                '\'' => {
                    out.push(c);
                    self.i += 1;
                    while let Some(n) = self.peek() {
                        out.push(n);
                        self.i += 1;
                        if n == '\'' {
                            break;
                        }
                    }
                }
                '"' => {
                    out.push(c);
                    self.i += 1;
                    while let Some(n) = self.peek() {
                        out.push(n);
                        self.i += 1;
                        if n == '\\' {
                            if let Some(m) = self.peek() {
                                out.push(m);
                                self.i += 1;
                            }
                            continue;
                        }
                        if n == '"' {
                            break;
                        }
                    }
                }
                '\\' => {
                    out.push(c);
                    self.i += 1;
                    if let Some(n) = self.peek() {
                        out.push(n);
                        self.i += 1;
                    }
                }
                '(' => {
                    out.push(c);
                    self.i += 1;
                    depth += 1;
                }
                ')' => {
                    out.push(c);
                    self.i += 1;
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {
                    out.push(c);
                    self.i += 1;
                }
            }
        }
        if depth != 0 {
            self.error.get_or_insert(ShellError::UnexpectedEof {
                expected: "`)`".to_string(),
            });
        }
        out
    }

    fn read_balanced_paren(&mut self) -> String {
        // assumes current char is '('
        let mut out = String::new();
        let mut depth = 0;
        while let Some(c) = self.peek() {
            out.push(c);
            self.i += 1;
            if c == '(' {
                depth += 1;
            } else if c == ')' {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
        }
        if depth != 0 {
            self.error.get_or_insert(ShellError::UnexpectedEof {
                expected: "`)`".to_string(),
            });
        }
        out
    }
}

// ===================== Parser =====================

struct Parser {
    toks: Vec<Tok>,
    i: usize,
    error: Option<ShellError>,
}

const RESERVED: &[&str] = &[
    "if", "then", "elif", "else", "fi", "for", "while", "until", "do", "done", "case", "esac",
    "in", "function", "{", "}", "!", "[[", "]]",
];

fn redirect_pipeline_stderr(node: Node) -> Node {
    let redirect = Redirect {
        fd: 2,
        op: RedirOp::DupOut,
        target: "&1".into(),
    };
    match node {
        Node::Command {
            assigns,
            words,
            mut redirects,
        } => {
            redirects.push(redirect);
            Node::Command {
                assigns,
                words,
                redirects,
            }
        }
        Node::Redirected(node, mut redirects) => {
            redirects.push(redirect);
            Node::Redirected(node, redirects)
        }
        node => Node::Redirected(Box::new(node), vec![redirect]),
    }
}

impl Parser {
    fn new(toks: Vec<Tok>) -> Self {
        Parser {
            toks,
            i: 0,
            error: None,
        }
    }

    fn peek(&self) -> &Tok {
        self.toks.get(self.i).unwrap_or(&Tok::Eof)
    }

    fn next(&mut self) -> Tok {
        let t = self.toks.get(self.i).cloned().unwrap_or(Tok::Eof);
        self.i += 1;
        t
    }

    fn skip_newlines(&mut self) {
        while matches!(self.peek(), Tok::Op(o) if o == "\n" || o == ";") {
            self.i += 1;
        }
    }
    fn skip_blank_newlines(&mut self) {
        while matches!(self.peek(), Tok::Op(o) if o == "\n") {
            self.i += 1;
        }
    }

    fn word_is(&self, kw: &str) -> bool {
        matches!(self.peek(), Tok::Word(w) if w == kw)
    }

    fn parse_program(&mut self) -> Node {
        let mut nodes = Vec::new();
        self.skip_newlines();
        while !matches!(self.peek(), Tok::Eof) {
            if self.error.is_some() {
                break;
            }
            if matches!(self.peek(), Tok::Op(o) if o == ")" ) {
                break;
            }
            let position = self.i;
            let n = self.parse_and_or();
            nodes.push(n);
            if self.i == position {
                self.expected("command");
                break;
            }
            self.skip_terminators();
            // stop at block enders
            if self.at_block_end() {
                break;
            }
        }
        if nodes.len() == 1 {
            nodes.pop().unwrap()
        } else {
            Node::Seq(nodes)
        }
    }

    fn at_block_end(&self) -> bool {
        matches!(self.peek(), Tok::Word(w) if ["then","elif","else","fi","do","done","esac","}",")"].contains(&w.as_str()))
            || matches!(self.peek(), Tok::Op(o) if o == ")" || o == ";;")
    }

    fn skip_terminators(&mut self) {
        while matches!(self.peek(), Tok::Op(o) if o == "\n" || o == ";") {
            self.i += 1;
        }
    }

    fn parse_and_or(&mut self) -> Node {
        let mut left = self.parse_pipeline();
        loop {
            match self.peek() {
                Tok::Op(o) if o == "&&" => {
                    self.i += 1;
                    self.skip_blank_newlines();
                    let right = self.parse_pipeline();
                    left = Node::And(Box::new(left), Box::new(right));
                }
                Tok::Op(o) if o == "||" => {
                    self.i += 1;
                    self.skip_blank_newlines();
                    let right = self.parse_pipeline();
                    left = Node::Or(Box::new(left), Box::new(right));
                }
                Tok::Op(o) if o == "&" => {
                    self.i += 1;
                    left = Node::Background(Box::new(left));
                }
                _ => break,
            }
        }
        left
    }

    fn parse_pipeline(&mut self) -> Node {
        // optional leading !
        let mut negate = false;
        if self.word_is("!") {
            negate = true;
            self.i += 1;
        }
        let mut stages = vec![self.parse_command()];
        while matches!(self.peek(), Tok::Op(o) if o == "|" || o == "|&") {
            let redirect_stderr = matches!(self.peek(), Tok::Op(o) if o == "|&");
            self.i += 1;
            if redirect_stderr {
                let stage = stages.pop().expect("a pipeline always has a left stage");
                stages.push(redirect_pipeline_stderr(stage));
            }
            self.skip_blank_newlines();
            stages.push(self.parse_command());
        }
        let node = if stages.len() == 1 {
            stages.pop().unwrap()
        } else {
            Node::Pipeline(stages)
        };
        if negate {
            // model `! cmd` as Or-trick: handled in exec via a wrapper command; simplest: wrap in subshell marker
            // We encode negation as a Pipeline of one with a sentinel; instead reuse If-free approach:
            Node::Not(Box::new(node))
        } else {
            node
        }
    }

    fn parse_command(&mut self) -> Node {
        self.skip_blank_newlines();
        if matches!(self.peek(), Tok::Eof)
            || matches!(self.peek(), Tok::Op(op) if [")", "|", "&&", "||"].contains(&op.as_str()))
        {
            self.expected("command");
            return Node::Empty;
        }
        // compound commands (may carry a trailing redirect, e.g. `while …; done < file`)
        if self.word_is("if") {
            let n = self.parse_if();
            return self.attach_redirects(n);
        }
        if self.word_is("for") {
            let n = self.parse_for();
            return self.attach_redirects(n);
        }
        if self.word_is("while") {
            let n = self.parse_while(false);
            return self.attach_redirects(n);
        }
        if self.word_is("until") {
            let n = self.parse_while(true);
            return self.attach_redirects(n);
        }
        if self.word_is("case") {
            let n = self.parse_case();
            return self.attach_redirects(n);
        }
        if self.word_is("function") {
            self.i += 1;
            return self.parse_funcdef_named();
        }
        if let Tok::Arithmetic(expression) = self.peek().clone() {
            self.i += 1;
            return self.attach_redirects(Node::Arithmetic(expression));
        }
        if matches!(self.peek(), Tok::Op(o) if o == "(") {
            self.i += 1;
            let body = self.parse_program();
            self.expect_op(")");
            return self.attach_redirects(Node::Subshell(Box::new(body)));
        }
        if self.word_is("{") {
            self.i += 1;
            let body = self.parse_program();
            self.expect_word("}");
            return self.attach_redirects(Node::Group(Box::new(body)));
        }
        // function def:  name () { ... }
        if let Tok::Word(name) = self.peek().clone() {
            if !RESERVED.contains(&name.as_str())
                && matches!(self.toks.get(self.i + 1), Some(Tok::Op(o)) if o == "(")
                && matches!(self.toks.get(self.i + 2), Some(Tok::Op(o)) if o == ")")
            {
                self.i += 3;
                self.skip_blank_newlines();
                let body = self.parse_command();
                return Node::FuncDef {
                    name,
                    body: Box::new(body),
                };
            }
        }
        if matches!(self.peek(), Tok::Word(word) if RESERVED.contains(&word.as_str()) && word != "[[")
        {
            self.expected("command");
            return Node::Empty;
        }
        self.parse_simple()
    }

    fn parse_funcdef_named(&mut self) -> Node {
        let name = if let Tok::Word(n) = self.next() {
            n
        } else {
            String::new()
        };
        // optional ()
        if matches!(self.peek(), Tok::Op(o) if o == "(") {
            self.i += 1;
            self.expect_op(")");
        }
        self.skip_blank_newlines();
        let body = self.parse_command();
        Node::FuncDef {
            name,
            body: Box::new(body),
        }
    }

    fn parse_simple(&mut self) -> Node {
        let mut assigns = Vec::new();
        let mut words = Vec::new();
        let mut redirects = Vec::new();
        // leading assignments
        loop {
            if let Tok::Word(w) = self.peek() {
                if words.is_empty() && is_assignment(w) {
                    let (k, v) = split_assignment(w);
                    assigns.push((k, v));
                    self.i += 1;
                    continue;
                }
            }
            break;
        }
        loop {
            match self.peek().clone() {
                Tok::Word(w) => {
                    if words.is_empty() && RESERVED.contains(&w.as_str()) && w != "[[" {
                        break;
                    }
                    words.push(w);
                    self.i += 1;
                }
                Tok::Less => {
                    self.i += 1;
                    let t = self.take_word();
                    redirects.push(Redirect {
                        fd: 0,
                        op: RedirOp::Read,
                        target: t,
                    });
                }
                Tok::Great => {
                    self.i += 1;
                    let t = self.take_word();
                    redirects.push(Redirect {
                        fd: 1,
                        op: RedirOp::Write,
                        target: t,
                    });
                }
                Tok::DGreat => {
                    self.i += 1;
                    let t = self.take_word();
                    redirects.push(Redirect {
                        fd: 1,
                        op: RedirOp::Append,
                        target: t,
                    });
                }
                Tok::GreatAmp(n) => {
                    self.i += 1;
                    redirects.push(Redirect {
                        fd: 1,
                        op: RedirOp::DupOut,
                        target: format!("&{n}"),
                    });
                }
                Tok::CloseOut => {
                    self.i += 1;
                    redirects.push(Redirect {
                        fd: 1,
                        op: RedirOp::Close,
                        target: "-".into(),
                    });
                }
                Tok::RedirFd(fd, op) => {
                    self.i += 1;
                    if op == "&>" || op == "&>>" {
                        let t = self.take_word();
                        redirects.push(Redirect {
                            fd: 1,
                            op: if op == "&>>" {
                                RedirOp::Append
                            } else {
                                RedirOp::Write
                            },
                            target: t.clone(),
                        });
                        redirects.push(Redirect {
                            fd: 2,
                            op: RedirOp::DupOut,
                            target: "&1".into(),
                        });
                    } else if op == ">&-" || op == "<&-" {
                        redirects.push(Redirect {
                            fd,
                            op: RedirOp::Close,
                            target: "-".into(),
                        });
                    } else if let Some(rest) = op.strip_prefix(">&") {
                        redirects.push(Redirect {
                            fd,
                            op: RedirOp::DupOut,
                            target: format!("&{rest}"),
                        });
                    } else if let Some(rest) = op.strip_prefix("<&") {
                        redirects.push(Redirect {
                            fd,
                            op: RedirOp::DupOut,
                            target: format!("&{rest}"),
                        });
                    } else if op == "<>" {
                        let t = self.take_word();
                        redirects.push(Redirect {
                            fd,
                            op: RedirOp::ReadWrite,
                            target: t,
                        });
                    } else if op == ">" {
                        let t = self.take_word();
                        redirects.push(Redirect {
                            fd,
                            op: RedirOp::Write,
                            target: t,
                        });
                    } else if op == ">>" {
                        let t = self.take_word();
                        redirects.push(Redirect {
                            fd,
                            op: RedirOp::Append,
                            target: t,
                        });
                    } else if op == "<" {
                        let t = self.take_word();
                        redirects.push(Redirect {
                            fd,
                            op: RedirOp::Read,
                            target: t,
                        });
                    }
                }
                Tok::Heredoc(body, quoted) => {
                    self.i += 1;
                    redirects.push(Redirect {
                        fd: 0,
                        op: if quoted {
                            RedirOp::HeredocRaw
                        } else {
                            RedirOp::Heredoc
                        },
                        target: body,
                    });
                }
                Tok::HereString(target) => {
                    self.i += 1;
                    redirects.push(Redirect {
                        fd: 0,
                        op: RedirOp::HereString,
                        target,
                    });
                }
                _ => break,
            }
        }
        Node::Command {
            assigns,
            words,
            redirects,
        }
    }

    fn take_word(&mut self) -> String {
        match self.peek().clone() {
            Tok::Word(w) => {
                self.i += 1;
                w
            }
            _ => {
                self.expected("word");
                String::new()
            }
        }
    }

    fn attach_redirects(&mut self, node: Node) -> Node {
        // Compound commands accept the same ordinary descriptor redirects as simple commands.
        let mut redirs = Vec::new();
        loop {
            match self.peek().clone() {
                Tok::Great => {
                    self.i += 1;
                    redirs.push(Redirect {
                        fd: 1,
                        op: RedirOp::Write,
                        target: self.take_word(),
                    });
                }
                Tok::DGreat => {
                    self.i += 1;
                    redirs.push(Redirect {
                        fd: 1,
                        op: RedirOp::Append,
                        target: self.take_word(),
                    });
                }
                Tok::Less => {
                    self.i += 1;
                    redirs.push(Redirect {
                        fd: 0,
                        op: RedirOp::Read,
                        target: self.take_word(),
                    });
                }
                Tok::GreatAmp(source) => {
                    self.i += 1;
                    redirs.push(Redirect {
                        fd: 1,
                        op: RedirOp::DupOut,
                        target: format!("&{source}"),
                    });
                }
                Tok::CloseOut => {
                    self.i += 1;
                    redirs.push(Redirect {
                        fd: 1,
                        op: RedirOp::Close,
                        target: "-".into(),
                    });
                }
                Tok::RedirFd(fd, op) => {
                    self.i += 1;
                    let (op, target) = match op.as_str() {
                        "&>" | "&>>" => {
                            let target = self.take_word();
                            redirs.push(Redirect {
                                fd: 1,
                                op: if op == "&>>" {
                                    RedirOp::Append
                                } else {
                                    RedirOp::Write
                                },
                                target,
                            });
                            redirs.push(Redirect {
                                fd: 2,
                                op: RedirOp::DupOut,
                                target: "&1".into(),
                            });
                            continue;
                        }
                        ">" => (RedirOp::Write, self.take_word()),
                        ">>" => (RedirOp::Append, self.take_word()),
                        "<" => (RedirOp::Read, self.take_word()),
                        "<>" => (RedirOp::ReadWrite, self.take_word()),
                        ">&-" | "<&-" => (RedirOp::Close, "-".into()),
                        value if value.starts_with(">&") => {
                            (RedirOp::DupOut, format!("&{}", &value[2..]))
                        }
                        value if value.starts_with("<&") => {
                            (RedirOp::DupOut, format!("&{}", &value[2..]))
                        }
                        _ => unreachable!("lexer emitted an unknown descriptor redirect"),
                    };
                    redirs.push(Redirect { fd, op, target });
                }
                _ => break,
            }
        }
        if redirs.is_empty() {
            node
        } else {
            Node::Redirected(Box::new(node), redirs)
        }
    }

    fn expect_op(&mut self, op: &str) {
        if matches!(self.peek(), Tok::Op(o) if o == op) {
            self.i += 1;
        } else {
            self.expected(&format!("`{op}`"));
        }
    }

    fn expect_word(&mut self, kw: &str) {
        self.skip_newlines();
        if self.word_is(kw) {
            self.i += 1;
        } else {
            self.expected(&format!("`{kw}`"));
        }
    }

    fn expected(&mut self, expected: &str) {
        if self.error.is_some() {
            return;
        }
        self.error = Some(match self.peek() {
            Tok::Eof => ShellError::UnexpectedEof {
                expected: expected.to_string(),
            },
            token => ShellError::UnexpectedToken {
                found: token_description(token),
                expected: expected.to_string(),
            },
        });
    }

    fn parse_if(&mut self) -> Node {
        self.i += 1; // if
        let cond = self.parse_program();
        self.expect_word("then");
        let then = self.parse_program();
        let mut elifs = Vec::new();
        let mut els = None;
        loop {
            self.skip_newlines();
            if self.word_is("elif") {
                self.i += 1;
                let c = self.parse_program();
                self.expect_word("then");
                let b = self.parse_program();
                elifs.push((c, b));
            } else if self.word_is("else") {
                self.i += 1;
                els = Some(Box::new(self.parse_program()));
            } else {
                break;
            }
        }
        self.expect_word("fi");
        Node::If {
            cond: Box::new(cond),
            then: Box::new(then),
            elifs,
            els,
        }
    }

    fn parse_for(&mut self) -> Node {
        self.i += 1; // for
        if let Tok::Arithmetic(expression) = self.peek().clone() {
            self.i += 1;
            let [init, cond, update] = split_c_for_expression(&expression);
            self.skip_terminators();
            self.expect_word("do");
            let body = self.parse_program();
            self.expect_word("done");
            return Node::CFor {
                init,
                cond,
                update,
                body: Box::new(body),
            };
        }
        let var = self.take_word();
        self.skip_newlines();
        let mut words = Vec::new();
        if self.word_is("in") {
            self.i += 1;
            while let Tok::Word(w) = self.peek().clone() {
                if RESERVED.contains(&w.as_str()) {
                    break;
                }
                words.push(w);
                self.i += 1;
            }
        } else {
            words.push("\"$@\"".to_string());
        }
        self.skip_terminators();
        self.expect_word("do");
        let body = self.parse_program();
        self.expect_word("done");
        Node::For {
            var,
            words,
            body: Box::new(body),
        }
    }

    fn parse_while(&mut self, until: bool) -> Node {
        self.i += 1;
        let cond = self.parse_program();
        self.expect_word("do");
        let body = self.parse_program();
        self.expect_word("done");
        Node::While {
            cond: Box::new(cond),
            body: Box::new(body),
            until,
        }
    }

    fn parse_case(&mut self) -> Node {
        self.i += 1; // case
        let word = self.take_word();
        self.expect_word("in");
        self.skip_newlines();
        let mut arms = Vec::new();
        while !self.word_is("esac") && !matches!(self.peek(), Tok::Eof) {
            if self.error.is_some() {
                break;
            }
            let position = self.i;
            // optional leading (
            if matches!(self.peek(), Tok::Op(o) if o == "(") {
                self.i += 1;
            }
            let mut pats = Vec::new();
            loop {
                let w = self.take_word();
                pats.push(w);
                if matches!(self.peek(), Tok::Op(o) if o == "|") {
                    self.i += 1;
                } else {
                    break;
                }
            }
            self.expect_op(")");
            self.skip_newlines();
            let body = if matches!(self.peek(), Tok::Op(operator) if operator == ";;") {
                Node::Empty
            } else {
                self.parse_program()
            };
            arms.push((pats, body));
            self.skip_newlines();
            if matches!(self.peek(), Tok::Op(o) if o == ";;") {
                self.i += 1;
            }
            self.skip_newlines();
            if self.i == position {
                self.expected("case arm");
                break;
            }
        }
        self.expect_word("esac");
        Node::Case { word, arms }
    }
}

fn split_c_for_expression(expression: &str) -> [String; 3] {
    let mut boundaries = Vec::with_capacity(2);
    let mut quote = None;
    let mut escaped = false;
    let mut parens = 0_u32;
    let mut braces = 0_u32;
    let mut brackets = 0_u32;
    for (index, ch) in expression.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && quote != Some('\'') {
            escaped = true;
            continue;
        }
        if matches!(ch, '\'' | '"' | '`') {
            quote = if quote == Some(ch) {
                None
            } else if quote.is_none() {
                Some(ch)
            } else {
                quote
            };
            continue;
        }
        if quote.is_some() {
            continue;
        }
        match ch {
            '(' => parens = parens.saturating_add(1),
            ')' => parens = parens.saturating_sub(1),
            '{' => braces = braces.saturating_add(1),
            '}' => braces = braces.saturating_sub(1),
            '[' => brackets = brackets.saturating_add(1),
            ']' => brackets = brackets.saturating_sub(1),
            ';' if parens == 0 && braces == 0 && brackets == 0 => {
                boundaries.push(index);
                if boundaries.len() == 2 {
                    break;
                }
            }
            _ => {}
        }
    }
    let first = boundaries.first().copied().unwrap_or(expression.len());
    let second = boundaries.get(1).copied().unwrap_or(expression.len());
    [
        expression[..first].trim().to_string(),
        expression[first.saturating_add(1).min(expression.len())..second]
            .trim()
            .to_string(),
        expression[second.saturating_add(1).min(expression.len())..]
            .trim()
            .to_string(),
    ]
}

fn token_description(token: &Tok) -> String {
    match token {
        Tok::Word(word) => format!("`{word}`"),
        Tok::Op(op) => format!("`{op}`"),
        Tok::Less => "`<`".to_string(),
        Tok::Great => "`>`".to_string(),
        Tok::DGreat => "`>>`".to_string(),
        Tok::Heredoc(_, _) => "heredoc".to_string(),
        Tok::Arithmetic(_) => "arithmetic command".to_string(),
        Tok::HereString(_) => "here-string".to_string(),
        Tok::GreatAmp(_) | Tok::CloseOut | Tok::RedirFd(_, _) => "redirection".to_string(),
        Tok::Eof => "end of file".to_string(),
    }
}

fn is_assignment(w: &str) -> bool {
    // Forms: name=val, name+=val, name[sub]=val, name[sub]+=val, name=( … ), name+=( … ).
    let eq = match w.find('=') {
        Some(0) | None => return false,
        Some(e) => e,
    };
    // The left side is everything before '='; strip a trailing '+' (for +=).
    let mut lhs = &w[..eq];
    if let Some(stripped) = lhs.strip_suffix('+') {
        lhs = stripped;
    }
    // Optional `[subscript]` suffix.
    let name = if let Some(br) = lhs.find('[') {
        if !lhs.ends_with(']') {
            return false;
        }
        &lhs[..br]
    } else {
        lhs
    };
    if name.is_empty() {
        return false;
    }
    name.chars()
        .enumerate()
        .all(|(i, c)| c == '_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit()))
}

fn split_assignment(w: &str) -> (String, String) {
    // The key retains any `[subscript]` and a trailing `+` (append), decoded later in exec.
    let eq = w.find('=').unwrap();
    (w[..eq].to_string(), w[eq + 1..].to_string())
}

// extra AST nodes referenced above
impl Node {}

// We added Not and Redirected variants in code; declare them here by extending enum is not
// possible after definition, so they are part of the enum below via re-export. To keep the
// single definition, add them to the enum at top. (See additions.)

/// Parse a complete shell action without executing any partial syntax tree.
pub fn parse(src: &str) -> Result<Node, ShellError> {
    let (toks, _, lexer_error) = Lexer::new(src).tokenize();
    if let Some(error) = lexer_error {
        return Err(error);
    }
    let mut p = Parser::new(toks);
    let node = p.parse_program();
    if p.error.is_none() && !matches!(p.peek(), Tok::Eof) {
        p.expected("end of input");
    }
    p.error.map_or(Ok(node), Err)
}

/// Return whether `src` ends inside a construct that more input could complete: an open
/// quote, an unterminated heredoc, or a compound command, list, or pipeline cut off at the end.
pub fn needs_more_input(src: &str) -> bool {
    !heredocs_complete(src)
        || matches!(
            parse(src),
            Err(ShellError::UnexpectedEof { .. } | ShellError::UnclosedQuote(_))
        )
}

/// Return whether every heredoc opened in `src` has its terminating delimiter.
///
/// The action console uses this narrow completeness check to collect a standard pasted heredoc
/// before executing it. Other multiline shell constructs remain complete-action inputs.
pub fn heredocs_complete(src: &str) -> bool {
    let (_, complete, _) = Lexer::new(src).tokenize();
    complete
}

// ===================== entry on Interp =====================

impl Interp {
    /// Parse and execute an action with an explicit input stream.
    ///
    /// The environment, including shell variables and the working directory, persists after the
    /// action completes. `stdin` belongs only to this action and is overridden by an explicit
    /// shell input redirect such as a pipe, `< file`, or heredoc.
    pub fn run_script_capture_with_stdin(
        &mut self,
        src: &str,
        stdin: &[u8],
    ) -> (crate::resources::RunOutcome, Vec<u8>, Vec<u8>) {
        self.run_script_capture_inner(src, stdin.to_vec())
    }

    /// Parse and execute an action with closed stdin, without writing to the host console.
    ///
    /// Use [`Interp::run_script_capture_with_stdin`] when a harness supplies input bytes for the
    /// action.
    pub fn run_script_capture(
        &mut self,
        src: &str,
    ) -> (crate::resources::RunOutcome, Vec<u8>, Vec<u8>) {
        self.run_script_capture_inner(src, Vec::new())
    }

    fn run_script_capture_inner(
        &mut self,
        src: &str,
        stdin: Vec<u8>,
    ) -> (crate::resources::RunOutcome, Vec<u8>, Vec<u8>) {
        let mut out = Vec::new();
        let mut err = Vec::new();
        if let Some(status) = self.termination_status() {
            return (self.outcome(status), out, err);
        }
        // A no-argument `python` command transfers the foreground session to the minimal REPL.
        // Keep feeding later actions to it until `exit()`/`quit()` returns control to the shell.
        if let Some(mut repl) = self.python_repl.take() {
            let cpu_before = self.resources.cpu_used();
            let disk_before = self.vfs.disk_used();
            let output_remaining = self.resources.output_remaining();
            let (code, stay) =
                crate::python::run_repl_line(self, &mut repl, src, &mut out, &mut err);
            if stay && !self.resources.is_stopped() {
                self.python_repl = Some(repl);
            }
            let produced = out.len().saturating_add(err.len()) as u64;
            if produced > output_remaining {
                let allowed_out = out.len().min(output_remaining as usize);
                out.truncate(allowed_out);
                let remaining = output_remaining.saturating_sub(allowed_out as u64) as usize;
                err.truncate(err.len().min(remaining));
            }
            let _ = self.resources.charge_output(produced);
            self.cmd_trace.record("python:repl");
            self.resources.record_command(
                "python:repl",
                cpu_before,
                disk_before,
                self.vfs.disk_used(),
            );
            self.last_status = code;
            return (self.outcome(code), out, err);
        }
        let code = match self.parse_shell_action(src) {
            Ok(ast) => crate::exec::exec(self, &ast, stdin, &mut out, &mut err),
            Err((status, diagnostic)) => {
                err.extend_from_slice(&diagnostic);
                status
            }
        };
        let status = self.exiting.unwrap_or(code);
        (self.outcome(status), out, err)
    }

    /// Parse one complete action under the shared deterministic parser budget.
    pub(crate) fn parse_shell_action(&mut self, src: &str) -> Result<Node, (i32, Vec<u8>)> {
        let parser_memory = 8 * 1024 + (src.len() as u64).saturating_mul(2);
        if !self.resources.reserve_memory(parser_memory) {
            let status = self
                .resources
                .stop_reason()
                .map_or(137, |reason| reason.exit_status());
            return Err((status, Vec::new()));
        }
        let parsed = if self.resources.charge_cpu(src.len() as u64) {
            parse(src)
                .map_err(|error| (2, format!("shellsim: syntax error: {error}\n").into_bytes()))
        } else {
            Err((
                self.resources
                    .stop_reason()
                    .map_or(137, |reason| reason.exit_status()),
                Vec::new(),
            ))
        };
        self.resources.release_memory(parser_memory);
        if let Err((status, _)) = &parsed {
            self.last_status = *status;
        }
        parsed
    }

    /// Parse and run a whole script, returning the final exit status.
    pub fn run_script(&mut self, src: &str) -> i32 {
        let (outcome, out, err) = self.run_script_capture(src);
        // anything left on stdout/stderr goes to the real console of the simulator
        self.flush_console(&out, &err);
        outcome.exit_status
    }

    fn flush_console(&mut self, out: &[u8], err: &[u8]) {
        use std::io::Write;
        if !out.is_empty() {
            let _ = std::io::stdout().write_all(out);
        }
        if !err.is_empty() {
            let _ = std::io::stderr().write_all(err);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{needs_more_input, parse, ShellError};

    #[test]
    fn stdin_programs_wait_for_complete_commands() {
        for partial in [
            "if true; then",
            "echo \"a",
            "cat <<EOF\nbody",
            "true &&",
            "f() {",
        ] {
            assert!(needs_more_input(partial), "{partial:?}");
        }
        for complete in ["echo a", "{ echo a; }", "cat <<EOF\nbody\nEOF\n", "fi"] {
            assert!(!needs_more_input(complete), "{complete:?}");
        }
    }

    #[test]
    fn incomplete_compound_commands_are_parse_errors() {
        for source in [
            "if true; then echo no",
            "for item in one; do echo no",
            "while true; do echo no",
            "until false; do echo no",
            "(echo no",
        ] {
            assert!(
                matches!(parse(source), Err(ShellError::UnexpectedEof { .. })),
                "source unexpectedly parsed: {source:?}"
            );
        }
    }

    #[test]
    fn unterminated_quotes_are_parse_errors() {
        for (source, quote) in [("echo 'no", '\''), ("echo \"no", '"'), ("echo `no", '`')] {
            assert!(matches!(
                parse(source),
                Err(ShellError::UnclosedQuote(found)) if found == quote
            ));
        }
    }

    #[test]
    fn complete_compound_commands_still_parse() {
        assert!(parse(
            "if true; then for item in one; do (echo \"$item\"); done; else echo no; fi"
        )
        .is_ok());
    }

    #[test]
    fn case_arm_terminators_stop_the_arm_body() {
        assert!(parse("case value in v*) printf yes;; *) printf no;; esac").is_ok());
        assert!(parse("case value in v*) ;; *)\n;; esac").is_ok());
    }
}
