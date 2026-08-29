//! A pragmatic bash-subset parser and executor.
//!
//! Scope includes pipelines, `&& || ; &` lists, redirects + heredocs, `if/for/while/until/case`,
//! function definitions, subshells/groups, and word expansion (quoting, `$VAR`/`${...}`
//! parameter expansion, `$(...)`/backtick command substitution, `$((...))` arithmetic,
//! tilde and globbing). It is not a complete bash, but it is faithful where it matters.

use crate::interp::Interp;

// ===================== AST =====================

#[derive(Clone, Debug)]
pub enum Node {
    Command {
        assigns: Vec<(String, String)>,
        words: Vec<String>,
        redirects: Vec<Redirect>,
    },
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
    DupOut,     // >&N  / N>&M
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
    RedirFd(i32, String),  // e.g. 2> with op ; we keep simple
    Eof,
}

struct Lexer {
    chars: Vec<char>,
    i: usize,
    toks: Vec<Tok>,
}

impl Lexer {
    fn new(src: &str) -> Self {
        Lexer {
            chars: src.chars().collect(),
            i: 0,
            toks: Vec::new(),
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.i).copied()
    }
    fn at(&self, o: usize) -> Option<char> {
        self.chars.get(self.i + o).copied()
    }

    fn tokenize(mut self) -> Vec<Tok> {
        // pending heredocs: (delim, quoted, token-index placeholder)
        let mut pending: Vec<(String, bool, usize)> = Vec::new();
        while let Some(c) = self.peek() {
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
                    if self.at(1) == Some('&') {
                        self.toks.push(Tok::Op("&&".into()));
                        self.i += 2;
                    } else if self.at(1) == Some('>') {
                        // &> file  → redirect both
                        self.i += 2;
                        self.toks.push(Tok::RedirFd(1, "&>".into()));
                    } else {
                        self.toks.push(Tok::Op("&".into()));
                        self.i += 1;
                    }
                }
                '|' => {
                    if self.at(1) == Some('|') {
                        self.toks.push(Tok::Op("||".into()));
                        self.i += 2;
                    } else {
                        self.toks.push(Tok::Op("|".into()));
                        self.i += 1;
                    }
                }
                '(' => {
                    if self.at(1) == Some('(') {
                        let expression = self.read_arithmetic_command();
                        self.toks.push(Tok::Arithmetic(expression));
                    } else {
                        self.toks.push(Tok::Op("(".into()));
                        self.i += 1;
                    }
                }
                ')' => {
                    self.toks.push(Tok::Op(")".into()));
                    self.i += 1;
                }
                '<' => {
                    if self.at(1) == Some('<') && self.at(2) == Some('<') {
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
                    if self.at(1) == Some('>') {
                        self.toks.push(Tok::DGreat);
                        self.i += 2;
                    } else if self.at(1) == Some('&') {
                        // >&N
                        self.i += 2;
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
                c if c.is_ascii_digit() && (self.at(1) == Some('>') || self.at(1) == Some('<')) => {
                    // fd-prefixed redirect like 2> 2>> 1> 2>&1
                    let fd = c.to_digit(10).unwrap() as i32;
                    self.i += 1;
                    if self.peek() == Some('>') {
                        if self.at(1) == Some('>') {
                            self.i += 2;
                            self.toks.push(Tok::RedirFd(fd, ">>".into()));
                        } else if self.at(1) == Some('&') {
                            self.i += 2;
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
                        // <
                        self.i += 1;
                        self.toks.push(Tok::RedirFd(fd, "<".into()));
                    }
                }
                _ => {
                    let w = self.read_word();
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
        self.toks.push(Tok::Eof);
        self.toks
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
                    while let Some(n) = self.peek() {
                        w.push(n);
                        self.i += 1;
                        if n == '\'' {
                            break;
                        }
                    }
                }
                '"' => {
                    w.push(c);
                    self.i += 1;
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
                            break;
                        }
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
                    while let Some(n) = self.peek() {
                        w.push(n);
                        self.i += 1;
                        if n == '`' {
                            break;
                        }
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
        out
    }
}

// ===================== Parser =====================

struct Parser {
    toks: Vec<Tok>,
    i: usize,
}

const RESERVED: &[&str] = &[
    "if", "then", "elif", "else", "fi", "for", "while", "until", "do", "done", "case", "esac",
    "in", "function", "{", "}", "!", "[[", "]]",
];

impl Parser {
    fn new(toks: Vec<Tok>) -> Self {
        Parser { toks, i: 0 }
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
            if matches!(self.peek(), Tok::Op(o) if o == ")" ) {
                break;
            }
            let n = self.parse_and_or();
            nodes.push(n);
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
        matches!(self.peek(), Tok::Word(w) if ["then","elif","else","fi","do","done","esac","}",")",";;"].contains(&w.as_str()))
            || matches!(self.peek(), Tok::Op(o) if o == ")" )
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
        while matches!(self.peek(), Tok::Op(o) if o == "|") {
            self.i += 1;
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
            if self.word_is("}") {
                self.i += 1;
            }
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
                Tok::RedirFd(fd, op) => {
                    self.i += 1;
                    if op == "&>" {
                        let t = self.take_word();
                        redirects.push(Redirect {
                            fd: 1,
                            op: RedirOp::Write,
                            target: t.clone(),
                        });
                        redirects.push(Redirect {
                            fd: 2,
                            op: RedirOp::DupOut,
                            target: "&1".into(),
                        });
                    } else if let Some(rest) = op.strip_prefix(">&") {
                        redirects.push(Redirect {
                            fd,
                            op: RedirOp::DupOut,
                            target: format!("&{rest}"),
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
        match self.next() {
            Tok::Word(w) => w,
            _ => String::new(),
        }
    }

    fn attach_redirects(&mut self, node: Node) -> Node {
        // optionally consume redirects after compound commands; we ignore for now except
        // for groups feeding pipelines (rare in corpus). Return as-is.
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
        }
    }

    fn expect_word(&mut self, kw: &str) {
        self.skip_newlines();
        if self.word_is(kw) {
            self.i += 1;
        }
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
            let mut parts = expression.splitn(3, ';');
            let init = parts.next().unwrap_or_default().trim().to_string();
            let cond = parts.next().unwrap_or_default().trim().to_string();
            let update = parts.next().unwrap_or_default().trim().to_string();
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
            let body = self.parse_program();
            arms.push((pats, body));
            self.skip_newlines();
            if matches!(self.peek(), Tok::Op(o) if o == ";;") {
                self.i += 1;
            }
            self.skip_newlines();
        }
        self.expect_word("esac");
        Node::Case { word, arms }
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

pub fn parse(src: &str) -> Node {
    let toks = Lexer::new(src).tokenize();
    let mut p = Parser::new(toks);
    p.parse_program()
}

// ===================== entry on Interp =====================

impl Interp {
    /// Parse and execute without writing to the host console.
    pub fn run_script_capture(
        &mut self,
        src: &str,
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
            self.cmd_trace.push("python:repl".to_string());
            self.resources.record_command(
                "python:repl",
                cpu_before,
                disk_before,
                self.vfs.disk_used(),
            );
            self.last_status = code;
            return (self.outcome(code), out, err);
        }
        let memory_mark = self.resources.memory_mark();
        let parser_memory = 8 * 1024 + (src.len() as u64).saturating_mul(2);
        let code = if self.resources.reserve_memory(parser_memory)
            && self.resources.charge_cpu(src.len() as u64)
        {
            let ast = parse(src);
            crate::exec::exec(self, &ast, Vec::new(), &mut out, &mut err)
        } else {
            self.resources
                .stop_reason()
                .map_or(137, |reason| reason.exit_status())
        };
        self.resources.restore_memory(memory_mark);
        let status = self.exiting.unwrap_or(code);
        (self.outcome(status), out, err)
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
