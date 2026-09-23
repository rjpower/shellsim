//! Word expansion: quoting, parameter/variable expansion, command & arithmetic
//! substitution, tilde, word-splitting and pathname globbing.

use crate::interp::{Interp, ShellExpansionError};
use crate::vfs::resolve_against;

/// One syntactic command substitution within a shell word.
pub(crate) struct CommandSubstitution {
    pub start: usize,
    pub end: usize,
    pub source: String,
}

/// Find the first command substitution that is active under shell quoting rules.
pub(crate) fn find_command_substitution(word: &str) -> Option<CommandSubstitution> {
    let bytes = word.as_bytes();
    let mut index = 0usize;
    let mut quote = None;
    while index < bytes.len() {
        match bytes[index] {
            b'\\' if quote != Some(b'\'') => index = index.saturating_add(2),
            b'\'' if quote != Some(b'"') => {
                quote = if quote == Some(b'\'') {
                    None
                } else {
                    Some(b'\'')
                };
                index += 1;
            }
            b'"' if quote != Some(b'\'') => {
                quote = if quote == Some(b'"') {
                    None
                } else {
                    Some(b'"')
                };
                index += 1;
            }
            b'`' if quote != Some(b'\'') => {
                let start = index;
                index += 1;
                let body_start = index;
                while index < bytes.len() {
                    if bytes[index] == b'\\' {
                        index = index.saturating_add(2);
                    } else if bytes[index] == b'`' {
                        return Some(CommandSubstitution {
                            start,
                            end: index + 1,
                            source: word[body_start..index].to_string(),
                        });
                    } else {
                        index += 1;
                    }
                }
            }
            b'$' if quote != Some(b'\'')
                && bytes.get(index + 1) == Some(&b'(')
                && bytes.get(index + 2) != Some(&b'(') =>
            {
                let start = index;
                let body_start = index + 2;
                index = body_start;
                let mut depth = 1usize;
                let mut inner_quote = None;
                while index < bytes.len() {
                    match bytes[index] {
                        b'\\' if inner_quote != Some(b'\'') => index = index.saturating_add(2),
                        b'\'' if inner_quote != Some(b'"') => {
                            inner_quote = if inner_quote == Some(b'\'') {
                                None
                            } else {
                                Some(b'\'')
                            };
                            index += 1;
                        }
                        b'"' if inner_quote != Some(b'\'') => {
                            inner_quote = if inner_quote == Some(b'"') {
                                None
                            } else {
                                Some(b'"')
                            };
                            index += 1;
                        }
                        b'`' if inner_quote.is_none() => {
                            index += 1;
                            while index < bytes.len() {
                                if bytes[index] == b'\\' {
                                    index = index.saturating_add(2);
                                } else if bytes[index] == b'`' {
                                    index += 1;
                                    break;
                                } else {
                                    index += 1;
                                }
                            }
                        }
                        b'(' if inner_quote.is_none() => {
                            depth += 1;
                            index += 1;
                        }
                        b')' if inner_quote.is_none() => {
                            depth -= 1;
                            if depth == 0 {
                                return Some(CommandSubstitution {
                                    start,
                                    end: index + 1,
                                    source: word[body_start..index].to_string(),
                                });
                            }
                            index += 1;
                        }
                        _ => index += 1,
                    }
                }
            }
            _ => index += 1,
        }
    }
    None
}

/// A piece of an expanding word, tracking whether it came from a quoted context
/// (quoted text is never word-split or glob-expanded).
struct Part {
    text: String,
    quoted: bool,
    has_glob: bool,
    /// Force a field boundary before this part (used by `"${arr[@]}"` to keep elements as
    /// separate words even though each is individually quoted).
    field_break: bool,
}

pub fn expand_word(interp: &mut Interp, word: &str, do_split_glob: bool) -> Vec<String> {
    let mut out = Vec::new();
    for expanded in brace_expand(word, 0) {
        let parts = expand_to_parts(interp, &expanded);
        out.extend(assemble(interp, parts, do_split_glob));
    }
    out
}

/// Expand a list of words (a command's argv) into the final field list.
pub fn expand_words(interp: &mut Interp, words: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for w in words {
        out.extend(expand_word(interp, w, true));
    }
    out
}

fn scalar_value(interp: &mut Interp, name: &str) -> String {
    match interp.get_var(name) {
        Some(value) => value,
        None => {
            if interp.opt_nounset && !matches!(name, "@" | "*") {
                interp.expansion_error.get_or_insert_with(|| {
                    ShellExpansionError::parameter(format!("shellsim: {name}: unbound variable\n"))
                });
            }
            String::new()
        }
    }
}

/// Bash-style pre-expansion for the common `{a,b}` and `{1..5[..step]}` forms. Braces inside
/// quotes and parameter expansions are left alone. The recursion and result count are bounded so
/// an adversarial expansion cannot consume unmetered host memory.
fn brace_expand(word: &str, depth: usize) -> Vec<String> {
    if depth >= 8 {
        return vec![word.to_string()];
    }
    let chars: Vec<char> = word.chars().collect();
    let mut quote = None;
    let mut escaped = false;
    let mut open = None;
    for (index, ch) in chars.iter().copied().enumerate() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        if matches!(ch, '\'' | '"') {
            quote = if quote == Some(ch) {
                None
            } else if quote.is_none() {
                Some(ch)
            } else {
                quote
            };
            continue;
        }
        if ch == '{' && quote.is_none() && (index == 0 || chars[index - 1] != '$') {
            open = Some(index);
            break;
        }
    }
    let Some(open) = open else {
        return vec![word.to_string()];
    };
    let mut nested = 0usize;
    let mut close = None;
    for (index, ch) in chars.iter().copied().enumerate().skip(open + 1) {
        match ch {
            '{' => nested += 1,
            '}' if nested > 0 => nested -= 1,
            '}' => {
                close = Some(index);
                break;
            }
            _ => {}
        }
    }
    let Some(close) = close else {
        return vec![word.to_string()];
    };
    let inner: String = chars[open + 1..close].iter().collect();
    let alternatives = brace_alternatives(&inner);
    if alternatives.is_empty() {
        return vec![word.to_string()];
    }
    let prefix: String = chars[..open].iter().collect();
    let suffix: String = chars[close + 1..].iter().collect();
    let mut out = Vec::new();
    for alternative in alternatives.into_iter().take(1_024) {
        let combined = format!("{prefix}{alternative}{suffix}");
        out.extend(brace_expand(&combined, depth + 1));
        if out.len() >= 1_024 {
            out.truncate(1_024);
            break;
        }
    }
    out
}

fn brace_alternatives(inner: &str) -> Vec<String> {
    let comma_parts = split_brace_commas(inner);
    if comma_parts.len() > 1 {
        return comma_parts;
    }
    let range: Vec<&str> = inner.split("..").collect();
    if !(2..=3).contains(&range.len()) {
        return Vec::new();
    }
    let step = range
        .get(2)
        .and_then(|value| value.parse::<i64>().ok())
        .filter(|step| *step != 0);
    if let (Ok(start), Ok(end)) = (range[0].parse::<i64>(), range[1].parse::<i64>()) {
        let width = range[0]
            .trim_start_matches('-')
            .len()
            .max(range[1].trim_start_matches('-').len());
        let step = step.unwrap_or(if start <= end { 1 } else { -1 });
        if (end - start).signum() != step.signum() && start != end {
            return Vec::new();
        }
        let mut result = Vec::new();
        let mut value = start;
        while result.len() < 1_024 && if step > 0 { value <= end } else { value >= end } {
            let sign = if value < 0 { "-" } else { "" };
            result.push(format!(
                "{sign}{:0width$}",
                value.unsigned_abs(),
                width = width
            ));
            value = value.saturating_add(step);
        }
        return result;
    }
    let start = range[0].chars().collect::<Vec<_>>();
    let end = range[1].chars().collect::<Vec<_>>();
    if start.len() == 1 && end.len() == 1 {
        let start = start[0] as i64;
        let end = end[0] as i64;
        let step = step.unwrap_or(if start <= end { 1 } else { -1 });
        let mut result = Vec::new();
        let mut value = start;
        while result.len() < 1_024 && if step > 0 { value <= end } else { value >= end } {
            if let Some(ch) = char::from_u32(value as u32) {
                result.push(ch.to_string());
            }
            value = value.saturating_add(step);
        }
        return result;
    }
    Vec::new()
}

fn split_brace_commas(inner: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (index, ch) in inner.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(inner[start..index].to_string());
                start = index + 1;
            }
            _ => {}
        }
    }
    if !parts.is_empty() {
        parts.push(inner[start..].to_string());
    }
    parts
}

fn assemble(interp: &mut Interp, parts: Vec<Part>, do_split_glob: bool) -> Vec<String> {
    let ifs = interp.get_var("IFS").unwrap_or_else(|| " \t\n".to_string());
    let mut fields: Vec<(String, bool)> = Vec::new(); // (text, has_unquoted_glob)
    let mut cur = String::new();
    let mut cur_glob = false;
    let mut started = false;
    let mut after_ifs_whitespace = false;
    for p in parts {
        if p.field_break && started {
            fields.push((std::mem::take(&mut cur), cur_glob));
            cur_glob = false;
            started = false;
            after_ifs_whitespace = false;
        }
        if p.quoted || !do_split_glob {
            cur.push_str(&p.text);
            started = true;
            after_ifs_whitespace = false;
            cur_glob |= p.has_glob && !p.quoted;
        } else {
            // POSIX distinguishes IFS whitespace from other delimiters. Whitespace runs collapse
            // and trim, while each non-whitespace delimiter can delimit an empty field.
            for c in p.text.chars() {
                if ifs.contains(c) && matches!(c, ' ' | '\t' | '\n') {
                    if started {
                        fields.push((std::mem::take(&mut cur), cur_glob));
                        cur_glob = false;
                        started = false;
                    }
                    after_ifs_whitespace = true;
                } else if ifs.contains(c) {
                    if started {
                        fields.push((std::mem::take(&mut cur), cur_glob));
                        cur_glob = false;
                        started = false;
                    } else if !after_ifs_whitespace {
                        fields.push((String::new(), false));
                    }
                    after_ifs_whitespace = false;
                } else {
                    cur.push(c);
                    started = true;
                    after_ifs_whitespace = false;
                }
            }
            cur_glob |= p.has_glob;
        }
    }
    if started {
        fields.push((cur, cur_glob));
    }
    // globbing
    let mut out = Vec::new();
    for (f, glob) in fields {
        if do_split_glob && glob {
            let matches = glob_vfs(interp, &f);
            if matches.is_empty() {
                out.push(f);
            } else {
                out.extend(matches);
            }
        } else {
            out.push(f);
        }
    }
    out
}

fn expand_to_parts(interp: &mut Interp, word: &str) -> Vec<Part> {
    let chars: Vec<char> = word.chars().collect();
    let mut i = 0;
    let mut parts: Vec<Part> = Vec::new();
    let mut buf = String::new();
    let mut buf_glob = false;
    macro_rules! flush_unquoted {
        () => {
            if !buf.is_empty() {
                parts.push(Part {
                    text: std::mem::take(&mut buf),
                    quoted: false,
                    has_glob: buf_glob,
                    field_break: false,
                });
                buf_glob = false;
            }
        };
    }
    // tilde at start
    if chars.first() == Some(&'~') {
        let home = interp
            .get_var("HOME")
            .unwrap_or_else(|| "/root".to_string());
        // ~ or ~/...
        if chars.get(1).map(|c| *c == '/').unwrap_or(true) {
            buf.push_str(&home);
            i = 1;
        }
    }
    while i < chars.len() {
        let c = chars[i];
        match c {
            '\\' => {
                if i + 1 < chars.len() {
                    buf.push(chars[i + 1]);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            '\'' => {
                i += 1;
                let start = i;
                while i < chars.len() && chars[i] != '\'' {
                    i += 1;
                }
                let s: String = chars[start..i].iter().collect();
                i += 1; // closing
                flush_unquoted!();
                parts.push(Part {
                    text: s,
                    quoted: true,
                    has_glob: false,
                    field_break: false,
                });
            }
            '"' => {
                i += 1;
                let (dparts, consumed) = expand_double(interp, &chars[i..]);
                i += consumed;
                flush_unquoted!();
                parts.extend(dparts);
            }
            '$' => {
                // `${arr[@]}` / `${arr[*]}` unquoted: elements joined by a space, then the
                // surrounding IFS word-splitting turns them into separate fields.
                if let Some((words, consumed)) = try_array_words(interp, &chars[i..], false) {
                    i += consumed;
                    if !buf.is_empty() {
                        buf.push(' ');
                    }
                    buf.push_str(&words.join(" "));
                    continue;
                }
                let (val, consumed, _glob) = expand_dollar(interp, &chars[i..], false);
                i += consumed;
                buf.push_str(&val);
            }
            '`' => {
                i += 1;
                let start = i;
                while i < chars.len() && chars[i] != '`' {
                    i += 1;
                }
                let cmd: String = chars[start..i].iter().collect();
                i += 1;
                let out = run_capture(interp, &cmd);
                buf.push_str(&out);
            }
            '*' | '?' => {
                buf.push(c);
                buf_glob = true;
                i += 1;
            }
            '[' => {
                buf.push(c);
                buf_glob = true;
                i += 1;
            }
            _ => {
                buf.push(c);
                i += 1;
            }
        }
    }
    if !buf.is_empty() {
        parts.push(Part {
            text: buf,
            quoted: false,
            has_glob: buf_glob,
            field_break: false,
        });
    }
    parts
}

/// Expand the inside of a double-quoted string until the closing quote.
/// Returns (quoted parts, chars_consumed_including_closing_quote). Normally this is a single
/// part, but `"${arr[@]}"` expands to one quoted part per element (like `"$@"`), so the result
/// can hold several parts (the surrounding text merges with the first/last as in bash).
fn expand_double(interp: &mut Interp, chars: &[char]) -> (Vec<Part>, usize) {
    let mut parts: Vec<Part> = Vec::new();
    let mut out = String::new();
    let mut i = 0;
    // Track whether the quoted string ever contributed literal text or an array element, so a
    // sole empty `"${arr[@]}"` yields zero fields (rather than one empty field).
    let mut any_content = false;
    // Whether the field currently accumulating in `out` should start a new word.
    let mut cur_break = false;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '"' => {
                i += 1;
                break;
            }
            '\\' => {
                if i + 1 < chars.len() {
                    let n = chars[i + 1];
                    if matches!(n, '"' | '\\' | '$' | '`') {
                        out.push(n);
                        i += 2;
                    } else {
                        out.push('\\');
                        i += 1;
                    }
                } else {
                    i += 1;
                }
            }
            '$' => {
                // `"${arr[@]}"` / `"${arr[*]}"`: splice each element as its own field, with the
                // preceding text joined to the first element and following text to the last.
                if let Some((words, consumed)) = try_array_words(interp, &chars[i..], true) {
                    i += consumed;
                    if !words.is_empty() {
                        any_content = true;
                        for (k, wv) in words.into_iter().enumerate() {
                            if k == 0 {
                                out.push_str(&wv);
                            } else {
                                // close current field, then begin a fresh word for this element
                                parts.push(Part {
                                    text: std::mem::take(&mut out),
                                    quoted: true,
                                    has_glob: false,
                                    field_break: cur_break,
                                });
                                cur_break = true;
                                out.push_str(&wv);
                            }
                        }
                    }
                    continue;
                }
                let (val, consumed, _) = expand_dollar(interp, &chars[i..], true);
                out.push_str(&val);
                any_content = true;
                i += consumed;
            }
            '`' => {
                i += 1;
                let start = i;
                while i < chars.len() && chars[i] != '`' {
                    i += 1;
                }
                let cmd: String = chars[start..i].iter().collect();
                i += 1;
                out.push_str(&run_capture(interp, &cmd));
                any_content = true;
            }
            _ => {
                out.push(c);
                any_content = true;
                i += 1;
            }
        }
    }
    if !any_content && parts.is_empty() && out.is_empty() {
        // sole empty array expansion → zero fields
        return (Vec::new(), i);
    }
    parts.push(Part {
        text: out,
        quoted: true,
        has_glob: false,
        field_break: cur_break,
    });
    (parts, i)
}

/// If the `$...` at `chars` is a multi-valued array expansion (`${arr[@]}`, `${arr[*]}`,
/// `${!arr[@]}`, `${!arr[*]}`, or an `@`/`*` slice), return the element/key list and the number
/// of chars consumed. Returns `None` for everything else (handled by `expand_dollar`).
fn try_array_words(
    interp: &mut Interp,
    chars: &[char],
    quoted: bool,
) -> Option<(Vec<String>, usize)> {
    if chars.first() != Some(&'$') {
        return None;
    }
    if matches!(chars.get(1), Some('@' | '*')) {
        let parameter = chars[1];
        let items = interp.positional.clone();
        if quoted && parameter == '*' {
            let ifs = interp.get_var("IFS").unwrap_or_else(|| " \t\n".to_string());
            let separator = ifs.chars().next().unwrap_or(' ').to_string();
            return Some((vec![items.join(&separator)], 2));
        }
        return Some((items, 2));
    }
    if chars.get(1) != Some(&'{') {
        return None;
    }
    let (inner, consumed) = read_balanced(&chars[1..], '{', '}');
    let total = 1 + consumed;

    // ${!name[@]} / ${!name[*]} → keys/indices
    let (want_keys, body) = match inner.strip_prefix('!') {
        Some(rest) => (true, rest),
        None => (false, inner.as_str()),
    };
    // Need a subscripted name: name[ @ | * ] possibly followed by a slice `:a:b`.
    let br = body.find('[')?;
    let name = &body[..br];
    if name.is_empty() || !is_name(name) {
        return None;
    }
    let after = &body[br + 1..];
    // subscript up to ']'
    let close = after.find(']')?;
    let sub = &after[..close];
    let tail = &after[close + 1..]; // e.g. ":1:2"
    if sub != "@" && sub != "*" {
        return None;
    }
    if !interp.is_array(name) {
        // not an array: @/* on a scalar yields the scalar (if set) as one element
        return match interp.get_var(name) {
            Some(v) if !v.is_empty() => Some((vec![v], total)),
            _ => Some((Vec::new(), total)),
        };
    }
    let mut items = if want_keys {
        interp.array_keys(name)
    } else {
        interp.array_all(name)
    };
    // optional slice `:offset:len` or `:offset`
    if let Some(spec) = tail.strip_prefix(':') {
        let (off, len) = parse_slice(interp, spec);
        let n = items.len() as i64;
        let start = if off < 0 {
            (n + off).max(0)
        } else {
            off.min(n)
        } as usize;
        items = items.into_iter().skip(start).collect();
        if let Some(l) = len {
            let l = if l < 0 {
                (items.len() as i64 + l).max(0)
            } else {
                l
            } as usize;
            items.truncate(l);
        }
    }
    // For `${arr[*]}` (unquoted) bash joins with IFS[0]; but the caller's word-splitting will
    // re-split anyway, so returning the elements is equivalent except inside quotes. When
    // quoted and `*`, join into a single element with IFS first char.
    if quoted && sub == "*" {
        let ifs = interp.get_var("IFS").unwrap_or_else(|| " \t\n".to_string());
        let sep = ifs.chars().next().unwrap_or(' ').to_string();
        return Some((vec![items.join(&sep)], total));
    }
    Some((items, total))
}

fn is_name(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .enumerate()
            .all(|(i, c)| c == '_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit()))
}

/// Parse a slice spec `offset[:len]` (each an arithmetic expression). A leading space before a
/// negative offset (`${arr[@]: -1}`) is allowed.
fn parse_slice(interp: &mut Interp, spec: &str) -> (i64, Option<i64>) {
    match spec.split_once(':') {
        Some((a, b)) => (
            eval_arith(interp, a.trim()),
            Some(eval_arith(interp, b.trim())),
        ),
        None => (eval_arith(interp, spec.trim()), None),
    }
}

/// Expand only the `$…`/`${…}`/`$(…)` substitutions in an arithmetic expression, leaving
/// operators, numbers and bare identifiers untouched (those are resolved by the arith parser).
fn expand_arith_subs(interp: &mut Interp, expr: &str) -> String {
    let chars: Vec<char> = expr.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '$' {
            let (val, consumed, _) = expand_dollar(interp, &chars[i..], true);
            out.push_str(&val);
            i += consumed;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// Expand a `$...` starting at chars[0] == '$'. Returns (value, consumed, was_glob).
fn expand_dollar(interp: &mut Interp, chars: &[char], _in_quotes: bool) -> (String, usize, bool) {
    if chars.len() < 2 {
        return ("$".to_string(), 1, false);
    }
    match chars[1] {
        '(' => {
            if chars.get(2) == Some(&'(') {
                // arithmetic $(( )): first expand `$…`/`${…}` substitutions (so `${arr[i]}`,
                // `${m[k]}` and `$var` resolve), then evaluate the resulting expression.
                let (inner, consumed) = read_double_paren(&chars[1..]);
                let expanded = expand_arith_subs(interp, &inner);
                let val = eval_arith(interp, &expanded);
                (val.to_string(), 1 + consumed, false)
            } else {
                let (inner, consumed) = read_balanced(&chars[1..], '(', ')');
                let out = run_capture(interp, &inner);
                (out, 1 + consumed, false)
            }
        }
        '{' => {
            // ${...}
            let (inner, consumed) = read_balanced(&chars[1..], '{', '}');
            let val = expand_param(interp, &inner);
            (val, 1 + consumed, false)
        }
        c if matches!(c, '?' | '$' | '#' | '@' | '*' | '!') => {
            let val = scalar_value(interp, &c.to_string());
            (val, 2, false)
        }
        c if c.is_ascii_alphabetic() || c == '_' => {
            let mut name = String::new();
            let mut i = 1;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                name.push(chars[i]);
                i += 1;
            }
            let val = scalar_value(interp, &name);
            (val, i, false)
        }
        c if c.is_ascii_digit() => {
            let mut name = String::new();
            let mut i = 1;
            while i < chars.len() && chars[i].is_ascii_digit() {
                name.push(chars[i]);
                i += 1;
            }
            let val = scalar_value(interp, &name);
            (val, i, false)
        }
        _ => ("$".to_string(), 1, false),
    }
}

/// Handle the inside of `${...}`: name plus optional operator. Single-valued result (the
/// multi-valued `@`/`*` array forms are intercepted earlier by `try_array_words`).
fn expand_param(interp: &mut Interp, inner: &str) -> String {
    // ${#name} / ${#name[@]} / ${#name[idx]}
    if let Some(rest) = inner.strip_prefix('#') {
        if let Some((name, sub)) = parse_subscript(rest) {
            if sub == "@" || sub == "*" {
                return interp.array_len(name).to_string();
            }
            let key = expand_word(interp, sub, false).join(" ");
            let v = interp.array_get(name, &key).unwrap_or_default();
            return v.chars().count().to_string();
        }
        // ${#@} → number of positional params; ${#var} → length
        if rest == "@" || rest == "*" {
            return interp.positional.len().to_string();
        }
        let v = interp.get_var(rest).unwrap_or_default();
        return v.chars().count().to_string();
    }
    // ${!name[@]} keys handled by try_array_words; here support ${!name} indirection minimally.
    // Subscripted reference: ${name[subscript]}[op...]
    if let Some((name, sub, tail)) = parse_subscript_tail(inner) {
        let key = expand_word(interp, sub, false).join(" ");
        let cur = if sub == "@" || sub == "*" {
            // single-valued context (e.g. inside $(...)): join with space
            Some(interp.array_all(name).join(" "))
        } else {
            interp.array_get(name, &key)
        };
        if tail.is_empty() {
            return cur.unwrap_or_default();
        }
        return apply_op_from_rest(interp, name, tail, cur);
    }
    // find operator
    // name is leading [A-Za-z0-9_@*] run
    let name_end = inner
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '@' || c == '*'))
        .unwrap_or(inner.len());
    let name = &inner[..name_end];
    let rest = &inner[name_end..];
    let cur = interp.get_var(name);
    if rest.is_empty() {
        return cur.unwrap_or_else(|| scalar_value(interp, name));
    }
    apply_op_from_rest(interp, name, rest, cur)
}

/// Apply the operator portion `rest` (e.g. `:-default`, `%.gz`, `:1:2`) of a `${name<rest>}`.
fn apply_op_from_rest(interp: &mut Interp, name: &str, rest: &str, cur: Option<String>) -> String {
    // substring slice `${var:offset:len}` (offset not one of the named ops)
    let ops = [
        ":-", ":=", ":+", ":?", "-", "+", "?", "##", "#", "%%", "%", "//", "/", "^^", "^", ",,",
        ",",
    ];
    for op in ops {
        if let Some(arg) = rest.strip_prefix(op) {
            let arg_expanded = expand_word(interp, arg, false).join(" ");
            return apply_param_op(interp, name, op, &arg_expanded, cur);
        }
    }
    if let Some(spec) = rest.strip_prefix(':') {
        // substring expansion ${var:offset:len}
        let v = cur.unwrap_or_default();
        let (off, len) = parse_slice(interp, spec);
        let chars: Vec<char> = v.chars().collect();
        let n = chars.len() as i64;
        let start = if off < 0 {
            (n + off).max(0)
        } else {
            off.min(n)
        } as usize;
        let end = match len {
            Some(l) if l < 0 => (n + l).max(start as i64) as usize,
            Some(l) => (start + l as usize).min(chars.len()),
            None => chars.len(),
        };
        return chars[start..end.max(start)].iter().collect();
    }
    cur.unwrap_or_default()
}

/// Parse `name[subscript]` exactly (no trailing operator). Returns (name, subscript).
fn parse_subscript(s: &str) -> Option<(&str, &str)> {
    let br = s.find('[')?;
    if !s.ends_with(']') {
        return None;
    }
    let name = &s[..br];
    if !is_name(name) {
        return None;
    }
    Some((name, &s[br + 1..s.len() - 1]))
}

/// Parse `name[subscript]<tail>` where tail is an optional operator. Returns (name, sub, tail).
fn parse_subscript_tail(s: &str) -> Option<(&str, &str, &str)> {
    let br = s.find('[')?;
    let name = &s[..br];
    if !is_name(name) {
        return None;
    }
    // find matching ']' (subscripts here don't nest brackets in practice)
    let close = s[br + 1..].find(']')? + br + 1;
    Some((name, &s[br + 1..close], &s[close + 1..]))
}

fn apply_param_op(
    interp: &mut Interp,
    name: &str,
    op: &str,
    arg: &str,
    cur: Option<String>,
) -> String {
    let is_set = cur.as_ref().map(|s| !s.is_empty()).unwrap_or(false);
    let exists = cur.is_some();
    match op {
        ":-" => {
            if is_set {
                cur.unwrap()
            } else {
                arg.to_string()
            }
        }
        "-" => {
            if exists {
                cur.unwrap()
            } else {
                arg.to_string()
            }
        }
        ":=" => {
            if is_set {
                cur.unwrap()
            } else {
                interp.set_var(name, arg);
                arg.to_string()
            }
        }
        ":+" => {
            if is_set {
                arg.to_string()
            } else {
                String::new()
            }
        }
        "+" => {
            if exists {
                arg.to_string()
            } else {
                String::new()
            }
        }
        ":?" => {
            if is_set {
                cur.unwrap()
            } else {
                let message = if arg.is_empty() {
                    format!("{name}: parameter null or not set")
                } else {
                    arg.to_string()
                };
                interp.expansion_error.get_or_insert_with(|| {
                    ShellExpansionError::parameter(format!("shellsim: {message}\n"))
                });
                String::new()
            }
        }
        "?" => {
            if exists {
                cur.unwrap()
            } else {
                let message = if arg.is_empty() {
                    format!("{name}: parameter not set")
                } else {
                    arg.to_string()
                };
                interp.expansion_error.get_or_insert_with(|| {
                    ShellExpansionError::parameter(format!("shellsim: {message}\n"))
                });
                String::new()
            }
        }
        "#" => strip_prefix_glob(&cur.unwrap_or_default(), arg, false),
        "##" => strip_prefix_glob(&cur.unwrap_or_default(), arg, true),
        "%" => strip_suffix_glob(&cur.unwrap_or_default(), arg, false),
        "%%" => strip_suffix_glob(&cur.unwrap_or_default(), arg, true),
        "/" | "//" => {
            let v = cur.unwrap_or_default();
            let (pat, rep) = match arg.split_once('/') {
                Some((p, r)) => (p.to_string(), r.to_string()),
                None => (arg.to_string(), String::new()),
            };
            if pat.is_empty() {
                return v;
            }
            if op == "//" {
                v.replace(&pat, &rep)
            } else {
                v.replacen(&pat, &rep, 1)
            }
        }
        "^^" => cur.unwrap_or_default().to_uppercase(),
        "^" => uppercase_first(&cur.unwrap_or_default()),
        ",," => cur.unwrap_or_default().to_lowercase(),
        "," => lowercase_first(&cur.unwrap_or_default()),
        _ => cur.unwrap_or_default(),
    }
}

fn uppercase_first(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}
fn lowercase_first(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_lowercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

fn glob_to_regex(pat: &str) -> String {
    let mut re = String::from("^");
    for c in pat.chars() {
        match c {
            '*' => re.push_str(".*"),
            '?' => re.push('.'),
            '.' | '+' | '(' | ')' | '|' | '^' | '$' | '{' | '}' | '\\' => {
                re.push('\\');
                re.push(c);
            }
            _ => re.push(c),
        }
    }
    re.push('$');
    re
}

fn strip_prefix_glob(s: &str, pat: &str, greedy: bool) -> String {
    let Ok(pattern) = regex::Regex::new(&glob_to_regex(pat)) else {
        return s.to_string();
    };
    let mut boundaries = s.char_indices().map(|(index, _)| index).collect::<Vec<_>>();
    boundaries.push(s.len());
    if greedy {
        boundaries.reverse();
    }
    for boundary in boundaries {
        if pattern.is_match(&s[..boundary]) {
            return s[boundary..].to_string();
        }
    }
    s.to_string()
}

fn strip_suffix_glob(s: &str, pat: &str, greedy: bool) -> String {
    let Ok(pattern) = regex::Regex::new(&glob_to_regex(pat)) else {
        return s.to_string();
    };
    let mut boundaries = s.char_indices().map(|(index, _)| index).collect::<Vec<_>>();
    boundaries.push(s.len());
    if !greedy {
        boundaries.reverse();
    }
    for boundary in boundaries {
        if pattern.is_match(&s[boundary..]) {
            return s[..boundary].to_string();
        }
    }
    s.to_string()
}

fn read_balanced(chars: &[char], open: char, close: char) -> (String, usize) {
    // chars[0] == open
    let mut depth = 0;
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == open {
            depth += 1;
            if depth > 1 {
                out.push(c);
            }
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                i += 1;
                break;
            }
            out.push(c);
        } else {
            out.push(c);
        }
        i += 1;
    }
    (out, i)
}

fn read_double_paren(chars: &[char]) -> (String, usize) {
    // chars[0..2] == "((" ; read until "))"
    let mut out = String::new();
    let mut i = 2;
    let mut depth = 1;
    while i < chars.len() {
        if chars[i] == '(' {
            depth += 1;
            out.push('(');
            i += 1;
        } else if chars[i] == ')' {
            depth -= 1;
            if depth == 0 {
                // expect another ')'
                i += 1;
                if chars.get(i) == Some(&')') {
                    i += 1;
                }
                break;
            }
            out.push(')');
            i += 1;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    (out, i)
}

fn run_capture(interp: &mut Interp, src: &str) -> String {
    interp.last_status = 125;
    interp.pending_stderr.extend_from_slice(
        format!(
            "shellsim: internal error: command substitution reached synchronous expansion: {src}\n"
        )
        .as_bytes(),
    );
    String::new()
}

pub fn eval_arith(interp: &mut Interp, expr: &str) -> i64 {
    let expr = expr.trim();
    if expr.is_empty() {
        return 0;
    }
    // Bash arithmetic commands commonly mutate a loop variable. Handle the useful assignment
    // and increment forms before handing pure expressions to the precedence parser below.
    if let Some(name) = expr.strip_suffix("++").map(str::trim) {
        if is_name(name) {
            let old = interp
                .get_var(name)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0i64);
            interp.set_var(name, old.saturating_add(1).to_string());
            return old;
        }
    }
    if let Some(name) = expr.strip_suffix("--").map(str::trim) {
        if is_name(name) {
            let old = interp
                .get_var(name)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0i64);
            interp.set_var(name, old.saturating_sub(1).to_string());
            return old;
        }
    }
    for (prefix, delta) in [("++", 1i64), ("--", -1i64)] {
        if let Some(name) = expr.strip_prefix(prefix).map(str::trim) {
            if is_name(name) {
                let old = interp
                    .get_var(name)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0i64);
                let value = old.saturating_add(delta);
                interp.set_var(name, value.to_string());
                return value;
            }
        }
    }
    if let Some((name, op, rhs)) = arithmetic_assignment(expr) {
        let rhs = eval_arith(interp, rhs);
        let old = interp
            .get_var(name)
            .and_then(|v| v.parse().ok())
            .unwrap_or(0i64);
        let value = match op {
            "=" => rhs,
            "+=" => old.saturating_add(rhs),
            "-=" => old.saturating_sub(rhs),
            "*=" => old.saturating_mul(rhs),
            "/=" if rhs != 0 => old / rhs,
            "%=" if rhs != 0 => old % rhs,
            _ => old,
        };
        interp.set_var(name, value.to_string());
        return value;
    }
    let mut p = ArithParser {
        interp,
        chars: expr.chars().collect(),
        i: 0,
    };
    p.expr()
}

fn arithmetic_assignment(expr: &str) -> Option<(&str, &str, &str)> {
    let bytes = expr.as_bytes();
    let mut depth = 0usize;
    for i in 0..bytes.len() {
        match bytes[i] {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth = depth.saturating_sub(1),
            _ if depth != 0 => continue,
            _ => {}
        }
        for op in ["+=", "-=", "*=", "/=", "%=", "="] {
            if expr[i..].starts_with(op) {
                if op == "="
                    && (i > 0 && matches!(bytes[i - 1], b'!' | b'<' | b'>' | b'=')
                        || bytes.get(i + 1) == Some(&b'='))
                {
                    continue;
                }
                let name = expr[..i].trim();
                if is_name(name) {
                    return Some((name, op, expr[i + op.len()..].trim()));
                }
            }
        }
    }
    None
}

struct ArithParser<'a> {
    interp: &'a mut Interp,
    chars: Vec<char>,
    i: usize,
}

impl ArithParser<'_> {
    fn skip_ws(&mut self) {
        while self.i < self.chars.len() && self.chars[self.i].is_whitespace() {
            self.i += 1;
        }
    }
    fn peek(&mut self) -> Option<char> {
        self.skip_ws();
        self.chars.get(self.i).copied()
    }
    fn expr(&mut self) -> i64 {
        self.ternary()
    }
    fn ternary(&mut self) -> i64 {
        let c = self.logical_or();
        if self.peek() == Some('?') {
            self.i += 1;
            let a = self.logical_or();
            self.skip_ws();
            if self.peek() == Some(':') {
                self.i += 1;
            }
            let b = self.logical_or();
            if c != 0 {
                a
            } else {
                b
            }
        } else {
            c
        }
    }

    fn logical_or(&mut self) -> i64 {
        let mut value = self.logical_and();
        while self.consume("||") {
            let right = self.logical_and();
            value = i64::from(value != 0 || right != 0);
        }
        value
    }

    fn logical_and(&mut self) -> i64 {
        let mut value = self.comparison();
        while self.consume("&&") {
            let right = self.comparison();
            value = i64::from(value != 0 && right != 0);
        }
        value
    }

    fn comparison(&mut self) -> i64 {
        let mut value = self.add_sub();
        loop {
            let op = ["==", "!=", "<=", ">=", "<", ">"]
                .into_iter()
                .find(|op| self.starts_with(op));
            let Some(op) = op else { break };
            self.i += op.len();
            let right = self.add_sub();
            value = i64::from(match op {
                "==" => value == right,
                "!=" => value != right,
                "<=" => value <= right,
                ">=" => value >= right,
                "<" => value < right,
                ">" => value > right,
                _ => false,
            });
        }
        value
    }

    fn starts_with(&mut self, value: &str) -> bool {
        self.skip_ws();
        let wanted: Vec<char> = value.chars().collect();
        self.chars.get(self.i..self.i + wanted.len()) == Some(wanted.as_slice())
    }

    fn consume(&mut self, value: &str) -> bool {
        if self.starts_with(value) {
            self.i += value.chars().count();
            true
        } else {
            false
        }
    }
    fn add_sub(&mut self) -> i64 {
        let mut v = self.mul_div();
        loop {
            match self.peek() {
                Some('+') => {
                    self.i += 1;
                    v += self.mul_div();
                }
                Some('-') => {
                    self.i += 1;
                    v -= self.mul_div();
                }
                _ => break,
            }
        }
        v
    }
    fn mul_div(&mut self) -> i64 {
        let mut v = self.unary();
        loop {
            match self.peek() {
                Some('*') => {
                    self.i += 1;
                    v *= self.unary();
                }
                Some('/') => {
                    self.i += 1;
                    let d = self.unary();
                    if d != 0 {
                        v /= d;
                    }
                }
                Some('%') => {
                    self.i += 1;
                    let d = self.unary();
                    if d != 0 {
                        v %= d;
                    }
                }
                _ => break,
            }
        }
        v
    }
    fn unary(&mut self) -> i64 {
        match self.peek() {
            Some('-') => {
                self.i += 1;
                -self.unary()
            }
            Some('+') => {
                self.i += 1;
                self.unary()
            }
            Some('!') => {
                self.i += 1;
                if self.unary() == 0 {
                    1
                } else {
                    0
                }
            }
            _ => self.atom(),
        }
    }
    fn atom(&mut self) -> i64 {
        self.skip_ws();
        match self.peek() {
            Some('(') => {
                self.i += 1;
                let v = self.expr();
                self.skip_ws();
                if self.peek() == Some(')') {
                    self.i += 1;
                }
                v
            }
            Some(c) if c.is_ascii_digit() => {
                let mut n = String::new();
                while let Some(d) = self.chars.get(self.i) {
                    if d.is_ascii_digit() {
                        n.push(*d);
                        self.i += 1;
                    } else {
                        break;
                    }
                }
                n.parse().unwrap_or(0)
            }
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {
                let mut name = String::new();
                while let Some(d) = self.chars.get(self.i) {
                    if d.is_ascii_alphanumeric() || *d == '_' {
                        name.push(*d);
                        self.i += 1;
                    } else {
                        break;
                    }
                }
                // bare array subscript inside arithmetic: name[index]
                if self.chars.get(self.i) == Some(&'[') {
                    self.i += 1;
                    let idx = self.expr();
                    if self.peek() == Some(']') {
                        self.i += 1;
                    }
                    return self
                        .interp
                        .array_get(&name, &idx.to_string())
                        .and_then(|v| v.trim().parse().ok())
                        .unwrap_or(0);
                }
                self.interp
                    .get_var(&name)
                    .and_then(|v| v.trim().parse().ok())
                    .unwrap_or(0)
            }
            Some('$') => {
                self.i += 1;
                self.atom()
            }
            _ => 0,
        }
    }
}

fn glob_vfs(interp: &Interp, pattern: &str) -> Vec<String> {
    // Only glob the basename components that contain metacharacters, against the VFS.
    // Split pattern into directory part and a per-component glob walk.
    let absolute = pattern.starts_with('/');
    let base = if absolute {
        "/".to_string()
    } else {
        interp.cwd.clone()
    };
    let comps: Vec<&str> = pattern.split('/').filter(|c| !c.is_empty()).collect();
    let mut current = vec![base];
    for comp in &comps {
        let mut next = Vec::new();
        let has_meta = comp.contains('*') || comp.contains('?') || comp.contains('[');
        for dir in &current {
            if has_meta {
                if let Ok(entries) = interp.vfs.list_dir("/", dir) {
                    let re = regex::Regex::new(&glob_to_regex(comp)).ok();
                    for e in entries {
                        // skip hidden unless pattern starts with .
                        if e.starts_with('.') && !comp.starts_with('.') {
                            continue;
                        }
                        let matched = re.as_ref().map(|r| r.is_match(&e)).unwrap_or(false);
                        if matched {
                            next.push(format!("{}/{}", dir.trim_end_matches('/'), e));
                        }
                    }
                }
            } else {
                let cand = format!("{}/{}", dir.trim_end_matches('/'), comp);
                if interp.vfs.lexists("/", &cand) {
                    next.push(cand);
                }
            }
        }
        current = next;
        if current.is_empty() {
            return Vec::new();
        }
    }
    // produce results relative to cwd if pattern was relative
    let mut out: Vec<String> = current
        .into_iter()
        .map(|p| {
            if absolute {
                p
            } else {
                let prefix = format!("{}/", interp.cwd.trim_end_matches('/'));
                p.strip_prefix(&prefix).map(|s| s.to_string()).unwrap_or(p)
            }
        })
        .collect();
    out.sort();
    let _ = resolve_against; // keep import used
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(src: &str) -> Vec<String> {
        let mut i = Interp::new();
        i.set_var("FOO", "bar");
        i.set_var("EMPTY", "");
        expand_word(&mut i, src, true)
    }

    #[test]
    fn vars_and_quotes() {
        assert_eq!(ex("$FOO"), vec!["bar"]);
        assert_eq!(ex("\"$FOO\""), vec!["bar"]);
        assert_eq!(ex("'$FOO'"), vec!["$FOO"]);
        assert_eq!(ex("a${FOO}b"), vec!["abarb"]);
        assert_eq!(ex("${EMPTY:-def}"), vec!["def"]);
        assert_eq!(ex("${MISSING:-x}"), vec!["x"]);
        assert_eq!(ex("${#FOO}"), vec!["3"]);
    }

    #[test]
    fn arithmetic() {
        assert_eq!(ex("$((1+2*3))"), vec!["7"]);
        assert_eq!(ex("$(( (1+2)*3 ))"), vec!["9"]);
    }

    #[test]
    fn suffix_prefix() {
        let mut i = Interp::new();
        i.set_var("F", "file.tar.gz");
        assert_eq!(expand_word(&mut i, "${F%.gz}", true), vec!["file.tar"]);
        assert_eq!(expand_word(&mut i, "${F%%.*}", true), vec!["file"]);
        assert_eq!(expand_word(&mut i, "${F#*.}", true), vec!["tar.gz"]);
        assert_eq!(expand_word(&mut i, "${F##*.}", true), vec!["gz"]);
    }

    #[test]
    fn non_whitespace_ifs_delimiters_preserve_empty_fields() {
        let mut i = Interp::new();
        i.set_var("IFS", ":");
        i.set_var("VALUE", ":a::b:");
        assert_eq!(expand_word(&mut i, "$VALUE", true), vec!["", "a", "", "b"]);
    }

    #[test]
    fn parameter_errors_request_shell_abort_with_failure_status() {
        let mut i = Interp::new();
        assert!(expand_word(&mut i, "${MISSING:?required}", true).is_empty());
        let error = i.expansion_error.take().unwrap();
        assert_eq!(error.status, 1);
        assert!(error.abort_shell);
        assert!(error.message.contains("required"));
    }

    #[test]
    fn command_substitution_scanner_obeys_quotes_and_nesting() {
        assert!(find_command_substitution("'$(ignored)'").is_none());
        assert!(find_command_substitution("$((1 + 2))").is_none());
        let found = find_command_substitution("prefix$(printf '%s' \"a)b\")suffix").unwrap();
        assert_eq!(found.source, "printf '%s' \"a)b\"");
        assert_eq!(
            &"prefix$(printf '%s' \"a)b\")suffix"[found.start..found.end],
            "$(printf '%s' \"a)b\")"
        );
        assert_eq!(
            find_command_substitution("x`echo y`z").unwrap().source,
            "echo y"
        );
    }
}
