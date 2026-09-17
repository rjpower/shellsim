//! Small compatibility translations between POSIX utility regexes and Rust regex syntax.

/// Translate commonly used POSIX basic regular-expression operators to Rust regex syntax.
///
/// BRE makes the extended operators literal unless escaped. Backreferences remain unsupported
/// because Rust's linear-time regex engine does not implement them.
pub(crate) fn basic_regex_to_rust(pattern: &str) -> Result<String, String> {
    let mut translated = String::with_capacity(pattern.len());
    let mut characters = pattern.chars();
    let mut in_class = false;
    while let Some(character) = characters.next() {
        if character == '\\' {
            let Some(escaped) = characters.next() else {
                return Err("trailing backslash".to_string());
            };
            if !in_class && escaped.is_ascii_digit() {
                return Err("backreferences are not supported".to_string());
            }
            if !in_class && matches!(escaped, '+' | '?' | '|' | '(' | ')' | '{' | '}') {
                translated.push(escaped);
            } else if !in_class && matches!(escaped, '<' | '>') {
                translated.push_str(r"\b");
            } else {
                translated.push('\\');
                translated.push(escaped);
            }
            continue;
        }
        if character == '[' && !in_class {
            in_class = true;
        } else if character == ']' && in_class {
            in_class = false;
        }
        if !in_class && matches!(character, '+' | '?' | '|' | '(' | ')' | '{' | '}') {
            translated.push('\\');
        }
        translated.push(character);
    }
    Ok(translated)
}

#[cfg(test)]
mod tests {
    use super::basic_regex_to_rust;

    #[test]
    fn preserves_the_bre_boundary() {
        assert_eq!(basic_regex_to_rust("a+b?(c)").unwrap(), r"a\+b\?\(c\)");
        assert_eq!(basic_regex_to_rust(r"a\+b\{2,3\}").unwrap(), "a+b{2,3}");
        assert_eq!(basic_regex_to_rust(r"\<word\>").unwrap(), r"\bword\b");
        assert_eq!(
            basic_regex_to_rust(r"\(a\)\1").unwrap_err(),
            "backreferences are not supported"
        );
    }
}
