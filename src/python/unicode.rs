//! Python's Unicode character classes and case mappings for `str` methods.
//!
//! Rust's `char` methods provide the Uppercase, Lowercase and full case-mapping data that
//! CPython also uses, but not the numeric types, Python's whitespace set, the title-case
//! category and mappings, or Case_Ignorable. [`tables`] records those from CPython's Unicode
//! database; regenerate it with `infra/generate-unicode-tables.py`.
//!
//! The case transforms follow `Objects/unicodeobject.c`: `title` and `capitalize` title-case
//! a letter that does not follow a cased letter and lower-case the rest, and every lower-case
//! mapping of capital sigma applies the final-sigma rule.

mod tables;

/// Whether `character` falls in one of the sorted inclusive `ranges`.
fn in_ranges(ranges: &[(u32, u32)], character: char) -> bool {
    let code = u32::from(character);
    ranges
        .binary_search_by(|&(start, end)| {
            if end < code {
                std::cmp::Ordering::Less
            } else if start > code {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// `str.isdecimal` for one character: Numeric_Type=Decimal.
pub(super) fn is_decimal(character: char) -> bool {
    in_ranges(tables::DECIMAL, character)
}

/// `str.isdigit` for one character: Numeric_Type Decimal or Digit, which includes
/// superscripts that `isdecimal` rejects.
pub(super) fn is_digit(character: char) -> bool {
    in_ranges(tables::DIGIT, character)
}

/// `str.isnumeric` for one character: any Numeric_Type, including vulgar fractions and CJK
/// numerals.
pub(super) fn is_numeric(character: char) -> bool {
    in_ranges(tables::NUMERIC, character)
}

/// `str.isspace` for one character. Python counts the ASCII separators U+001C..U+001F, which
/// Rust's White_Space-based `char::is_whitespace` does not.
pub(super) fn is_space(character: char) -> bool {
    in_ranges(tables::SPACE, character)
}

/// `str.isalnum` for one character.
pub(super) fn is_alphanumeric(character: char) -> bool {
    character.is_alphabetic() || is_numeric(character)
}

fn is_title_letter(character: char) -> bool {
    in_ranges(tables::TITLE_LETTER, character)
}

/// Unicode's Cased property, which decides where words begin for `title` and `istitle`.
fn is_cased(character: char) -> bool {
    character.is_uppercase() || character.is_lowercase() || is_title_letter(character)
}

fn push_title(character: char, output: &mut String) {
    let code = u32::from(character);
    match tables::TITLE_CASE.binary_search_by_key(&code, |&(key, _)| key) {
        Ok(index) => output.push_str(tables::TITLE_CASE[index].1),
        Err(_) => output.extend(character.to_uppercase()),
    }
}

/// Push the lower-case mapping of `characters[index]`, choosing final sigma for a capital
/// sigma that ends a cased word.
fn push_lower(characters: &[char], index: usize, output: &mut String) {
    let character = characters[index];
    if character != '\u{03a3}' {
        output.extend(character.to_lowercase());
        return;
    }
    let significant = |character: &&char| !in_ranges(tables::CASE_IGNORABLE, **character);
    let follows_cased = characters[..index]
        .iter()
        .rev()
        .find(significant)
        .is_some_and(|character| is_cased(*character));
    let precedes_cased = characters[index + 1..]
        .iter()
        .find(significant)
        .is_some_and(|character| is_cased(*character));
    output.push(if follows_cased && !precedes_cased {
        '\u{03c2}'
    } else {
        '\u{03c3}'
    });
}

/// `str.title`: upper-case the first cased letter of each run of cased letters.
///
/// ```text
/// title("hello wORLD 3rd") == "Hello World 3Rd"
/// ```
pub(super) fn title(text: &str) -> String {
    let characters = text.chars().collect::<Vec<_>>();
    let mut output = String::with_capacity(text.len());
    let mut previous_cased = false;
    for (index, &character) in characters.iter().enumerate() {
        if previous_cased {
            push_lower(&characters, index, &mut output);
        } else {
            push_title(character, &mut output);
        }
        previous_cased = is_cased(character);
    }
    output
}

/// `str.capitalize`: title-case the first character and lower-case the rest.
pub(super) fn capitalize(text: &str) -> String {
    let characters = text.chars().collect::<Vec<_>>();
    let mut output = String::with_capacity(text.len());
    for (index, &character) in characters.iter().enumerate() {
        if index == 0 {
            push_title(character, &mut output);
        } else {
            push_lower(&characters, index, &mut output);
        }
    }
    output
}

/// `str.swapcase`. Title-case letters such as U+01C5 are neither upper nor lower case, so they
/// stay unchanged.
pub(super) fn swapcase(text: &str) -> String {
    let characters = text.chars().collect::<Vec<_>>();
    let mut output = String::with_capacity(text.len());
    for (index, &character) in characters.iter().enumerate() {
        if character.is_uppercase() {
            push_lower(&characters, index, &mut output);
        } else if character.is_lowercase() {
            output.extend(character.to_uppercase());
        } else {
            output.push(character);
        }
    }
    output
}

/// `str.istitle`: every run of cased letters starts with an upper- or title-case letter and
/// continues in lower case, and at least one cased letter exists.
pub(super) fn is_title(text: &str) -> bool {
    let mut cased = false;
    let mut previous_cased = false;
    for character in text.chars() {
        if character.is_uppercase() || is_title_letter(character) {
            if previous_cased {
                return false;
            }
            previous_cased = true;
            cased = true;
        } else if character.is_lowercase() {
            if !previous_cased {
                return false;
            }
            previous_cased = true;
            cased = true;
        } else {
            previous_cased = false;
        }
    }
    cased
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_classes_follow_numeric_type() {
        assert!(is_decimal('7') && is_digit('7') && is_numeric('7'));
        assert!(!is_decimal('\u{00b2}') && is_digit('\u{00b2}') && is_numeric('\u{00b2}'));
        assert!(!is_digit('\u{00bd}') && is_numeric('\u{00bd}'));
        assert!(is_numeric('\u{4e00}') && !is_numeric('a'));
    }

    #[test]
    fn python_whitespace_includes_ascii_separators() {
        assert!(is_space('\u{1c}') && is_space('\u{3000}') && !is_space('\u{200b}'));
    }

    #[test]
    fn title_uses_title_case_mappings_and_final_sigma() {
        assert_eq!(
            title("hello wORLD 3rd o\u{2019}neil"),
            "Hello World 3Rd O\u{2019}Neil"
        );
        assert_eq!(title("\u{01c6}a \u{fb01}x \u{00df}"), "\u{01c5}a Fix Ss");
        assert_eq!(
            title("\u{03a3}\u{0391}\u{03a3}"),
            "\u{03a3}\u{03b1}\u{03c2}"
        );
        assert_eq!(
            capitalize("\u{03a3}\u{0391}\u{03a3}. x"),
            "\u{03a3}\u{03b1}\u{03c2}. x"
        );
        assert_eq!(
            capitalize("\u{03a3}\u{0391}\u{03a3}.x"),
            "\u{03a3}\u{03b1}\u{03c3}.x"
        );
        assert_eq!(
            capitalize("\u{0391}\u{03a3}'\u{03b1}"),
            "\u{0391}\u{03c3}'\u{03b1}"
        );
    }

    #[test]
    fn swapcase_and_istitle_treat_title_letters_as_cased() {
        assert_eq!(swapcase("aB\u{00df}\u{01c5}"), "Ab\u{53}\u{53}\u{01c5}");
        assert!(is_title("\u{01c5}a Ab"));
        assert!(!is_title("AB") && !is_title("") && !is_title("aB"));
    }
}
