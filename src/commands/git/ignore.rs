//! A small `.gitignore` matcher for the simulated Git porcelain.
//!
//! The matcher supports the rules agents actually write: comments, blank lines, negation with
//! `!`, directory-only patterns with a trailing `/`, anchoring with an embedded `/`, and the
//! `*`, `?`, and `**` wildcards. Ranges, `\` escapes, and `.git/info/exclude` are out of scope.
//! Pattern files are read once per command and bounded so that a hostile tree cannot make
//! matching unbounded.

use crate::interp::Interp;
use crate::vfs::NodeKind;

/// The most pattern files and patterns one command will consider.
const MAX_FILES: usize = 64;
const MAX_PATTERNS: usize = 1_024;

/// One `.gitignore` line, pre-classified.
#[derive(Debug)]
struct Pattern {
    /// Directory the owning `.gitignore` lives in, relative to the repository root, with a
    /// trailing `/` unless it is the root itself.
    base: String,
    text: String,
    negated: bool,
    directory_only: bool,
    anchored: bool,
    /// Where the line came from, as `git check-ignore -v` reports it.
    source: String,
    line: usize,
    original: String,
}

/// All ignore patterns that apply inside one repository.
#[derive(Debug, Default)]
pub(crate) struct IgnoreRules {
    patterns: Vec<Pattern>,
}

impl IgnoreRules {
    /// Whether `relative` (a repository-relative file path) is ignored.
    ///
    /// A file is ignored when it matches a pattern directly or when any ancestor directory
    /// matches a directory pattern. The last matching pattern wins, so a later `!` line
    /// re-includes a path an earlier line excluded.
    pub fn is_ignored(&self, relative: &str) -> bool {
        self.decide(relative)
            .is_some_and(|pattern| !pattern.negated)
    }

    /// The `source:line:pattern` description `git check-ignore -v` prints for an ignored path.
    pub fn describe(&self, relative: &str) -> Option<String> {
        let pattern = self.decide(relative).filter(|pattern| !pattern.negated)?;
        Some(format!(
            "{}:{}:{}",
            pattern.source, pattern.line, pattern.original
        ))
    }

    /// The pattern that settles `relative`, which is the last one to match it.
    fn decide(&self, relative: &str) -> Option<&Pattern> {
        let mut decision: Option<&Pattern> = None;
        for (candidate, is_directory) in ancestor_candidates(relative) {
            for pattern in &self.patterns {
                if pattern.directory_only && !is_directory {
                    continue;
                }
                if pattern.matches(&candidate) {
                    decision = Some(pattern);
                }
            }
            // Git does not descend into an ignored directory, so nothing below it can be
            // re-included by a later pattern.
            if is_directory && decision.is_some_and(|pattern| !pattern.negated) {
                return decision;
            }
        }
        decision
    }
}

/// Yield each ancestor directory of `relative` and then the path itself.
fn ancestor_candidates(relative: &str) -> Vec<(String, bool)> {
    let mut out = Vec::new();
    let mut prefix = String::new();
    let components: Vec<&str> = relative.split('/').collect();
    for component in &components[..components.len().saturating_sub(1)] {
        prefix.push_str(component);
        out.push((prefix.clone(), true));
        prefix.push('/');
    }
    out.push((relative.to_string(), false));
    out
}

impl Pattern {
    fn matches(&self, candidate: &str) -> bool {
        let Some(rest) = candidate.strip_prefix(self.base.as_str()) else {
            return false;
        };
        if self.anchored {
            return glob_match(&self.text, rest);
        }
        // An unanchored pattern matches at any depth below its own directory.
        if glob_match(&self.text, rest) {
            return true;
        }
        rest.match_indices('/')
            .any(|(index, _)| glob_match(&self.text, &rest[index + 1..]))
    }
}

/// Match one `.gitignore` pattern against a path fragment.
///
/// `*`, `?`, and `[abc]` stop at a `/`; `**` crosses separators. Backtracking is linear in the
/// text length per wildcard, which is bounded by the pattern and path limits the caller enforces.
pub(crate) fn glob_match(pattern: &str, text: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let text: Vec<char> = text.chars().collect();
    matches_from(&pattern, &text)
}

fn matches_from(pattern: &[char], text: &[char]) -> bool {
    let mut p = 0;
    let mut t = 0;
    // Backtracking state for the most recent single-star, which cannot cross `/`.
    let mut star: Option<(usize, usize)> = None;
    while t < text.len() {
        if p < pattern.len() {
            match pattern[p] {
                '*' if pattern.get(p + 1) == Some(&'*') => {
                    let mut next = p + 2;
                    if pattern.get(next) == Some(&'/') {
                        next += 1;
                    }
                    // `**` may consume any number of characters, separators included.
                    for skip in 0..=text.len() - t {
                        if matches_from(&pattern[next..], &text[t + skip..]) {
                            return true;
                        }
                    }
                    return false;
                }
                '*' => {
                    star = Some((p, t));
                    p += 1;
                    continue;
                }
                '[' if text[t] != '/' && class_end(pattern, p).is_some() => {
                    let end = class_end(pattern, p).unwrap_or(p);
                    if class_contains(pattern, p, end, text[t]) {
                        p = end + 1;
                        t += 1;
                        continue;
                    }
                }
                '?' if text[t] != '/' => {
                    p += 1;
                    t += 1;
                    continue;
                }
                candidate if candidate == text[t] => {
                    p += 1;
                    t += 1;
                    continue;
                }
                _ => {}
            }
        }
        match star {
            Some((star_p, star_t)) if text[star_t] != '/' => {
                p = star_p + 1;
                t = star_t + 1;
                star = Some((star_p, star_t + 1));
            }
            _ => return false,
        }
    }
    pattern[p..].iter().all(|character| *character == '*')
}

/// Locate the `]` that closes a class opened at `start`, if the class is terminated.
fn class_end(pattern: &[char], start: usize) -> Option<usize> {
    let mut index = start + 1;
    if matches!(pattern.get(index), Some('!' | '^')) {
        index += 1;
    }
    // A `]` immediately after the opening bracket is a literal member, not the terminator.
    if pattern.get(index) == Some(&']') {
        index += 1;
    }
    while index < pattern.len() {
        if pattern[index] == ']' {
            return Some(index);
        }
        index += 1;
    }
    None
}

/// Whether `candidate` belongs to the class spanning `start..=end`.
fn class_contains(pattern: &[char], start: usize, end: usize, candidate: char) -> bool {
    let mut index = start + 1;
    let negated = matches!(pattern.get(index), Some('!' | '^'));
    if negated {
        index += 1;
    }
    let mut found = false;
    while index < end {
        if index + 2 < end && pattern[index + 1] == '-' {
            found |= pattern[index] <= candidate && candidate <= pattern[index + 2];
            index += 3;
        } else {
            found |= pattern[index] == candidate;
            index += 1;
        }
    }
    found != negated
}

/// Whether a repository-relative `path` is named by one pathspec.
///
/// A plain pathspec names a file or a directory prefix. One holding a wildcard is matched against
/// the whole path, and, as in Git, its wildcards cross `/`: `*.py` matches `src/main.py`.
pub(crate) fn matches_pathspec(spec: &str, path: &str) -> bool {
    if spec.is_empty() {
        return true;
    }
    if !spec.contains(['*', '?', '[']) {
        return path == spec || path.starts_with(&format!("{spec}/"));
    }
    glob_match(&widen(spec), path)
}

/// Turn every run of `*` into `**` so pathspec wildcards cross directory separators.
fn widen(spec: &str) -> String {
    let mut out = String::new();
    let mut characters = spec.chars().peekable();
    while let Some(character) = characters.next() {
        if character != '*' {
            out.push(character);
            continue;
        }
        while characters.peek() == Some(&'*') {
            characters.next();
        }
        out.push_str("**");
    }
    out
}

/// Read `.git/info/exclude` and every `.gitignore` file in the working tree below `root`.
pub(crate) fn load(interp: &Interp, root: &str) -> IgnoreRules {
    let mut rules = IgnoreRules::default();
    let mut files = 0;
    // Repository-local excludes rank below every `.gitignore`, so they are read first.
    if let Ok(data) = interp
        .vfs
        .read("/", &super::repo::git_path(root, "info/exclude"))
    {
        files += 1;
        add_patterns(
            &mut rules,
            "",
            ".git/info/exclude",
            &String::from_utf8_lossy(&data),
        );
    }
    for (path, node) in interp.vfs.all_paths() {
        if files >= MAX_FILES || rules.patterns.len() >= MAX_PATTERNS {
            break;
        }
        if !matches!(node.kind, NodeKind::File(_)) || !path.ends_with(".gitignore") {
            continue;
        }
        let Some(relative) = super::repo::relative_path(root, path) else {
            continue;
        };
        if super::repo::is_git_path(root, path) {
            continue;
        }
        let base = relative
            .strip_suffix(".gitignore")
            .unwrap_or_default()
            .to_string();
        let NodeKind::File(data) = &node.kind else {
            continue;
        };
        if data.len() > 64 * 1024 {
            continue;
        }
        files += 1;
        add_patterns(&mut rules, &base, &relative, &String::from_utf8_lossy(data));
    }
    rules
}

fn add_patterns(rules: &mut IgnoreRules, base: &str, source: &str, text: &str) {
    for (number, line) in text.lines().enumerate() {
        if rules.patterns.len() >= MAX_PATTERNS {
            return;
        }
        let original = line.trim_end().to_string();
        let line = original.as_str();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (negated, line) = match line.strip_prefix('!') {
            Some(rest) => (true, rest),
            None => (false, line),
        };
        let (directory_only, line) = match line.strip_suffix('/') {
            Some(rest) => (true, rest),
            None => (false, line),
        };
        let anchored = line.contains('/');
        let line = line.strip_prefix('/').unwrap_or(line);
        if line.is_empty() {
            continue;
        }
        rules.patterns.push(Pattern {
            base: base.to_string(),
            text: line.to_string(),
            negated,
            directory_only,
            anchored,
            source: source.to_string(),
            line: number + 1,
            original,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::{add_patterns, IgnoreRules};

    fn rules(base: &str, text: &str) -> IgnoreRules {
        let mut rules = IgnoreRules::default();
        add_patterns(&mut rules, base, "test", text);
        rules
    }

    #[test]
    fn matches_names_at_any_depth() {
        let rules = rules("", "*.log\n");
        assert!(rules.is_ignored("a.log"));
        assert!(rules.is_ignored("deep/nested/a.log"));
        assert!(!rules.is_ignored("a.txt"));
    }

    #[test]
    fn anchors_patterns_containing_a_separator() {
        let rules = rules("", "/build\nsrc/generated\n");
        assert!(rules.is_ignored("build/out.o"));
        assert!(rules.is_ignored("src/generated/api.rs"));
        assert!(!rules.is_ignored("vendor/build/out.o"));
    }

    #[test]
    fn honors_directory_only_and_negation() {
        let rules = rules("", "target/\n!target/keep.txt\n");
        assert!(rules.is_ignored("target/debug/app"));
        // Git cannot re-include a file below an ignored directory.
        assert!(rules.is_ignored("target/keep.txt"));
        assert!(!rules.is_ignored("target.txt"));
    }

    #[test]
    fn applies_nested_pattern_files_only_below_their_directory() {
        let mut rules = rules("docs/", "draft.md\n");
        add_patterns(&mut rules, "", "test", "top.md\n");
        assert!(rules.is_ignored("docs/draft.md"));
        assert!(!rules.is_ignored("draft.md"));
        assert!(rules.is_ignored("top.md"));
    }

    #[test]
    fn supports_double_star() {
        let rules = rules("", "**/node_modules/**\n");
        assert!(rules.is_ignored("a/b/node_modules/pkg/index.js"));
    }
}
