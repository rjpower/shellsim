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
        if self.patterns.is_empty() {
            return false;
        }
        let mut ignored = false;
        for (candidate, is_directory) in ancestor_candidates(relative) {
            for pattern in &self.patterns {
                if pattern.directory_only && !is_directory {
                    continue;
                }
                if pattern.matches(&candidate) {
                    ignored = !pattern.negated;
                }
            }
            if ignored && is_directory {
                // Git does not descend into an ignored directory, so nothing below it can be
                // re-included by a later pattern.
                return true;
            }
        }
        ignored
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
/// `*` and `?` stop at a `/`; `**` crosses separators. Backtracking is linear in the text
/// length per wildcard, which is bounded by the pattern and path limits the caller enforces.
fn glob_match(pattern: &str, text: &str) -> bool {
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

/// Read every `.gitignore` file in the working tree below `root`.
pub(crate) fn load(interp: &Interp, root: &str) -> IgnoreRules {
    let mut rules = IgnoreRules::default();
    let mut files = 0;
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
        add_patterns(&mut rules, &base, &String::from_utf8_lossy(data));
    }
    rules
}

fn add_patterns(rules: &mut IgnoreRules, base: &str, text: &str) {
    for line in text.lines() {
        if rules.patterns.len() >= MAX_PATTERNS {
            return;
        }
        let line = line.trim_end();
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
        });
    }
}

#[cfg(test)]
mod tests {
    use super::{add_patterns, IgnoreRules};

    fn rules(base: &str, text: &str) -> IgnoreRules {
        let mut rules = IgnoreRules::default();
        add_patterns(&mut rules, base, text);
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
        add_patterns(&mut rules, "", "top.md\n");
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
