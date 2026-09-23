//! Standalone WASI word-count command used to exercise shellsim's guest process ABI.
//!
//! This intentionally uses only Rust's WASI standard library. It is built separately from the
//! kernel and cannot call shellsim's in-process command context.

use std::fs;
use std::io::{self, Read, Write};

#[derive(Clone, Copy, Default)]
struct Selected {
    lines: bool,
    words: bool,
    bytes: bool,
    characters: bool,
}

impl Selected {
    fn columns(self) -> [bool; 4] {
        let none = !self.lines && !self.words && !self.bytes && !self.characters;
        [
            self.lines || none,
            self.words || none,
            self.bytes || none,
            self.characters,
        ]
    }
}

#[derive(Clone, Copy, Default)]
struct Counts {
    lines: usize,
    words: usize,
    bytes: usize,
    characters: usize,
}

impl Counts {
    fn of(data: &[u8]) -> Self {
        let text = String::from_utf8_lossy(data);
        Self {
            lines: data.iter().filter(|&&byte| byte == b'\n').count(),
            words: text.split_whitespace().count(),
            bytes: data.len(),
            characters: text.chars().count(),
        }
    }

    fn add(&mut self, other: Self) {
        self.lines = self.lines.saturating_add(other.lines);
        self.words = self.words.saturating_add(other.words);
        self.bytes = self.bytes.saturating_add(other.bytes);
        self.characters = self.characters.saturating_add(other.characters);
    }

    fn values(self) -> [usize; 4] {
        [self.lines, self.words, self.bytes, self.characters]
    }
}

fn print_counts(
    out: &mut impl Write,
    counts: Counts,
    selected: Selected,
    label: &str,
) -> io::Result<()> {
    let columns = selected.columns();
    let values = counts.values();
    let parts = values
        .into_iter()
        .zip(columns)
        .filter_map(|(value, enabled)| enabled.then(|| format!("{value:>7}")))
        .collect::<Vec<_>>();
    let mut line = parts.join(" ").trim_start().to_string();
    if !label.is_empty() {
        line.push(' ');
        line.push_str(label);
    }
    writeln!(out, "{line}")
}

fn run() -> i32 {
    let mut selected = Selected::default();
    let mut paths = Vec::new();
    let mut options = true;
    for argument in std::env::args().skip(1) {
        if options && argument == "--" {
            options = false;
        } else if options && argument.starts_with('-') && argument.len() > 1 {
            for flag in argument[1..].chars() {
                match flag {
                    'l' => selected.lines = true,
                    'w' => selected.words = true,
                    'c' => selected.bytes = true,
                    'm' => selected.characters = true,
                    _ => {
                        eprintln!("wc: unimplemented option");
                        return 2;
                    }
                }
            }
        } else {
            paths.push(argument);
        }
    }
    let mut out = io::stdout().lock();
    if paths.is_empty() {
        let mut data = Vec::new();
        if let Err(error) = io::stdin().read_to_end(&mut data) {
            eprintln!("wc: stdin: {error}");
            return 1;
        }
        return if print_counts(&mut out, Counts::of(&data), selected, "").is_ok() {
            0
        } else {
            1
        };
    }
    let mut status = 0;
    let mut totals = Counts::default();
    for path in &paths {
        match fs::read(path) {
            Ok(data) => {
                let counts = Counts::of(&data);
                totals.add(counts);
                if print_counts(&mut out, counts, selected, path).is_err() {
                    return 1;
                }
            }
            Err(error) => {
                eprintln!("wc: {path}: {error}");
                status = 1;
            }
        }
    }
    if paths.len() > 1 && print_counts(&mut out, totals, selected, "total").is_err() {
        return 1;
    }
    status
}

fn main() {
    std::process::exit(run());
}
