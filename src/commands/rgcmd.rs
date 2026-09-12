//! Bounded ripgrep-style recursive search over the virtual filesystem.
//!
//! This module implements the common agent-facing `rg` surface without invoking a host binary.
//! Traversal order is stable, binary detection is explicit, and every examined line consumes
//! deterministic CPU fuel.

use std::collections::HashMap;

use crate::commands::util::{ewln, glob_eq, wln};
use crate::commands::{CommandContext, CommandSpec, Io, Trust};

pub fn register(commands: &mut HashMap<&'static str, CommandSpec>) {
    super::reg(commands, &["rg"], Trust::Partial, cmd_rg);
}

#[derive(Default)]
struct Options {
    ignore_case: bool,
    invert: bool,
    fixed: bool,
    word: bool,
    line: bool,
    count: bool,
    files_with: bool,
    files_without: bool,
    quiet: bool,
    hidden: bool,
    text: bool,
    force_filename: Option<bool>,
    list_files: bool,
    globs: Vec<String>,
    types: Vec<String>,
    patterns: Vec<String>,
    paths: Vec<String>,
}

fn cmd_rg(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let options = match parse_options(args) {
        Ok(options) => options,
        Err(error) => {
            ewln(io.err, &format!("rg: {error}"));
            return 2;
        }
    };
    let matcher = if options.list_files {
        None
    } else {
        match build_matcher(&options) {
            Ok(matcher) => Some(matcher),
            Err(error) => {
                ewln(io.err, &format!("rg: {error}"));
                return 2;
            }
        }
    };
    let (inputs, reserved) = match collect_inputs(interp, &options, &io.stdin) {
        Ok(inputs) => inputs,
        Err(error) => {
            ewln(io.err, &format!("rg: {error}"));
            return 2;
        }
    };
    let status = search_inputs(interp, &options, matcher.as_ref(), inputs, io);
    interp.resources.release_memory(reserved);
    status
}

fn search_inputs(
    interp: &mut CommandContext<'_>,
    options: &Options,
    matcher: Option<&Matcher>,
    inputs: Vec<Input>,
    io: &mut Io,
) -> i32 {
    let show_filename = options
        .force_filename
        .unwrap_or(inputs.len() > 1 || inputs.iter().any(|input| input.recursive));
    let mut any = false;
    for input in inputs {
        if !interp.charge_cpu(1) {
            return 137;
        }
        if options.list_files {
            wln(io.out, &input.label);
            any = true;
            continue;
        }
        if !options.text && input.bytes.contains(&0) {
            continue;
        }
        let matcher = matcher.expect("search mode has a matcher");
        let text = String::from_utf8_lossy(&input.bytes);
        let mut matched_count = 0_usize;
        for (index, line) in text.lines().enumerate() {
            if !interp.charge_cpu(1_u64.saturating_add(line.len() as u64)) {
                return 137;
            }
            if !(matcher.is_match(line) ^ options.invert) {
                continue;
            }
            matched_count = matched_count.saturating_add(1);
            if options.quiet || options.count || options.files_with || options.files_without {
                continue;
            }
            let mut prefix = String::new();
            if show_filename {
                prefix.push_str(&input.label);
                prefix.push(':');
            }
            if options.line {
                prefix.push_str(&(index + 1).to_string());
                prefix.push(':');
            }
            wln(io.out, &format!("{prefix}{line}"));
        }
        let file_matches = matched_count > 0;
        let selected = if options.files_without {
            !file_matches
        } else {
            file_matches
        };
        any |= selected;
        if options.quiet && selected {
            return 0;
        }
        if options.files_with && file_matches || options.files_without && !file_matches {
            wln(io.out, &input.label);
        } else if options.count {
            if show_filename {
                wln(io.out, &format!("{}:{matched_count}", input.label));
            } else {
                wln(io.out, &matched_count.to_string());
            }
        }
    }
    i32::from(!any)
}

struct Matcher(regex::Regex);

impl Matcher {
    fn is_match(&self, line: &str) -> bool {
        self.0.is_match(line)
    }
}

fn build_matcher(options: &Options) -> Result<Matcher, String> {
    if options.patterns.is_empty() {
        return Err("a pattern is required".to_string());
    }
    let patterns = options.patterns.iter().map(|pattern| {
        let mut pattern = if options.fixed {
            regex::escape(pattern)
        } else {
            pattern.clone()
        };
        if options.word {
            pattern = format!(r"\b(?:{pattern})\b");
        }
        format!("(?:{pattern})")
    });
    regex::RegexBuilder::new(&patterns.collect::<Vec<_>>().join("|"))
        .case_insensitive(options.ignore_case)
        .build()
        .map(Matcher)
        .map_err(|error| format!("invalid regular expression: {error}"))
}

struct Input {
    label: String,
    bytes: Vec<u8>,
    recursive: bool,
}

fn collect_inputs(
    interp: &mut CommandContext<'_>,
    options: &Options,
    stdin: &[u8],
) -> Result<(Vec<Input>, u64), String> {
    if options.paths.is_empty() && !stdin.is_empty() {
        if !interp.reserve_memory(stdin.len() as u64) {
            return Err("memory limit exceeded".to_string());
        }
        return Ok((
            vec![Input {
                label: "-".to_string(),
                bytes: stdin.to_vec(),
                recursive: false,
            }],
            stdin.len() as u64,
        ));
    }
    let paths = if options.paths.is_empty() {
        vec![".".to_string()]
    } else {
        options.paths.clone()
    };
    let mut inputs = Vec::new();
    let mut reserved = 0_u64;
    for operand in paths {
        let absolute = crate::vfs::resolve_against(&interp.cwd, &operand);
        if interp.vfs.is_file("/", &absolute) {
            if let Err(error) = push_file(
                interp,
                options,
                &absolute,
                operand,
                false,
                &mut inputs,
                &mut reserved,
            ) {
                interp.resources.release_memory(reserved);
                return Err(error);
            }
            continue;
        }
        if !interp.vfs.is_dir("/", &absolute) {
            interp.resources.release_memory(reserved);
            return Err(format!("{operand}: No such file or directory"));
        }
        let mut paths = interp.vfs.walk(&absolute);
        paths.sort();
        for path in paths {
            if !interp.vfs.is_file("/", &path) {
                continue;
            }
            let label = display_path(&interp.cwd, &path);
            if let Err(error) = push_file(
                interp,
                options,
                &path,
                label,
                true,
                &mut inputs,
                &mut reserved,
            ) {
                interp.resources.release_memory(reserved);
                return Err(error);
            }
        }
    }
    Ok((inputs, reserved))
}

fn push_file(
    interp: &mut CommandContext<'_>,
    options: &Options,
    absolute: &str,
    label: String,
    recursive: bool,
    inputs: &mut Vec<Input>,
    reserved: &mut u64,
) -> Result<(), String> {
    if !options.hidden
        && label
            .split('/')
            .any(|part| part.starts_with('.') && part != ".")
    {
        return Ok(());
    }
    if !matches_globs(&options.globs, &label) || !matches_types(&options.types, &label) {
        return Ok(());
    }
    let bytes = interp
        .fs_read_limited("/", absolute, crate::descriptors::MAX_CAPTURE_BYTES)
        .map_err(|error| error.to_string())?;
    if !interp.reserve_memory(bytes.len() as u64) {
        return Err("memory limit exceeded".to_string());
    }
    *reserved = reserved.saturating_add(bytes.len() as u64);
    inputs.push(Input {
        label,
        bytes,
        recursive,
    });
    Ok(())
}

fn display_path(cwd: &str, absolute: &str) -> String {
    if cwd == "/" {
        absolute.trim_start_matches('/').to_string()
    } else {
        absolute
            .strip_prefix(&format!("{}/", cwd.trim_end_matches('/')))
            .unwrap_or(absolute)
            .to_string()
    }
}

fn matches_globs(globs: &[String], path: &str) -> bool {
    let mut included = !globs.iter().any(|glob| !glob.starts_with('!'));
    for glob in globs {
        let (exclude, pattern) = glob
            .strip_prefix('!')
            .map_or((false, glob.as_str()), |pattern| (true, pattern));
        if glob_eq(pattern, path) || glob_eq(pattern, crate::vfs::basename(path)) {
            included = !exclude;
        }
    }
    included
}

fn matches_types(types: &[String], path: &str) -> bool {
    types.is_empty()
        || types.iter().any(|kind| {
            let extensions: &[&str] = match kind.as_str() {
                "rust" => &["rs"],
                "py" | "python" => &["py", "pyi"],
                "js" => &["js", "jsx", "mjs", "cjs"],
                "ts" => &["ts", "tsx"],
                "json" => &["json"],
                "toml" => &["toml"],
                "yaml" => &["yaml", "yml"],
                "md" | "markdown" => &["md", "markdown"],
                "sh" | "shell" => &["sh", "bash", "zsh"],
                "c" => &["c", "h"],
                "cpp" => &["cc", "cpp", "cxx", "hpp"],
                _ => &[],
            };
            path.rsplit_once('.')
                .is_some_and(|(_, extension)| extensions.contains(&extension))
        })
}

fn parse_options(args: &[String]) -> Result<Options, String> {
    let mut options = Options::default();
    let mut index = 0;
    let mut positional = Vec::new();
    while index < args.len() {
        let argument = &args[index];
        match argument.as_str() {
            "--" => {
                positional.extend_from_slice(&args[index + 1..]);
                break;
            }
            "-e" | "--regexp" => options
                .patterns
                .push(take_value(args, &mut index, argument)?),
            "-g" | "--glob" => options.globs.push(take_value(args, &mut index, argument)?),
            "-t" | "--type" => options.types.push(take_value(args, &mut index, argument)?),
            "-n" | "--line-number" => options.line = true,
            "-i" | "--ignore-case" => options.ignore_case = true,
            "-v" | "--invert-match" => options.invert = true,
            "-F" | "--fixed-strings" => options.fixed = true,
            "-w" | "--word-regexp" => options.word = true,
            "-c" | "--count" => options.count = true,
            "-l" | "--files-with-matches" => options.files_with = true,
            "--files-without-match" => options.files_without = true,
            "-q" | "--quiet" => options.quiet = true,
            "--hidden" => options.hidden = true,
            "-a" | "--text" => options.text = true,
            "-H" | "--with-filename" => options.force_filename = Some(true),
            "-h" | "--no-filename" => options.force_filename = Some(false),
            "--files" => options.list_files = true,
            "--no-ignore" | "--no-heading" | "--color=never" => {}
            value if value.starts_with("--glob=") => options.globs.push(value[7..].to_string()),
            value if value.starts_with("--type=") => options.types.push(value[7..].to_string()),
            value if value.starts_with("--regexp=") => {
                options.patterns.push(value[9..].to_string())
            }
            value if value.starts_with("-g") && value.len() > 2 => {
                options.globs.push(value[2..].to_string())
            }
            value if value.starts_with("-t") && value.len() > 2 => {
                options.types.push(value[2..].to_string())
            }
            value if value.starts_with("-e") && value.len() > 2 => {
                options.patterns.push(value[2..].to_string())
            }
            value if value.starts_with('-') => {
                for flag in value[1..].chars() {
                    match flag {
                        'n' => options.line = true,
                        'i' => options.ignore_case = true,
                        'v' => options.invert = true,
                        'F' => options.fixed = true,
                        'w' => options.word = true,
                        'c' => options.count = true,
                        'l' => options.files_with = true,
                        'q' => options.quiet = true,
                        'H' => options.force_filename = Some(true),
                        'h' => options.force_filename = Some(false),
                        _ => return Err(format!("unsupported option {argument}")),
                    }
                }
            }
            _ => positional.push(argument.clone()),
        }
        index += 1;
    }
    if !options.list_files && options.patterns.is_empty() && !positional.is_empty() {
        options.patterns.push(positional.remove(0));
    }
    options.paths = positional;
    Ok(options)
}

fn take_value(args: &[String], index: &mut usize, option: &str) -> Result<String, String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("option {option} requires a value"))
}
