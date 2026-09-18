//! Atomic, bounded patch application against the simulated filesystem.
//!
//! The implementation accepts ordinary unified diffs through `patch` and the compact
//! `*** Begin Patch` format commonly used by coding agents through `apply_patch`. Patches are
//! parsed and applied entirely in memory, then committed to the VFS as one transaction. It never
//! invokes a host patch program or accesses a host path.

use std::collections::HashMap;

use crate::commands::{reg_costed, CommandContext, CommandSpec, Io, Trust};
use crate::vfs::{parent_of, resolve_against, NodeKind};

const MAX_PATCH_BYTES: usize = 8 * 1024 * 1024;
const MAX_FILE_PATCHES: usize = 10_000;

pub fn register(commands: &mut HashMap<&'static str, CommandSpec>) {
    reg_costed(
        commands,
        &["apply_patch"],
        Trust::Partial,
        100,
        16 * 1024,
        cmd_apply_patch,
    );
    reg_costed(
        commands,
        &["patch"],
        Trust::Partial,
        100,
        16 * 1024,
        cmd_patch,
    );
}

#[derive(Debug)]
struct FilePatch {
    old_path: Option<String>,
    new_path: Option<String>,
    hunks: Vec<Hunk>,
}

#[derive(Debug)]
struct Hunk {
    old_start: Option<usize>,
    lines: Vec<PatchLine>,
}

#[derive(Debug)]
enum PatchLine {
    Context(String),
    Remove(String),
    Add(String),
}

fn cmd_apply_patch(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let cwd = ctx.cwd.clone();
    run_patch(ctx, args, io, true, &cwd)
}

fn cmd_patch(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let cwd = ctx.cwd.clone();
    run_patch(ctx, args, io, false, &cwd)
}

fn run_patch(
    ctx: &mut CommandContext<'_>,
    args: &[String],
    io: &mut Io,
    agent_command: bool,
    cwd: &str,
) -> i32 {
    let (strip, input_path) = match parse_options(agent_command, args) {
        Ok(options) => options,
        Err(error) => return fail(io, 2, &error),
    };
    let bytes = if let Some(path) = input_path {
        match ctx.fs_read_limited(&ctx.cwd, &path, MAX_PATCH_BYTES) {
            Ok(bytes) => bytes,
            Err(error) => return fail(io, 2, &format!("cannot read patch: {error}")),
        }
    } else {
        std::mem::take(&mut io.stdin)
    };
    if bytes.len() > MAX_PATCH_BYTES {
        return fail(io, 2, "patch input exceeds the 8 MiB limit");
    }
    let Ok(text) = std::str::from_utf8(&bytes) else {
        return fail(io, 2, "patch input must be UTF-8 text");
    };
    let reserved = ctx
        .vfs
        .disk_used()
        .saturating_mul(4)
        .saturating_add((bytes.len() as u64).saturating_mul(4));
    if !ctx.reserve_memory(reserved) {
        return resource_failure(ctx, io);
    }
    if !ctx.charge_cpu(bytes.len() as u64) {
        ctx.resources.release_memory(reserved);
        return resource_failure(ctx, io);
    }
    let parsed = if text.starts_with("*** Begin Patch") {
        parse_agent_patch(text)
    } else {
        parse_unified_patch(text, strip)
    };
    let status = match parsed {
        Ok(patches) if patches.len() <= MAX_FILE_PATCHES => {
            apply_transaction(ctx, patches, io, cwd)
        }
        Ok(_) => fail(io, 2, "too many files in patch"),
        Err(error) => fail(io, 2, &error),
    };
    ctx.resources.release_memory(reserved);
    status
}

/// Apply a unified diff on behalf of `git apply`.
///
/// Git strips one leading path component by default, and `--check` verifies the patch without
/// keeping the result, which is done here by restoring the filesystem snapshot afterwards.
pub(crate) fn apply_unified_diff(
    ctx: &mut CommandContext<'_>,
    args: &[String],
    io: &mut Io,
) -> i32 {
    let mut check = false;
    let mut forwarded = vec!["-p1".to_string()];
    for argument in args {
        match argument.as_str() {
            "--check" | "--summary" | "--stat" => check = true,
            "-v" | "--verbose" | "--index" | "--cached" | "--3way" | "--whitespace=nowarn" => {}
            value if value.starts_with("-p") || value.starts_with("--unsafe-paths") => {
                forwarded.insert(0, value.to_string());
            }
            value if value.starts_with('-') => {
                io.err.extend_from_slice(
                    format!("git: unsupported apply option: {value}\n").as_bytes(),
                );
                return 2;
            }
            value => forwarded.push(value.to_string()),
        }
    }
    let cwd = ctx.cwd.clone();
    let before = check.then(|| ctx.vfs.clone());
    let status = run_patch(ctx, &forwarded, io, false, &cwd);
    if let Some(before) = before {
        ctx.vfs = before;
    }
    status
}

/// Apply one trusted harness patch atomically beneath `/work` using the command's parser and
/// resource model.
pub(crate) fn apply_harness_patch(
    interp: &mut crate::interp::Interp,
    patch: &str,
    strip: usize,
) -> Result<(), String> {
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut io = Io {
        stdin: patch.as_bytes().to_vec(),
        out: &mut stdout,
        err: &mut stderr,
    };
    let mut context = CommandContext {
        env: interp,
        command_name: "patch",
    };
    let status = run_patch(
        &mut context,
        &[format!("-p{strip}")],
        &mut io,
        false,
        "/work",
    );
    if status == 0 {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&stderr).trim().to_string())
    }
}

fn parse_options(agent_command: bool, args: &[String]) -> Result<(usize, Option<String>), String> {
    if agent_command {
        return match args {
            [] => Ok((1, None)),
            [path] if !path.starts_with('-') => Ok((1, Some(path.clone()))),
            _ => Err("usage: apply_patch [PATCH_FILE]".to_string()),
        };
    }
    let mut strip = 0;
    let mut input = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "-i" | "--input" => {
                index += 1;
                let Some(path) = args.get(index) else {
                    return Err("-i requires a patch file".to_string());
                };
                input = Some(path.clone());
            }
            value if value.starts_with("-p") => {
                let number = if value == "-p" {
                    index += 1;
                    args.get(index).map(String::as_str)
                } else {
                    Some(&value[2..])
                };
                strip = number
                    .and_then(|value| value.parse().ok())
                    .ok_or_else(|| "-p requires a non-negative integer".to_string())?;
            }
            value if value.starts_with('-') => {
                return Err(format!("unsupported option: {value}"));
            }
            value if input.is_none() => input = Some(value.to_string()),
            _ => return Err("only one patch file may be supplied".to_string()),
        }
        index += 1;
    }
    Ok((strip, input))
}

fn parse_unified_patch(text: &str, strip: usize) -> Result<Vec<FilePatch>, String> {
    let lines = physical_lines(text);
    let mut patches = Vec::new();
    let mut index = 0;
    while index < lines.len() {
        if !line_text(&lines[index]).starts_with("--- ") {
            index += 1;
            continue;
        }
        let old = header_path(line_text(&lines[index]), "--- ", strip)?;
        index += 1;
        let Some(next) = lines.get(index) else {
            return Err("missing +++ file header".to_string());
        };
        let new = header_path(line_text(next), "+++ ", strip)?;
        index += 1;
        let mut hunks = Vec::new();
        while index < lines.len() {
            let line = line_text(&lines[index]);
            if line.starts_with("--- ") || line.starts_with("diff --git ") {
                break;
            }
            if !line.starts_with("@@ ") {
                index += 1;
                continue;
            }
            let (old_start, old_count, new_count) = parse_hunk_header(line)?;
            index += 1;
            hunks.push(parse_hunk_body(
                &lines,
                &mut index,
                Some((old_count, new_count)),
                Some(old_start),
            )?);
        }
        if hunks.is_empty() {
            return Err("file patch contains no hunks".to_string());
        }
        patches.push(FilePatch {
            old_path: old,
            new_path: new,
            hunks,
        });
    }
    if patches.is_empty() {
        return Err("no unified file patches found".to_string());
    }
    Ok(patches)
}

fn parse_agent_patch(text: &str) -> Result<Vec<FilePatch>, String> {
    let lines = physical_lines(text);
    if line_text(lines.last().ok_or("empty patch")?) != "*** End Patch" {
        return Err("missing *** End Patch marker".to_string());
    }
    let mut patches = Vec::new();
    let mut index = 1;
    while index + 1 < lines.len() {
        let header = line_text(&lines[index]);
        if header.is_empty() {
            index += 1;
            continue;
        }
        if let Some(path) = header.strip_prefix("*** Add File: ") {
            index += 1;
            let mut added = Vec::new();
            while index < lines.len() && !line_text(&lines[index]).starts_with("*** ") {
                let raw = &lines[index];
                let Some(content) = raw.strip_prefix('+') else {
                    return Err(format!("added file '{path}' contains a non-addition line"));
                };
                added.push(PatchLine::Add(content.to_string()));
                index += 1;
            }
            patches.push(FilePatch {
                old_path: None,
                new_path: Some(clean_agent_path(path)?),
                hunks: vec![Hunk {
                    old_start: Some(0),
                    lines: added,
                }],
            });
            continue;
        }
        if let Some(path) = header.strip_prefix("*** Delete File: ") {
            patches.push(FilePatch {
                old_path: Some(clean_agent_path(path)?),
                new_path: None,
                hunks: Vec::new(),
            });
            index += 1;
            continue;
        }
        let Some(path) = header.strip_prefix("*** Update File: ") else {
            return Err(format!("unsupported patch directive: {header}"));
        };
        let path = clean_agent_path(path)?;
        index += 1;
        let mut hunks = Vec::new();
        while index < lines.len() && !line_text(&lines[index]).starts_with("*** ") {
            if !line_text(&lines[index]).starts_with("@@") {
                return Err(format!("expected @@ hunk header for '{path}'"));
            }
            index += 1;
            hunks.push(parse_hunk_body(&lines, &mut index, None, None)?);
        }
        if hunks.is_empty() {
            return Err(format!("file patch for '{path}' contains no hunks"));
        }
        patches.push(FilePatch {
            old_path: Some(path.clone()),
            new_path: Some(path),
            hunks,
        });
    }
    if patches.is_empty() {
        return Err("patch contains no file operations".to_string());
    }
    Ok(patches)
}

fn parse_hunk_body(
    lines: &[String],
    index: &mut usize,
    expected_counts: Option<(usize, usize)>,
    old_start: Option<usize>,
) -> Result<Hunk, String> {
    let mut body = Vec::new();
    let mut old_seen = 0_usize;
    let mut new_seen = 0_usize;
    while *index < lines.len() {
        let raw = &lines[*index];
        let plain = line_text(raw);
        let counts_complete =
            expected_counts.is_some_and(|(old, new)| old_seen == old && new_seen == new);
        if counts_complete && plain != "\\ No newline at end of file" {
            break;
        }
        if expected_counts.is_none()
            && (plain.starts_with("@@")
                || plain.starts_with("diff --git ")
                || plain.starts_with("*** "))
        {
            break;
        }
        if plain == "\\ No newline at end of file" {
            let Some(previous) = body.last_mut() else {
                return Err("orphaned no-newline marker".to_string());
            };
            remove_trailing_newline(previous);
            *index += 1;
            continue;
        }
        let Some(marker) = raw.as_bytes().first() else {
            return Err("unprefixed empty line in hunk".to_string());
        };
        let content = raw[1..].to_string();
        body.push(match marker {
            b' ' => {
                old_seen = old_seen.saturating_add(1);
                new_seen = new_seen.saturating_add(1);
                PatchLine::Context(content)
            }
            b'-' => {
                old_seen = old_seen.saturating_add(1);
                PatchLine::Remove(content)
            }
            b'+' => {
                new_seen = new_seen.saturating_add(1);
                PatchLine::Add(content)
            }
            _ => return Err(format!("invalid hunk line prefix: {}", *marker as char)),
        });
        *index += 1;
    }
    if expected_counts.is_some_and(|(old, new)| old_seen != old || new_seen != new) {
        return Err("hunk body does not match its declared line counts".to_string());
    }
    Ok(Hunk {
        old_start,
        lines: body,
    })
}

fn remove_trailing_newline(line: &mut PatchLine) {
    let value = match line {
        PatchLine::Context(value) | PatchLine::Remove(value) | PatchLine::Add(value) => value,
    };
    if value.ends_with('\n') {
        value.pop();
        if value.ends_with('\r') {
            value.pop();
        }
    }
}

fn physical_lines(text: &str) -> Vec<String> {
    text.split_inclusive('\n').map(str::to_string).collect()
}

fn line_text(line: &str) -> &str {
    let line = line.strip_suffix('\n').unwrap_or(line);
    line.strip_suffix('\r').unwrap_or(line)
}

fn header_path(line: &str, prefix: &str, strip: usize) -> Result<Option<String>, String> {
    let value = line
        .strip_prefix(prefix)
        .ok_or_else(|| format!("expected {prefix}file header"))?
        .split(['\t', ' '])
        .next()
        .unwrap_or_default();
    if value == "/dev/null" {
        return Ok(None);
    }
    let components = value
        .split('/')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    if strip >= components.len() {
        return Err(format!("cannot strip {strip} components from '{value}'"));
    }
    clean_agent_path(&components[strip..].join("/")).map(Some)
}

fn clean_agent_path(path: &str) -> Result<String, String> {
    let path = path.trim();
    if path.is_empty() || path.starts_with('/') || path.split('/').any(|part| part == "..") {
        return Err(format!("unsafe patch path: '{path}'"));
    }
    Ok(path.trim_start_matches("./").to_string())
}

fn parse_hunk_header(header: &str) -> Result<(usize, usize, usize), String> {
    let (old_range, rest) = header
        .strip_prefix("@@ -")
        .and_then(|rest| rest.split_once(' '))
        .ok_or_else(|| format!("invalid hunk header: {header}"))?;
    let new_range = rest
        .strip_prefix('+')
        .and_then(|rest| rest.split_once(' '))
        .map(|(range, _)| range)
        .ok_or_else(|| format!("invalid hunk header: {header}"))?;
    let (old_start, old_count) = parse_range(old_range, header)?;
    let (_, new_count) = parse_range(new_range, header)?;
    Ok((old_start, old_count, new_count))
}

fn parse_range(range: &str, header: &str) -> Result<(usize, usize), String> {
    let (start, count) = range.split_once(',').map_or((range, "1"), |pair| pair);
    let start = start
        .parse()
        .map_err(|_| format!("invalid range in hunk: {header}"))?;
    let count = count
        .parse()
        .map_err(|_| format!("invalid range in hunk: {header}"))?;
    Ok((start, count))
}

fn apply_transaction(
    ctx: &mut CommandContext<'_>,
    patches: Vec<FilePatch>,
    io: &mut Io,
    cwd: &str,
) -> i32 {
    let before = ctx.vfs.clone();
    for patch in patches {
        if let Err(error) = apply_file_patch(ctx, patch, cwd) {
            ctx.vfs = before;
            return fail(io, 1, &error);
        }
    }
    0
}

fn apply_file_patch(
    ctx: &mut CommandContext<'_>,
    patch: FilePatch,
    cwd: &str,
) -> Result<(), String> {
    let source = patch.old_path.as_deref();
    let destination = patch.new_path.as_deref();
    if source.is_none() && destination.is_none() {
        return Err("patch has neither source nor destination".to_string());
    }
    let source_bytes = if let Some(path) = source {
        ctx.vfs
            .read(cwd, path)
            .map_err(|error| format!("{path}: {error}"))?
    } else {
        Vec::new()
    };
    if !ctx.charge_cpu(source_bytes.len() as u64) {
        return Err("resource limit exceeded".to_string());
    }
    let directive_delete = destination.is_none() && patch.hunks.is_empty();
    let mut lines = split_bytes_lines(&source_bytes)?;
    let mut offset = 0_isize;
    let mut search_from = 0;
    for hunk in patch.hunks {
        let old = hunk
            .lines
            .iter()
            .filter_map(|line| match line {
                PatchLine::Context(value) | PatchLine::Remove(value) => Some(value.clone()),
                PatchLine::Add(_) => None,
            })
            .collect::<Vec<_>>();
        let new = hunk
            .lines
            .into_iter()
            .filter_map(|line| match line {
                PatchLine::Context(value) | PatchLine::Add(value) => Some(value),
                PatchLine::Remove(_) => None,
            })
            .collect::<Vec<_>>();
        let position = if let Some(start) = hunk.old_start {
            let base = start.saturating_sub(1);
            usize::try_from((base as isize).saturating_add(offset))
                .map_err(|_| "hunk position underflow".to_string())?
        } else {
            find_exact(&lines, &old, search_from)
                .ok_or_else(|| "hunk context was not found exactly".to_string())?
        };
        let end = position
            .checked_add(old.len())
            .ok_or_else(|| "hunk range overflow".to_string())?;
        if end > lines.len() || lines[position..end] != old {
            return Err("hunk does not apply at its declared location".to_string());
        }
        let old_len = old.len();
        let new_len = new.len();
        lines.splice(position..end, new);
        offset = offset.saturating_add(new_len as isize - old_len as isize);
        search_from = position.saturating_add(new_len);
    }
    let output = lines.concat().into_bytes();
    if destination.is_none() {
        let path = source.expect("deletion has a source");
        if !directive_delete && !output.is_empty() {
            return Err(format!(
                "deletion patch for '{path}' does not remove all content"
            ));
        }
        ctx.sync_vfs_time();
        return ctx
            .vfs
            .remove_file(cwd, path)
            .map_err(|error| format!("{path}: {error}"));
    }
    let path = destination.expect("non-deletion has a destination");
    let absolute = resolve_against(cwd, path);
    if let Some(parent) = parent_of(&absolute) {
        ctx.vfs
            .mkdir_all("/", &parent)
            .map_err(|error| format!("{path}: {error}"))?;
    }
    let mode = source
        .and_then(|path| ctx.vfs.metadata(cwd, path, true).ok())
        .and_then(|metadata| match metadata.kind {
            NodeKind::File(_) => Some(metadata.mode),
            _ => None,
        })
        .unwrap_or(0o644);
    ctx.sync_vfs_time();
    ctx.vfs
        .write("/", &absolute, &output, mode)
        .map_err(|error| format!("{path}: {error}"))?;
    if source.is_some_and(|source| source != path) {
        ctx.vfs
            .remove_file(cwd, source.unwrap())
            .map_err(|error| format!("{}: {error}", source.unwrap()))?;
    }
    Ok(())
}

fn split_bytes_lines(bytes: &[u8]) -> Result<Vec<String>, String> {
    let text = std::str::from_utf8(bytes).map_err(|_| "patched files must be UTF-8 text")?;
    Ok(physical_lines(text))
}

fn find_exact(haystack: &[String], needle: &[String], start: usize) -> Option<usize> {
    if needle.is_empty() {
        return Some(start.min(haystack.len()));
    }
    if needle.len() > haystack.len() || start > haystack.len() - needle.len() {
        return None;
    }
    (start..=haystack.len().saturating_sub(needle.len()))
        .find(|position| haystack[*position..*position + needle.len()] == *needle)
}

fn fail(io: &mut Io, status: i32, message: &str) -> i32 {
    io.err
        .extend_from_slice(format!("patch: {message}\n").as_bytes());
    status
}

fn resource_failure(ctx: &CommandContext<'_>, io: &mut Io) -> i32 {
    fail(
        io,
        ctx.resources
            .stop_reason()
            .map_or(137, |reason| reason.exit_status()),
        "resource limit exceeded",
    )
}
