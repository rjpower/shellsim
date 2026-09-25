//! Atomic, bounded patch application against the simulated filesystem.
//!
//! The implementation accepts ordinary unified diffs through `patch` and the compact
//! `*** Begin Patch` format commonly used by coding agents through `apply_patch`. Patches are
//! parsed and applied entirely in memory, then committed to the VFS as one transaction. It never
//! invokes a host patch program or accesses a host path.

use std::collections::{BTreeMap, HashMap};

use crate::commands::{CommandContext, CommandSpec, Io, Trust};
use crate::exec::ShellPoll;
use crate::program::ProcessContext;
use crate::syscalls::{FileChange, FileKind, System};
use crate::vfs::{parent_of, resolve_against};

const MAX_PATCH_BYTES: usize = 8 * 1024 * 1024;
const MAX_FILE_PATCHES: usize = 10_000;

pub fn register(commands: &mut HashMap<&'static str, CommandSpec>) {
    super::reg_system_poll_costed(
        commands,
        "/usr/bin/apply_patch",
        Trust::Partial,
        100,
        16 * 1024,
        cmd_apply_patch,
    );
    super::reg_system_poll_costed(
        commands,
        "/usr/bin/patch",
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

fn cmd_apply_patch(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    run_native_patch(context, io, true)
}

fn cmd_patch(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    run_native_patch(context, io, false)
}

fn run_native_patch(
    context: &mut ProcessContext<'_>,
    io: &mut Io,
    agent_command: bool,
) -> ShellPoll {
    if parse_options(agent_command, context.args)
        .ok()
        .is_some_and(|(_, path)| path.is_none())
    {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let cwd = context.system.cwd().to_string();
    ShellPoll::Ready(run_patch(
        context.system,
        context.args,
        io,
        agent_command,
        &cwd,
        false,
    ))
}

fn run_patch(
    system: &mut dyn System,
    args: &[String],
    io: &mut Io,
    agent_command: bool,
    cwd: &str,
    check: bool,
) -> i32 {
    let (strip, input_path) = match parse_options(agent_command, args) {
        Ok(options) => options,
        Err(error) => return fail(io, 2, &error),
    };
    let bytes = if let Some(path) = input_path {
        match system.read_file_limited(cwd, &path, MAX_PATCH_BYTES) {
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
    let reserved = system
        .disk_used()
        .saturating_mul(4)
        .saturating_add((bytes.len() as u64).saturating_mul(4));
    if !system.reserve_memory(reserved) {
        return resource_failure(system, io);
    }
    if !system.charge_cpu(bytes.len() as u64) {
        system.release_memory(reserved);
        return resource_failure(system, io);
    }
    let parsed = if text.starts_with("*** Begin Patch") {
        parse_agent_patch(text)
    } else {
        parse_unified_patch(text, strip)
    };
    let status = match parsed {
        Ok(patches) if patches.len() <= MAX_FILE_PATCHES => {
            apply_transaction(system, patches, io, cwd, check)
        }
        Ok(_) => fail(io, 2, "too many files in patch"),
        Err(error) => fail(io, 2, &error),
    };
    system.release_memory(reserved);
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
    let mut strip = "-p1".to_string();
    let mut forwarded: Vec<String> = Vec::new();
    for argument in args {
        match argument.as_str() {
            "--check" => check = true,
            "-v" | "--verbose" | "--3way" | "--whitespace=nowarn" => {}
            // A lone `-` names standard input, which is already where an operandless patch reads.
            "-" => {}
            value if value.starts_with("-p") => strip = value.to_string(),
            value if value.starts_with("--unsafe-paths") => {}
            value if value.starts_with('-') => {
                io.print_err(&format!("git: unsupported apply option: {value}\n"));
                return 2;
            }
            value => forwarded.push(value.to_string()),
        }
    }
    forwarded.insert(0, strip);
    let cwd = ctx.cwd.clone();
    // A failure is reported in Git's words, since that is what callers match on.
    let mut errors = Vec::new();
    let status = {
        let mut inner = Io {
            stdin: std::mem::take(&mut io.stdin),
            out: io.out,
            err: &mut errors,
        };
        let status = run_patch(
            &mut ctx.system(),
            &forwarded,
            &mut inner,
            false,
            &cwd,
            check,
        );
        io.stdin = std::mem::take(&mut inner.stdin);
        status
    };
    if status == 0 {
        io.err.extend_from_slice(&errors);
        return 0;
    }
    for line in String::from_utf8_lossy(&errors).lines() {
        let message = line.strip_prefix("patch: ").unwrap_or(line);
        io.print_err(&format!("error: {message}\n"));
    }
    io.print_err("error: patch does not apply\n");
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
    let status = run_patch(
        &mut crate::syscalls::ActiveSystem::new(interp),
        &[format!("-p{strip}")],
        &mut io,
        false,
        "/work",
        false,
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
    system: &mut dyn System,
    patches: Vec<FilePatch>,
    io: &mut Io,
    cwd: &str,
    check: bool,
) -> i32 {
    let mut staged = BTreeMap::new();
    let mut changes = Vec::new();
    for patch in patches {
        if let Err(error) = apply_file_patch(system, patch, cwd, &mut staged, &mut changes) {
            return fail(io, 1, &error);
        }
    }
    if check {
        return 0;
    }
    match system.apply_file_batch("/", changes) {
        Ok(()) => 0,
        Err(error) => fail(io, 1, &error.to_string()),
    }
}

fn apply_file_patch(
    system: &mut dyn System,
    patch: FilePatch,
    cwd: &str,
    staged: &mut BTreeMap<String, Option<(Vec<u8>, u32)>>,
    changes: &mut Vec<FileChange>,
) -> Result<(), String> {
    let source = patch.old_path.as_deref();
    let destination = patch.new_path.as_deref();
    if source.is_none() && destination.is_none() {
        return Err("patch has neither source nor destination".to_string());
    }
    let source_bytes = if let Some(path) = source {
        let absolute = resolve_against(cwd, path);
        match staged.get(&absolute) {
            Some(Some((bytes, _))) => bytes.clone(),
            Some(None) => return Err(format!("{path}: no such file")),
            None => system
                .read_file_limited(cwd, path, MAX_PATCH_BYTES)
                .map_err(|error| format!("{path}: {error}"))?,
        }
    } else {
        Vec::new()
    };
    if !system.charge_cpu(source_bytes.len() as u64) {
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
        // A hunk with no trailing context ran to the end of the file it was made from, and one
        // that starts at the first line began there; Git holds a patch to both, which is what
        // stops a stale patch from being dropped into the middle of a file that has since grown.
        let trailing = hunk
            .lines
            .iter()
            .rev()
            .take_while(|line| matches!(line, PatchLine::Context(_)))
            .count();
        let new = hunk
            .lines
            .into_iter()
            .filter_map(|line| match line {
                PatchLine::Context(value) | PatchLine::Add(value) => Some(value),
                PatchLine::Remove(_) => None,
            })
            .collect::<Vec<_>>();
        // Only a unified diff carries the line numbers these anchors rely on; the agent patch
        // format names no positions at all and is matched purely by context.
        let anchored_to_end = trailing == 0 && hunk.old_start.is_some();
        let anchored_to_start = hunk.old_start.is_some_and(|start| start <= 1);
        let fits = |position: usize| {
            let Some(end) = position.checked_add(old.len()) else {
                return false;
            };
            end <= lines.len()
                && lines[position..end] == old
                && (!anchored_to_end || end == lines.len())
                && (!anchored_to_start || position == 0)
        };
        let declared = match hunk.old_start {
            Some(start) => {
                usize::try_from((start.saturating_sub(1) as isize).saturating_add(offset))
                    .map_err(|_| "hunk position underflow".to_string())?
            }
            None => search_from,
        };
        let position = if fits(declared) {
            declared
        } else {
            // Git looks for somewhere else the hunk fits before giving up.
            (search_from..=lines.len().saturating_sub(old.len().min(lines.len())))
                .find(|position| fits(*position))
                .ok_or_else(|| match hunk.old_start {
                    Some(line) => format!("patch failed at line {line}"),
                    None => "hunk context was not found exactly".to_string(),
                })?
        };
        let end = position
            .checked_add(old.len())
            .ok_or_else(|| "hunk range overflow".to_string())?;
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
        let absolute = resolve_against(cwd, path);
        staged.insert(absolute.clone(), None);
        changes.push(FileChange::RemoveFile(absolute));
        return Ok(());
    }
    let path = destination.expect("non-deletion has a destination");
    let absolute = resolve_against(cwd, path);
    if matches!(
        system.metadata("/", &absolute, false).map(|info| info.kind),
        Ok(FileKind::Symlink)
    ) {
        return Err(format!(
            "{path}: symbolic-link destinations are not supported"
        ));
    }
    if let Some(parent) = parent_of(&absolute) {
        changes.push(FileChange::MkdirAll(parent));
    }
    let mode = source
        .and_then(|path| {
            let absolute = resolve_against(cwd, path);
            match staged.get(&absolute) {
                Some(Some((_, mode))) => Some(*mode),
                Some(None) => None,
                None => system.metadata(cwd, path, true).ok().and_then(|metadata| {
                    (metadata.kind == FileKind::File).then_some(metadata.mode)
                }),
            }
        })
        .unwrap_or(0o644);
    changes.push(FileChange::PutFile {
        path: absolute.clone(),
        bytes: output.clone(),
        mode,
    });
    staged.insert(absolute, Some((output, mode)));
    if source.is_some_and(|source| source != path) {
        let old = resolve_against(cwd, source.expect("rename has a source"));
        staged.insert(old.clone(), None);
        changes.push(FileChange::RemoveFile(old));
    }
    Ok(())
}

fn split_bytes_lines(bytes: &[u8]) -> Result<Vec<String>, String> {
    let text = std::str::from_utf8(bytes).map_err(|_| "patched files must be UTF-8 text")?;
    Ok(physical_lines(text))
}

fn fail(io: &mut Io, status: i32, message: &str) -> i32 {
    io.print_err(&format!("patch: {message}\n"));
    status
}

fn resource_failure(system: &dyn System, io: &mut Io) -> i32 {
    fail(io, system.stop_status(), "resource limit exceeded")
}
