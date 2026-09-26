//! Filesystem commands operating on the VFS: listing (ls), tree mutation
//! (mkdir/rmdir/rm/cp/mv/touch/ln), permissions (chmod/chown), path math
//! (basename/dirname/realpath/readlink), inspection (stat/file/find/du/tree), installation,
//! truncation, and temporary files.

use std::collections::HashMap;

use crate::commands::util::{ewln, split_flags, wln};
use crate::commands::{CommandSpec, Io, Trust};
use crate::syscalls::{FileKind, System};
use crate::vfs::resolve_against;

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    use super::reg_system;
    reg_system(m, "/usr/bin/ls", Trust::Real, run_ls);
    reg_system(m, "/usr/bin/mkdir", Trust::Real, run_mkdir);
    reg_system(m, "/usr/bin/rmdir", Trust::Real, run_rmdir);
    reg_system(m, "/usr/bin/rm", Trust::Real, run_rm);
    reg_system(m, "/usr/bin/cp", Trust::Real, run_cp);
    reg_system(m, "/usr/bin/mv", Trust::Real, run_mv);
    reg_system(m, "/usr/bin/touch", Trust::Real, run_touch);
    reg_system(m, "/usr/bin/ln", Trust::Real, run_ln);
    reg_system(m, "/usr/bin/chmod", Trust::Real, run_chmod);
    reg_system(m, "/usr/bin/chown", Trust::Real, run_chown);
    reg_system(m, "/usr/bin/chgrp", Trust::Real, run_chown);
    reg_system(m, "/usr/bin/basename", Trust::Real, run_basename);
    reg_system(m, "/usr/bin/dirname", Trust::Real, run_dirname);
    reg_system(m, "/usr/bin/realpath", Trust::Real, run_realpath);
    reg_system(m, "/usr/bin/readlink", Trust::Real, run_readlink);
    reg_system(m, "/usr/bin/stat", Trust::Real, run_stat);
    reg_system(m, "/usr/bin/du", Trust::Real, run_du);
    reg_system(m, "/usr/bin/mktemp", Trust::Real, run_mktemp);
    reg_system(m, "/usr/bin/file", Trust::Real, run_file);
    reg_system(m, "/usr/bin/install", Trust::Real, run_install);
    reg_system(m, "/usr/bin/truncate", Trust::Real, run_truncate);
    reg_system(m, "/usr/bin/tree", Trust::Real, run_tree);
}

fn run_ls(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let system = &mut *context.system;
    let mut flags = Vec::new();
    let mut ops = Vec::new();
    let mut options = true;
    for arg in args {
        if options && arg == "--" {
            options = false;
        } else if options && arg.starts_with('-') && arg != "-" {
            if arg.starts_with("--") {
                ewln(io.err, &format!("ls: unimplemented option '{arg}'"));
                return 2;
            }
            for flag in arg[1..].chars() {
                if !matches!(flag, 'l' | 'a' | 'A' | '1' | 'R' | 'd') {
                    ewln(io.err, &format!("ls: unimplemented option '-{flag}'"));
                    return 2;
                }
                flags.push(flag);
            }
        } else {
            ops.push(arg);
        }
    }
    let long = flags.contains(&'l');
    let all = flags.contains(&'a');
    let almost_all = flags.contains(&'A');
    let recursive = flags.contains(&'R');
    let directory_as_file = flags.contains(&'d');
    let cwd = system.cwd().to_string();
    let paths: Vec<String> = if ops.is_empty() {
        vec![cwd.clone()]
    } else {
        ops.iter().map(|s| s.to_string()).collect()
    };
    let mut status = 0;
    for p in &paths {
        if directory_as_file {
            if system.metadata(&cwd, p, false).is_ok() {
                emit_listing(
                    system,
                    ".",
                    std::slice::from_ref(p),
                    long,
                    ListingTotal::Omit,
                    io.out,
                );
            } else {
                ewln(
                    io.err,
                    &format!("ls: cannot access '{p}': No such file or directory"),
                );
                status = 2;
            }
            continue;
        }
        if matches!(
            system.metadata(&cwd, p, true),
            Ok(crate::syscalls::FileInfo {
                kind: FileKind::Directory,
                ..
            })
        ) {
            let mut entries = match system.list_dir(&cwd, p) {
                Ok(e) => e,
                Err(e) => {
                    ewln(io.err, &format!("ls: {e}"));
                    status = 2;
                    continue;
                }
            };
            if !all && !almost_all {
                entries.retain(|entry| !entry.starts_with('.'));
            } else if all {
                entries.insert(0, "..".into());
                entries.insert(0, ".".into());
            }
            if paths.len() > 1 || recursive {
                wln(io.out, &format!("{p}:"));
            }
            emit_listing(system, p, &entries, long, ListingTotal::Show, io.out);
            if recursive {
                let base = resolve_against(&cwd, p);
                let Ok(all_paths) = system.walk(&cwd, p) else {
                    status = 2;
                    continue;
                };
                for sub in all_paths.into_iter().skip(1) {
                    if !matches!(
                        system.metadata("/", &sub, false),
                        Ok(crate::syscalls::FileInfo {
                            kind: FileKind::Directory,
                            ..
                        })
                    ) {
                        continue;
                    }
                    // Headers name the directory the way the operand did, and a hidden directory
                    // is not descended into unless hidden entries were asked for.
                    let relative = sub
                        .strip_prefix(&base)
                        .unwrap_or(&sub)
                        .trim_start_matches('/');
                    if !all && !almost_all && relative.split('/').any(|part| part.starts_with('.'))
                    {
                        continue;
                    }
                    let label = format!("{}/{relative}", p.trim_end_matches('/'));
                    let mut sub_entries = system.list_dir("/", &sub).unwrap_or_default();
                    if !all && !almost_all {
                        sub_entries.retain(|entry| !entry.starts_with('.'));
                    } else if all {
                        sub_entries.insert(0, "..".into());
                        sub_entries.insert(0, ".".into());
                    }
                    wln(io.out, "");
                    wln(io.out, &format!("{label}:"));
                    emit_listing(system, &sub, &sub_entries, long, ListingTotal::Show, io.out);
                }
            }
        } else if system.metadata(&cwd, p, false).is_ok() {
            // A file operand is listed the same way an entry of a directory is.
            emit_listing(
                system,
                ".",
                std::slice::from_ref(p),
                long,
                ListingTotal::Omit,
                io.out,
            );
        } else {
            ewln(
                io.err,
                &format!("ls: cannot access '{p}': No such file or directory"),
            );
            status = 2;
        }
    }
    status
}

/// Whether a long listing starts with GNU's `total N` line: directory contents do, file
/// operands do not.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ListingTotal {
    Show,
    Omit,
}

fn emit_listing(
    system: &mut dyn System,
    dir: &str,
    entries: &[String],
    long: bool,
    total: ListingTotal,
    out: &mut Vec<u8>,
) {
    let cwd = system.cwd().to_string();
    if long {
        let mut lines = Vec::with_capacity(entries.len());
        // `total` counts allocated space in 1 KiB units; `blocks` is in 512-byte units.
        let mut total_kib = 0u64;
        for e in entries {
            let full = if e.starts_with('/') {
                e.clone()
            } else if e == "." {
                dir.to_string()
            } else if e == ".." {
                crate::vfs::parent_of(&crate::vfs::resolve_against(&cwd, dir))
                    .unwrap_or_else(|| "/".into())
            } else {
                format!("{}/{}", dir.trim_end_matches('/'), e)
            };
            let (typ, mode, size, target) = match system.metadata(&cwd, &full, false) {
                Ok(n) => {
                    let t = match n.kind {
                        FileKind::Directory => 'd',
                        FileKind::Symlink => 'l',
                        _ => '-',
                    };
                    // A long listing names what a link points at rather than following it.
                    let target = n
                        .link_target
                        .map_or_else(String::new, |target| format!(" -> {target}"));
                    total_kib = total_kib.saturating_add(n.blocks.div_ceil(2));
                    (t, n.mode, n.size, target)
                }
                Err(_) => ('-', 0o644, 0, String::new()),
            };
            lines.push(format!(
                "{}{} 1 root root {:>6} Jan  1 00:00 {}{}",
                typ,
                mode_str(mode),
                size,
                e,
                target
            ));
        }
        if total == ListingTotal::Show {
            wln(out, &format!("total {total_kib}"));
        }
        for line in lines {
            wln(out, &line);
        }
    } else {
        // Nothing here writes to a terminal, and `ls` writing to a pipe emits one name per line,
        // so `-1` is the only short form and needs no separate handling.
        for e in entries {
            wln(out, e);
        }
    }
}

fn mode_str(mode: u32) -> String {
    let bits = ['r', 'w', 'x'];
    let mut s = String::new();
    for shift in [6, 3, 0] {
        let g = (mode >> shift) & 0o7;
        for (i, b) in bits.iter().enumerate() {
            if g & (1 << (2 - i)) != 0 {
                s.push(*b);
            } else {
                s.push('-');
            }
        }
    }
    s
}

fn reject_options(
    command: &str,
    flags: &[char],
    allowed: &str,
    long: &[(&str, String)],
    io: &mut Io,
) -> bool {
    if let Some(flag) = flags.iter().find(|flag| !allowed.contains(**flag)) {
        ewln(
            io.err,
            &format!("{command}: unimplemented option '-{flag}'"),
        );
        return true;
    }
    if let Some((option, _)) = long.first() {
        ewln(
            io.err,
            &format!("{command}: unimplemented option '--{option}'"),
        );
        return true;
    }
    false
}

/// Execute directory creation through one process-scoped kernel handle.
fn run_mkdir(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let system = &mut *context.system;
    let (flags, ops, long) = split_flags(args);
    if reject_options("mkdir", &flags, "p", &long, io) {
        return 2;
    }
    if ops.is_empty() {
        ewln(io.err, "mkdir: missing operand");
        return 1;
    }
    let parents = flags.contains(&'p');
    let cwd = system.cwd().to_string();
    let mut status = 0;
    for d in &ops {
        let r = if parents {
            system.mkdir_all(&cwd, d)
        } else {
            system.mkdir(&cwd, d)
        };
        if let Err(e) = r {
            ewln(
                io.err,
                &format!("mkdir: cannot create directory '{d}': {e}"),
            );
            status = 1;
        }
    }
    status
}

fn run_rmdir(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let system = &mut *context.system;
    let (flags, ops, long) = split_flags(args);
    if reject_options("rmdir", &flags, "", &long, io) {
        return 2;
    }
    if ops.is_empty() {
        ewln(io.err, "rmdir: missing operand");
        return 1;
    }
    let cwd = system.cwd().to_string();
    let mut status = 0;
    for d in &ops {
        if let Err(e) = system.rmdir(&cwd, d) {
            ewln(io.err, &format!("rmdir: failed to remove '{d}': {e}"));
            status = 1;
        }
    }
    status
}

fn run_rm(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let system = &mut *context.system;
    let (flags, ops, long) = split_flags(args);
    if reject_options("rm", &flags, "rRf", &long, io) {
        return 2;
    }
    let recursive = flags.contains(&'r') || flags.contains(&'R');
    let force = flags.contains(&'f');
    if ops.is_empty() {
        if force {
            return 0;
        }
        ewln(io.err, "rm: missing operand");
        return 1;
    }
    let cwd = system.cwd().to_string();
    let mut status = 0;
    for t in &ops {
        let absolute = resolve_against(&cwd, t);
        if recursive && absolute == "/" {
            ewln(io.err, "rm: refusing to remove virtual root '/'");
            status = 1;
            continue;
        }
        let r = if recursive {
            system.walk(&cwd, t).and_then(|paths| {
                for path in paths.into_iter().rev() {
                    if !system.charge_cpu(1) {
                        return Err(crate::syscalls::SyscallError::ResourceExhausted);
                    }
                    let node = system.metadata("/", &path, false)?;
                    if node.kind == FileKind::Directory {
                        system.rmdir("/", &path)?;
                    } else {
                        system.unlink("/", &path)?;
                    }
                }
                Ok(())
            })
        } else {
            system.unlink(&cwd, t)
        };
        if let Err(e) = r {
            if matches!(e, crate::syscalls::SyscallError::ResourceExhausted) {
                return system.stop_status();
            }
            if !force {
                ewln(io.err, &format!("rm: cannot remove '{t}': {e}"));
                status = 1;
            }
        }
    }
    status
}

fn run_cp(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let system = &mut *context.system;
    let (flags, ops, long) = split_flags(args);
    if reject_options("cp", &flags, "rRafp", &long, io) {
        return 2;
    }
    let recursive = flags.contains(&'r') || flags.contains(&'R') || flags.contains(&'a');
    let preserve = flags.contains(&'p') || flags.contains(&'a');
    if ops.len() < 2 {
        ewln(io.err, "cp: missing destination operand");
        return 1;
    }
    let cwd = system.cwd().to_string();
    let dest = ops.last().unwrap();
    let sources = &ops[..ops.len() - 1];
    let destination_is_dir = matches!(
        system.metadata(&cwd, dest, true),
        Ok(crate::syscalls::FileInfo {
            kind: FileKind::Directory,
            ..
        })
    );
    if sources.len() > 1 && !destination_is_dir {
        ewln(io.err, &format!("cp: target '{dest}' is not a directory"));
        return 1;
    }
    let mut status = 0;
    for s in sources {
        if recursive
            && matches!(
                system.metadata(&cwd, s, false),
                Ok(crate::syscalls::FileInfo {
                    kind: FileKind::Symlink,
                    ..
                })
            )
        {
            ewln(io.err, "cp: unimplemented recursive copy of symbolic links");
            status = 2;
            continue;
        }
        let target = if destination_is_dir {
            format!("{}/{}", dest.trim_end_matches('/'), crate::vfs::basename(s))
        } else {
            (*dest).clone()
        };
        let r = if recursive {
            system.copy_recursive(&cwd, s, &target, preserve)
        } else {
            system.copy_file(&cwd, s, &target)
        };
        let r = r.and_then(|()| {
            if preserve && !recursive {
                let source = system.metadata(&cwd, s, true)?;
                system.chmod(&cwd, &target, source.mode)?;
                system.touch(&cwd, &target, source.mtime_ms)?;
            }
            Ok(())
        });
        match r {
            Ok(()) => {}
            Err(e) => {
                ewln(io.err, &format!("cp: cannot copy '{s}': {e}"));
                status = 1;
            }
        }
    }
    status
}

fn run_mv(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let system = &mut *context.system;
    let (flags, ops, long) = split_flags(args);
    if reject_options("mv", &flags, "f", &long, io) {
        return 2;
    }
    if ops.len() < 2 {
        ewln(io.err, "mv: missing destination operand");
        return 1;
    }
    let cwd = system.cwd().to_string();
    let dest = ops.last().unwrap();
    let destination_is_dir = matches!(
        system.metadata(&cwd, dest, true),
        Ok(crate::syscalls::FileInfo {
            kind: FileKind::Directory,
            ..
        })
    );
    if ops.len() > 2 && !destination_is_dir {
        ewln(io.err, &format!("mv: target '{dest}' is not a directory"));
        return 1;
    }
    let mut status = 0;
    for s in &ops[..ops.len() - 1] {
        let target = if destination_is_dir {
            format!("{}/{}", dest.trim_end_matches('/'), crate::vfs::basename(s))
        } else {
            (*dest).clone()
        };
        if let Err(e) = system.rename(&cwd, s, &target) {
            ewln(io.err, &format!("mv: cannot move '{s}': {e}"));
            status = 1;
        }
    }
    status
}

fn run_touch(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let system = &mut *context.system;
    if args.iter().any(|arg| arg.starts_with('-') && arg != "--") {
        ewln(io.err, "touch: unimplemented option");
        return 2;
    }
    let (_f, ops, _l) = split_flags(args);
    if ops.is_empty() {
        ewln(io.err, "touch: missing file operand");
        return 1;
    }
    let cwd = system.cwd().to_string();
    let now = system.wall_time_ms();
    let mut status = 0;
    for t in &ops {
        if let Err(e) = system.touch(&cwd, t, now) {
            ewln(io.err, &format!("touch: {e}"));
            status = 1;
        }
    }
    status
}

fn run_ln(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let system = &mut *context.system;
    let (flags, ops, long) = split_flags(args);
    if reject_options("ln", &flags, "sf", &long, io) {
        return 2;
    }
    let symbolic = flags.contains(&'s');
    if ops.len() < 2 {
        ewln(io.err, "ln: missing operand");
        return 1;
    }
    let cwd = system.cwd().to_string();
    let (target, link) = (ops[0], ops[1]);
    if !symbolic {
        ewln(io.err, "ln: unimplemented hard links");
        return 2;
    }
    let r = system.symlink(&cwd, target, link);
    if let Err(e) = r {
        ewln(io.err, &format!("ln: {e}"));
        1
    } else {
        0
    }
}

fn parse_mode(s: &str, cur: u32) -> u32 {
    if let Ok(oct) = u32::from_str_radix(s, 8) {
        if s.chars().all(|c| c.is_digit(8)) {
            return oct & 0o7777;
        }
    }
    // symbolic like u+x,g-w
    let mut mode = cur;
    for clause in s.split(',') {
        let (whoset, rest) = clause.split_at(clause.find(['+', '-', '=']).unwrap_or(0));
        if rest.is_empty() {
            continue;
        }
        let op = rest.chars().next().unwrap();
        let perms = &rest[1..];
        let mut mask = 0u32;
        for p in perms.chars() {
            mask |= match p {
                'r' => 0o444,
                'w' => 0o222,
                'x' => 0o111,
                _ => 0,
            };
        }
        let who_mask = if whoset.is_empty() || whoset.contains('a') {
            0o777
        } else {
            let mut m = 0;
            if whoset.contains('u') {
                m |= 0o700;
            }
            if whoset.contains('g') {
                m |= 0o070;
            }
            if whoset.contains('o') {
                m |= 0o007;
            }
            m
        };
        let bits = mask & who_mask;
        match op {
            '+' => mode |= bits,
            '-' => mode &= !bits,
            '=' => mode = (mode & !who_mask) | bits,
            _ => {}
        }
    }
    mode
}

fn run_chmod(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let system = &mut *context.system;
    let mut recursive = false;
    let mut mode_arg = None;
    let mut targets = Vec::new();
    for a in args {
        if a == "-R" || a == "--recursive" {
            recursive = true;
        } else if mode_arg.is_none()
            && (a.chars().all(|c| c.is_digit(8)) || a.contains(['+', '-', '=']))
            && !a.starts_with('/')
        {
            mode_arg = Some(a.clone());
        } else {
            targets.push(a.clone());
        }
    }
    let Some(mode_arg) = mode_arg else {
        ewln(io.err, "chmod: missing operand");
        return 1;
    };
    let cwd = system.cwd().to_string();
    let mut status = 0;
    for t in &targets {
        let paths = if recursive {
            let abs = crate::vfs::resolve_against(&cwd, t);
            match system.walk("/", &abs) {
                Ok(paths) => paths,
                Err(error) => {
                    ewln(io.err, &format!("chmod: {error}"));
                    status = 1;
                    continue;
                }
            }
        } else {
            vec![crate::vfs::resolve_against(&cwd, t)]
        };
        for p in paths {
            let cur = system
                .metadata("/", &p, true)
                .map(|n| n.mode)
                .unwrap_or(0o644);
            let m = parse_mode(&mode_arg, cur);
            if let Err(e) = system.chmod("/", &p, m) {
                ewln(io.err, &format!("chmod: {e}"));
                status = 1;
            }
        }
    }
    status
}

fn run_chown(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let system = &mut *context.system;
    let (flags, ops, long) = split_flags(args);
    if reject_options(context.command_name, &flags, "R", &long, io) {
        return 2;
    }
    let recursive = flags.contains(&'R');
    if ops.is_empty() {
        ewln(
            io.err,
            &format!("{}: missing operand", context.command_name),
        );
        return 1;
    }
    let spec = ops[0];
    let (uid, gid) = if context.command_name == "chgrp" {
        if spec.contains(':') {
            ewln(io.err, &format!("chgrp: invalid group '{spec}'"));
            return 1;
        }
        (None, parse_owner(spec).0)
    } else {
        parse_owner(spec)
    };
    let cwd = system.cwd().to_string();
    let mut status = 0;
    for t in &ops[1..] {
        let paths = if recursive {
            match system.walk(&cwd, t) {
                Ok(paths) => paths,
                Err(error) => {
                    ewln(io.err, &format!("chown: {error}"));
                    status = 1;
                    continue;
                }
            }
        } else {
            vec![crate::vfs::resolve_against(&cwd, t)]
        };
        for p in paths {
            if let Err(e) = system.chown("/", &p, uid, gid) {
                ewln(io.err, &format!("chown: {e}"));
                status = 1;
            }
        }
    }
    status
}

fn parse_owner(spec: &str) -> (Option<u32>, Option<u32>) {
    let name_to_uid = |n: &str| -> Option<u32> {
        n.parse().ok().or(match n {
            "root" => Some(0),
            "" => None,
            _ => Some(1000),
        })
    };
    if let Some((u, g)) = spec.split_once(':') {
        (name_to_uid(u), name_to_uid(g))
    } else {
        (name_to_uid(spec), None)
    }
}

fn run_basename(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let Some(p) = args.first() else { return 1 };
    let trimmed = p.trim_end_matches('/');
    let mut base = if trimmed.is_empty() && p.starts_with('/') {
        "/".to_string()
    } else {
        crate::vfs::basename(trimmed).to_string()
    };
    if let Some(suffix) = args.get(1) {
        if base.ends_with(suffix.as_str()) && &base != suffix {
            base = base[..base.len() - suffix.len()].to_string();
        }
    }
    wln(io.out, &base);
    0
}

fn run_dirname(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let Some(p) = args.first() else { return 1 };
    let p = p.trim_end_matches('/');
    if p.is_empty() {
        wln(io.out, "/");
        return 0;
    }
    let d = match p.rfind('/') {
        Some(0) => "/".to_string(),
        Some(i) => p[..i].to_string(),
        None => ".".to_string(),
    };
    wln(io.out, &d);
    0
}

fn run_realpath(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    realpath_impl(context, "realpath", io)
}

fn run_readlink(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    realpath_impl(context, "readlink", io)
}

fn realpath_impl(context: &mut crate::program::ProcessContext<'_>, cmd: &str, io: &mut Io) -> i32 {
    let args = context.args;
    let system = &mut *context.system;
    let cwd = system.cwd().to_string();
    let (flags, ops, long) = split_flags(args);
    let allowed = if cmd == "readlink" { "f" } else { "m" };
    if reject_options(cmd, &flags, allowed, &long, io) {
        return 2;
    }
    if ops.is_empty() {
        ewln(io.err, &format!("{cmd}: missing operand"));
        return 1;
    }
    for p in &ops {
        if cmd == "readlink" {
            if flags.contains(&'f') {
                match system.canonicalize(&cwd, p, true) {
                    Ok(r) => wln(io.out, &r),
                    Err(_) => return 1,
                }
            } else {
                match system.read_link(&cwd, p) {
                    Ok(t) => wln(io.out, &t),
                    Err(_) => {
                        ewln(io.err, &format!("readlink: {p}: Invalid argument"));
                        return 1;
                    }
                }
            }
        } else {
            let resolved = system.canonicalize(&cwd, p, !flags.contains(&'m'));
            match resolved {
                Ok(r) => wln(io.out, &r),
                Err(error) => {
                    ewln(io.err, &format!("realpath: {error}"));
                    return 1;
                }
            }
        }
    }
    0
}

fn run_stat(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let system = &mut *context.system;
    let cwd = system.cwd().to_string();
    let (_f, ops, long) = split_flags(args);
    let fmt = long
        .iter()
        .find(|(k, _)| *k == "format" || *k == "printf")
        .map(|(_, v)| v.clone());
    // also handle -c FORMAT
    let mut format = fmt;
    let mut files = Vec::new();
    let follow = args.iter().any(|arg| arg == "-L" || arg == "--dereference");
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "-c" || a == "--format" {
            format = it.next().cloned();
        } else if let Some(value) = a
            .strip_prefix("--format=")
            .or_else(|| a.strip_prefix("--printf="))
        {
            format = Some(value.to_string());
        } else if a == "-L" || a == "--dereference" {
        } else if !a.starts_with('-') {
            files.push(a.clone());
        } else {
            ewln(io.err, &format!("stat: unimplemented option '{a}'"));
            return 2;
        }
    }
    let _ = ops;
    if files.is_empty() {
        ewln(io.err, "stat: missing operand");
        return 1;
    }
    if let Some(value) = &format {
        let mut chars = value.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '%' {
                let Some(code) = chars.next() else { break };
                if !matches!(
                    code,
                    '%' | 's' | 'n' | 'a' | 'U' | 'u' | 'g' | 'Y' | 'F' | 'N' | 'b' | 'B' | 'o'
                ) {
                    ewln(
                        io.err,
                        &format!("stat: unimplemented format directive '%{code}'"),
                    );
                    return 2;
                }
            }
        }
    }
    for f in &files {
        match system.metadata(&cwd, f, follow) {
            Ok(n) => {
                let size = n.size;
                if let Some(fmt) = &format {
                    let s = fmt
                        .replace("%%", "\0")
                        .replace("%s", &size.to_string())
                        .replace("%b", &n.blocks.to_string())
                        .replace("%B", "512")
                        .replace("%o", "4096")
                        .replace("%n", f)
                        .replace("%a", &format!("{:o}", n.mode))
                        .replace("%U", "root")
                        .replace("%u", &n.uid.to_string())
                        .replace("%g", &n.gid.to_string())
                        .replace("%Y", &(n.mtime_ms / 1000).to_string())
                        .replace(
                            "%F",
                            match n.kind {
                                FileKind::File if n.native_executable => "native executable",
                                FileKind::File => "regular file",
                                FileKind::Directory => "directory",
                                FileKind::Symlink => "symbolic link",
                            },
                        )
                        .replace("%N", f)
                        .replace('\0', "%");
                    wln(io.out, &s);
                } else {
                    wln(io.out, &format!("  File: {f}\n  Size: {size}"));
                }
            }
            Err(e) => {
                ewln(io.err, &format!("stat: {e}"));
                return 1;
            }
        }
    }
    0
}

/// Output units for `du`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DuUnits {
    Kibibytes,
    Bytes,
    Human,
}

/// Options for `du`, following GNU coreutils.
struct DuOptions {
    all: bool,
    summarize: bool,
    grand_total: bool,
    apparent: bool,
    units: DuUnits,
    max_depth: Option<usize>,
}

fn parse_du_options(args: &[String]) -> Result<(DuOptions, Vec<String>), String> {
    let mut options = DuOptions {
        all: false,
        summarize: false,
        grand_total: false,
        apparent: false,
        units: DuUnits::Kibibytes,
        max_depth: None,
    };
    let mut operands = Vec::new();
    let mut arguments = args.iter();
    let parse_depth = |value: &str| {
        value
            .parse::<usize>()
            .map_err(|_| format!("invalid maximum depth '{value}'"))
    };
    while let Some(argument) = arguments.next() {
        if argument == "--" {
            operands.extend(arguments.by_ref().cloned());
            break;
        }
        if let Some(long) = argument.strip_prefix("--") {
            match long.split_once('=') {
                Some(("max-depth", value)) => options.max_depth = Some(parse_depth(value)?),
                None if long == "all" => options.all = true,
                None if long == "summarize" => options.summarize = true,
                None if long == "total" => options.grand_total = true,
                None if long == "apparent-size" => options.apparent = true,
                None if long == "bytes" => {
                    options.apparent = true;
                    options.units = DuUnits::Bytes;
                }
                None if long == "human-readable" => options.units = DuUnits::Human,
                _ => return Err(format!("unrecognized option '{argument}'")),
            }
            continue;
        }
        let Some(flags) = argument.strip_prefix('-').filter(|flags| !flags.is_empty()) else {
            operands.push(argument.clone());
            continue;
        };
        for (index, flag) in flags.char_indices() {
            match flag {
                'a' => options.all = true,
                's' => options.summarize = true,
                'c' => options.grand_total = true,
                'k' => options.units = DuUnits::Kibibytes,
                'h' => options.units = DuUnits::Human,
                'b' => {
                    options.apparent = true;
                    options.units = DuUnits::Bytes;
                }
                'd' => {
                    let rest = &flags[index + 1..];
                    let value = if rest.is_empty() {
                        arguments
                            .next()
                            .ok_or_else(|| "option requires an argument -- 'd'".to_string())?
                            .clone()
                    } else {
                        rest.to_string()
                    };
                    options.max_depth = Some(parse_depth(&value)?);
                    break;
                }
                other => return Err(format!("invalid option -- '{other}'")),
            }
        }
    }
    if options.summarize && options.all {
        return Err("cannot both summarize and show all entries".to_string());
    }
    if operands.is_empty() {
        operands.push(".".to_string());
    }
    Ok((options, operands))
}

/// Format a byte count the way GNU `du -h` does: bare below 1 KiB, one decimal rounded up
/// below 10 units, otherwise a whole number rounded up.
fn du_human(bytes: u64) -> String {
    if bytes < 1024 {
        return bytes.to_string();
    }
    let units = ['K', 'M', 'G', 'T', 'P', 'E'];
    let mut scale = 1024u64;
    for (index, unit) in units.iter().enumerate() {
        let next = scale.saturating_mul(1024);
        if bytes >= next && index + 1 < units.len() {
            scale = next;
            continue;
        }
        let tenths = u128::from(bytes)
            .saturating_mul(10)
            .div_ceil(u128::from(scale));
        if tenths < 100 {
            return format!("{}.{}{unit}", tenths / 10, tenths % 10);
        }
        let whole = u128::from(bytes).div_ceil(u128::from(scale));
        if whole >= 1024 && index + 1 < units.len() {
            scale = next;
            continue;
        }
        return format!("{whole}{unit}");
    }
    unreachable!("the last unit always formats")
}

fn du_amount(bytes: u64, units: DuUnits) -> String {
    match units {
        DuUnits::Bytes => bytes.to_string(),
        DuUnits::Kibibytes => bytes.div_ceil(1024).to_string(),
        DuUnits::Human => du_human(bytes),
    }
}

/// Usage of one node in bytes: allocated blocks by default, `st_size` for `--apparent-size`.
fn du_usage(info: &crate::syscalls::FileInfo, apparent: bool) -> u64 {
    if apparent {
        info.size
    } else {
        info.blocks.saturating_mul(512)
    }
}

/// A directory whose children are still being summed during `du`'s post-order walk.
struct DuDirectory {
    path: String,
    depth: usize,
    total: u64,
    pending: std::collections::VecDeque<String>,
}

fn run_du(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let system = &mut *context.system;
    let cwd = system.cwd().to_string();
    let (options, operands) = match parse_du_options(args) {
        Ok(parsed) => parsed,
        Err(message) => {
            ewln(io.err, &format!("du: {message}"));
            return 1;
        }
    };
    let shows = |depth: usize, directory: bool| {
        let within = options.max_depth.is_none_or(|limit| depth <= limit);
        if options.summarize {
            depth == 0
        } else {
            within && (directory || options.all || depth == 0)
        }
    };
    let mut status = 0;
    let mut grand_total = 0u64;
    for operand in &operands {
        let info = match system.metadata(&cwd, operand, false) {
            Ok(info) => info,
            Err(error) => {
                let reason = match &error {
                    crate::syscalls::SyscallError::File(error) => error.reason().to_string(),
                    other => other.to_string(),
                };
                ewln(io.err, &format!("du: cannot access '{operand}': {reason}"));
                status = 1;
                continue;
            }
        };
        if info.kind != FileKind::Directory {
            let usage = du_usage(&info, options.apparent);
            grand_total = grand_total.saturating_add(usage);
            wln(
                io.out,
                &format!("{}\t{operand}", du_amount(usage, options.units)),
            );
            continue;
        }
        // An explicit stack gives the post-order output GNU du uses (children before their
        // directory) without host recursion over untrusted tree depth.
        let mut stack = vec![DuDirectory {
            path: operand.clone(),
            depth: 0,
            total: du_usage(&info, options.apparent),
            pending: Vec::new().into(),
        }];
        match system.list_dir(&cwd, operand) {
            Ok(entries) => stack[0].pending = entries.into(),
            Err(error) => {
                ewln(
                    io.err,
                    &format!("du: cannot read directory '{operand}': {error}"),
                );
                status = 1;
            }
        }
        while let Some(top) = stack.last_mut() {
            let Some(name) = top.pending.pop_front() else {
                let done = stack.pop().expect("stack has a top entry");
                if shows(done.depth, true) {
                    wln(
                        io.out,
                        &format!("{}\t{}", du_amount(done.total, options.units), done.path),
                    );
                }
                match stack.last_mut() {
                    Some(parent) => parent.total = parent.total.saturating_add(done.total),
                    None => grand_total = grand_total.saturating_add(done.total),
                }
                continue;
            };
            if !system.charge_cpu(1) {
                return 137;
            }
            let path = format!("{}/{name}", top.path.trim_end_matches('/'));
            let depth = top.depth + 1;
            let Ok(child) = system.metadata(&cwd, &path, false) else {
                continue;
            };
            let usage = du_usage(&child, options.apparent);
            if child.kind == FileKind::Directory {
                let pending = match system.list_dir(&cwd, &path) {
                    Ok(entries) => entries.into(),
                    Err(error) => {
                        ewln(
                            io.err,
                            &format!("du: cannot read directory '{path}': {error}"),
                        );
                        status = 1;
                        Vec::new().into()
                    }
                };
                stack.push(DuDirectory {
                    path,
                    depth,
                    total: usage,
                    pending,
                });
            } else {
                top.total = top.total.saturating_add(usage);
                if shows(depth, false) {
                    wln(
                        io.out,
                        &format!("{}\t{path}", du_amount(usage, options.units)),
                    );
                }
            }
        }
    }
    if options.grand_total {
        wln(
            io.out,
            &format!("{}\ttotal", du_amount(grand_total, options.units)),
        );
    }
    status
}

fn run_mktemp(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let system = &mut *context.system;
    let (flags, ops, long) = split_flags(args);
    if reject_options("mktemp", &flags, "d", &long, io) {
        return 2;
    }
    let dir = flags.contains(&'d');
    let tmpl = ops.first().map(|s| s.as_str()).unwrap_or("tmp.XXXXXX");
    // Identity is deterministic but deliberately separate from time: creating a name is not a
    // temporal effect and must not perturb deadlines.
    let Some(n) = system.allocate_temp_id() else {
        ewln(io.err, "mktemp: exhausted deterministic name space");
        return 1;
    };
    let suffix = format!("{n:06}");
    let name = tmpl.replace("XXXXXX", &suffix);
    let path = if name.starts_with('/') {
        name
    } else {
        format!("/tmp/{name}")
    };
    if let Err(error) = system.mkdir_all("/", "/tmp") {
        ewln(io.err, &format!("mktemp: {error}"));
        return 1;
    }
    let result = if dir {
        system.mkdir("/", &path)
    } else {
        system
            .open_file(
                "/",
                &path,
                crate::syscalls::OpenFile {
                    readable: false,
                    writable: true,
                    create: true,
                    exclusive: true,
                    truncate: false,
                    append: false,
                },
            )
            .and_then(|fd| {
                let result = system.chmod("/", &path, 0o600);
                let close = system.close(fd);
                result.and(close)
            })
    };
    if let Err(error) = result {
        ewln(io.err, &format!("mktemp: {error}"));
        return 1;
    }
    wln(io.out, &path);
    0
}

fn run_file(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let system = &mut *context.system;
    let (flags, ops, long) = split_flags(args);
    if reject_options("file", &flags, "", &long, io) {
        return 2;
    }
    if ops.is_empty() {
        ewln(io.err, "file: missing operand");
        return 1;
    }
    let cwd = system.cwd().to_string();
    let read_limit = system.limits().memory.min(64 * 1024 * 1024) as usize;
    let mut status = 0;
    for p in &ops {
        let desc = match system.read_file_limited(&cwd, p, read_limit) {
            Ok(d) if d.is_empty() => "empty".to_string(),
            Ok(d)
                if d.iter().all(|b| b.is_ascii() || *b >= 0x80)
                    && std::str::from_utf8(&d).is_ok() =>
            {
                "ASCII text".to_string()
            }
            Ok(_) => "data".to_string(),
            Err(_)
                if matches!(
                    system.metadata(&cwd, p, false),
                    Ok(crate::syscalls::FileInfo {
                        kind: FileKind::Directory,
                        ..
                    })
                ) =>
            {
                "directory".to_string()
            }
            Err(error) => {
                status = 1;
                format!("cannot open ({error})")
            }
        };
        wln(io.out, &format!("{p}: {desc}"));
    }
    status
}

fn run_install(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let system = &mut *context.system;
    let mut create_parents = false;
    let mut directories = false;
    let mut mode_arg: Option<String> = None;
    let mut operands = Vec::new();
    let mut options = true;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if options && arg == "--" {
            options = false;
        } else if options && arg == "-D" {
            create_parents = true;
        } else if options && (arg == "-d" || arg == "--directory") {
            directories = true;
        } else if options && arg == "-Dm" {
            create_parents = true;
            index += 1;
            let Some(value) = args.get(index) else {
                ewln(io.err, "install: option requires an argument -- 'm'");
                return 2;
            };
            mode_arg = Some(value.clone());
        } else if options && arg.starts_with("-Dm") && arg.len() > 3 {
            create_parents = true;
            mode_arg = Some(arg[3..].to_string());
        } else if options && (arg == "-m" || arg == "--mode") {
            index += 1;
            let Some(value) = args.get(index) else {
                ewln(io.err, "install: option requires an argument -- 'm'");
                return 2;
            };
            mode_arg = Some(value.clone());
        } else if options && arg.starts_with("-m") && arg.len() > 2 {
            mode_arg = Some(arg[2..].to_string());
        } else if options && arg.starts_with("--mode=") {
            mode_arg = Some(arg[7..].to_string());
        } else if options && arg.starts_with('-') && arg != "-" {
            ewln(io.err, &format!("install: unimplemented option '{arg}'"));
            return 2;
        } else {
            operands.push(arg.clone());
        }
        index += 1;
    }

    if create_parents && directories {
        ewln(io.err, "install: options '-D' and '-d' cannot be combined");
        return 2;
    }
    let default_mode = 0o755;
    let Some(mode) = mode_arg.as_deref().map_or(Some(default_mode), |value| {
        install_mode(value, default_mode)
    }) else {
        ewln(
            io.err,
            &format!(
                "install: invalid mode '{}'",
                mode_arg.as_deref().unwrap_or_default()
            ),
        );
        return 1;
    };

    let cwd = system.cwd().to_string();
    if directories {
        if operands.is_empty() {
            ewln(io.err, "install: missing operand");
            return 1;
        }
        let mut status = 0;
        for path in operands {
            if !system.charge_cpu(1) {
                return system.stop_status();
            }
            if let Err(error) = system
                .mkdir_all(&cwd, &path)
                .and_then(|()| system.chmod(&cwd, &path, mode))
            {
                ewln(
                    io.err,
                    &format!("install: cannot create directory '{path}': {error}"),
                );
                status = 1;
            }
        }
        return status;
    }

    if operands.len() < 2 {
        ewln(io.err, "install: missing destination file operand");
        return 1;
    }
    let destination = operands.last().expect("operand count checked");
    let destination_is_dir = matches!(
        system.metadata(&cwd, destination, true),
        Ok(crate::syscalls::FileInfo {
            kind: FileKind::Directory,
            ..
        })
    );
    let sources = &operands[..operands.len() - 1];
    if sources.len() > 1 && (!destination_is_dir || create_parents) {
        ewln(
            io.err,
            &format!("install: target '{destination}' is not a directory"),
        );
        return 1;
    }

    let mut status = 0;
    for source in sources {
        let target = if destination_is_dir {
            format!(
                "{}/{}",
                destination.trim_end_matches('/'),
                crate::vfs::basename(source)
            )
        } else {
            destination.clone()
        };
        let read_limit = system.limits().memory.min(64 * 1024 * 1024) as usize;
        let data = match system.read_file_limited(&cwd, source, read_limit) {
            Ok(data) => data,
            Err(error) => {
                ewln(io.err, &format!("install: cannot stat '{source}': {error}"));
                status = 1;
                continue;
            }
        };
        if !system.charge_cpu(data.len() as u64) {
            return system.stop_status();
        }
        let result = if create_parents {
            let absolute = resolve_against(&cwd, &target);
            if matches!(
                system.metadata("/", &absolute, false),
                Ok(crate::syscalls::FileInfo {
                    kind: FileKind::Directory,
                    ..
                })
            ) {
                Err(crate::syscalls::SyscallError::IsDirectory)
            } else {
                // `put_file` plans the missing parent directories and the file replacement as
                // one quota-checked mutation. A failed `install -D` therefore leaves no parents.
                system.put_file_with_parents("/", &absolute, data, mode)
            }
        } else {
            system
                .write_file(&cwd, &target, &data, mode)
                .and_then(|()| system.chmod(&cwd, &target, mode))
        };
        if let Err(error) = result {
            ewln(
                io.err,
                &format!("install: cannot create regular file '{target}': {error}"),
            );
            status = 1;
        }
    }
    status
}

fn install_mode(value: &str, default: u32) -> Option<u32> {
    if !value.is_empty() && value.chars().all(|ch| ch.is_digit(8)) {
        return u32::from_str_radix(value, 8).ok().map(|mode| mode & 0o7777);
    }
    if value.split(',').all(|clause| {
        let Some(operator) = clause.find(['+', '-', '=']) else {
            return false;
        };
        let (who, permission) = clause.split_at(operator);
        who.chars().all(|ch| matches!(ch, 'u' | 'g' | 'o' | 'a'))
            && permission.len() > 1
            && permission[1..]
                .chars()
                .all(|ch| matches!(ch, 'r' | 'w' | 'x'))
    }) {
        Some(parse_mode(value, default))
    } else {
        None
    }
}

fn run_truncate(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let system = &mut *context.system;
    let mut no_create = false;
    let mut size_arg = None;
    let mut files = Vec::new();
    let mut options = true;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if options && arg == "--" {
            options = false;
        } else if options && (arg == "-c" || arg == "--no-create") {
            no_create = true;
        } else if options && (arg == "-s" || arg == "--size") {
            index += 1;
            let Some(value) = args.get(index) else {
                ewln(io.err, "truncate: option requires an argument -- 's'");
                return 2;
            };
            size_arg = Some(value.clone());
        } else if options && arg.starts_with("-s") && arg.len() > 2 {
            size_arg = Some(arg[2..].to_string());
        } else if options && arg.starts_with("--size=") {
            size_arg = Some(arg[7..].to_string());
        } else if options && arg.starts_with('-') && arg != "-" {
            ewln(io.err, &format!("truncate: unimplemented option '{arg}'"));
            return 2;
        } else {
            files.push(arg.clone());
        }
        index += 1;
    }
    let Some(size_arg) = size_arg else {
        ewln(io.err, "truncate: missing operand");
        return 1;
    };
    let Some(change) = parse_size_change(&size_arg) else {
        ewln(io.err, &format!("truncate: invalid number: '{size_arg}'"));
        return 1;
    };
    if files.is_empty() {
        ewln(io.err, "truncate: missing file operand");
        return 1;
    }

    let cwd = system.cwd().to_string();
    let mut status = 0;
    for file in files {
        let (existing_size, mode, existing) = match system.metadata(&cwd, &file, true) {
            Ok(node) => match node.kind {
                FileKind::File if !node.native_executable => (node.size, node.mode, true),
                _ => {
                    ewln(
                        io.err,
                        &format!("truncate: cannot open '{file}': Not a file"),
                    );
                    status = 1;
                    continue;
                }
            },
            Err(_) if no_create => continue,
            Err(_) => (0, 0o666, false),
        };
        let Ok(existing_size) = usize::try_from(existing_size) else {
            ewln(
                io.err,
                &format!("truncate: cannot open '{file}': file too large"),
            );
            status = 1;
            continue;
        };
        let Some(new_len) = change.apply(existing_size) else {
            ewln(io.err, &format!("truncate: invalid number: '{size_arg}'"));
            status = 1;
            continue;
        };
        if !system.charge_cpu(new_len.saturating_sub(existing_size) as u64) {
            return system.stop_status();
        }
        let reservation = existing_size.max(new_len) as u64;
        if !system.reserve_memory(reservation) {
            return system.stop_status();
        }
        let data = if existing {
            system.read_file_limited(&cwd, &file, existing_size)
        } else {
            Ok(Vec::new())
        };
        let mut data = match data {
            Ok(data) => data,
            Err(error) => {
                system.release_memory(reservation);
                ewln(io.err, &format!("truncate: cannot open '{file}': {error}"));
                status = 1;
                continue;
            }
        };
        data.resize(new_len, 0);
        let result = system.write_file(&cwd, &file, &data, mode);
        system.release_memory(reservation);
        if let Err(error) = result {
            ewln(io.err, &format!("truncate: cannot open '{file}': {error}"));
            status = 1;
        }
    }
    status
}

#[derive(Clone, Copy)]
enum SizeChange {
    Absolute(usize),
    Increase(usize),
    Decrease(usize),
}

impl SizeChange {
    fn apply(self, current: usize) -> Option<usize> {
        match self {
            Self::Absolute(size) => Some(size),
            Self::Increase(size) => current.checked_add(size),
            Self::Decrease(size) => Some(current.saturating_sub(size)),
        }
    }
}

fn parse_size_change(value: &str) -> Option<SizeChange> {
    let (kind, amount) = match value.as_bytes().first() {
        Some(b'+') => (1, &value[1..]),
        Some(b'-') => (2, &value[1..]),
        _ => (0, value),
    };
    if amount.is_empty() {
        return None;
    }
    let digits = amount.trim_end_matches(|ch: char| ch.is_ascii_alphabetic());
    let suffix = &amount[digits.len()..];
    let multiplier = match suffix {
        "" => 1usize,
        "K" | "KiB" => 1024,
        "M" | "MiB" => 1024 * 1024,
        "G" | "GiB" => 1024 * 1024 * 1024,
        "KB" => 1000,
        "MB" => 1000 * 1000,
        "GB" => 1000 * 1000 * 1000,
        _ => return None,
    };
    let size = digits.parse::<usize>().ok()?.checked_mul(multiplier)?;
    Some(match kind {
        1 => SizeChange::Increase(size),
        2 => SizeChange::Decrease(size),
        _ => SizeChange::Absolute(size),
    })
}

fn run_tree(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let system = &mut *context.system;
    let cwd = system.cwd().to_string();
    let mut all = false;
    let mut max_depth = None;
    let mut operands = Vec::new();
    let mut options = true;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if options && arg == "--" {
            options = false;
        } else if options && (arg == "-a" || arg == "--all") {
            all = true;
        } else if options && (arg == "-L" || arg == "--level") {
            index += 1;
            let Some(value) = args.get(index) else {
                ewln(io.err, "tree: option requires an argument -- 'L'");
                return 2;
            };
            let Ok(depth) = value.parse::<usize>() else {
                ewln(io.err, "tree: invalid level, must be greater than 0");
                return 2;
            };
            if depth == 0 {
                ewln(io.err, "tree: invalid level, must be greater than 0");
                return 2;
            }
            max_depth = Some(depth);
        } else if options && arg.starts_with('-') && arg != "-" {
            ewln(io.err, &format!("tree: unimplemented option '{arg}'"));
            return 2;
        } else {
            operands.push(arg.clone());
        }
        index += 1;
    }
    if operands.len() > 1 {
        ewln(io.err, "tree: too many operands");
        return 2;
    }
    let root = operands.first().map_or(".", String::as_str);
    let node = match system.metadata(&cwd, root, false) {
        Ok(node) => node,
        Err(error) => {
            ewln(io.err, &format!("tree: {root}: {error}"));
            return 1;
        }
    };
    wln(io.out, root);
    let mut directories = 0usize;
    let mut files = 0usize;
    if matches!(node.kind, FileKind::Directory) {
        let status = tree_directory(
            system,
            root,
            "",
            1,
            max_depth,
            all,
            &mut directories,
            &mut files,
            io,
        );
        if status != 0 {
            return status;
        }
    } else {
        files = 1;
    }
    wln(io.out, "");
    wln(
        io.out,
        &format!(
            "{directories} director{}, {files} file{}",
            if directories == 1 { "y" } else { "ies" },
            if files == 1 { "" } else { "s" }
        ),
    );
    0
}

#[allow(clippy::too_many_arguments)]
fn tree_directory(
    system: &mut dyn System,
    path: &str,
    prefix: &str,
    depth: usize,
    max_depth: Option<usize>,
    all: bool,
    directories: &mut usize,
    files: &mut usize,
    io: &mut Io,
) -> i32 {
    let cwd = system.cwd().to_string();
    let mut entries = match system.list_dir(&cwd, path) {
        Ok(entries) => entries,
        Err(error) => {
            ewln(io.err, &format!("tree: {path}: {error}"));
            return 1;
        }
    };
    entries.sort();
    if !all {
        entries.retain(|entry| !entry.starts_with('.'));
    }
    for (position, entry) in entries.iter().enumerate() {
        if !system.charge_cpu(1) {
            return system.stop_status();
        }
        let last = position + 1 == entries.len();
        let child = if path == "/" {
            format!("/{entry}")
        } else {
            format!("{}/{entry}", path.trim_end_matches('/'))
        };
        let node = match system.metadata(&cwd, &child, false) {
            Ok(node) => node,
            Err(error) => {
                ewln(io.err, &format!("tree: {child}: {error}"));
                return 1;
            }
        };
        let suffix = node
            .link_target
            .as_ref()
            .map_or_else(String::new, |target| format!(" -> {target}"));
        wln(
            io.out,
            &format!(
                "{prefix}{} {entry}{suffix}",
                if last { "└──" } else { "├──" }
            ),
        );
        match node.kind {
            FileKind::Directory => {
                *directories = directories.saturating_add(1);
                if max_depth.is_none_or(|maximum| depth < maximum) {
                    let next_prefix = format!("{prefix}{}", if last { "    " } else { "│   " });
                    let status = tree_directory(
                        system,
                        &child,
                        &next_prefix,
                        depth + 1,
                        max_depth,
                        all,
                        directories,
                        files,
                        io,
                    );
                    if status != 0 {
                        return status;
                    }
                }
            }
            _ => *files = files.saturating_add(1),
        }
    }
    0
}
