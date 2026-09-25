//! Filesystem commands operating on the VFS: listing (ls), tree mutation
//! (mkdir/rmdir/rm/cp/mv/touch/ln), permissions (chmod/chown), path math
//! (basename/dirname/realpath/readlink), inspection (stat/file/find/du/tree), installation,
//! truncation, and temporary files.

use std::collections::HashMap;

use crate::commands::util::{ewln, split_flags, wln};
use crate::commands::{CommandContext, CommandSpec, Io, Trust};
use crate::interp::Interp;
use crate::syscalls::{FileKind, System};
use crate::vfs::resolve_against;

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    use super::reg;
    reg(m, &["ls"], Trust::Real, cmd_ls);
    reg(m, &["mkdir"], Trust::Real, cmd_mkdir);
    reg(m, &["rmdir"], Trust::Real, cmd_rmdir);
    reg(m, &["rm"], Trust::Real, cmd_rm);
    reg(m, &["cp"], Trust::Real, cmd_cp);
    reg(m, &["mv"], Trust::Real, cmd_mv);
    reg(m, &["touch"], Trust::Real, cmd_touch);
    reg(m, &["ln"], Trust::Real, cmd_ln);
    reg(m, &["chmod"], Trust::Real, cmd_chmod);
    reg(m, &["chown", "chgrp"], Trust::Real, cmd_chown);
    reg(m, &["basename"], Trust::Real, cmd_basename);
    reg(m, &["dirname"], Trust::Real, cmd_dirname);
    reg(m, &["realpath"], Trust::Real, cmd_realpath);
    reg(m, &["readlink"], Trust::Real, cmd_readlink);
    reg(m, &["stat"], Trust::Real, cmd_stat);
    reg(m, &["du"], Trust::Real, cmd_du);
    reg(m, &["mktemp"], Trust::Real, cmd_mktemp);
    reg(m, &["file"], Trust::Real, cmd_file);
    reg(m, &["install"], Trust::Real, cmd_install);
    reg(m, &["truncate"], Trust::Real, cmd_truncate);
    reg(m, &["tree"], Trust::Real, cmd_tree);
}

fn cmd_ls(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    run_ls(&mut interp.system(), args, io)
}

fn run_ls(system: &mut impl System, args: &[String], io: &mut Io) -> i32 {
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
                emit_listing(system, ".", std::slice::from_ref(p), long, io.out);
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
            emit_listing(system, p, &entries, long, io.out);
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
                    emit_listing(system, &sub, &sub_entries, long, io.out);
                }
            }
        } else if system.metadata(&cwd, p, false).is_ok() {
            // A file operand is listed the same way an entry of a directory is.
            emit_listing(system, ".", std::slice::from_ref(p), long, io.out);
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

fn emit_listing(
    system: &mut impl System,
    dir: &str,
    entries: &[String],
    long: bool,
    out: &mut Vec<u8>,
) {
    let cwd = system.cwd().to_string();
    if long {
        for e in entries {
            let full = if e == "." {
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
                    (t, n.mode, n.size, target)
                }
                Err(_) => ('-', 0o644, 0, String::new()),
            };
            wln(
                out,
                &format!(
                    "{}{} 1 root root {:>6} Jan  1 00:00 {}{}",
                    typ,
                    mode_str(mode),
                    size,
                    e,
                    target
                ),
            );
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

fn cmd_mkdir(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    run_mkdir(&mut interp.system(), args, io)
}

/// Execute directory creation through one process-scoped kernel handle.
pub(crate) fn run_mkdir(system: &mut impl System, args: &[String], io: &mut Io) -> i32 {
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

fn cmd_rmdir(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (flags, ops, long) = split_flags(args);
    if reject_options("rmdir", &flags, "", &long, io) {
        return 2;
    }
    if ops.is_empty() {
        ewln(io.err, "rmdir: missing operand");
        return 1;
    }
    let mut system = interp.system();
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

fn cmd_rm(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
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
    let cwd = interp.cwd.clone();
    let mut status = 0;
    for t in &ops {
        let r = if recursive {
            interp.vfs.remove_all(&cwd, t)
        } else {
            interp.vfs.remove_file(&cwd, t)
        };
        if let Err(e) = r {
            if !force {
                ewln(io.err, &format!("rm: cannot remove '{t}': {e}"));
                status = 1;
            }
        }
    }
    status
}

fn cmd_cp(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
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
    let cwd = interp.cwd.clone();
    let dest = ops.last().unwrap();
    let sources = &ops[..ops.len() - 1];
    let destination_is_dir = matches!(
        interp.fs_metadata(&cwd, dest, true),
        Ok(crate::vfs::Node {
            kind: crate::vfs::NodeKind::Dir,
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
                interp.fs_metadata(&cwd, s, false),
                Ok(crate::vfs::Node {
                    kind: crate::vfs::NodeKind::Symlink(_),
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
            interp.vfs.copy_recursive(&cwd, s, &target, preserve)
        } else {
            interp.vfs.copy_file(&cwd, s, &target)
        };
        let r = r.and_then(|()| {
            if preserve && !recursive {
                let source = interp.vfs.metadata(&cwd, s, true)?;
                interp.vfs.chmod(&cwd, &target, source.mode)?;
                interp.vfs.touch(&cwd, &target, source.mtime)?;
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

fn cmd_mv(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (flags, ops, long) = split_flags(args);
    if reject_options("mv", &flags, "f", &long, io) {
        return 2;
    }
    if ops.len() < 2 {
        ewln(io.err, "mv: missing destination operand");
        return 1;
    }
    let cwd = interp.cwd.clone();
    let dest = ops.last().unwrap();
    let destination_is_dir = matches!(
        interp.fs_metadata(&cwd, dest, true),
        Ok(crate::vfs::Node {
            kind: crate::vfs::NodeKind::Dir,
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
        if let Err(e) = interp.vfs.rename(&cwd, s, &target) {
            ewln(io.err, &format!("mv: cannot move '{s}': {e}"));
            status = 1;
        }
    }
    status
}

fn cmd_touch(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    if args.iter().any(|arg| arg.starts_with('-') && arg != "--") {
        ewln(io.err, "touch: unimplemented option");
        return 2;
    }
    let (_f, ops, _l) = split_flags(args);
    if ops.is_empty() {
        ewln(io.err, "touch: missing file operand");
        return 1;
    }
    let cwd = interp.cwd.clone();
    let now = interp.clock.unix_ms();
    let mut status = 0;
    for t in &ops {
        if let Err(e) = interp.vfs.touch(&cwd, t, now) {
            ewln(io.err, &format!("touch: {e}"));
            status = 1;
        }
    }
    status
}

fn cmd_ln(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (flags, ops, long) = split_flags(args);
    if reject_options("ln", &flags, "sf", &long, io) {
        return 2;
    }
    let symbolic = flags.contains(&'s');
    if ops.len() < 2 {
        ewln(io.err, "ln: missing operand");
        return 1;
    }
    let cwd = interp.cwd.clone();
    let (target, link) = (ops[0], ops[1]);
    if !symbolic {
        ewln(io.err, "ln: unimplemented hard links");
        return 2;
    }
    let r = interp.vfs.symlink(&cwd, target, link);
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

fn cmd_chmod(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
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
    let cwd = interp.cwd.clone();
    let mut status = 0;
    for t in &targets {
        let paths = if recursive {
            let abs = crate::vfs::resolve_against(&cwd, t);
            interp.vfs.walk(&abs)
        } else {
            vec![crate::vfs::resolve_against(&cwd, t)]
        };
        for p in paths {
            let cur = interp
                .vfs
                .metadata("/", &p, true)
                .map(|n| n.mode)
                .unwrap_or(0o644);
            let m = parse_mode(&mode_arg, cur);
            if let Err(e) = interp.vfs.chmod("/", &p, m) {
                ewln(io.err, &format!("chmod: {e}"));
                status = 1;
            }
        }
    }
    status
}

fn cmd_chown(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (flags, ops, _l) = split_flags(args);
    let recursive = flags.contains(&'R');
    if ops.is_empty() {
        return 0;
    }
    let spec = ops[0];
    let (uid, gid) = parse_owner(spec);
    let cwd = interp.cwd.clone();
    let mut status = 0;
    for t in &ops[1..] {
        let paths = if recursive {
            interp.vfs.walk(&crate::vfs::resolve_against(&cwd, t))
        } else {
            vec![crate::vfs::resolve_against(&cwd, t)]
        };
        for p in paths {
            if let Err(e) = interp.vfs.chown("/", &p, uid, gid) {
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

fn cmd_basename(_interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
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

fn cmd_dirname(_interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
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

fn cmd_realpath(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    realpath_impl(interp, "realpath", args, io)
}

fn cmd_readlink(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    realpath_impl(interp, "readlink", args, io)
}

fn realpath_impl(interp: &mut Interp, cmd: &str, args: &[String], io: &mut Io) -> i32 {
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
                let abs = crate::vfs::resolve_against(&interp.cwd, p);
                match interp.fs_realpath("/", &abs, true) {
                    Ok(r) => wln(io.out, &r),
                    Err(_) => return 1,
                }
            } else {
                match interp.fs_read_link(&interp.cwd, p) {
                    Ok(t) => wln(io.out, &t),
                    Err(_) => {
                        ewln(io.err, &format!("readlink: {p}: Invalid argument"));
                        return 1;
                    }
                }
            }
        } else {
            let abs = crate::vfs::resolve_against(&interp.cwd, p);
            let resolved = if flags.contains(&'m') {
                // GNU `-m` permits missing components. The VFS resolver still follows every
                // modeled symlink in an existing prefix and normalizes the untouched suffix.
                interp.vfs.realpath(&abs, true)
            } else {
                interp.fs_realpath("/", &abs, true)
            };
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

fn cmd_stat(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
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
                    '%' | 's' | 'n' | 'a' | 'U' | 'u' | 'g' | 'Y' | 'F' | 'N'
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
        match interp.fs_metadata(&interp.cwd, f, follow) {
            Ok(n) => {
                let size = match &n.kind {
                    crate::vfs::NodeKind::File(d) => d.len(),
                    _ => 0,
                };
                if let Some(fmt) = &format {
                    let s = fmt
                        .replace("%%", "\0")
                        .replace("%s", &size.to_string())
                        .replace("%n", f)
                        .replace("%a", &format!("{:o}", n.mode))
                        .replace("%U", "root")
                        .replace("%u", &n.uid.to_string())
                        .replace("%g", &n.gid.to_string())
                        .replace("%Y", &(n.mtime / 1000).to_string())
                        .replace(
                            "%F",
                            match n.kind {
                                crate::vfs::NodeKind::File(_) => "regular file",
                                crate::vfs::NodeKind::Dir => "directory",
                                crate::vfs::NodeKind::Symlink(_) => "symbolic link",
                                crate::vfs::NodeKind::NativeExecutable(_) => "native executable",
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

fn cmd_du(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (flags, ops, long) = split_flags(args);
    if flags.iter().any(|flag| !matches!(flag, 's' | 'b')) || !long.is_empty() {
        ewln(io.err, "du: unimplemented option");
        return 2;
    }
    let bytes = flags.contains(&'b');
    let paths = if ops.is_empty() {
        vec!["."]
    } else {
        ops.iter().map(|path| path.as_str()).collect()
    };
    let mut status = 0;
    for path in paths {
        match interp.fs_walk(&interp.cwd, path) {
            Ok(nodes) => {
                let total = nodes.into_iter().fold(0usize, |sum, node| {
                    sum.saturating_add(match interp.fs_metadata("/", &node, false) {
                        Ok(crate::vfs::Node {
                            kind: crate::vfs::NodeKind::File(data),
                            ..
                        }) => data.len(),
                        _ => 0,
                    })
                });
                let amount = if bytes {
                    total
                } else {
                    total.saturating_add(1023) / 1024
                };
                wln(io.out, &format!("{amount}\t{path}"));
            }
            Err(error) => {
                ewln(io.err, &format!("du: {error}"));
                status = 1;
            }
        }
    }
    status
}

fn cmd_mktemp(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (flags, ops, long) = split_flags(args);
    if reject_options("mktemp", &flags, "d", &long, io) {
        return 2;
    }
    let dir = flags.contains(&'d');
    let tmpl = ops.first().map(|s| s.as_str()).unwrap_or("tmp.XXXXXX");
    // Identity is deterministic but deliberately separate from time: creating a name is not a
    // temporal effect and must not perturb deadlines.
    let Some(n) = interp.next_temp_id() else {
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
    if let Err(error) = interp.vfs.mkdir_all("/", "/tmp") {
        ewln(io.err, &format!("mktemp: {error}"));
        return 1;
    }
    let result = if dir {
        interp.vfs.mkdir("/", &path)
    } else {
        interp.vfs.write("/", &path, b"", 0o600)
    };
    if let Err(error) = result {
        ewln(io.err, &format!("mktemp: {error}"));
        return 1;
    }
    wln(io.out, &path);
    0
}

fn cmd_file(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (flags, ops, long) = split_flags(args);
    if reject_options("file", &flags, "", &long, io) {
        return 2;
    }
    if ops.is_empty() {
        ewln(io.err, "file: missing operand");
        return 1;
    }
    let mut status = 0;
    for p in &ops {
        let desc = match interp.fs_read(&interp.cwd, p) {
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
                    interp.fs_metadata(&interp.cwd, p, false),
                    Ok(crate::vfs::Node {
                        kind: crate::vfs::NodeKind::Dir,
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

fn cmd_install(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
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

    let cwd = interp.cwd.clone();
    if directories {
        if operands.is_empty() {
            ewln(io.err, "install: missing operand");
            return 1;
        }
        let mut status = 0;
        for path in operands {
            if !interp.charge_cpu(1) {
                return 137;
            }
            if let Err(error) = interp
                .vfs
                .mkdir_all(&cwd, &path)
                .and_then(|()| interp.vfs.chmod(&cwd, &path, mode))
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
        interp.fs_metadata(&cwd, destination, true),
        Ok(crate::vfs::Node {
            kind: crate::vfs::NodeKind::Dir,
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
        let data = match interp.fs_read(&cwd, source) {
            Ok(data) => data,
            Err(error) => {
                ewln(io.err, &format!("install: cannot stat '{source}': {error}"));
                status = 1;
                continue;
            }
        };
        if !interp.charge_cpu(data.len() as u64) {
            return 137;
        }
        let result = if create_parents {
            let absolute = resolve_against(&cwd, &target);
            if matches!(
                interp.fs_metadata("/", &absolute, false),
                Ok(crate::vfs::Node {
                    kind: crate::vfs::NodeKind::Dir,
                    ..
                })
            ) {
                Err(crate::vfs::VfsError::IsADir(target.clone()))
            } else {
                // `put_file` plans the missing parent directories and the file replacement as
                // one quota-checked mutation. A failed `install -D` therefore leaves no parents.
                interp.vfs.put_file(&absolute, data, mode)
            }
        } else {
            interp
                .vfs
                .write(&cwd, &target, &data, mode)
                .and_then(|()| interp.vfs.chmod(&cwd, &target, mode))
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

fn cmd_truncate(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
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

    let cwd = interp.cwd.clone();
    let mut status = 0;
    for file in files {
        let (mut data, mode) = match interp.fs_metadata(&cwd, &file, true) {
            Ok(node) => match node.kind {
                crate::vfs::NodeKind::File(data) => (data, node.mode),
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
            Err(_) => (Vec::new(), 0o666),
        };
        let Some(new_len) = change.apply(data.len()) else {
            ewln(io.err, &format!("truncate: invalid number: '{size_arg}'"));
            status = 1;
            continue;
        };
        if !interp.charge_cpu(new_len.saturating_sub(data.len()) as u64) {
            return 137;
        }
        if !interp.reserve_memory(new_len as u64) {
            return 137;
        }
        data.resize(new_len, 0);
        let result = interp.vfs.write(&cwd, &file, &data, mode);
        interp.resources.release_memory(new_len as u64);
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

fn cmd_tree(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
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
    let node = match interp.fs_metadata(&interp.cwd, root, false) {
        Ok(node) => node,
        Err(error) => {
            ewln(io.err, &format!("tree: {root}: {error}"));
            return 1;
        }
    };
    wln(io.out, root);
    let mut directories = 0usize;
    let mut files = 0usize;
    if matches!(node.kind, crate::vfs::NodeKind::Dir) {
        let status = tree_directory(
            interp,
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
    interp: &mut CommandContext<'_>,
    path: &str,
    prefix: &str,
    depth: usize,
    max_depth: Option<usize>,
    all: bool,
    directories: &mut usize,
    files: &mut usize,
    io: &mut Io,
) -> i32 {
    let mut entries = match interp.fs_list_dir(&interp.cwd, path) {
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
        if !interp.charge_cpu(1) {
            return 137;
        }
        let last = position + 1 == entries.len();
        let child = if path == "/" {
            format!("/{entry}")
        } else {
            format!("{}/{entry}", path.trim_end_matches('/'))
        };
        let node = match interp.fs_metadata(&interp.cwd, &child, false) {
            Ok(node) => node,
            Err(error) => {
                ewln(io.err, &format!("tree: {child}: {error}"));
                return 1;
            }
        };
        let suffix = match &node.kind {
            crate::vfs::NodeKind::Symlink(target) => format!(" -> {target}"),
            _ => String::new(),
        };
        wln(
            io.out,
            &format!(
                "{prefix}{} {entry}{suffix}",
                if last { "└──" } else { "├──" }
            ),
        );
        match node.kind {
            crate::vfs::NodeKind::Dir => {
                *directories = directories.saturating_add(1);
                if max_depth.is_none_or(|maximum| depth < maximum) {
                    let next_prefix = format!("{prefix}{}", if last { "    " } else { "│   " });
                    let status = tree_directory(
                        interp,
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
