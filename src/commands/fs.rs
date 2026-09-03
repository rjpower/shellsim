//! Filesystem commands operating on the VFS: listing (ls), tree mutation
//! (mkdir/rmdir/rm/cp/mv/touch/ln), permissions (chmod/chown), path math
//! (basename/dirname/realpath/readlink), inspection (stat/file/find/du), and mktemp.

use std::collections::HashMap;

use crate::commands::util::{ewln, glob_eq, split_flags, wln};
use crate::commands::{CommandContext, CommandSpec, Io, Trust};
use crate::interp::Interp;

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
    reg(m, &["find"], Trust::Real, cmd_find);
    reg(m, &["du"], Trust::Partial, cmd_du);
    reg(m, &["mktemp"], Trust::Real, cmd_mktemp);
    reg(m, &["file"], Trust::Real, cmd_file);
}

fn cmd_ls(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (flags, ops, _l) = split_flags(args);
    let long = flags.contains(&'l');
    let all = flags.contains(&'a');
    let one = flags.contains(&'1') || long;
    let recursive = flags.contains(&'R');
    let paths: Vec<String> = if ops.is_empty() {
        vec![interp.cwd.clone()]
    } else {
        ops.iter().map(|s| s.to_string()).collect()
    };
    let mut status = 0;
    for p in &paths {
        if interp.vfs.is_dir(&interp.cwd, p) {
            let mut entries = match interp.vfs.list_dir(&interp.cwd, p) {
                Ok(e) => e,
                Err(e) => {
                    ewln(io.err, &format!("ls: {e}"));
                    status = 2;
                    continue;
                }
            };
            if all {
                entries.insert(0, "..".into());
                entries.insert(0, ".".into());
            }
            if paths.len() > 1 {
                wln(io.out, &format!("{p}:"));
            }
            emit_listing(interp, p, &entries, long, one, io.out);
            if recursive {
                for e in &entries {
                    if e == "." || e == ".." {
                        continue;
                    }
                    let sub = format!("{}/{}", p.trim_end_matches('/'), e);
                    if interp.vfs.is_dir(&interp.cwd, &sub) {
                        wln(io.out, "");
                        wln(io.out, &format!("{sub}:"));
                        if let Ok(se) = interp.vfs.list_dir(&interp.cwd, &sub) {
                            emit_listing(interp, &sub, &se, long, one, io.out);
                        }
                    }
                }
            }
        } else if interp.vfs.lexists(&interp.cwd, p) {
            wln(io.out, p);
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
    interp: &Interp,
    dir: &str,
    entries: &[String],
    long: bool,
    one: bool,
    out: &mut Vec<u8>,
) {
    if long {
        for e in entries {
            let full = if e == "." {
                dir.to_string()
            } else if e == ".." {
                crate::vfs::parent_of(&crate::vfs::resolve_against(&interp.cwd, dir))
                    .unwrap_or_else(|| "/".into())
            } else {
                format!("{}/{}", dir.trim_end_matches('/'), e)
            };
            let (typ, mode, size) = match interp.vfs.metadata(&interp.cwd, &full, false) {
                Ok(n) => {
                    let t = match n.kind {
                        crate::vfs::NodeKind::Dir => 'd',
                        crate::vfs::NodeKind::Symlink(_) => 'l',
                        _ => '-',
                    };
                    let sz = match &n.kind {
                        crate::vfs::NodeKind::File(d) => d.len(),
                        _ => 0,
                    };
                    (t, n.mode, sz)
                }
                Err(_) => ('-', 0o644, 0),
            };
            wln(
                out,
                &format!(
                    "{}{} 1 root root {:>6} Jan  1 00:00 {}",
                    typ,
                    mode_str(mode),
                    size,
                    e
                ),
            );
        }
    } else if one {
        for e in entries {
            wln(out, e);
        }
    } else {
        wln(out, &entries.join("  "));
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

fn cmd_mkdir(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (flags, ops, _l) = split_flags(args);
    let parents = flags.contains(&'p');
    let cwd = interp.cwd.clone();
    let mut status = 0;
    for d in &ops {
        let r = if parents {
            interp.vfs.mkdir_all(&cwd, d)
        } else {
            interp.vfs.mkdir(&cwd, d)
        };
        if let Err(e) = r {
            if !parents {
                ewln(
                    io.err,
                    &format!("mkdir: cannot create directory '{d}': {e}"),
                );
                status = 1;
            }
        }
    }
    status
}

fn cmd_rmdir(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (_f, ops, _l) = split_flags(args);
    let cwd = interp.cwd.clone();
    let mut status = 0;
    for d in &ops {
        if let Err(e) = interp.vfs.rmdir(&cwd, d) {
            ewln(io.err, &format!("rmdir: failed to remove '{d}': {e}"));
            status = 1;
        }
    }
    status
}

fn cmd_rm(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (flags, ops, _l) = split_flags(args);
    let recursive = flags.contains(&'r') || flags.contains(&'R');
    let force = flags.contains(&'f');
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
    let (flags, ops, _l) = split_flags(args);
    let recursive = flags.contains(&'r') || flags.contains(&'R') || flags.contains(&'a');
    if ops.len() < 2 {
        ewln(io.err, "cp: missing destination operand");
        return 1;
    }
    let cwd = interp.cwd.clone();
    let dest = ops.last().unwrap();
    let sources = &ops[..ops.len() - 1];
    let mut status = 0;
    for s in sources {
        let r = if recursive {
            interp.vfs.copy_recursive(&cwd, s, dest)
        } else {
            interp.vfs.copy_file(&cwd, s, dest)
        };
        if let Err(e) = r {
            ewln(io.err, &format!("cp: cannot copy '{s}': {e}"));
            status = 1;
        }
    }
    status
}

fn cmd_mv(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (_f, ops, _l) = split_flags(args);
    if ops.len() < 2 {
        ewln(io.err, "mv: missing destination operand");
        return 1;
    }
    let cwd = interp.cwd.clone();
    let dest = ops.last().unwrap();
    let mut status = 0;
    for s in &ops[..ops.len() - 1] {
        if let Err(e) = interp.vfs.rename(&cwd, s, dest) {
            ewln(io.err, &format!("mv: cannot move '{s}': {e}"));
            status = 1;
        }
    }
    status
}

fn cmd_touch(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (_f, ops, _l) = split_flags(args);
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
    let (flags, ops, _l) = split_flags(args);
    let symbolic = flags.contains(&'s');
    if ops.len() < 2 {
        ewln(io.err, "ln: missing operand");
        return 1;
    }
    let cwd = interp.cwd.clone();
    let (target, link) = (ops[0], ops[1]);
    let r = if symbolic {
        interp.vfs.symlink(&cwd, target, link)
    } else {
        interp.vfs.copy_file(&cwd, target, link)
    };
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
    let mut base = crate::vfs::basename(p.trim_end_matches('/')).to_string();
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
    let (flags, ops, _l) = split_flags(args);
    for p in &ops {
        if cmd == "readlink" {
            if flags.contains(&'f') {
                let abs = crate::vfs::resolve_against(&interp.cwd, p);
                match interp.vfs.realpath(&abs, true) {
                    Ok(r) => wln(io.out, &r),
                    Err(_) => return 1,
                }
            } else {
                match interp.vfs.read_link(&interp.cwd, p) {
                    Ok(t) => wln(io.out, &t),
                    Err(_) => {
                        ewln(io.err, &format!("readlink: {p}: Invalid argument"));
                        return 1;
                    }
                }
            }
        } else {
            let abs = crate::vfs::resolve_against(&interp.cwd, p);
            match interp.vfs.realpath(&abs, true) {
                Ok(r) => wln(io.out, &r),
                Err(_) => wln(io.out, &abs),
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
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "-c" || a == "--format" {
            format = it.next().cloned();
        } else if !a.starts_with('-') {
            files.push(a.clone());
        }
    }
    let _ = ops;
    for f in &files {
        match interp.vfs.metadata(&interp.cwd, f, true) {
            Ok(n) => {
                let size = match &n.kind {
                    crate::vfs::NodeKind::File(d) => d.len(),
                    _ => 0,
                };
                if let Some(fmt) = &format {
                    let s = fmt
                        .replace("%s", &size.to_string())
                        .replace("%n", f)
                        .replace("%a", &format!("{:o}", n.mode))
                        .replace("%U", "root")
                        .replace("%u", &n.uid.to_string())
                        .replace("%g", &n.gid.to_string())
                        .replace("%Y", &(n.mtime / 1000).to_string());
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

fn cmd_find(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    // supports: find [paths...] [-type f|d] [-name PAT] [-maxdepth N] [-path PAT]
    let mut paths = Vec::new();
    let mut typ: Option<char> = None;
    let mut name_pat: Option<String> = None;
    let mut path_pat: Option<String> = None;
    let mut maxdepth: Option<usize> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-type" => {
                typ = args.get(i + 1).and_then(|s| s.chars().next());
                i += 2;
            }
            "-name" => {
                name_pat = args.get(i + 1).cloned();
                i += 2;
            }
            "-path" => {
                path_pat = args.get(i + 1).cloned();
                i += 2;
            }
            "-maxdepth" => {
                maxdepth = args.get(i + 1).and_then(|s| s.parse().ok());
                i += 2;
            }
            "-print" | "-print0" => {
                i += 1;
            }
            s if s.starts_with('-') => {
                i += 2; // skip unknown predicate + arg
            }
            s => {
                paths.push(s.to_string());
                i += 1;
            }
        }
    }
    if paths.is_empty() {
        paths.push(".".to_string());
    }
    for start in &paths {
        let abs = crate::vfs::resolve_against(&interp.cwd, start);
        let base_depth = abs.matches('/').count();
        let mut all = interp.vfs.walk(&abs);
        all.sort();
        for p in all {
            if let Some(md) = maxdepth {
                let depth = p.matches('/').count().saturating_sub(base_depth);
                if depth > md {
                    continue;
                }
            }
            let is_dir = matches!(
                interp.vfs.metadata("/", &p, false).map(|n| n.kind),
                Ok(crate::vfs::NodeKind::Dir)
            );
            if let Some(t) = typ {
                let ok = match t {
                    'd' => is_dir,
                    'f' => interp.vfs.is_file("/", &p),
                    'l' => interp.vfs.is_symlink("/", &p),
                    _ => true,
                };
                if !ok {
                    continue;
                }
            }
            if let Some(pat) = &name_pat {
                if !glob_eq(pat, crate::vfs::basename(&p)) {
                    continue;
                }
            }
            if let Some(pat) = &path_pat {
                if !glob_eq(pat, &p) {
                    continue;
                }
            }
            // print relative to the start path the way find does
            let display = if start == "." {
                if p == abs {
                    ".".to_string()
                } else {
                    format!(".{}", &p[abs.len()..])
                }
            } else {
                let prefix = format!("{}/", interp.cwd.trim_end_matches('/'));
                if !start.starts_with('/') {
                    p.strip_prefix(&prefix)
                        .map(|s| s.to_string())
                        .unwrap_or(p.clone())
                } else {
                    p.clone()
                }
            };
            wln(io.out, &display);
        }
    }
    0
}

fn cmd_du(_interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    wln(
        io.out,
        &format!("0\t{}", args.last().cloned().unwrap_or_else(|| ".".into())),
    );
    0
}

fn cmd_mktemp(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (flags, ops, _l) = split_flags(args);
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
    let _ = interp.vfs.mkdir_all("/", "/tmp");
    if dir {
        let _ = interp.vfs.mkdir_all("/", &path);
    } else {
        let _ = interp.vfs.write("/", &path, b"", 0o600);
    }
    wln(io.out, &path);
    0
}

fn cmd_file(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (_f, ops, _l) = split_flags(args);
    for p in &ops {
        let desc = match interp.vfs.read(&interp.cwd, p) {
            Ok(d) if d.is_empty() => "empty".to_string(),
            Ok(d)
                if d.iter().all(|b| b.is_ascii() || *b >= 0x80)
                    && std::str::from_utf8(&d).is_ok() =>
            {
                "ASCII text".to_string()
            }
            Ok(_) => "data".to_string(),
            Err(_) if interp.vfs.is_dir(&interp.cwd, p) => "directory".to_string(),
            Err(_) => "cannot open".to_string(),
        };
        wln(io.out, &format!("{p}: {desc}"));
    }
    0
}
