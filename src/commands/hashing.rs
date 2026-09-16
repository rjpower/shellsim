//! Hashing and encoding commands: the sha*/md5/cksum digests, base64/base32, hex dumps,
//! and `strings`. Digests are byte-exact (see [`crate::hashes`]).

use std::collections::HashMap;

use crate::commands::util::{ewln, read_inputs, split_flags, wln};
use crate::commands::{CommandContext, CommandSpec, Io, Trust};
use crate::interp::Interp;

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    use super::reg;
    reg(m, &["sha256sum"], Trust::Real, cmd_sha256sum);
    reg(m, &["sha1sum"], Trust::Real, cmd_sha1sum);
    reg(m, &["sha512sum"], Trust::Real, cmd_sha512sum);
    reg(m, &["md5sum"], Trust::Real, cmd_md5sum);
    reg(m, &["cksum"], Trust::Real, cmd_cksum);
    reg(m, &["base64", "base32"], Trust::Real, cmd_base64);
    reg(m, &["xxd", "hexdump", "od"], Trust::Partial, cmd_hexdump);
    reg(m, &["strings"], Trust::Real, cmd_strings);
}

fn cmd_sha256sum(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    hash_impl(interp, "sha256", args, io)
}
fn cmd_sha1sum(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    hash_impl(interp, "sha1", args, io)
}
fn cmd_sha512sum(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    hash_impl(interp, "sha512", args, io)
}
fn cmd_md5sum(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    hash_impl(interp, "md5", args, io)
}
fn cmd_cksum(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    hash_impl(interp, "crc32", args, io)
}

fn hash_impl(interp: &mut Interp, algo: &str, args: &[String], io: &mut Io) -> i32 {
    let (flags, ops, _l) = split_flags(args);
    let check = flags.contains(&'c');
    if flags.iter().any(|flag| !matches!(flag, 'b' | 'c' | 't')) {
        ewln(io.err, &format!("{algo}sum: unimplemented option"));
        return 2;
    }
    let compute = |data: &[u8]| -> String {
        match algo {
            "sha256" => crate::hashes::sha256_hex(data),
            "sha1" => crate::hashes::sha1_hex(data),
            "sha512" => crate::hashes::sha512_hex(data),
            "md5" => crate::hashes::md5_hex(data),
            "crc32" => {
                let (c, n) = crate::hashes::cksum(data);
                format!("{c} {n}")
            }
            _ => String::new(),
        }
    };
    if check {
        let manifests = if ops.is_empty() {
            vec!["-"]
        } else {
            ops.iter().map(|value| value.as_str()).collect()
        };
        let mut status = 0;
        for manifest in manifests {
            let data = if manifest == "-" {
                io.stdin.clone()
            } else {
                match interp.fs_read(&interp.cwd, manifest) {
                    Ok(data) => data,
                    Err(error) => {
                        ewln(io.err, &format!("{algo}sum: {manifest}: {error}"));
                        status = 1;
                        continue;
                    }
                }
            };
            for line in String::from_utf8_lossy(&data).lines() {
                let Some((expected, filename)) = line.split_once(char::is_whitespace) else {
                    ewln(io.err, &format!("{algo}sum: malformed checksum line"));
                    status = 1;
                    continue;
                };
                let filename = filename.trim_start().trim_start_matches('*');
                match interp.fs_read(&interp.cwd, filename) {
                    Ok(contents) if compute(&contents) == expected => {
                        wln(io.out, &format!("{filename}: OK"))
                    }
                    Ok(_) => {
                        wln(io.out, &format!("{filename}: FAILED"));
                        status = 1;
                    }
                    Err(error) => {
                        ewln(io.err, &format!("{algo}sum: {filename}: {error}"));
                        status = 1;
                    }
                }
            }
        }
        status
    } else if ops.is_empty() {
        wln(io.out, &format!("{}  -", compute(&io.stdin)));
        0
    } else {
        let mut status = 0;
        for f in &ops {
            match interp.fs_read(&interp.cwd, f) {
                Ok(d) => wln(io.out, &format!("{}  {}", compute(&d), f)),
                Err(error) => {
                    ewln(io.err, &format!("{algo}sum: {f}: {error}"));
                    status = 1;
                }
            }
        }
        status
    }
}

fn cmd_base64(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (flags, ops, long) = split_flags(args);
    if flags.iter().any(|flag| *flag != 'd') || long.iter().any(|(option, _)| *option != "decode") {
        ewln(io.err, "base64: unimplemented option");
        return 2;
    }
    let decode = flags.contains(&'d') || long.iter().any(|(k, _)| *k == "decode");
    let (data, errors) = read_inputs(interp, &ops, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("base64: {error}"));
        return 1;
    }
    if decode {
        let s: String = String::from_utf8_lossy(&data)
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        match crate::hashes::base64_decode(&s) {
            Some(d) => io.out.extend_from_slice(&d),
            None => return 1,
        }
    } else {
        let encoded = crate::hashes::base64_encode(&data);
        wln(io.out, &encoded);
    }
    0
}

fn cmd_hexdump(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (_f, ops, _l) = split_flags(args);
    let (data, errors) = read_inputs(interp, &ops, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("hexdump: {error}"));
        return 1;
    }
    let hex: String = data.iter().map(|b| format!("{b:02x}")).collect();
    wln(io.out, &hex);
    0
}

fn cmd_strings(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (_f, ops, _l) = split_flags(args);
    let (data, errors) = read_inputs(interp, &ops, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("strings: {error}"));
        return 1;
    }
    let mut cur = String::new();
    for &b in &data {
        if b.is_ascii_graphic() || b == b' ' {
            cur.push(b as char);
        } else {
            if cur.len() >= 4 {
                wln(io.out, &cur);
            }
            cur.clear();
        }
    }
    if cur.len() >= 4 {
        wln(io.out, &cur);
    }
    0
}
