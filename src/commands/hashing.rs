//! Hashing and encoding commands: the sha*/md5/cksum digests, base64/base32, hex dumps,
//! and `strings`. Digests are byte-exact (see [`crate::hashes`]).

use std::collections::HashMap;

use crate::commands::util::{read_inputs, split_flags, wln};
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
    let _ = check;
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
    if ops.is_empty() {
        wln(io.out, &format!("{}  -", compute(&io.stdin)));
    } else {
        for f in &ops {
            match interp.vfs.read(&interp.cwd, f) {
                Ok(d) => wln(io.out, &format!("{}  {}", compute(&d), f)),
                Err(_) => return 1,
            }
        }
    }
    0
}

fn cmd_base64(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (flags, ops, long) = split_flags(args);
    let decode = flags.contains(&'d') || long.iter().any(|(k, _)| *k == "decode");
    let (data, _e) = read_inputs(interp, &ops, &io.stdin);
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
    let (data, _e) = read_inputs(interp, &ops, &io.stdin);
    let hex: String = data.iter().map(|b| format!("{b:02x}")).collect();
    wln(io.out, &hex);
    0
}

fn cmd_strings(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let (_f, ops, _l) = split_flags(args);
    let (data, _e) = read_inputs(interp, &ops, &io.stdin);
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
