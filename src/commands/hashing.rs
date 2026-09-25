//! Hashing and encoding commands: the sha*/md5/cksum digests, base64/base32, hex dumps,
//! and `strings`. Digests are byte-exact (see [`crate::hashes`]).

use std::collections::HashMap;

use crate::commands::util::{ewln, read_inputs_system, split_flags, uses_standard_input, wln};
use crate::commands::{CommandSpec, Io, Trust};
use crate::exec::ShellPoll;
use crate::program::ProcessContext;
use crate::syscalls::{SyscallError, System};

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    use super::reg_system_poll;
    reg_system_poll(m, "/usr/bin/sha256sum", Trust::Real, cmd_sha256sum);
    reg_system_poll(m, "/usr/bin/sha1sum", Trust::Real, cmd_sha1sum);
    reg_system_poll(m, "/usr/bin/sha512sum", Trust::Real, cmd_sha512sum);
    reg_system_poll(m, "/usr/bin/md5sum", Trust::Real, cmd_md5sum);
    reg_system_poll(m, "/usr/bin/cksum", Trust::Real, cmd_cksum);
    reg_system_poll(m, "/usr/bin/base64", Trust::Real, cmd_base64);
    reg_system_poll(m, "/usr/bin/base32", Trust::Real, cmd_base64);
    reg_system_poll(m, "/usr/bin/xxd", Trust::Partial, cmd_hexdump);
    reg_system_poll(m, "/usr/bin/hexdump", Trust::Partial, cmd_hexdump);
    reg_system_poll(m, "/usr/bin/od", Trust::Partial, cmd_hexdump);
    reg_system_poll(m, "/usr/bin/strings", Trust::Real, cmd_strings);
}

fn cmd_sha256sum(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    hash_impl(context, "sha256", io)
}
fn cmd_sha1sum(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    hash_impl(context, "sha1", io)
}
fn cmd_sha512sum(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    hash_impl(context, "sha512", io)
}
fn cmd_md5sum(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    hash_impl(context, "md5", io)
}
fn cmd_cksum(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    hash_impl(context, "crc32", io)
}

fn read_named(system: &mut dyn System, path: &str) -> Result<Vec<u8>, SyscallError> {
    let cwd = system.cwd().to_string();
    let maximum = usize::try_from(system.limits().memory).unwrap_or(usize::MAX);
    system.read_file_limited(&cwd, path, maximum)
}

/// Charge for byte-oriented work that the old command dispatcher metered after execution.
fn charge_input(system: &mut dyn System, length: usize) -> Result<(), ShellPoll> {
    let units = u64::try_from(length).unwrap_or(u64::MAX);
    if system.charge_cpu(units) {
        Ok(())
    } else {
        Err(ShellPoll::Ready(system.stop_status()))
    }
}

fn hash_impl(context: &mut ProcessContext<'_>, algo: &str, io: &mut Io) -> ShellPoll {
    let args = context.args;
    let (flags, ops, _l) = split_flags(args);
    let check = flags.contains(&'c');
    if flags.iter().any(|flag| !matches!(flag, 'b' | 'c' | 't')) {
        ewln(io.err, &format!("{algo}sum: unimplemented option"));
        return ShellPoll::Ready(2);
    }
    if ops.is_empty() || (check && ops.iter().any(|operand| operand.as_str() == "-")) {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
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
                match read_named(context.system, manifest) {
                    Ok(data) => data,
                    Err(error) => {
                        ewln(io.err, &format!("{algo}sum: {manifest}: {error}"));
                        status = 1;
                        continue;
                    }
                }
            };
            if let Err(poll) = charge_input(context.system, data.len()) {
                return poll;
            }
            for line in String::from_utf8_lossy(&data).lines() {
                let Some((expected, filename)) = line.split_once(char::is_whitespace) else {
                    ewln(io.err, &format!("{algo}sum: malformed checksum line"));
                    status = 1;
                    continue;
                };
                let filename = filename.trim_start().trim_start_matches('*');
                match read_named(context.system, filename) {
                    Ok(contents) => {
                        if let Err(poll) = charge_input(context.system, contents.len()) {
                            return poll;
                        }
                        if compute(&contents) == expected {
                            wln(io.out, &format!("{filename}: OK"));
                        } else {
                            wln(io.out, &format!("{filename}: FAILED"));
                            status = 1;
                        }
                    }
                    Err(error) => {
                        ewln(io.err, &format!("{algo}sum: {filename}: {error}"));
                        status = 1;
                    }
                }
            }
        }
        ShellPoll::Ready(status)
    } else if ops.is_empty() {
        if let Err(poll) = charge_input(context.system, io.stdin.len()) {
            return poll;
        }
        wln(io.out, &format!("{}  -", compute(&io.stdin)));
        ShellPoll::Ready(0)
    } else {
        let mut status = 0;
        for f in &ops {
            match read_named(context.system, f) {
                Ok(data) => {
                    if let Err(poll) = charge_input(context.system, data.len()) {
                        return poll;
                    }
                    wln(io.out, &format!("{}  {}", compute(&data), f));
                }
                Err(error) => {
                    ewln(io.err, &format!("{algo}sum: {f}: {error}"));
                    status = 1;
                }
            }
        }
        ShellPoll::Ready(status)
    }
}

fn cmd_base64(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    let args = context.args;
    let (flags, ops, long) = split_flags(args);
    if flags.iter().any(|flag| *flag != 'd') || long.iter().any(|(option, _)| *option != "decode") {
        ewln(io.err, "base64: unimplemented option");
        return ShellPoll::Ready(2);
    }
    if uses_standard_input(&ops) {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let decode = flags.contains(&'d') || long.iter().any(|(k, _)| *k == "decode");
    let (data, errors) = read_inputs_system(context.system, &ops, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("base64: {error}"));
        return ShellPoll::Ready(1);
    }
    if let Err(poll) = charge_input(context.system, data.len()) {
        return poll;
    }
    if decode {
        let s: String = String::from_utf8_lossy(&data)
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        match crate::hashes::base64_decode(&s) {
            Some(d) => io.out.extend_from_slice(&d),
            None => return ShellPoll::Ready(1),
        }
    } else {
        let encoded = crate::hashes::base64_encode(&data);
        wln(io.out, &encoded);
    }
    ShellPoll::Ready(0)
}

fn cmd_hexdump(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    let args = context.args;
    let (_f, ops, _l) = split_flags(args);
    if uses_standard_input(&ops) {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let (data, errors) = read_inputs_system(context.system, &ops, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("hexdump: {error}"));
        return ShellPoll::Ready(1);
    }
    if let Err(poll) = charge_input(context.system, data.len()) {
        return poll;
    }
    let hex: String = data.iter().map(|b| format!("{b:02x}")).collect();
    wln(io.out, &hex);
    ShellPoll::Ready(0)
}

fn cmd_strings(context: &mut ProcessContext<'_>, io: &mut Io) -> ShellPoll {
    let args = context.args;
    let (_f, ops, _l) = split_flags(args);
    if uses_standard_input(&ops) {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let (data, errors) = read_inputs_system(context.system, &ops, &io.stdin);
    if let Some(error) = errors.first() {
        ewln(io.err, &format!("strings: {error}"));
        return ShellPoll::Ready(1);
    }
    if let Err(poll) = charge_input(context.system, data.len()) {
        return poll;
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
    ShellPoll::Ready(0)
}
