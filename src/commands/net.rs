//! Virtual networking: `curl`/`wget` resolve the route table (no real egress), and the
//! `net` control command wires up fake URLs / probes the request log from a script.

use std::collections::HashMap;

use crate::commands::util::{ewln, wln};
use crate::commands::{CommandContext, CommandSpec, Io, Trust};

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    use super::reg;
    reg(m, &["curl"], Trust::Real, cmd_curl);
    reg(m, &["wget"], Trust::Real, cmd_wget);
    reg(m, &["net"], Trust::Real, cmd_net);
}

fn cmd_curl(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    crate::netcmd::curl(interp, args, io.out, io.err)
}

fn cmd_wget(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    crate::netcmd::wget(interp, args, io.out, io.err)
}

/// `net` — register fake URLs / probe the virtual network from a script.
///   net route <url-pattern> [status] [body...]   register a static response (`*` globs)
///   net route-file <url-pattern> <vfs-path>      serve a VFS file as the response body
///   net listen <host:port>                       mark a service as up
///   net log                                      print the request log
fn cmd_net(interp: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    match args.first().map(|s| s.as_str()) {
        Some("route") => {
            let Some(pattern) = args.get(1) else {
                ewln(io.err, "net route: missing url pattern");
                return 2;
            };
            let status = args
                .get(2)
                .and_then(|s| s.parse::<u16>().ok())
                .unwrap_or(200);
            let body = if args.len() > 3 {
                args[3..].join(" ")
            } else {
                String::new()
            };
            interp.net.route_static(pattern, status, body.into_bytes());
            0
        }
        Some("route-file") => match (args.get(1), args.get(2)) {
            (Some(pattern), Some(path)) => {
                let abs = crate::vfs::resolve_against(&interp.cwd, path);
                interp.net.route_vfs(pattern, &abs);
                0
            }
            _ => {
                ewln(
                    io.err,
                    "net route-file: usage: net route-file <pattern> <vfs-path>",
                );
                2
            }
        },
        Some("listen") => {
            if let Some(hp) = args.get(1) {
                interp.net.listen(hp);
            }
            0
        }
        Some("log") => {
            for (m, u) in &interp.net.log {
                wln(io.out, &format!("{m} {u}"));
            }
            0
        }
        _ => {
            ewln(io.err, "net: usage: net route|route-file|listen|log …");
            2
        }
    }
}
