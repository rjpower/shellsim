//! Virtual networking: `curl`/`wget` resolve the route table (no real egress), and the
//! `net` control command wires up fake URLs / probes the request log from a script.

use std::collections::HashMap;

use crate::commands::util::{ewln, wln};
use crate::commands::{CommandSpec, Io, Trust};
use crate::program::ProcessContext;

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    use super::reg_system;
    reg_system(m, "/usr/bin/curl", Trust::Real, cmd_curl);
    reg_system(m, "/usr/bin/wget", Trust::Real, cmd_wget);
    reg_system(m, "/usr/bin/net", Trust::Real, cmd_net);
}

fn cmd_curl(context: &mut ProcessContext<'_>, io: &mut Io) -> i32 {
    crate::netcmd::curl(context.system, context.args, io.out, io.err)
}

fn cmd_wget(context: &mut ProcessContext<'_>, io: &mut Io) -> i32 {
    crate::netcmd::wget(context.system, context.args, io.out, io.err)
}

/// `net` — register fake URLs / probe the virtual network from a script.
///   net route <url-pattern> [status] [body...]   register a static response (`*` globs)
///   net route-file <url-pattern> <vfs-path>      serve a VFS file as the response body
///   net listen <host:port>                       mark a service as up
///   net log                                      print the request log
fn cmd_net(context: &mut ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
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
            match context
                .system
                .http_route_static(pattern, status, body.into_bytes())
            {
                Ok(()) => 0,
                Err(error) => {
                    ewln(io.err, &format!("net route: {error}"));
                    2
                }
            }
        }
        Some("route-file") => match (args.get(1), args.get(2)) {
            (Some(pattern), Some(path)) => {
                let abs = crate::vfs::resolve_against(context.system.cwd(), path);
                match context.system.http_route_file(pattern, &abs) {
                    Ok(()) => 0,
                    Err(error) => {
                        ewln(io.err, &format!("net route-file: {error}"));
                        2
                    }
                }
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
                context.system.network_listen(hp);
            }
            0
        }
        Some("log") => {
            for index in 0..context.system.network_request_count() {
                if let Some(request) = context.system.network_request_at(index) {
                    wln(io.out, &format!("{} {}", request.method, request.url));
                }
            }
            0
        }
        _ => {
            ewln(io.err, "net: usage: net route|route-file|listen|log …");
            2
        }
    }
}
