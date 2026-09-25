//! Echo formatting for shell-local invocation and the native process adapter.

use std::collections::HashMap;

use super::util::{unescape, w};
use super::{reg_system_costed, CommandSpec, Io, Trust};

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    reg_system_costed(m, "/usr/bin/echo", Trust::Real, 20, run);
}

fn run(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let args = context.args;
    let mut newline = true;
    let mut interpret = false;
    let mut start = 0;
    for a in args {
        match a.as_str() {
            "-n" => {
                newline = false;
                start += 1;
            }
            "-e" => {
                interpret = true;
                start += 1;
            }
            "-E" => {
                interpret = false;
                start += 1;
            }
            "-ne" | "-en" => {
                newline = false;
                interpret = true;
                start += 1;
            }
            _ => break,
        }
    }
    let text = args[start..].join(" ");
    let text = if interpret { unescape(&text) } else { text };
    w(io.out, &text);
    if newline {
        io.out.push(b'\n');
    }
    0
}
