use std::collections::HashMap;

use super::util::{unescape, w};
use super::{reg_costed, CommandContext, CommandSpec, Io, Trust};

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    reg_costed(m, &["echo"], Trust::Real, 20, 10 * 1024, run);
}

fn run(_env: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
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
