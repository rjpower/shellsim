//! "Process-ish" commands: date/sync, `timeout` argument parsing shared with the native image,
//! and jq.

use std::collections::HashMap;

use crate::clock::NANOS_PER_SECOND;
use crate::commands::util::{ewln, parse_duration_ns, wln};
use crate::commands::{CommandSpec, Io, Trust};

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    // time / scheduling (virtual clock, never blocks)
    super::reg_system(m, "/usr/bin/date", Trust::Real, cmd_date);
    super::reg_system(m, "/usr/bin/sync", Trust::Real, |_, _| 0);

    super::reg_system_poll(m, "/usr/bin/jq", Trust::Partial, cmd_jq);
}

pub(crate) struct TimeoutInvocation {
    pub(crate) duration: u64,
    pub(crate) preserve_status: bool,
    /// Keep the child in timeout's process group and signal only the child.
    pub(crate) foreground: bool,
    pub(crate) signal: crate::process::Signal,
    /// Send KILL this long after the first signal if the command is still running.
    pub(crate) kill_after: Option<u64>,
    pub(crate) argv: Vec<String>,
}

pub(crate) fn parse_timeout(args: &[String]) -> Result<TimeoutInvocation, String> {
    let mut index = 0;
    let mut preserve_status = false;
    let mut foreground = false;
    let mut kill_after = None;
    let mut signal = crate::process::Signal::Terminate;
    while index < args.len() {
        match args[index].as_str() {
            "--preserve-status" => {
                preserve_status = true;
                index += 1;
            }
            "--foreground" => {
                foreground = true;
                index += 1;
            }
            "-s" | "--signal" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "option requires a signal".to_string())?;
                signal = crate::process::Signal::parse(value)
                    .filter(|signal| signal.terminates())
                    .ok_or_else(|| format!("unsupported signal '{value}'"))?;
                index += 2;
            }
            option if option.starts_with("--signal=") => {
                let value = option.trim_start_matches("--signal=");
                signal = crate::process::Signal::parse(value)
                    .filter(|signal| signal.terminates())
                    .ok_or_else(|| format!("unsupported signal '{value}'"))?;
                index += 1;
            }
            "-k" | "--kill-after" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "option requires an argument -- 'k'".to_string())?;
                kill_after = Some(parse_duration_ns(value)?);
                index += 2;
            }
            option if option.starts_with("--kill-after=") => {
                kill_after = Some(parse_duration_ns(
                    option.trim_start_matches("--kill-after="),
                )?);
                index += 1;
            }
            "--" => {
                index += 1;
                break;
            }
            option if option.starts_with('-') => {
                return Err(format!("unsupported option '{option}'"));
            }
            _ => break,
        }
    }
    let duration = parse_duration_ns(
        args.get(index)
            .ok_or_else(|| "missing operand".to_string())?,
    )?;
    let argv = args[index + 1..].to_vec();
    if argv.is_empty() {
        return Err("missing command".to_string());
    }
    Ok(TimeoutInvocation {
        duration,
        preserve_status,
        foreground,
        signal,
        kill_after,
        argv,
    })
}

fn cmd_date(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> i32 {
    let unix_ns = match context.system.wall_time_signed_ns() {
        Ok(value) => value,
        Err(error) => {
            ewln(io.err, &format!("date: {error}"));
            return 1;
        }
    };
    let rendered = if let Some(fmt) = context.args.iter().find(|a| a.starts_with('+')) {
        format_date(unix_ns, &fmt[1..])
    } else {
        format_date(unix_ns, "%a %b %e %H:%M:%S UTC %Y")
    };
    match rendered {
        Ok(rendered) => {
            wln(io.out, &rendered);
            0
        }
        Err(error) => {
            ewln(io.err, &format!("date: {error}"));
            1
        }
    }
}

pub(crate) fn format_date(unix_ns: i128, fmt: &str) -> Result<String, String> {
    // convert unix secs to UTC fields (proleptic Gregorian)
    let secs = unix_ns.div_euclid(i128::from(NANOS_PER_SECOND));
    let subsecond_ns = unix_ns.rem_euclid(i128::from(NANOS_PER_SECOND));
    let days = i64::try_from(secs.div_euclid(86_400))
        .map_err(|_| "timestamp is outside the supported calendar range".to_string())?;
    let tod = secs.rem_euclid(86_400) as u64;
    let (h, mi, s) = (tod / 3600, (tod % 3600) / 60, tod % 60);
    let (y, mo, d) = civil_from_days(days);
    let weekdays = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let weekday = weekdays[(days + 4).rem_euclid(7) as usize];
    let month = months[(mo - 1) as usize];
    let mut output = String::new();
    let mut chars = fmt.chars();
    while let Some(character) = chars.next() {
        if character != '%' {
            output.push(character);
            continue;
        }
        // `%-X` suppresses the usual zero padding, as GNU date allows.
        let (unpadded, specifier) = match chars.next() {
            Some('-') => (true, chars.next()),
            other => (false, other),
        };
        if unpadded {
            match specifier {
                Some('Y') => output.push_str(&y.to_string()),
                Some('m') => output.push_str(&mo.to_string()),
                Some('d') | Some('e') => output.push_str(&d.to_string()),
                Some('H') => output.push_str(&h.to_string()),
                Some('M') => output.push_str(&mi.to_string()),
                Some('S') => output.push_str(&s.to_string()),
                Some(other) => {
                    output.push_str("%-");
                    output.push(other);
                }
                None => output.push_str("%-"),
            }
            continue;
        }
        match specifier {
            Some('%') => output.push('%'),
            Some('Y') => output.push_str(&format!("{y:04}")),
            Some('m') => output.push_str(&format!("{mo:02}")),
            Some('d') => output.push_str(&format!("{d:02}")),
            Some('e') => output.push_str(&format!("{d:2}")),
            Some('H') => output.push_str(&format!("{h:02}")),
            Some('M') => output.push_str(&format!("{mi:02}")),
            Some('S') => output.push_str(&format!("{s:02}")),
            Some('N') => output.push_str(&format!("{subsecond_ns:09}")),
            Some('a') => output.push_str(weekday),
            Some('b') => output.push_str(month),
            Some('s') => output.push_str(&secs.to_string()),
            Some('z') => output.push_str("+0000"),
            Some('Z') => output.push_str("UTC"),
            Some('F') => output.push_str(&format!("{y:04}-{mo:02}-{d:02}")),
            Some('T') => output.push_str(&format!("{h:02}:{mi:02}:{s:02}")),
            Some(other) => {
                output.push('%');
                output.push(other);
            }
            None => output.push('%'),
        }
    }
    Ok(output)
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn cmd_jq(context: &mut crate::program::ProcessContext<'_>, io: &mut Io) -> crate::exec::ShellPoll {
    if crate::jqcmd::reads_standard_input(context.args) {
        if let Err(poll) = context.read_standard_input(io) {
            return poll;
        }
    }
    let stdin = std::mem::take(&mut io.stdin);
    crate::exec::ShellPoll::Ready(crate::jqcmd::jq(
        context.system,
        context.args,
        stdin,
        io.out,
        io.err,
    ))
}

#[cfg(test)]
mod time_tests {
    use super::format_date;
    use crate::clock::NANOS_PER_SECOND;
    use std::process::Command;

    #[test]
    fn utc_formatting_matches_coreutils_when_available() {
        let format = "%a %b %F %T %z %Z %s";
        for seconds in [0_i128, 951_827_696, 1_735_689_600, 1_772_368_496] {
            let expected = Command::new("date")
                .args(["-u", &format!("--date=@{seconds}"), &format!("+{format}")])
                .output();
            let Ok(expected) = expected else {
                return;
            };
            assert!(expected.status.success());
            assert_eq!(
                format_date(seconds * i128::from(NANOS_PER_SECOND), format).unwrap(),
                String::from_utf8_lossy(&expected.stdout).trim_end()
            );
        }
    }
}
