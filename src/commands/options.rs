//! Declarative option scanning shared by simulated command-line utilities.
//!
//! Command modules provide closed option tables. The scanner recognizes only those spellings,
//! which keeps unsupported flags visible instead of silently approximating their semantics.

/// Whether one command-line option accepts an argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OptionValue {
    None,
    Required,
    /// Accept a value only when attached to the option, as in -i.bak.
    OptionalAttached,
}

/// One command-local option spelling mapped to a canonical key.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OptionSpec {
    pub key: &'static str,
    pub short: Option<char>,
    pub long: Option<&'static str>,
    pub value: OptionValue,
}

impl OptionSpec {
    pub const fn flag(key: &'static str, short: Option<char>, long: Option<&'static str>) -> Self {
        Self {
            key,
            short,
            long,
            value: OptionValue::None,
        }
    }

    pub const fn required(
        key: &'static str,
        short: Option<char>,
        long: Option<&'static str>,
    ) -> Self {
        Self {
            key,
            short,
            long,
            value: OptionValue::Required,
        }
    }

    pub const fn optional_attached(
        key: &'static str,
        short: Option<char>,
        long: Option<&'static str>,
    ) -> Self {
        Self {
            key,
            short,
            long,
            value: OptionValue::OptionalAttached,
        }
    }
}

/// One parsed option occurrence. Repeated options remain repeated and ordered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ParsedOption {
    pub key: &'static str,
    pub value: Option<String>,
}

/// Options and operands produced by the scanner.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ParsedArgs {
    pub options: Vec<ParsedOption>,
    pub operands: Vec<String>,
}

/// Scan ordinary Unix utility options using a small command-local specification table.
///
/// The scanner handles the operand boundary, short clusters, attached or separate required
/// values, long options, and attached long values. It does not attach semantics to names.
pub(crate) fn parse_options(args: &[String], specs: &[OptionSpec]) -> Result<ParsedArgs, String> {
    let mut parsed = ParsedArgs::default();
    let mut operands_only = false;
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        if operands_only || argument == "-" || !argument.starts_with('-') {
            parsed.operands.push(argument.clone());
            index += 1;
            continue;
        }
        if argument == "--" {
            operands_only = true;
            index += 1;
            continue;
        }
        if let Some(long) = argument.strip_prefix("--") {
            let (name, attached) = long
                .split_once('=')
                .map_or((long, None), |(name, value)| (name, Some(value)));
            let Some(spec) = specs.iter().find(|spec| spec.long == Some(name)) else {
                return Err(format!("unsupported option '--{name}'"));
            };
            let value = match spec.value {
                OptionValue::None if attached.is_some() => {
                    return Err(format!("option '--{name}' does not take an argument"));
                }
                OptionValue::None => None,
                OptionValue::OptionalAttached => attached.map(str::to_string),
                OptionValue::Required => match attached {
                    Some(value) => Some(value.to_string()),
                    None => {
                        index += 1;
                        Some(
                            args.get(index)
                                .ok_or_else(|| format!("option '--{name}' requires an argument"))?
                                .clone(),
                        )
                    }
                },
            };
            parsed.options.push(ParsedOption {
                key: spec.key,
                value,
            });
            index += 1;
            continue;
        }

        let cluster = &argument[1..];
        for (offset, short) in cluster.char_indices() {
            let Some(spec) = specs.iter().find(|spec| spec.short == Some(short)) else {
                return Err(format!("unsupported option '-{short}'"));
            };
            let remainder_start = offset + short.len_utf8();
            let remainder = &cluster[remainder_start..];
            let value = match spec.value {
                OptionValue::None => None,
                OptionValue::OptionalAttached => {
                    (!remainder.is_empty()).then(|| remainder.to_string())
                }
                OptionValue::Required if !remainder.is_empty() => Some(remainder.to_string()),
                OptionValue::Required => {
                    index += 1;
                    Some(
                        args.get(index)
                            .ok_or_else(|| format!("option '-{short}' requires an argument"))?
                            .clone(),
                    )
                }
            };
            parsed.options.push(ParsedOption {
                key: spec.key,
                value,
            });
            if spec.value != OptionValue::None {
                break;
            }
        }
        index += 1;
    }
    Ok(parsed)
}

/// Parse one command's options and emit its standard diagnostic on failure.
pub(crate) fn parse_options_or_report(
    command: &str,
    args: &[String],
    specs: &[OptionSpec],
    stderr: &mut Vec<u8>,
) -> Option<ParsedArgs> {
    match parse_options(args, specs) {
        Ok(parsed) => Some(parsed),
        Err(error) => {
            stderr.extend_from_slice(format!("{command}: {error}\n").as_bytes());
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_options, OptionSpec, ParsedOption};

    #[test]
    fn handles_clusters_values_and_operand_boundaries() {
        let specs = [
            OptionSpec::flag("ignore_case", Some('i'), Some("ignore-case")),
            OptionSpec::flag("verbose", Some('v'), Some("verbose")),
            OptionSpec::required("expression", Some('e'), Some("regexp")),
            OptionSpec::optional_attached("in_place", Some('I'), Some("in-place")),
        ];
        let args = [
            "-iv".to_string(),
            "-evalue".to_string(),
            "--in-place=.bak".to_string(),
            "first".to_string(),
            "--".to_string(),
            "-v".to_string(),
        ];
        let parsed = parse_options(&args, &specs).unwrap();
        assert_eq!(
            parsed.options,
            [
                ParsedOption {
                    key: "ignore_case",
                    value: None,
                },
                ParsedOption {
                    key: "verbose",
                    value: None,
                },
                ParsedOption {
                    key: "expression",
                    value: Some("value".into()),
                },
                ParsedOption {
                    key: "in_place",
                    value: Some(".bak".into()),
                },
            ]
        );
        assert_eq!(parsed.operands, ["first", "-v"]);
    }

    #[test]
    fn rejects_unknown_and_malformed_options() {
        let specs = [
            OptionSpec::flag("quiet", Some('q'), Some("quiet")),
            OptionSpec::required("expression", Some('e'), Some("regexp")),
        ];
        assert_eq!(
            parse_options(&["--colour".into()], &specs).unwrap_err(),
            "unsupported option '--colour'"
        );
        assert_eq!(
            parse_options(&["-e".into()], &specs).unwrap_err(),
            "option '-e' requires an argument"
        );
        assert_eq!(
            parse_options(&["--quiet=yes".into()], &specs).unwrap_err(),
            "option '--quiet' does not take an argument"
        );
    }
}
