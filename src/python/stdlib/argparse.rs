//! Bounded, capability-free :mod:`argparse` compatibility.
//!
//! The implementation intentionally stores plain argument definitions and walks them linearly.
//! It covers ordinary options, positionals, help, known-argument parsing, and one subcommand
//! level. Unsupported actions and nested subcommands fail explicitly rather than approximating a
//! larger parser framework.

use super::super::native::PyValue as Value;
use super::super::native::{
    CallArgs, FunctionDef, MethodDef, ModuleDef, NativeTypeDef, OwnedPyString, PyArgumentParser,
    PyArgumentParserData, PyArgumentSpec, PyError, PyKind, PyMarker, PyResult, PyRuntime,
    PySubcommandSpec, PySubparsersSpec, PyValueCast,
};

type NamespaceValues = Vec<(String, Value)>;
type ParsedArguments = Result<(NamespaceValues, Vec<String>), PyError>;

pub(crate) static ARGUMENT_PARSER_TYPE: NativeTypeDef = NativeTypeDef {
    name: "argparse.ArgumentParser",
    methods: &[
        MethodDef {
            type_name: "argparse.ArgumentParser",
            name: "add_argument",
            call: add_argument,
        },
        MethodDef {
            type_name: "argparse.ArgumentParser",
            name: "add_subparsers",
            call: add_subparsers,
        },
        MethodDef {
            type_name: "argparse.ArgumentParser",
            name: "add_parser",
            call: add_parser,
        },
        MethodDef {
            type_name: "argparse.ArgumentParser",
            name: "parse_args",
            call: parse_args,
        },
        MethodDef {
            type_name: "argparse.ArgumentParser",
            name: "parse_known_args",
            call: parse_known_args,
        },
        MethodDef {
            type_name: "argparse.ArgumentParser",
            name: "format_help",
            call: format_help_method,
        },
        MethodDef {
            type_name: "argparse.ArgumentParser",
            name: "print_help",
            call: print_help,
        },
    ],
};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "argparse",
    functions: &[
        FunctionDef {
            module: "argparse",
            name: "ArgumentParser",
            call: argument_parser,
        },
        FunctionDef {
            module: "argparse",
            name: "Namespace",
            call: namespace,
        },
    ],
    values: &[],
};

fn argument_parser(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("ArgumentParser", 0, 0)?;
    args.reject_unknown_keywords("ArgumentParser", &["prog", "description", "add_help"])?;
    let program = optional_string(runtime, &args, "ArgumentParser", "prog")?
        .unwrap_or_else(|| runtime.argv0());
    let description = optional_string(runtime, &args, "ArgumentParser", "description")?;
    let add_help = boolean_keyword(runtime, &args, "ArgumentParser", "add_help", true)?;
    runtime.new_argument_parser(program, description, add_help, false)
}

fn namespace(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("Namespace", 0, 0)?;
    runtime.new_namespace(args.into_parts().1)
}

fn add_argument(runtime: &mut dyn PyRuntime, receiver: Value, args: CallArgs) -> PyResult {
    if args.positional().is_empty() {
        return Err(PyError::type_error(
            "add_argument() requires at least one name",
        ));
    }
    args.reject_unknown_keywords(
        "add_argument",
        &[
            "dest", "required", "action", "type", "default", "help", "choices",
        ],
    )?;
    let parser = receiver.cast::<PyArgumentParser>(runtime)?;
    let names = args
        .positional()
        .iter()
        .copied()
        .map(|value| value.cast::<OwnedPyString>(runtime).map(|value| value.0))
        .collect::<PyResult<Vec<_>>>()?;
    let optional = names.iter().any(|name| name.starts_with('-'));
    let dest = optional_string(runtime, &args, "add_argument", "dest")?.unwrap_or_else(|| {
        names
            .iter()
            .find(|name| name.starts_with("--"))
            .unwrap_or(&names[0])
            .trim_start_matches('-')
            .replace('-', "_")
    });
    let required = boolean_keyword(runtime, &args, "add_argument", "required", !optional)?;
    let action = optional_string(runtime, &args, "add_argument", "action")?;
    let (store_true, store_false) = match action.as_deref() {
        None | Some("store") => (false, false),
        Some("store_true") => (true, false),
        Some("store_false") => (false, true),
        Some(action) => {
            return Err(PyError::value_error(format!(
                "unsupported argparse action {action:?}"
            )))
        }
    };
    let integer = match args.keyword("add_argument", "type")? {
        None => false,
        Some(value) if runtime.is_integer_type(value) => true,
        Some(value) if runtime.is_string_type(value) => false,
        Some(_) => {
            return Err(PyError::value_error(
                "argparse type must be int or str in shellsim",
            ))
        }
    };
    let default = args
        .keyword("add_argument", "default")?
        .copied()
        .unwrap_or(if store_true {
            Value::Bool(false)
        } else if store_false {
            Value::Bool(true)
        } else {
            Value::None
        });
    let mut choices = Vec::new();
    if let Some(value) = args.keyword("add_argument", "choices")? {
        let iterator = runtime.iterator(*value)?;
        while let Some(value) = runtime.iterator_next(iterator)? {
            choices.push(value);
        }
    }
    let help = optional_string(runtime, &args, "add_argument", "help")?;
    runtime.append_argument(
        parser,
        PyArgumentSpec {
            names,
            dest,
            required,
            default,
            store_true,
            store_false,
            integer,
            choices,
            help,
        },
    )?;
    Ok(Value::None)
}

fn add_subparsers(runtime: &mut dyn PyRuntime, receiver: Value, args: CallArgs) -> PyResult {
    args.expect_positional("add_subparsers", 0, 0)?;
    args.reject_unknown_keywords("add_subparsers", &["dest", "required", "help"])?;
    let parser = receiver.cast::<PyArgumentParser>(runtime)?;
    let dest = optional_string(runtime, &args, "add_subparsers", "dest")?;
    let required = boolean_keyword(runtime, &args, "add_subparsers", "required", false)?;
    let help = optional_string(runtime, &args, "add_subparsers", "help")?;
    runtime.configure_subparsers(
        parser,
        PySubparsersSpec {
            dest,
            required,
            help,
            commands: Vec::new(),
        },
    )?;
    // The parent parser is also the deliberately tiny subparser builder. `add_parser` validates
    // that `add_subparsers` initialized it before accepting a command.
    Ok(receiver)
}

fn add_parser(runtime: &mut dyn PyRuntime, receiver: Value, args: CallArgs) -> PyResult {
    args.expect_positional("add_parser", 1, 1)?;
    args.reject_unknown_keywords("add_parser", &["help", "description", "add_help"])?;
    let parent = receiver.cast::<PyArgumentParser>(runtime)?;
    let OwnedPyString(name) = args.positional()[0].cast(runtime)?;
    let help = optional_string(runtime, &args, "add_parser", "help")?;
    let description = optional_string(runtime, &args, "add_parser", "description")?;
    let add_help = boolean_keyword(runtime, &args, "add_parser", "add_help", true)?;
    let parent_data = runtime.argument_parser_parts(parent)?;
    let child = runtime.new_argument_parser(
        format!("{} {name}", parent_data.prog),
        description,
        add_help,
        true,
    )?;
    let child_parser = child.cast::<PyArgumentParser>(runtime)?;
    runtime.append_subcommand(
        parent,
        PySubcommandSpec {
            name,
            help,
            parser: child_parser,
        },
    )?;
    Ok(child)
}

fn parse_args(runtime: &mut dyn PyRuntime, receiver: Value, args: CallArgs) -> PyResult {
    let (values, unknown) = parse(runtime, receiver, args, false)?;
    debug_assert!(unknown.is_empty());
    runtime.new_namespace(values)
}

fn parse_known_args(runtime: &mut dyn PyRuntime, receiver: Value, args: CallArgs) -> PyResult {
    let (values, unknown) = parse(runtime, receiver, args, true)?;
    let namespace = runtime.new_namespace(values)?;
    let mut unknown_values = Vec::with_capacity(unknown.len());
    for value in unknown {
        unknown_values.push(runtime.new_string(value)?);
    }
    let unknown = runtime.new_list(unknown_values)?;
    runtime.new_tuple(vec![namespace, unknown])
}

fn parse(
    runtime: &mut dyn PyRuntime,
    receiver: Value,
    args: CallArgs,
    allow_unknown: bool,
) -> ParsedArguments {
    let operation = if allow_unknown {
        "parse_known_args"
    } else {
        "parse_args"
    };
    args.expect_positional(operation, 0, 1)?;
    args.reject_keywords(operation)?;
    let parser = receiver.cast::<PyArgumentParser>(runtime)?;
    let data = runtime.argument_parser_parts(parser)?;
    let input = input_arguments(runtime, &args)?;
    parse_values(runtime, &data, &input, allow_unknown)
}

fn input_arguments(runtime: &mut dyn PyRuntime, args: &CallArgs) -> PyResult<Vec<String>> {
    let Some(value) = args.positional().first() else {
        return Ok(runtime.command_arguments());
    };
    if value.is_none() {
        return Ok(runtime.command_arguments());
    }
    let iterator = runtime.iterator(*value)?;
    let mut input = Vec::new();
    while let Some(value) = runtime.iterator_next(iterator)? {
        input.push(value.cast::<OwnedPyString>(runtime)?.0);
    }
    Ok(input)
}

fn parse_values(
    runtime: &mut dyn PyRuntime,
    data: &PyArgumentParserData,
    input: &[String],
    allow_unknown: bool,
) -> ParsedArguments {
    let mut values = data
        .arguments
        .iter()
        .map(|spec| (spec.dest.clone(), spec.default))
        .collect::<Vec<_>>();
    if let Some(dest) = data
        .subparsers
        .as_ref()
        .and_then(|subparsers| subparsers.dest.as_ref())
    {
        values.push((dest.clone(), Value::None));
    }
    let positionals = data
        .arguments
        .iter()
        .filter(|spec| !spec.names.iter().any(|name| name.starts_with('-')))
        .collect::<Vec<_>>();
    let mut positional_index = 0;
    let mut unknown = Vec::new();
    let mut options = true;
    let mut index = 0;
    while index < input.len() {
        runtime.charge_cpu(1)?;
        let token = &input[index];
        if options && token == "--" {
            options = false;
            index += 1;
            continue;
        }
        if options && data.add_help && matches!(token.as_str(), "-h" | "--help") {
            write_help(runtime, data, PyMarker::Stdout)?;
            return Err(PyError::exit(0));
        }
        let (name, attached) = token
            .split_once('=')
            .map_or((token.as_str(), None), |(name, value)| (name, Some(value)));
        if options {
            if let Some(spec) = data
                .arguments
                .iter()
                .find(|spec| spec.names.iter().any(|candidate| candidate == name))
            {
                if spec.store_true || spec.store_false {
                    if attached.is_some() {
                        return parser_error(
                            runtime,
                            data,
                            format!("argument {name}: ignored explicit argument"),
                        );
                    }
                    set_value(&mut values, &spec.dest, Value::Bool(spec.store_true))?;
                    index += 1;
                    continue;
                }
                let raw = if let Some(attached) = attached {
                    attached.to_string()
                } else {
                    index += 1;
                    input.get(index).cloned().ok_or_else(|| {
                        PyError::value_error(format!("argument {name:?} expected one value"))
                    })?
                };
                let value = argument_value(runtime, spec, name, raw)?;
                set_value(&mut values, &spec.dest, value)?;
                index += 1;
                continue;
            }
        }
        if let Some(subparsers) = &data.subparsers {
            if let Some(command) = subparsers
                .commands
                .iter()
                .find(|command| command.name == *token)
            {
                if let Some(dest) = &subparsers.dest {
                    let command_name = runtime.new_string(command.name.clone())?;
                    set_value(&mut values, dest, command_name)?;
                }
                let child = runtime.argument_parser_parts(command.parser)?;
                let (child_values, child_unknown) =
                    parse_values(runtime, &child, &input[index + 1..], allow_unknown)?;
                for (name, value) in child_values {
                    set_or_push(&mut values, name, value);
                }
                unknown.extend(child_unknown);
                validate_required(runtime, data, &values)?;
                return Ok((values, unknown));
            }
        }
        if options && token.starts_with('-') {
            if allow_unknown {
                unknown.push(token.clone());
                index += 1;
                continue;
            }
            return parser_error(runtime, data, format!("unrecognized argument {token:?}"));
        }
        if let Some(spec) = positionals.get(positional_index) {
            let value = argument_value(runtime, spec, &spec.dest, token.clone())?;
            set_value(&mut values, &spec.dest, value)?;
            positional_index += 1;
        } else if allow_unknown {
            unknown.push(token.clone());
        } else if data.subparsers.is_some() {
            return parser_error(runtime, data, format!("unknown command {token:?}"));
        } else {
            return parser_error(runtime, data, format!("unrecognized argument {token:?}"));
        }
        index += 1;
    }
    validate_required(runtime, data, &values)?;
    if data
        .subparsers
        .as_ref()
        .is_some_and(|subparsers| subparsers.required)
    {
        return parser_error(runtime, data, "a command is required".into());
    }
    Ok((values, unknown))
}

fn argument_value(
    runtime: &mut dyn PyRuntime,
    spec: &PyArgumentSpec,
    name: &str,
    raw: String,
) -> PyResult<Value> {
    let value = if spec.integer {
        runtime
            .new_integer(&raw)
            .map_err(|_| PyError::value_error(format!("argument {name:?} must be an integer")))?
    } else {
        runtime.new_string(raw)?
    };
    if !spec.choices.is_empty() {
        let mut accepted = false;
        for choice in &spec.choices {
            if runtime.equals(&value, choice)? {
                accepted = true;
                break;
            }
        }
        if !accepted {
            return Err(PyError::value_error(format!(
                "invalid choice for argument {name:?}"
            )));
        }
    }
    Ok(value)
}

fn validate_required(
    runtime: &mut dyn PyRuntime,
    data: &PyArgumentParserData,
    values: &[(String, Value)],
) -> PyResult<()> {
    for spec in &data.arguments {
        runtime.charge_cpu(1)?;
        if spec.required
            && values
                .iter()
                .find(|(dest, _)| dest == &spec.dest)
                .is_none_or(|(_, value)| value.is_none())
        {
            return parser_error(
                runtime,
                data,
                format!(
                    "the following arguments are required: {}",
                    spec.names.join(", ")
                ),
            );
        }
    }
    Ok(())
}

fn format_help_method(runtime: &mut dyn PyRuntime, receiver: Value, args: CallArgs) -> PyResult {
    args.expect_positional("format_help", 0, 0)?;
    args.reject_keywords("format_help")?;
    let parser = receiver.cast::<PyArgumentParser>(runtime)?;
    let data = runtime.argument_parser_parts(parser)?;
    let help = format_help(runtime, &data)?;
    runtime.new_string(help)
}

fn print_help(runtime: &mut dyn PyRuntime, receiver: Value, args: CallArgs) -> PyResult {
    args.expect_positional("print_help", 0, 0)?;
    args.reject_keywords("print_help")?;
    let parser = receiver.cast::<PyArgumentParser>(runtime)?;
    let data = runtime.argument_parser_parts(parser)?;
    write_help(runtime, &data, PyMarker::Stdout)?;
    Ok(Value::None)
}

fn parser_error<T>(
    runtime: &mut dyn PyRuntime,
    data: &PyArgumentParserData,
    message: String,
) -> Result<T, PyError> {
    let usage = format_usage(data);
    runtime.reserve_memory(usage.len().saturating_add(message.len()).saturating_add(16))?;
    let stream = runtime.marker(PyMarker::Stderr);
    runtime.write_stream(
        &stream,
        &format!("{usage}\n{}: error: {message}\n", data.prog),
    )?;
    Err(PyError::exit(2))
}

fn write_help(
    runtime: &mut dyn PyRuntime,
    data: &PyArgumentParserData,
    stream: PyMarker,
) -> PyResult<()> {
    let help = format_help(runtime, data)?;
    let stream = runtime.marker(stream);
    runtime.write_stream(&stream, &help)?;
    Ok(())
}

fn format_help(runtime: &mut dyn PyRuntime, data: &PyArgumentParserData) -> PyResult<String> {
    let estimate = data
        .arguments
        .len()
        .saturating_mul(96)
        .saturating_add(
            data.subparsers
                .as_ref()
                .map_or(0, |subparsers| subparsers.commands.len().saturating_mul(96)),
        )
        .saturating_add(data.prog.len())
        .saturating_add(data.description.as_ref().map_or(0, String::len))
        .saturating_add(256);
    runtime.reserve_memory(estimate)?;
    runtime.charge_cpu(u64::try_from(estimate).unwrap_or(u64::MAX))?;
    let mut output = format!("{}\n", format_usage(data));
    if let Some(description) = &data.description {
        output.push_str(&format!("\n{description}\n"));
    }
    let positionals = data
        .arguments
        .iter()
        .filter(|spec| !spec.names.iter().any(|name| name.starts_with('-')))
        .collect::<Vec<_>>();
    if !positionals.is_empty() || data.subparsers.is_some() {
        output.push_str("\npositional arguments:\n");
        for spec in positionals {
            help_line(&mut output, &spec.dest, spec.help.as_deref());
        }
        if let Some(subparsers) = &data.subparsers {
            let names = subparsers
                .commands
                .iter()
                .map(|command| command.name.as_str())
                .collect::<Vec<_>>()
                .join(",");
            help_line(
                &mut output,
                &format!("{{{names}}}"),
                subparsers.help.as_deref(),
            );
            for command in &subparsers.commands {
                help_line(
                    &mut output,
                    &format!("  {}", command.name),
                    command.help.as_deref(),
                );
            }
        }
    }
    let options = data
        .arguments
        .iter()
        .filter(|spec| spec.names.iter().any(|name| name.starts_with('-')))
        .collect::<Vec<_>>();
    if data.add_help || !options.is_empty() {
        output.push_str("\noptions:\n");
        if data.add_help {
            help_line(
                &mut output,
                "-h, --help",
                Some("show this help message and exit"),
            );
        }
        for spec in options {
            let mut label = spec.names.join(", ");
            if !spec.store_true && !spec.store_false {
                label.push(' ');
                label.push_str(&spec.dest.to_ascii_uppercase());
            }
            help_line(&mut output, &label, spec.help.as_deref());
        }
    }
    Ok(output)
}

fn format_usage(data: &PyArgumentParserData) -> String {
    let mut usage = format!("usage: {}", data.prog);
    if data.add_help {
        usage.push_str(" [-h]");
    }
    for spec in &data.arguments {
        let optional = spec.names.iter().any(|name| name.starts_with('-'));
        let mut part = if optional {
            spec.names
                .iter()
                .find(|name| name.starts_with("--"))
                .unwrap_or(&spec.names[0])
                .clone()
        } else {
            spec.dest.clone()
        };
        if optional && !spec.store_true && !spec.store_false {
            part.push(' ');
            part.push_str(&spec.dest.to_ascii_uppercase());
        }
        if optional && !spec.required {
            usage.push_str(&format!(" [{part}]"));
        } else {
            usage.push(' ');
            usage.push_str(&part);
        }
    }
    if let Some(subparsers) = &data.subparsers {
        let names = subparsers
            .commands
            .iter()
            .map(|command| command.name.as_str())
            .collect::<Vec<_>>()
            .join(",");
        usage.push_str(&format!(" {{{names}}} ..."));
    }
    usage
}

fn help_line(output: &mut String, label: &str, help: Option<&str>) {
    output.push_str("  ");
    output.push_str(label);
    if let Some(help) = help {
        let padding = 24usize.saturating_sub(label.len());
        output.extend(std::iter::repeat_n(' ', padding.max(2)));
        output.push_str(help);
    }
    output.push('\n');
}

fn optional_string(
    runtime: &mut dyn PyRuntime,
    args: &CallArgs,
    operation: &str,
    keyword: &str,
) -> PyResult<Option<String>> {
    args.keyword(operation, keyword)?
        .copied()
        .map(|value| value.cast::<OwnedPyString>(runtime).map(|value| value.0))
        .transpose()
}

fn boolean_keyword(
    runtime: &mut dyn PyRuntime,
    args: &CallArgs,
    operation: &str,
    keyword: &str,
    default: bool,
) -> PyResult<bool> {
    args.keyword(operation, keyword)?
        .map(|value| {
            if matches!(runtime.kind(value)?, PyKind::Bool | PyKind::Int) {
                runtime.truth(value)
            } else {
                Err(PyError::type_error(format!("{keyword} must be a boolean")))
            }
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn set_value(values: &mut [(String, Value)], dest: &str, value: Value) -> PyResult<()> {
    let slot = values
        .iter_mut()
        .find(|(name, _)| name == dest)
        .ok_or_else(|| PyError::runtime_error("invalid parser state"))?;
    slot.1 = value;
    Ok(())
}

fn set_or_push(values: &mut Vec<(String, Value)>, name: String, value: Value) {
    if let Some((_, slot)) = values.iter_mut().find(|(candidate, _)| candidate == &name) {
        *slot = value;
    } else {
        values.push((name, value));
    }
}
