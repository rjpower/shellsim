//! Constructors for the bounded, capability-free :mod:`argparse` compatibility surface.
//!
//! The default program name comes from the modeled Python invocation. Parser mutation and parsing
//! remain object protocols, so this module receives no filesystem or process capability.

use super::super::native::PyValue as Value;
use super::super::native::{
    CallArgs, FunctionDef, MethodDef, ModuleDef, NativeTypeDef, PyArgumentParser, PyArgumentSpec,
    PyError, PyKind, PyResult, PyRuntime, PyString, PyValueCast,
};

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
            name: "parse_args",
            call: parse_args,
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
    args.reject_unknown_keywords("ArgumentParser", &["prog", "description"])?;
    let program = args
        .keyword("ArgumentParser", "prog")?
        .cloned()
        .map(|value| value.cast::<PyString>(runtime).map(|value| value.0))
        .transpose()?
        .unwrap_or_else(|| runtime.argv0());
    runtime.new_argument_parser(program)
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
        .cloned()
        .map(|value| value.cast::<PyString>(runtime).map(|value| value.0))
        .collect::<PyResult<Vec<_>>>()?;
    let dest = args
        .keyword("add_argument", "dest")?
        .cloned()
        .map(|value| value.cast::<PyString>(runtime).map(|value| value.0))
        .transpose()?
        .unwrap_or_else(|| {
            names
                .iter()
                .find(|name| name.starts_with("--"))
                .unwrap_or(&names[0])
                .trim_start_matches('-')
                .replace('-', "_")
        });
    let required = args
        .keyword("add_argument", "required")?
        .map(|value| {
            if matches!(runtime.kind(value)?, PyKind::Bool | PyKind::Int) {
                runtime.truth(value)
            } else {
                Err(PyError::type_error("required must be a boolean"))
            }
        })
        .transpose()?
        .unwrap_or(false);
    let store_true = args
        .keyword("add_argument", "action")?
        .cloned()
        .map(|value| {
            value
                .cast::<PyString>(runtime)
                .map(|value| value.0 == "store_true")
        })
        .transpose()?
        .unwrap_or(false);
    let integer = args
        .keyword("add_argument", "type")?
        .is_some_and(|value| runtime.is_integer_type(value));
    let default = args
        .keyword("add_argument", "default")?
        .cloned()
        .unwrap_or(if store_true {
            Value::Bool(false)
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
    runtime.append_argument(
        parser,
        PyArgumentSpec {
            names,
            dest,
            required,
            default,
            store_true,
            integer,
            choices,
        },
    )?;
    Ok(Value::None)
}

fn parse_args(runtime: &mut dyn PyRuntime, receiver: Value, args: CallArgs) -> PyResult {
    args.expect_positional("parse_args", 0, 1)?;
    args.reject_keywords("parse_args")?;
    let parser = receiver.cast::<PyArgumentParser>(runtime)?;
    let (_program, specs) = runtime.argument_parser_parts(parser)?;
    let input = if let Some(value) = args.positional().first() {
        let iterator = runtime.iterator(*value)?;
        let mut input = Vec::new();
        while let Some(value) = runtime.iterator_next(iterator)? {
            input.push(value.cast::<PyString>(runtime)?.0);
        }
        input
    } else {
        runtime.command_arguments()
    };
    let values = parse_values(runtime, &specs, &input)?;
    runtime.new_namespace(values)
}

fn parse_values(
    runtime: &mut dyn PyRuntime,
    specs: &[PyArgumentSpec],
    input: &[String],
) -> PyResult<Vec<(String, Value)>> {
    let mut values = specs
        .iter()
        .map(|spec| (spec.dest.clone(), spec.default))
        .collect::<Vec<_>>();
    let mut positionals = specs
        .iter()
        .filter(|spec| !spec.names.iter().any(|name| name.starts_with('-')));
    let mut index = 0;
    while index < input.len() {
        runtime.charge_cpu(1)?;
        let token = &input[index];
        if token == "--" {
            index += 1;
            continue;
        }
        let (name, attached) = token
            .split_once('=')
            .map_or((token.as_str(), None), |(name, value)| (name, Some(value)));
        let spec = specs
            .iter()
            .find(|spec| spec.names.iter().any(|candidate| candidate == name));
        let (spec, raw) = if let Some(spec) = spec {
            if spec.store_true {
                set_value(&mut values, &spec.dest, Value::Bool(true))?;
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
            (spec, raw)
        } else if token.starts_with('-') {
            return Err(PyError::value_error(format!(
                "unrecognized argument {token:?}"
            )));
        } else {
            let spec = positionals
                .next()
                .ok_or_else(|| PyError::value_error(format!("unrecognized argument {token:?}")))?;
            (spec, token.clone())
        };
        let value = if spec.integer {
            runtime.new_integer(&raw).map_err(|_| {
                PyError::value_error(format!("argument {name:?} must be an integer"))
            })?
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
        set_value(&mut values, &spec.dest, value)?;
        index += 1;
    }
    for spec in specs {
        if spec.required
            && values
                .iter()
                .find(|(dest, _)| dest == &spec.dest)
                .is_none_or(|(_, value)| value.is_none())
        {
            return Err(PyError::value_error(format!(
                "the following arguments are required: {}",
                spec.names.join(", ")
            )));
        }
    }
    Ok(values)
}

fn set_value(values: &mut [(String, Value)], dest: &str, value: Value) -> PyResult<()> {
    let slot = values
        .iter_mut()
        .find(|(name, _)| name == dest)
        .ok_or_else(|| PyError::runtime_error("invalid parser state"))?;
    slot.1 = value;
    Ok(())
}
