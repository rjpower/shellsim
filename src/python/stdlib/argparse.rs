//! Constructors for the bounded, capability-free :mod:`argparse` compatibility surface.
//!
//! The default program name comes from the modeled Python invocation. Parser mutation and parsing
//! remain object protocols, so this module receives no filesystem or process capability.

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyResult, PyRuntime, PyString, PyValueCast,
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
