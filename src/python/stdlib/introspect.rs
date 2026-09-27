//! Private, read-only access to the parameter lists of Python functions.
//!
//! `parameters(callable)` lists a Python function's parameters in declaration order, as
//! `(name, kind, has_default, default)` tuples whose `kind` is the name of the matching
//! `inspect.Parameter` kind. A bound method omits its bound first parameter, as
//! `inspect.signature` does. Any other callable gives `None`. The helper reads only what the
//! compiler recorded for the function; no frame, local variable or host state crosses this
//! boundary. `scipy.stats` uses it to infer distribution shape parameters, where SciPy uses
//! `inspect.signature`.

use super::super::bytecode::ParameterKind;
use super::super::native::{CallArgs, FunctionDef, ModuleDef, PyResult, PyRuntime};
use super::super::Value;

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "_shellsim_introspect",
    functions: &[FunctionDef {
        module: "_shellsim_introspect",
        name: "parameters",
        call: parameters,
    }],
    values: &[],
};

fn kind_name(kind: ParameterKind) -> &'static str {
    match kind {
        ParameterKind::PositionalOnly => "POSITIONAL_ONLY",
        ParameterKind::Positional => "POSITIONAL_OR_KEYWORD",
        ParameterKind::Variadic => "VAR_POSITIONAL",
        ParameterKind::KeywordOnly => "KEYWORD_ONLY",
        ParameterKind::KeywordVariadic => "VAR_KEYWORD",
    }
}

fn parameters(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("parameters", 1, 1)?;
    args.reject_keywords("parameters")?;
    let Some(parameters) = runtime.function_parameters(&args.positional()[0])? else {
        return Ok(Value::None);
    };
    runtime.charge_cpu(u64::try_from(parameters.len()).unwrap_or(u64::MAX))?;
    let mut items = Vec::with_capacity(parameters.len());
    for parameter in parameters {
        let fields = vec![
            runtime.new_string(parameter.name)?,
            runtime.new_string(kind_name(parameter.kind).to_string())?,
            Value::Bool(parameter.default.is_some()),
            parameter.default.unwrap_or(Value::None),
        ];
        items.push(runtime.new_tuple(fields)?);
    }
    runtime.new_list(items)
}
