//! Native arithmetic core for the frozen :mod:`functools` compatibility layer.

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyCallable, PyError, PyList, PyResult, PyRuntime, PyValueCast,
};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "_functools",
    functions: &[FunctionDef {
        module: "_functools",
        name: "reduce",
        call: native_reduce,
    }],
    values: &[],
};

fn native_reduce<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("reduce", 2, 3)?;
    args.reject_keywords("reduce")?;
    let function = args.positional()[0].cast::<PyCallable>(runtime)?;
    let iterator = runtime.iterator(args.positional()[1])?;
    let initial = match args.positional().get(2) {
        Some(initial) => *initial,
        None => runtime.iterator_next(iterator)?.ok_or_else(|| {
            PyError::value_error("reduce() of empty sequence with no initial value")
        })?,
    };
    // The accumulator lives in a one-element list so each step can run in its own handle scope;
    // an unbounded iterator then holds a constant number of handles in this frame.
    let accumulator = runtime.new_list(vec![initial])?.cast::<PyList>(runtime)?;
    let mut exhausted = false;
    while !exhausted {
        runtime.nested(&mut |runtime, _| {
            let Some(value) = runtime.iterator_next(iterator)? else {
                exhausted = true;
                return Ok(());
            };
            runtime.charge_cpu(1)?;
            let current = runtime.list_items(accumulator)?[0];
            let next = function.call(runtime, CallArgs::new(vec![current, value], Vec::new()))?;
            runtime.replace_list_items(accumulator, vec![next])
        })?;
    }
    runtime.list_pop(accumulator, 0)
}

/// Fold a finite sequence through a caller-supplied operation.
///
/// The operation itself stays in the VM so Python's dynamic call and comparison rules remain in
/// one place.  This helper only gives the adapter a clear fold shape.
#[cfg(test)]
pub fn reduce_steps<T, F>(values: impl IntoIterator<Item = T>, mut accumulator: T, mut step: F) -> T
where
    F: FnMut(T, T) -> T,
{
    for value in values {
        accumulator = step(accumulator, value);
    }
    accumulator
}

#[cfg(test)]
mod tests {
    use super::reduce_steps;

    #[test]
    fn reduce_steps_folds_left_to_right() {
        assert_eq!(reduce_steps([2, 3, 4], 1, |a, b| a * b), 24);
    }
}
