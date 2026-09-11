//! Capability-free arithmetic helpers for the VM's small :mod:`functools` slice.

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyCallable, PyError, PyResult, PyRuntime, PyValueCast,
};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "functools",
    functions: &[FunctionDef {
        module: "functools",
        name: "reduce",
        call: native_reduce,
    }],
    values: &[],
};

fn native_reduce(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("reduce", 2, 3)?;
    args.reject_keywords("reduce")?;
    let function = args.positional()[0].clone().cast::<PyCallable>(runtime)?;
    let iterator = runtime.iterator(args.positional()[1].clone())?;
    let mut accumulator = match args.positional().get(2) {
        Some(initial) => initial.clone(),
        None => runtime.iterator_next(iterator)?.ok_or_else(|| {
            PyError::value_error("reduce() of empty sequence with no initial value")
        })?,
    };
    while let Some(value) = runtime.iterator_next(iterator)? {
        runtime.charge_cpu(1)?;
        accumulator = function
            .clone()
            .call(runtime, CallArgs::new(vec![accumulator, value], Vec::new()))?;
    }
    Ok(accumulator)
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
