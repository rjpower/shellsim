//! Small, capability-free cores for the finite portions of :mod:`itertools`.
//!
//! The VM adapter is responsible for representing Python's lazy iterators and for charging
//! resource usage.  These helpers only describe the arithmetic and finite slicing contract.

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyError, PyIndex, PyResult, PyRuntime, PyValueCast,
};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "itertools",
    functions: &[
        FunctionDef {
            module: "itertools",
            name: "count",
            call: native_count,
        },
        FunctionDef {
            module: "itertools",
            name: "islice",
            call: native_islice,
        },
    ],
    values: &[],
};

fn native_count(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("count", 0, 2)?;
    args.reject_keywords("count")?;
    let start = args
        .positional()
        .first()
        .cloned()
        .map(|value| value.cast::<PyIndex>(runtime).map(|value| value.0))
        .transpose()?
        .unwrap_or(0);
    let step = args
        .positional()
        .get(1)
        .cloned()
        .map(|value| value.cast::<PyIndex>(runtime).map(|value| value.0))
        .transpose()?
        .unwrap_or(1);
    runtime.new_count_iterator(start, step)
}

fn native_islice(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("islice", 2, 4)?;
    args.reject_keywords("islice")?;
    let values = args.positional();
    let iterable = values[0];
    let index =
        |value: &super::super::Value| (*value).cast::<PyIndex>(runtime).map(|value| value.0);
    let (start, stop, step) = match values {
        [_, stop] => (0, index(stop)?, 1),
        [_, start, stop] => (index(start)?, index(stop)?, 1),
        [_, start, stop, step] => (index(start)?, index(stop)?, index(step)?),
        _ => unreachable!(),
    };
    if start < 0 || stop < 0 {
        return Err(PyError::value_error("islice indices must be non-negative"));
    }
    if step <= 0 {
        return Err(PyError::value_error(
            "islice step must be greater than zero",
        ));
    }
    let start =
        usize::try_from(start).map_err(|_| PyError::overflow_error("islice start is too large"))?;
    let stop =
        usize::try_from(stop).map_err(|_| PyError::overflow_error("islice stop is too large"))?;
    let step =
        usize::try_from(step).map_err(|_| PyError::overflow_error("islice step is too large"))?;
    let iterator = runtime.iterator(iterable)?;
    let mut output = Vec::new();
    for position in 0..stop {
        runtime.charge_cpu(1)?;
        let Some(value) = runtime.iterator_next(iterator)? else {
            break;
        };
        if position >= start && (position - start) % step == 0 {
            runtime.reserve_memory(64)?;
            output.push(value);
        }
    }
    runtime.new_iterator(output)
}

/// Return the next value in a count sequence, checking the bounded integer domain.
pub fn count_next(current: i64, step: i64) -> Result<i64, &'static str> {
    current
        .checked_add(step)
        .ok_or("itertools.count exceeded the bounded integer range")
}

/// Produce a finite slice from an arithmetic count sequence.
#[cfg(test)]
pub fn count_slice(start: i64, step: i64, length: usize) -> Result<Vec<i64>, &'static str> {
    let mut values = Vec::with_capacity(length);
    let mut current = start;
    for _ in 0..length {
        values.push(current);
        current = count_next(current, step)?;
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::{count_next, count_slice};

    #[test]
    fn count_is_bounded_and_slice_is_finite() {
        assert_eq!(count_next(4, 2), Ok(6));
        assert_eq!(count_slice(3, 1, 3), Ok(vec![3, 4, 5]));
        assert!(count_next(i64::MAX, 1).is_err());
    }
}
