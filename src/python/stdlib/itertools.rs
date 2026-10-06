//! Small, capability-free cores for the finite portions of :mod:`itertools`.
//!
//! Infinite arithmetic iterators remain lazy. Combinatorial operations materialize a bounded
//! result so every path reserves memory and consumes CPU fuel before host work can grow.

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyError, PyIndex, PyIterator, PyList, PyResult, PyRuntime,
    PyValue, PyValueCast,
};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "_itertools",
    functions: &[
        FunctionDef {
            module: "itertools",
            name: "chain",
            call: native_chain,
        },
        FunctionDef {
            module: "itertools",
            name: "combinations",
            call: native_combinations,
        },
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
        FunctionDef {
            module: "itertools",
            name: "permutations",
            call: native_permutations,
        },
        FunctionDef {
            module: "itertools",
            name: "product",
            call: native_product,
        },
    ],
    values: &[],
};

fn native_chain(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.reject_keywords("chain")?;
    let output = new_output_list(runtime)?;
    for iterable in args.positional() {
        let iterator = runtime.iterator(*iterable)?;
        drain_into(runtime, iterator, output)?;
    }
    let output = runtime.list_items(output)?;
    runtime.new_iterator(output)
}

fn new_output_list(runtime: &mut dyn PyRuntime) -> PyResult<PyList> {
    runtime.new_list(Vec::new())?.cast::<PyList>(runtime)
}

/// Append the remaining items of `iterator` to `output`, charging each as a materialized element.
///
/// Each step runs in its own pin scope so a long iterator does not accumulate temporary
/// pins; the list keeps the collected items alive.
fn drain_into(runtime: &mut dyn PyRuntime, iterator: PyIterator, output: PyList) -> PyResult<()> {
    let mut exhausted = false;
    while !exhausted {
        runtime.nested(&mut |runtime| {
            let Some(value) = runtime.iterator_next(iterator)? else {
                exhausted = true;
                return Ok(());
            };
            runtime.charge_cpu(1)?;
            runtime.list_append(output, value)
        })?;
    }
    Ok(())
}

fn native_product(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.reject_unknown_keywords("product", &["repeat"])?;
    let repeat = args
        .keyword("product", "repeat")?
        .map(|value| value.cast::<PyIndex>(runtime).map(|value| value.0))
        .transpose()?
        .unwrap_or(1);
    let repeat = usize::try_from(repeat)
        .map_err(|_| PyError::value_error("repeat argument cannot be negative"))?;
    if repeat > 64 {
        return Err(PyError::resource_error("product repeat is too large"));
    }
    let mut base_pools = Vec::new();
    for value in args.positional() {
        let iterator = runtime.iterator(*value)?;
        let pool = new_output_list(runtime)?;
        drain_into(runtime, iterator, pool)?;
        base_pools.push(runtime.list_items(pool)?);
    }
    let mut pools = Vec::new();
    for _ in 0..repeat {
        pools.extend(base_pools.iter().cloned());
    }
    let count = pools.iter().try_fold(1usize, |count, pool| {
        count
            .checked_mul(pool.len())
            .ok_or_else(|| PyError::resource_error("product result is too large"))
    })?;
    reserve_rows(runtime, count)?;
    let mut rows = vec![Vec::new()];
    for pool in pools {
        let mut next = Vec::new();
        for row in &rows {
            for value in &pool {
                runtime.charge_cpu(1)?;
                let mut expanded = row.clone();
                expanded.push(*value);
                next.push(expanded);
            }
        }
        rows = next;
    }
    tuple_iterator(runtime, rows)
}

fn native_permutations(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("permutations", 1, 2)?;
    args.reject_keywords("permutations")?;
    let values = collect_iterable_bounded(runtime, args.positional()[0], 64)?;
    let length = optional_length(runtime, &args, values.len())?;
    if length > values.len() {
        return runtime.new_iterator(Vec::new());
    }
    let count = (0..length).try_fold(1usize, |count, index| {
        count
            .checked_mul(values.len().saturating_sub(index))
            .ok_or_else(|| PyError::resource_error("permutation result is too large"))
    })?;
    reserve_rows(runtime, count)?;
    let mut rows = Vec::with_capacity(count);
    let mut used = vec![false; values.len()];
    build_permutations(
        runtime,
        &values,
        length,
        &mut used,
        &mut Vec::new(),
        &mut rows,
    )?;
    tuple_iterator(runtime, rows)
}

fn build_permutations(
    runtime: &mut dyn PyRuntime,
    values: &[PyValue],
    length: usize,
    used: &mut [bool],
    row: &mut Vec<PyValue>,
    output: &mut Vec<Vec<PyValue>>,
) -> PyResult<()> {
    if row.len() == length {
        runtime.charge_cpu(1)?;
        output.push(row.clone());
        return Ok(());
    }
    for index in 0..values.len() {
        if used[index] {
            continue;
        }
        used[index] = true;
        row.push(values[index]);
        build_permutations(runtime, values, length, used, row, output)?;
        row.pop();
        used[index] = false;
    }
    Ok(())
}

fn native_combinations(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("combinations", 2, 2)?;
    args.reject_keywords("combinations")?;
    let values = collect_iterable_bounded(runtime, args.positional()[0], 64)?;
    let length = args.positional()[1].cast::<PyIndex>(runtime)?.0;
    let length = usize::try_from(length)
        .map_err(|_| PyError::value_error("combination length cannot be negative"))?;
    let count = combination_count(values.len(), length)?;
    reserve_rows(runtime, count)?;
    let mut rows = Vec::with_capacity(count);
    build_combinations(runtime, &values, length, 0, &mut Vec::new(), &mut rows)?;
    tuple_iterator(runtime, rows)
}

fn combination_count(total: usize, selected: usize) -> PyResult<usize> {
    if selected > total {
        return Ok(0);
    }
    let selected = selected.min(total - selected);
    let mut result = 1usize;
    for index in 0..selected {
        result = result
            .checked_mul(total - index)
            .ok_or_else(|| PyError::resource_error("combination result is too large"))?
            / (index + 1);
    }
    Ok(result)
}

fn build_combinations(
    runtime: &mut dyn PyRuntime,
    values: &[PyValue],
    length: usize,
    start: usize,
    row: &mut Vec<PyValue>,
    output: &mut Vec<Vec<PyValue>>,
) -> PyResult<()> {
    if row.len() == length {
        runtime.charge_cpu(1)?;
        output.push(row.clone());
        return Ok(());
    }
    for index in start..values.len() {
        row.push(values[index]);
        build_combinations(runtime, values, length, index + 1, row, output)?;
        row.pop();
    }
    Ok(())
}

fn reserve_rows(runtime: &mut dyn PyRuntime, count: usize) -> PyResult<()> {
    let bytes = count
        .checked_mul(64)
        .ok_or_else(|| PyError::resource_error("iterator result is too large"))?;
    runtime.reserve_memory(bytes)
}

fn collect_iterable_bounded(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
    maximum: usize,
) -> PyResult<Vec<PyValue>> {
    let iterator = runtime.iterator(value)?;
    let mut output = Vec::new();
    while let Some(value) = runtime.iterator_next(iterator)? {
        if output.len() == maximum {
            return Err(PyError::resource_error("iterator input is too large"));
        }
        runtime.charge_cpu(1)?;
        runtime.reserve_memory(64)?;
        output.push(value);
    }
    Ok(output)
}

fn optional_length(runtime: &dyn PyRuntime, args: &CallArgs, default: usize) -> PyResult<usize> {
    args.positional().get(1).map_or(Ok(default), |value| {
        let value = (*value).cast::<PyIndex>(runtime)?.0;
        usize::try_from(value).map_err(|_| PyError::value_error("length cannot be negative"))
    })
}

fn tuple_iterator(runtime: &mut dyn PyRuntime, rows: Vec<Vec<PyValue>>) -> PyResult {
    let mut values = Vec::with_capacity(rows.len());
    for row in rows {
        values.push(runtime.new_tuple(row)?);
    }
    runtime.new_iterator(values)
}

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
    let index = |value: &PyValue| (*value).cast::<PyIndex>(runtime).map(|value| value.0);
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
    let output = new_output_list(runtime)?;
    for position in 0..stop {
        let mut exhausted = false;
        runtime.nested(&mut |runtime| {
            runtime.charge_cpu(1)?;
            let Some(value) = runtime.iterator_next(iterator)? else {
                exhausted = true;
                return Ok(());
            };
            if position >= start && (position - start) % step == 0 {
                runtime.list_append(output, value)?;
            }
            Ok(())
        })?;
        if exhausted {
            break;
        }
    }
    let output = runtime.list_items(output)?;
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
