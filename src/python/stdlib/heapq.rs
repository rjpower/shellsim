//! Capability-free min-heap algorithms from Python's :mod:`heapq` module.
//!
//! The API accepts any `Ord` value, covering the VM adapter's integer and string values without
//! introducing a host process or filesystem capability.  Python's dynamic mixed-type failures
//! should be checked by that adapter before invoking these algorithms.

use std::cmp::Ordering;

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyError, PyList, PyResult, PyRuntime, PyValue, PyValueCast,
};
use super::super::Value;

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "heapq",
    functions: &[
        FunctionDef {
            module: "heapq",
            name: "heapify",
            call: native_heapify,
        },
        FunctionDef {
            module: "heapq",
            name: "heappop",
            call: native_heappop,
        },
    ],
    values: &[],
};

fn native_heapify(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("heapify", 1, 1)?;
    args.reject_keywords("heapify")?;
    let list = args.positional()[0].clone().cast::<PyList>(runtime)?;
    let mut values = list.items(runtime)?;
    dynamic_heapify(runtime, &mut values)?;
    runtime.replace_list_items(list, values)?;
    Ok(Value::None)
}

fn native_heappop(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("heappop", 1, 1)?;
    args.reject_keywords("heappop")?;
    let list = args.positional()[0].clone().cast::<PyList>(runtime)?;
    let mut values = list.items(runtime)?;
    let last = values
        .pop()
        .ok_or_else(|| PyError::value_error("index out of range"))?;
    if values.is_empty() {
        runtime.replace_list_items(list, values)?;
        return Ok(last);
    }
    let smallest = std::mem::replace(&mut values[0], last);
    dynamic_sift_down(runtime, &mut values, 0)?;
    runtime.replace_list_items(list, values)?;
    Ok(smallest)
}

fn dynamic_heapify(runtime: &mut dyn PyRuntime, values: &mut [PyValue]) -> PyResult<()> {
    if values.len() < 2 {
        return Ok(());
    }
    for index in (0..values.len() / 2).rev() {
        dynamic_sift_down(runtime, values, index)?;
    }
    Ok(())
}

fn dynamic_sift_down(
    runtime: &mut dyn PyRuntime,
    values: &mut [PyValue],
    mut parent: usize,
) -> PyResult<()> {
    loop {
        let left = parent
            .checked_mul(2)
            .and_then(|index| index.checked_add(1))
            .ok_or_else(|| PyError::resource_error("heap index overflow"))?;
        if left >= values.len() {
            return Ok(());
        }
        let right = left + 1;
        runtime.charge_cpu(1)?;
        let child = if right < values.len()
            && runtime.compare(&values[right], &values[left])? == Ordering::Less
        {
            right
        } else {
            left
        };
        runtime.charge_cpu(1)?;
        if runtime.compare(&values[parent], &values[child])? != Ordering::Greater {
            return Ok(());
        }
        values.swap(parent, child);
        parent = child;
    }
}

/// Error raised by Python's `heapq.heappop`/`heapreplace` on an empty heap.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeapEmpty;

/// Transform an arbitrary vector into a valid min-heap in place.
#[cfg(test)]
pub fn heapify<T: Ord>(heap: &mut [T]) {
    if heap.len() < 2 {
        return;
    }
    // Every node at or after this index is a leaf. Sift internal nodes down from the bottom up.
    for index in (0..heap.len() / 2).rev() {
        sift_down(heap, index);
    }
}

/// Push an item and restore the min-heap invariant.
#[cfg(test)]
pub fn heappush<T: Ord>(heap: &mut Vec<T>, item: T) {
    heap.push(item);
    let last = heap.len() - 1;
    sift_up(heap, last);
}

/// Pop the smallest item, or `HeapEmpty` for an empty vector.
#[cfg(test)]
pub fn heappop<T: Ord>(heap: &mut Vec<T>) -> Result<T, HeapEmpty> {
    let last = heap.pop().ok_or(HeapEmpty)?;
    if heap.is_empty() {
        return Ok(last);
    }
    let smallest = std::mem::replace(&mut heap[0], last);
    sift_down(heap, 0);
    Ok(smallest)
}

/// Replace the root and return the old smallest item.
#[cfg(test)]
pub fn heapreplace<T: Ord>(heap: &mut [T], item: T) -> Result<T, HeapEmpty> {
    if heap.is_empty() {
        return Err(HeapEmpty);
    }
    let smallest = std::mem::replace(&mut heap[0], item);
    sift_down(heap, 0);
    Ok(smallest)
}

/// Push an item then pop and return the smallest item, doing only one sift operation.
#[cfg(test)]
pub fn heappushpop<T: Ord>(heap: &mut [T], mut item: T) -> T {
    if let Some(root) = heap.first_mut() {
        if *root < item {
            std::mem::swap(root, &mut item);
            sift_down(heap, 0);
        }
    }
    item
}

#[cfg(test)]
fn sift_up<T: Ord>(heap: &mut [T], mut child: usize) {
    while child > 0 {
        let parent = (child - 1) / 2;
        if heap[parent] <= heap[child] {
            break;
        }
        heap.swap(parent, child);
        child = parent;
    }
}

#[cfg(test)]
fn sift_down<T: Ord>(heap: &mut [T], mut parent: usize) {
    loop {
        let left = parent * 2 + 1;
        if left >= heap.len() {
            return;
        }
        let right = left + 1;
        let child = if right < heap.len() && heap[right] < heap[left] {
            right
        } else {
            left
        };
        if heap[parent] <= heap[child] {
            return;
        }
        heap.swap(parent, child);
        parent = child;
    }
}

#[cfg(test)]
mod tests {
    use super::{heapify, heappop, heappush, heappushpop, heapreplace, HeapEmpty};

    fn is_heap<T: Ord>(heap: &[T]) -> bool {
        heap.iter().enumerate().all(|(parent, value)| {
            let left = parent * 2 + 1;
            let right = left + 1;
            heap.get(left).is_none_or(|child| value <= child)
                && heap.get(right).is_none_or(|child| value <= child)
        })
    }

    #[test]
    fn heapify_push_and_pop_are_min_ordered() {
        let mut heap = vec![5, 1, 4, 1, 3, 2];
        heapify(&mut heap);
        assert!(is_heap(&heap));
        heappush(&mut heap, 0);
        assert!(is_heap(&heap));
        let mut output = Vec::new();
        while !heap.is_empty() {
            output.push(heappop(&mut heap).unwrap());
        }
        assert_eq!(output, [0, 1, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn empty_and_replace_operations_match_python_shape() {
        let mut empty = Vec::<i64>::new();
        assert_eq!(heappop(&mut empty), Err(HeapEmpty));
        assert_eq!(heapreplace(&mut empty, 1), Err(HeapEmpty));
        let mut heap = vec![2, 4, 6];
        heapify(&mut heap);
        assert_eq!(heapreplace(&mut heap, 5), Ok(2));
        assert!(is_heap(&heap));
        assert_eq!(heappushpop(&mut heap, 1), 1);
        assert!(is_heap(&heap));
        assert_eq!(heappushpop(&mut heap, 9), 4);
        assert!(is_heap(&heap));
    }

    #[test]
    fn strings_are_supported_by_the_same_order_logic() {
        let mut heap = vec!["pear".to_string(), "apple".to_string(), "plum".to_string()];
        heapify(&mut heap);
        heappush(&mut heap, "banana".to_string());
        assert_eq!(heappop(&mut heap).unwrap(), "apple");
        assert_eq!(heappop(&mut heap).unwrap(), "banana");
    }
}
