//! Capability-free min-heap algorithms from Python's :mod:`heapq` module.
//!
//! The API accepts any `Ord` value, covering the VM adapter's integer and string values without
//! introducing a host process or filesystem capability.  Python's dynamic mixed-type failures
//! should be checked by that adapter before invoking these algorithms.

/// Error raised by Python's `heapq.heappop`/`heapreplace` on an empty heap.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeapEmpty;

/// Transform an arbitrary vector into a valid min-heap in place.
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
pub fn heappush<T: Ord>(heap: &mut Vec<T>, item: T) {
    heap.push(item);
    let last = heap.len() - 1;
    sift_up(heap, last);
}

/// Pop the smallest item, or `HeapEmpty` for an empty vector.
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
pub fn heapreplace<T: Ord>(heap: &mut [T], item: T) -> Result<T, HeapEmpty> {
    if heap.is_empty() {
        return Err(HeapEmpty);
    }
    let smallest = std::mem::replace(&mut heap[0], item);
    sift_down(heap, 0);
    Ok(smallest)
}

/// Push an item then pop and return the smallest item, doing only one sift operation.
pub fn heappushpop<T: Ord>(heap: &mut [T], mut item: T) -> T {
    if let Some(root) = heap.first_mut() {
        if *root < item {
            std::mem::swap(root, &mut item);
            sift_down(heap, 0);
        }
    }
    item
}

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
