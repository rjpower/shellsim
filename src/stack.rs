//! Bounded recursion over guest-controlled nesting.
//!
//! Parsers and evaluators recurse over syntax and data whose depth a simulated program chooses.
//! Two independent controls keep that recursion from overflowing the host stack, which would
//! abort the whole host process, including a Python process that embeds shellsim:
//!
//! * Each recursive domain keeps a deterministic depth counter with a fixed limit and reports an
//!   explicit error past it, so the same program fails the same way on every host.
//! * [`descend`] runs each level under [`stacker::maybe_grow`], which continues on a fresh heap
//!   segment when the host thread's stack runs low. The limits therefore need not be tuned to
//!   the host's stack size, build profile, or the embedding thread.
//!
//! Stack segments are not charged to the guest's memory budget: whether a segment is needed
//! depends on the host thread, so charging it would make accounting host dependent. The depth
//! limit already bounds that memory.
//!
//! Trees whose depth is bounded by these limits are still dropped, cloned, and compared by
//! derived recursive code that cannot grow the stack, so [`MAX_SYNTAX_DEPTH`] stays small
//! enough for that code to fit comfortably on a small thread stack.
//!
//! [`stacker`] tracks the bounds of the OS thread stack. It must not be reached from a Wasmtime
//! fiber stack, where its estimate of the remaining stack would be wrong.

/// Grow when less than this much stack remains; generous for debug-build frames.
const RED_ZONE: usize = 256 * 1024;
/// Size of each heap stack segment.
const SEGMENT: usize = 4 * 1024 * 1024;

/// Nesting limit for syntax trees built from guest source text.
///
/// Counts every level of the resulting tree, including links of left-associative chains such as
/// `a && b && c`, so code that walks the finished tree recursively is bounded too.
pub const MAX_SYNTAX_DEPTH: usize = 1000;

/// Run `body` one nesting level deeper, or return `None` when `depth` has reached `limit`.
///
/// `depth` selects the counter inside `owner`, which is restored after `body` returns. A panic
/// in `body` leaves the counter raised; callers do not reuse an owner after a panic.
///
/// ```ignore
/// fn group(&mut self) -> Result<Node, String> {
///     stack::descend(self, |p| &mut p.depth, stack::MAX_SYNTAX_DEPTH, Self::group_inner)
///         .ok_or_else(|| "nesting too deep".to_string())?
/// }
/// ```
pub(crate) fn descend<T: ?Sized, R>(
    owner: &mut T,
    depth: impl Fn(&mut T) -> &mut usize,
    limit: usize,
    body: impl FnOnce(&mut T) -> R,
) -> Option<R> {
    let level = depth(owner);
    if *level >= limit {
        return None;
    }
    *level += 1;
    let result = grow(|| body(owner));
    *depth(owner) -= 1;
    Some(result)
}

/// Run `body`, first moving to a fresh heap stack segment if the current stack is nearly full.
///
/// Use this directly for recursion already bounded by another counter.
pub(crate) fn grow<R>(body: impl FnOnce() -> R) -> R {
    stacker::maybe_grow(RED_ZONE, SEGMENT, body)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Counter {
        depth: usize,
    }

    fn recurse(counter: &mut Counter, remaining: usize) -> Option<usize> {
        if remaining == 0 {
            return Some(counter.depth);
        }
        // A large frame makes overflow certain without growth on a small thread.
        let padding = std::hint::black_box([0u8; 4096]);
        descend(
            counter,
            |c| &mut c.depth,
            usize::MAX,
            |c| recurse(c, remaining - 1),
        )?
        .map(|depth| depth + usize::from(padding[0]))
    }

    #[test]
    fn deep_recursion_grows_past_a_small_thread_stack() {
        let depth = std::thread::Builder::new()
            .stack_size(64 * 1024)
            .spawn(|| recurse(&mut Counter { depth: 0 }, 10_000))
            .unwrap()
            .join()
            .unwrap();
        assert_eq!(depth, Some(10_000));
    }

    #[test]
    fn limit_is_reported_and_counter_restored() {
        let mut counter = Counter { depth: 0 };
        assert_eq!(recurse(&mut counter, 5), Some(5));
        let mut limited = Counter { depth: 0 };
        let result = descend(
            &mut limited,
            |c| &mut c.depth,
            3,
            |c| {
                descend(
                    c,
                    |c| &mut c.depth,
                    3,
                    |c| {
                        descend(
                            c,
                            |c| &mut c.depth,
                            3,
                            |c| descend(c, |c| &mut c.depth, 3, |_| ()),
                        )
                    },
                )
            },
        );
        assert_eq!(result, Some(Some(Some(None))));
        assert_eq!(limited.depth, 0);
        assert_eq!(counter.depth, 0);
    }
}
