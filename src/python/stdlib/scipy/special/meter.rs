//! CPU accounting for iterative `scipy.special` kernels.
//!
//! Series, continued fractions and root finders need an amount of work that depends on their
//! arguments and can range from a handful of steps to millions (see `docs/scipy.md`, "Safety and
//! accounting"). Charging a fixed cost per element would either overcharge the common case or
//! let an adversarial input do unbounded work for a fixed price. Instead every iterative step
//! calls [`tick`], and [`evaluate`] runs the kernel under a growing allowance, charging CPU for
//! the steps actually taken:
//!
//! 1. Run the kernel with a starting allowance.
//! 2. If it never asked for a step beyond the allowance, charge for the steps it took and return
//!    the result.
//! 3. Otherwise charge for the steps it took, double the allowance, and run the kernel again from
//!    scratch.
//!
//! Kernels are pure functions of their arguments, so rerunning with a larger allowance reproduces
//! identical work up to the point where the previous run stopped, and the total work charged
//! across every attempt is less than twice the cost of the final, successful attempt. A kernel
//! that never converges still costs a bounded amount of CPU per doubling, so `runtime.charge_cpu`
//! eventually reports resource exhaustion rather than looping forever.
//!
//! The allowance is carried in thread-local cells rather than threaded through every kernel
//! call: kernels are plain `Fn(&[T]) -> T` closures shared with the ufunc loop (see
//! `numpy::ufunc::special_loop`), which leaves no room for an extra accounting parameter. Each
//! simulated Python process runs on one host thread, so the thread-local state never crosses
//! between unrelated evaluations.

use std::cell::Cell;

use super::super::super::super::native::{PyResult, PyRuntime};

thread_local! {
    static ALLOWANCE: Cell<u64> = const { Cell::new(0) };
    static STEPS: Cell<u64> = const { Cell::new(0) };
    static EXHAUSTED: Cell<bool> = const { Cell::new(false) };
}

/// Starting number of iterations a kernel may run before [`evaluate`] doubles its allowance.
const INITIAL_ALLOWANCE: u64 = 64;

/// Record one unit of iterative work (one series term, continued-fraction step, or root-finder
/// step). Returns `true` while the kernel stays under its current allowance and should keep
/// going; a kernel that sees `false` must stop and return its best estimate so far.
pub(in crate::python) fn tick() -> bool {
    STEPS.with(|steps| {
        let used = steps.get();
        if used >= ALLOWANCE.with(Cell::get) {
            EXHAUSTED.with(|exhausted| exhausted.set(true));
            return false;
        }
        steps.set(used + 1);
        true
    })
}

/// Run `kernel` under a growing iteration allowance, charging CPU for the steps it actually
/// took. `kernel` is called again, from scratch, each time it exhausts its allowance.
pub(in crate::python) fn evaluate<T>(
    runtime: &mut dyn PyRuntime,
    kernel: impl Fn() -> T,
) -> PyResult<T> {
    let mut allowance = INITIAL_ALLOWANCE;
    loop {
        ALLOWANCE.with(|cell| cell.set(allowance));
        STEPS.with(|cell| cell.set(0));
        EXHAUSTED.with(|cell| cell.set(false));
        let value = kernel();
        let steps = STEPS.with(Cell::get);
        // Every evaluation, even a closed-form one that never calls `tick`, costs at least one
        // unit so a tight loop of cheap calls still meters proportionally to element count.
        runtime.charge_cpu(steps.max(1))?;
        if !EXHAUSTED.with(Cell::get) {
            return Ok(value);
        }
        allowance = allowance.saturating_mul(2);
    }
}

/// Test-only escape hatch for calling a kernel directly, outside a `PyRuntime`: sets a generous
/// fixed allowance so `tick()`-gated loops (series, continued fractions, root finders) run to
/// their real convergence instead of the `0`-allowance default that applies when nothing has
/// called [`evaluate`] yet. Lets unit tests exercise tricky kernel-internal logic (see
/// `igam::tests`, `ibeta::tests`) without going through the ufunc dispatch and CPU-accounting
/// machinery those tests are not about.
#[cfg(test)]
pub(in crate::python) fn run_with_allowance_for_test<T>(
    allowance: u64,
    kernel: impl FnOnce() -> T,
) -> T {
    ALLOWANCE.with(|cell| cell.set(allowance));
    STEPS.with(|cell| cell.set(0));
    EXHAUSTED.with(|cell| cell.set(false));
    kernel()
}
