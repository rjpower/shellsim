//! Work metering for the iterative kernels.
//!
//! Series, continued fractions and root finders run from a handful to millions of iterations per
//! element depending on the arguments. The Hurwitz zeta sum for a negative non-integer `q`, for
//! example, runs about `|q|` terms, as SciPy's does. A flat per-element charge cannot bound that,
//! and the kernels are pure functions with no runtime to charge.
//!
//! So each loop whose trip count depends on the arguments calls [`step`] once per iteration, and
//! [`run`] evaluates a kernel under an allowance of steps. Once the allowance is spent, `step`
//! returns `false`, every loop stops early, and `run` reports that the evaluation did not
//! finish. The caller then charges the steps taken and retries with a larger allowance, so an
//! element may do as much work as the CPU budget allows but never more than it has been charged
//! for. Kernels are deterministic, so a retry computes what an unmetered evaluation would.
//!
//! The allowance is thread-local because it must reach loops deep inside kernels without
//! changing their signatures. Evaluations never nest, and outside [`run`] the allowance is
//! unlimited so unit tests can call kernels directly.

use std::cell::Cell;

#[derive(Clone, Copy)]
struct Allowance {
    remaining: u64,
    exhausted: bool,
}

const UNLIMITED: Allowance = Allowance {
    remaining: u64::MAX,
    exhausted: false,
};

thread_local! {
    static ALLOWANCE: Cell<Allowance> = const { Cell::new(UNLIMITED) };
}

/// Count one iteration. Returns `false` once the allowance is spent; the loop must then stop,
/// and [`run`] discards the kernel's result.
pub(super) fn step() -> bool {
    ALLOWANCE.with(|cell| {
        let mut allowance = cell.get();
        if allowance.remaining == 0 {
            allowance.exhausted = true;
            cell.set(allowance);
            return false;
        }
        allowance.remaining -= 1;
        cell.set(allowance);
        true
    })
}

/// Evaluate `kernel` with at most `steps` iterations. Returns the result, or `None` if the
/// kernel needed more steps, together with the steps taken.
pub(super) fn run<T>(steps: u64, kernel: impl FnOnce() -> T) -> (Option<T>, u64) {
    ALLOWANCE.with(|cell| {
        cell.set(Allowance {
            remaining: steps,
            exhausted: false,
        });
        let value = kernel();
        let allowance = cell.replace(UNLIMITED);
        let taken = steps - allowance.remaining;
        ((!allowance.exhausted).then_some(value), taken)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count_to(n: u64) -> u64 {
        let mut count = 0;
        while count < n && step() {
            count += 1;
        }
        count
    }

    #[test]
    fn a_kernel_that_fits_its_allowance_finishes() {
        assert_eq!(run(10, || count_to(10)), (Some(10), 10));
        assert_eq!(run(10, || count_to(3)), (Some(3), 3));
    }

    #[test]
    fn a_kernel_that_needs_more_steps_is_reported_unfinished() {
        assert_eq!(run(10, || count_to(11)), (None, 10));
        // The allowance is unlimited again afterwards.
        assert_eq!(count_to(1000), 1000);
    }
}
