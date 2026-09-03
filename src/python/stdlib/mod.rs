//! Deterministic, capability-free implementations of Python's small standard-library modules.
//!
//! The parent Python runtime deliberately wires these modules into its import table separately.
//! Keeping this directory self-contained makes each shim straightforward to review and test.

#[allow(dead_code)]
pub mod bisect;
#[allow(dead_code)]
pub mod functools;
#[allow(dead_code)]
pub mod heapq;
pub mod itertools;
pub mod math;
pub mod string;
