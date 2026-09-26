//! Python runtime integration tests share one crate so Cargo links the shellsim harness once.
//!
//! Private implementation invariants stay beside their production modules. These modules exercise
//! Python through shellsim's public interfaces; Python-only semantics live in source suites.

#[path = "python/asyncio.rs"]
mod asyncio;
#[path = "python/bytes.rs"]
mod bytes;
#[path = "python/cli.rs"]
mod cli;
#[path = "python/exceptions.rs"]
mod exceptions;
#[path = "python/execution.rs"]
mod execution;
#[path = "python/execution_representation.rs"]
mod execution_representation;
#[path = "python/first_order.rs"]
mod first_order;
#[path = "python/frozen_stdlib.rs"]
mod frozen_stdlib;
#[path = "python/function_arguments.rs"]
mod function_arguments;
#[path = "python/generators.rs"]
mod generators;
#[path = "python/language.rs"]
mod language;
#[path = "python/numeric_literals.rs"]
mod numeric_literals;
#[path = "python/numeric_lookaside.rs"]
mod numeric_lookaside;
#[path = "python/numpy.rs"]
mod numpy;
#[path = "python/object_model.rs"]
mod object_model;
#[path = "python/pytest.rs"]
mod pytest;
#[path = "python/resource_hardening.rs"]
mod resource_hardening;
#[path = "python/runtime.rs"]
mod runtime;
#[path = "python/source_suites.rs"]
mod source_suites;
#[path = "python/stdin_streaming.rs"]
mod stdin_streaming;
#[path = "python/stdlib.rs"]
mod stdlib;
#[path = "python/stdlib_differential.rs"]
mod stdlib_differential;
#[path = "python/subprocess.rs"]
mod subprocess;
#[path = "python/support.rs"]
mod support;
#[path = "python/tasktrove_differential.rs"]
mod tasktrove_differential;
#[path = "python/unittest.rs"]
mod unittest;
#[path = "python/unwind.rs"]
mod unwind;
