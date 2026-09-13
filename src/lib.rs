//! shellsim — a deterministic, resource-constrained BusyBox-like agent environment.
//!
//! The crate is organized as a small operating environment:
//!   * [`vfs`]   — in-memory filesystem (the single source of truth)
//!   * [`clock`] — virtual clock (sleep never blocks)
//!   * [`net`]   — virtual network (curl/wget against a route table)
//!   * [`interp`]— environment and shell/session state
//!   * [`resources`] — CPU, memory, disk/output limits and structured outcomes
//!   * [`shell`] / [`expand`] / [`exec`] — bash-subset parser, word expansion, executor
//!   * [`commands`] — native coreutils + builtins
//!   * [`python`] — metered Python 3.14 source-to-bytecode compatibility engine

pub mod clock;
pub mod commands;
pub mod descriptors;
pub mod exec;
pub mod expand;
pub mod harness;
pub mod harness_manager;
pub mod hashes;
pub mod host_ingest;
pub mod interp;
pub mod jqcmd;
pub mod net;
pub mod netcmd;
pub mod process;
pub mod pseudo_fs;
pub mod python;
pub mod resources;
pub mod sandbox;
pub mod scheduler;
pub mod shell;
pub mod telemetry;
pub mod vfs;

pub use clock::{
    BlockOutcome, EventId, EventKind, ScheduledEvent, Timeline, TimelineError, TimelineLimits,
};
pub use interp::{Environment, Interp, ProcessState};
pub use resources::{Limits, RunOutcome, StopReason, Usage};
pub use telemetry::{CommandTrust, InvocationEvent};
