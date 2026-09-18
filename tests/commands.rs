//! Command integration tests share one crate so Cargo compiles and links the shellsim harness once.
//!
//! Each command has a plainly named module under `tests/commands/`; closely coupled archive
//! formats share one module. Cross-command shell behavior remains in the top-level shell suites.

#[path = "commands/archive.rs"]
mod archive;
#[path = "commands/awk.rs"]
mod awk;
#[path = "commands/dd.rs"]
mod dd;
#[path = "commands/find.rs"]
mod find;
#[path = "commands/git.rs"]
mod git;
#[path = "commands/grep.rs"]
mod grep;
#[path = "commands/jq.rs"]
mod jq;
#[path = "commands/make.rs"]
mod make;
#[path = "commands/network.rs"]
mod network;
#[path = "commands/patch.rs"]
mod patch;
#[path = "commands/rg.rs"]
mod rg;
#[path = "commands/sed.rs"]
mod sed;
