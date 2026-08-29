//! Deterministic resource accounting for the simulated environment.
//!
//! CPU is monotonic fuel. Memory is a modeled concurrent working set and is restored at
//! command-frame boundaries. Output is a hard guardrail for materialized stdout/stderr.

use serde::{Deserialize, Serialize};

pub const COST_MODEL_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Limits {
    pub cpu: u64,
    pub memory: u64,
    pub disk: u64,
    pub output: u64,
}

impl Limits {
    pub const fn unlimited() -> Self {
        Self {
            cpu: u64::MAX,
            memory: u64::MAX,
            disk: u64::MAX,
            output: u64::MAX,
        }
    }
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            cpu: 10_000_000,
            memory: 64 * 1024 * 1024,
            disk: 64 * 1024 * 1024,
            output: 4 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    CpuExhausted,
    MemoryExhausted,
    OutputLimitExceeded,
}

impl StopReason {
    pub const fn exit_status(self) -> i32 {
        137
    }
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CpuExhausted => f.write_str("CPU exhausted"),
            Self::MemoryExhausted => f.write_str("memory exhausted"),
            Self::OutputLimitExceeded => f.write_str("output limit exceeded"),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    pub cpu_used: u64,
    pub memory_current: u64,
    pub memory_peak: u64,
    pub disk_current: u64,
    pub disk_peak: u64,
    pub output_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CommandUsage {
    pub command: String,
    pub cpu: u64,
    pub disk_delta: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunOutcome {
    pub exit_status: i32,
    pub stop_reason: Option<StopReason>,
    pub limits: Limits,
    pub usage: Usage,
    pub command_usage: Vec<CommandUsage>,
    pub cost_model_version: u32,
}

#[derive(Clone, Debug)]
pub struct Resources {
    limits: Limits,
    cpu_used: u64,
    memory_current: u64,
    memory_peak: u64,
    output_bytes: u64,
    stop_reason: Option<StopReason>,
    command_usage: Vec<CommandUsage>,
}

impl Resources {
    pub fn new(limits: Limits) -> Self {
        Self {
            limits,
            cpu_used: 0,
            memory_current: 0,
            memory_peak: 0,
            output_bytes: 0,
            stop_reason: None,
            command_usage: Vec::new(),
        }
    }

    pub fn limits(&self) -> Limits {
        self.limits
    }

    pub fn stop_reason(&self) -> Option<StopReason> {
        self.stop_reason
    }

    pub fn is_stopped(&self) -> bool {
        self.stop_reason.is_some()
    }

    pub fn charge_cpu(&mut self, units: u64) -> bool {
        if self.is_stopped() {
            return false;
        }
        let Some(next) = self.cpu_used.checked_add(units) else {
            self.cpu_used = self.limits.cpu;
            self.stop_reason = Some(StopReason::CpuExhausted);
            return false;
        };
        if next > self.limits.cpu {
            self.cpu_used = self.limits.cpu;
            self.stop_reason = Some(StopReason::CpuExhausted);
            false
        } else {
            self.cpu_used = next;
            true
        }
    }

    pub fn reserve_memory(&mut self, bytes: u64) -> bool {
        if self.is_stopped() {
            return false;
        }
        let Some(next) = self.memory_current.checked_add(bytes) else {
            self.stop_reason = Some(StopReason::MemoryExhausted);
            return false;
        };
        if next > self.limits.memory {
            self.stop_reason = Some(StopReason::MemoryExhausted);
            false
        } else {
            self.memory_current = next;
            self.memory_peak = self.memory_peak.max(next);
            true
        }
    }

    pub fn memory_mark(&self) -> u64 {
        self.memory_current
    }

    pub fn restore_memory(&mut self, mark: u64) {
        self.memory_current = mark.min(self.memory_current);
    }

    pub fn charge_output(&mut self, bytes: u64) -> bool {
        if self.is_stopped() {
            return false;
        }
        let Some(next) = self.output_bytes.checked_add(bytes) else {
            self.stop_reason = Some(StopReason::OutputLimitExceeded);
            return false;
        };
        if next > self.limits.output {
            self.output_bytes = self.limits.output;
            self.stop_reason = Some(StopReason::OutputLimitExceeded);
            false
        } else {
            self.output_bytes = next;
            true
        }
    }

    pub fn output_remaining(&self) -> u64 {
        self.limits.output.saturating_sub(self.output_bytes)
    }

    pub fn output_bytes(&self) -> u64 {
        self.output_bytes
    }

    pub fn cpu_used(&self) -> u64 {
        self.cpu_used
    }

    pub fn record_command(
        &mut self,
        command: &str,
        cpu_before: u64,
        disk_before: u64,
        disk_after: u64,
    ) {
        self.command_usage.push(CommandUsage {
            command: command.to_string(),
            cpu: self.cpu_used.saturating_sub(cpu_before),
            disk_delta: (disk_after as i128).saturating_sub(disk_before as i128) as i64,
        });
    }

    pub fn outcome(&self, exit_status: i32, disk_current: u64, disk_peak: u64) -> RunOutcome {
        RunOutcome {
            exit_status: self
                .stop_reason
                .map_or(exit_status, StopReason::exit_status),
            stop_reason: self.stop_reason,
            limits: self.limits,
            usage: Usage {
                cpu_used: self.cpu_used,
                memory_current: self.memory_current,
                memory_peak: self.memory_peak,
                disk_current,
                disk_peak,
                output_bytes: self.output_bytes,
            },
            command_usage: self.command_usage.clone(),
            cost_model_version: COST_MODEL_VERSION,
        }
    }
}

impl Default for Resources {
    fn default() -> Self {
        Self::new(Limits::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_is_monotonic_fuel() {
        let mut r = Resources::new(Limits {
            cpu: 10,
            ..Limits::unlimited()
        });
        assert!(r.charge_cpu(6));
        assert!(!r.charge_cpu(5));
        assert_eq!(r.cpu_used(), 10);
        assert_eq!(r.stop_reason(), Some(StopReason::CpuExhausted));
    }

    #[test]
    fn memory_can_be_restored_to_a_frame_mark() {
        let mut r = Resources::new(Limits {
            memory: 20,
            ..Limits::unlimited()
        });
        let mark = r.memory_mark();
        assert!(r.reserve_memory(12));
        r.restore_memory(mark);
        assert_eq!(r.memory_current, 0);
        assert_eq!(r.memory_peak, 12);
    }
}
