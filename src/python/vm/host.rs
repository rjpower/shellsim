//! Capability-scoped process, clock, and environment adapters for native modules.

use super::{
    PyClock, PyEnvironment, PyError, PyProcessHandle, PyProcessOutput, PyProcessPoll,
    PyProcessRunner, PyProcessStartRequest, PyResult, Vm,
};

impl PyProcessRunner for Vm<'_> {
    fn start(&mut self, request: PyProcessStartRequest) -> PyResult<PyProcessHandle> {
        super::super::process::start(self.interp, request)
    }

    fn poll(&mut self, handle: PyProcessHandle) -> PyResult<Option<i32>> {
        super::super::process::poll(self.interp, handle)
    }

    fn wait(
        &mut self,
        handle: PyProcessHandle,
        timeout_ns: Option<u64>,
    ) -> PyResult<PyProcessOutput> {
        let mut output = if self.mode.scheduler_owned && self.native_suspend_allowed {
            match super::super::process::wait_if_ready(self.interp, handle, timeout_ns)? {
                Ok(output) => output,
                Err(reason) => {
                    return Err(PyError::suspend(reason));
                }
            }
        } else {
            super::super::process::wait(self.interp, handle, timeout_ns)?
        };
        self.out
            .extend_from_slice(&std::mem::take(&mut output.inherited_stdout));
        self.err
            .extend_from_slice(&std::mem::take(&mut output.inherited_stderr));
        Ok(output)
    }

    fn communicate(
        &mut self,
        handle: PyProcessHandle,
        input: Vec<u8>,
        timeout_ns: Option<u64>,
    ) -> PyResult<PyProcessOutput> {
        let mut output = if self.mode.scheduler_owned && self.native_suspend_allowed {
            match super::super::process::communicate_if_ready(
                self.interp,
                handle,
                input,
                timeout_ns,
            )? {
                Ok(output) => output,
                Err(reason) => {
                    return Err(PyError::suspend(reason));
                }
            }
        } else {
            super::super::process::communicate(self.interp, handle, input, timeout_ns)?
        };
        self.out
            .extend_from_slice(&std::mem::take(&mut output.inherited_stdout));
        self.err
            .extend_from_slice(&std::mem::take(&mut output.inherited_stderr));
        Ok(output)
    }

    fn read_pipe(
        &mut self,
        handle: PyProcessHandle,
        fd: i32,
        amount: Option<usize>,
    ) -> PyResult<Vec<u8>> {
        if self.mode.scheduler_owned && self.native_suspend_allowed {
            return match super::super::process::read_pipe_if_ready(self.interp, handle, fd, amount)?
            {
                Ok(bytes) => Ok(bytes),
                Err(reason) => Err(PyError::suspend(reason)),
            };
        }
        super::super::process::read_pipe(self.interp, handle, fd, amount)
    }

    fn try_read_pipe(
        &mut self,
        handle: PyProcessHandle,
        fd: i32,
        amount: Option<usize>,
    ) -> PyResult<PyProcessPoll<Vec<u8>>> {
        match super::super::process::read_pipe_if_ready(self.interp, handle, fd, amount)? {
            Ok(bytes) => Ok(PyProcessPoll::Ready(bytes)),
            Err(reason) => Ok(PyProcessPoll::Blocked(reason)),
        }
    }

    fn write_pipe(&mut self, handle: PyProcessHandle, input: Vec<u8>) -> PyResult<usize> {
        if self.mode.scheduler_owned && self.native_suspend_allowed {
            return match super::super::process::write_pipe_if_ready(self.interp, handle, input)? {
                Ok(written) => Ok(written),
                Err(reason) => Err(PyError::suspend(reason)),
            };
        }
        super::super::process::write_pipe(self.interp, handle, input)
    }

    fn try_write_pipe(
        &mut self,
        handle: PyProcessHandle,
        input: Vec<u8>,
    ) -> PyResult<PyProcessPoll<usize>> {
        match super::super::process::write_pipe_if_ready(self.interp, handle, input)? {
            Ok(written) => Ok(PyProcessPoll::Ready(written)),
            Err(reason) => Ok(PyProcessPoll::Blocked(reason)),
        }
    }

    fn close_pipe(&mut self, handle: PyProcessHandle, fd: i32) -> PyResult<()> {
        super::super::process::close_pipe(self.interp, handle, fd)
    }

    fn send_signal(
        &mut self,
        handle: PyProcessHandle,
        signal: crate::process::Signal,
    ) -> PyResult<()> {
        super::super::process::send_signal(self.interp, handle, signal)
    }
}

impl PyClock for Vm<'_> {
    fn wall_time(&self) -> PyResult<f64> {
        self.interp
            .clock
            .wall_time_seconds()
            .map_err(|error| PyError::runtime_error(error.to_string()))
    }

    fn wall_time_ns(&self) -> PyResult<i64> {
        let nanos = self
            .interp
            .clock
            .wall_time_ns()
            .map_err(|error| PyError::runtime_error(error.to_string()))?;
        i64::try_from(nanos)
            .map_err(|_| PyError::overflow_error("wall clock is outside Python int range"))
    }

    fn monotonic(&self) -> f64 {
        self.interp.clock.monotonic_seconds()
    }

    fn monotonic_ns(&self) -> PyResult<i64> {
        i64::try_from(self.interp.clock.monotonic_ns())
            .map_err(|_| PyError::overflow_error("monotonic clock is outside Python int range"))
    }

    fn process_time(&self) -> f64 {
        self.interp.resources.process_time_seconds()
    }

    fn process_time_ns(&self) -> PyResult<i64> {
        i64::try_from(self.interp.resources.process_time_ns())
            .map_err(|_| PyError::overflow_error("process clock is outside Python int range"))
    }

    fn sleep(&mut self, seconds: f64) -> PyResult<()> {
        let nanos = seconds * crate::clock::NANOS_PER_SECOND as f64;
        if nanos > u64::MAX as f64 {
            return Err(PyError::overflow_error("time.sleep() length is too large"));
        }
        if self.mode.scheduler_owned && self.native_suspend_allowed {
            let event = self
                .interp
                .clock
                .schedule_wake_after(u64::from(self.interp.process.pid), nanos as u64)
                .map_err(|error| PyError::runtime_error(error.to_string()))?;
            self.pending_wait = Some(crate::scheduler::WaitReason::Timer(event.deadline_ns()));
            return Ok(());
        }
        match self
            .interp
            .clock
            .block_task(crate::clock::MAIN_TASK_ID, nanos as u64)
            .map_err(|error| PyError::runtime_error(error.to_string()))?
        {
            crate::clock::BlockOutcome::Completed => {}
            crate::clock::BlockOutcome::Interrupted(event) => {
                self.interp.deadline_interrupt = Some(event.id);
            }
        }
        Ok(())
    }
}

impl PyEnvironment for Vm<'_> {
    fn get(&self, name: &str) -> Option<String> {
        self.interp.get_var(name)
    }
}
