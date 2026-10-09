//! Checked access to ordinary and process-owned shared guest memory.
//!
//! Shared accesses use atomic bytes even though guest continuations are polled
//! serially. Reads copy only the requested bounded range, never the whole heap.

use std::borrow::Cow;
use std::sync::atomic::{AtomicU8, Ordering};
use wasmtime::{Caller, Error, Memory, SharedMemory};

use super::Host;

#[derive(Clone)]
pub(super) enum GuestMemory {
    Ordinary(Memory),
    Shared(SharedMemory),
}

impl GuestMemory {
    pub(super) fn data_size(&self, caller: &Caller<'_, Host>) -> usize {
        match self {
            Self::Ordinary(memory) => memory.data_size(caller),
            Self::Shared(memory) => memory.data_size(),
        }
    }

    pub(super) fn range(
        &self,
        caller: &Caller<'_, Host>,
        start: usize,
        length: usize,
    ) -> Result<(), Error> {
        if start
            .checked_add(length)
            .is_none_or(|end| end > self.data_size(caller))
        {
            return Err(Error::msg("guest memory access out of bounds"));
        }
        Ok(())
    }

    pub(super) fn read(
        &self,
        caller: &Caller<'_, Host>,
        start: usize,
        output: &mut [u8],
    ) -> Result<(), Error> {
        self.range(caller, start, output.len())?;
        match self {
            Self::Ordinary(memory) => memory.read(caller, start, output).map_err(Error::new),
            Self::Shared(memory) => {
                for (offset, byte) in output.iter_mut().enumerate() {
                    *byte = shared_byte(memory, start + offset).load(Ordering::SeqCst);
                }
                Ok(())
            }
        }
    }

    pub(super) fn write(
        &self,
        caller: &mut Caller<'_, Host>,
        start: usize,
        input: &[u8],
    ) -> Result<(), Error> {
        self.range(caller, start, input.len())?;
        match self {
            Self::Ordinary(memory) => memory.write(caller, start, input).map_err(Error::new),
            Self::Shared(memory) => {
                for (offset, byte) in input.iter().enumerate() {
                    shared_byte(memory, start + offset).store(*byte, Ordering::SeqCst);
                }
                Ok(())
            }
        }
    }

    /// Ordinary callers borrow their range; shared callers copy only that range.
    /// The caller must reserve and charge the requested length before allocation.
    pub(super) fn bytes<'a>(
        &self,
        caller: &'a Caller<'_, Host>,
        start: usize,
        length: usize,
    ) -> Result<Cow<'a, [u8]>, Error> {
        self.range(caller, start, length)?;
        match self {
            Self::Ordinary(memory) => {
                Ok(Cow::Borrowed(&memory.data(caller)[start..start + length]))
            }
            Self::Shared(_) => {
                let mut bytes = vec![0; length];
                self.read(caller, start, &mut bytes)?;
                Ok(Cow::Owned(bytes))
            }
        }
    }

    /// Scan at most `limit` bytes and stop at the first NUL. Shared memory reads
    /// do not allocate or scan beyond the caller's aggregate string budget.
    pub(super) fn c_string<'a>(
        &self,
        caller: &'a Caller<'_, Host>,
        start: usize,
        limit: usize,
    ) -> Result<Cow<'a, [u8]>, Error> {
        let available = self
            .data_size(caller)
            .checked_sub(start)
            .ok_or_else(|| Error::msg("guest string out of bounds"))?;
        let length = available.min(limit);
        match self {
            Self::Ordinary(memory) => {
                let bytes = &memory.data(caller)[start..start + length];
                let end = bytes
                    .iter()
                    .position(|byte| *byte == 0)
                    .ok_or_else(|| Error::msg("unterminated guest string"))?;
                Ok(Cow::Borrowed(&bytes[..end]))
            }
            Self::Shared(memory) => {
                let mut bytes = Vec::new();
                for offset in 0..length {
                    let byte = shared_byte(memory, start + offset).load(Ordering::SeqCst);
                    if byte == 0 {
                        return Ok(Cow::Owned(bytes));
                    }
                    bytes.push(byte);
                }
                Err(Error::msg("unterminated guest string"))
            }
        }
    }
}

#[allow(unsafe_code)]
fn shared_byte(memory: &SharedMemory, offset: usize) -> &AtomicU8 {
    // Every caller checked the range first. SharedMemory backing storage stays
    // stable as it grows, and AtomicU8 has byte alignment and no padding. The
    // process owner polls only one continuation at a time: no guest word access
    // runs concurrently with these host byte accesses, including across Stores.
    unsafe {
        &*memory
            .data()
            .as_ptr()
            .cast::<u8>()
            .add(offset)
            .cast::<AtomicU8>()
    }
}
