//! Bounded virtual framebuffer and input queue for simulated programs.
//!
//! Frames and events are machine-owned data, not handles to a host window or keyboard. One
//! display is sufficient for the first single-player guest while keeping ownership explicit.

use std::collections::VecDeque;

use crate::process::ProcessId;
use crate::resources::Resources;

const MAX_DIMENSION: u32 = 2048;
const MAX_EVENTS: usize = 256;
const EVENT_BYTES: u64 = 16;
pub(crate) const DISPLAY_HANDLE: u32 = 1;
pub(crate) const RGBA8888: u32 = 1;

/// One key transition in the virtual input queue. Codes are guest-defined, not host keycodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeyEvent {
    pub code: u32,
    pub pressed: bool,
}

/// Last frame submitted by a virtual process.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DisplayFrame {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// Errors at the virtual display boundary; callers translate them to their guest ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplayError {
    InvalidArgument,
    InvalidHandle,
    Busy,
    QueueFull,
    ResourceExhausted,
}

/// Environment-owned device state and its retained, metered frame.
#[derive(Clone, Debug, Default)]
pub struct VirtualDisplay {
    frame: Option<DisplayFrame>,
    owner: Option<ProcessId>,
    events: VecDeque<KeyEvent>,
}

impl VirtualDisplay {
    /// Inspect the last completed frame without granting the guest host-display access.
    pub fn frame(&self) -> Option<&DisplayFrame> {
        self.frame.as_ref()
    }

    /// Queue a bounded virtual key transition, including before the guest opens its display.
    pub(crate) fn inject_key(
        &mut self,
        resources: &mut Resources,
        event: KeyEvent,
    ) -> Result<(), DisplayError> {
        if event.code == 0 || event.code > 0xffff {
            return Err(DisplayError::InvalidArgument);
        }
        if self.events.len() >= MAX_EVENTS {
            return Err(DisplayError::QueueFull);
        }
        if !resources.reserve_memory(EVENT_BYTES) {
            return Err(DisplayError::ResourceExhausted);
        }
        self.events.push_back(event);
        Ok(())
    }

    pub(crate) fn open(
        &mut self,
        resources: &mut Resources,
        owner: ProcessId,
        width: u32,
        height: u32,
        format: u32,
    ) -> Result<u32, DisplayError> {
        if self.owner.is_some() {
            return Err(DisplayError::Busy);
        }
        if width == 0
            || height == 0
            || width > MAX_DIMENSION
            || height > MAX_DIMENSION
            || format != RGBA8888
        {
            return Err(DisplayError::InvalidArgument);
        }
        let bytes = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or(DisplayError::InvalidArgument)?;
        let old_bytes = self
            .frame
            .as_ref()
            .map_or(0, |frame| frame.pixels.len() as u64);
        if bytes > old_bytes {
            if !resources.reserve_memory(bytes - old_bytes) {
                return Err(DisplayError::ResourceExhausted);
            }
        } else {
            resources.release_memory(old_bytes - bytes);
        }
        self.frame = Some(DisplayFrame {
            width,
            height,
            pixels: vec![0; bytes as usize],
        });
        self.owner = Some(owner);
        Ok(DISPLAY_HANDLE)
    }

    pub(crate) fn present(
        &mut self,
        resources: &mut Resources,
        owner: ProcessId,
        handle: u32,
        pixels: &[u8],
        stride: u32,
    ) -> Result<(), DisplayError> {
        if handle != DISPLAY_HANDLE || self.owner != Some(owner) {
            return Err(DisplayError::InvalidHandle);
        }
        let frame = self.frame.as_mut().ok_or(DisplayError::InvalidHandle)?;
        let row_bytes = frame
            .width
            .checked_mul(4)
            .ok_or(DisplayError::InvalidArgument)?;
        if stride != row_bytes || pixels.len() != frame.pixels.len() {
            return Err(DisplayError::InvalidArgument);
        }
        if !resources.charge_cpu(pixels.len() as u64) {
            return Err(DisplayError::ResourceExhausted);
        }
        frame.pixels.copy_from_slice(pixels);
        Ok(())
    }

    pub(crate) fn poll_key(
        &mut self,
        resources: &mut Resources,
        owner: ProcessId,
        handle: u32,
    ) -> Result<Option<KeyEvent>, DisplayError> {
        if handle != DISPLAY_HANDLE || self.owner != Some(owner) {
            return Err(DisplayError::InvalidHandle);
        }
        let event = self.events.pop_front();
        if event.is_some() {
            resources.release_memory(EVENT_BYTES);
        }
        Ok(event)
    }

    pub(crate) fn close(&mut self, owner: ProcessId, handle: u32) -> Result<(), DisplayError> {
        if handle != DISPLAY_HANDLE || self.owner != Some(owner) {
            return Err(DisplayError::InvalidHandle);
        }
        self.owner = None;
        Ok(())
    }

    pub(crate) fn close_owner(&mut self, owner: ProcessId) {
        if self.owner == Some(owner) {
            self.owner = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::Limits;

    #[test]
    fn input_queue_is_bounded_and_releases_consumed_event_memory() {
        let mut display = VirtualDisplay::default();
        let mut resources = Resources::new(Limits::default());
        let event = KeyEvent {
            code: 27,
            pressed: true,
        };
        for _ in 0..MAX_EVENTS {
            display.inject_key(&mut resources, event).unwrap();
        }
        assert_eq!(
            display.inject_key(&mut resources, event),
            Err(DisplayError::QueueFull)
        );
        assert_eq!(resources.memory_mark(), MAX_EVENTS as u64 * EVENT_BYTES);
        display.open(&mut resources, 1, 2, 2, RGBA8888).unwrap();
        assert_eq!(
            display.poll_key(&mut resources, 1, DISPLAY_HANDLE),
            Ok(Some(event))
        );
        assert_eq!(
            resources.memory_mark(),
            MAX_EVENTS as u64 * EVENT_BYTES - EVENT_BYTES + 16
        );
    }

    #[test]
    fn display_rejects_wrong_owner_and_preserves_last_frame_on_close() {
        let mut display = VirtualDisplay::default();
        let mut resources = Resources::new(Limits::default());
        let handle = display.open(&mut resources, 1, 1, 1, RGBA8888).unwrap();
        assert_eq!(
            display.present(&mut resources, 2, handle, &[1, 2, 3, 4], 4),
            Err(DisplayError::InvalidHandle)
        );
        display
            .present(&mut resources, 1, handle, &[1, 2, 3, 4], 4)
            .unwrap();
        display.close(1, handle).unwrap();
        assert_eq!(display.frame().unwrap().pixels, [1, 2, 3, 4]);
        assert_eq!(
            display.poll_key(&mut resources, 1, handle),
            Err(DisplayError::InvalidHandle)
        );
    }
}
