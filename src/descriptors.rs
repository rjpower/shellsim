//! Bounded Unix-style descriptor and pipe primitives for logical processes.
//!
//! Descriptor numbers map to shared open descriptions. Duplication and fork retain the same
//! description, so cursor and endpoint lifetime are shared. Pipes are finite buffers and report
//! blocking instead of allocating without bound. This module owns no host handles.

use std::collections::{BTreeMap, VecDeque};

/// Descriptor number visible to a simulated process.
pub type Fd = i32;
/// Arena identity shared by duplicated descriptor entries.
pub type DescriptionId = u32;
/// Identity of one bounded pipe.
pub type PipeId = u32;

pub const MAX_FDS_PER_PROCESS: usize = 256;
pub const MAX_OPEN_DESCRIPTIONS: usize = 4_096;
pub const MAX_PIPES: usize = 1_024;
pub const DEFAULT_PIPE_CAPACITY: usize = 64 * 1024;
pub const MAX_CAPTURE_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescriptorError {
    InvalidFd,
    WrongAccess,
    DescriptorLimit,
    DescriptionLimit,
    PipeLimit,
    InvalidPipeCapacity,
    BrokenPipe,
    OutputLimit,
    ReferenceOverflow,
}

/// Result of an operation that may need the cooperative scheduler.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IoPoll<T> {
    Ready(T),
    Blocked,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum OpenDescription {
    Input { bytes: Vec<u8>, cursor: usize },
    Capture { bytes: Vec<u8> },
    Null,
    PipeReader(PipeId),
    PipeWriter(PipeId),
}

struct DescriptionEntry {
    description: OpenDescription,
    references: u32,
}

struct Pipe {
    bytes: VecDeque<u8>,
    capacity: usize,
    readers: u32,
    writers: u32,
}

/// Machine-owned descriptions and pipes shared by all process descriptor tables.
pub struct DescriptorArena {
    next_description: DescriptionId,
    next_pipe: PipeId,
    descriptions: BTreeMap<DescriptionId, DescriptionEntry>,
    pipes: BTreeMap<PipeId, Pipe>,
}

impl Default for DescriptorArena {
    fn default() -> Self {
        Self::new()
    }
}

impl DescriptorArena {
    pub fn new() -> Self {
        Self {
            next_description: 1,
            next_pipe: 1,
            descriptions: BTreeMap::new(),
            pipes: BTreeMap::new(),
        }
    }

    pub fn open_input(&mut self, bytes: Vec<u8>) -> Result<DescriptionId, DescriptorError> {
        if bytes.len() > MAX_CAPTURE_BYTES {
            return Err(DescriptorError::OutputLimit);
        }
        self.allocate(OpenDescription::Input { bytes, cursor: 0 })
    }

    pub fn open_capture(&mut self) -> Result<DescriptionId, DescriptorError> {
        self.allocate(OpenDescription::Capture { bytes: Vec::new() })
    }

    pub fn open_null(&mut self) -> Result<DescriptionId, DescriptorError> {
        self.allocate(OpenDescription::Null)
    }

    /// Create the two independently reference-counted descriptions for one pipe.
    pub fn open_pipe(
        &mut self,
        capacity: usize,
    ) -> Result<(DescriptionId, DescriptionId), DescriptorError> {
        if capacity == 0 || capacity > DEFAULT_PIPE_CAPACITY {
            return Err(DescriptorError::InvalidPipeCapacity);
        }
        if self.pipes.len() >= MAX_PIPES {
            return Err(DescriptorError::PipeLimit);
        }
        let pipe = self.next_pipe;
        self.next_pipe = self
            .next_pipe
            .checked_add(1)
            .ok_or(DescriptorError::PipeLimit)?;
        self.pipes.insert(
            pipe,
            Pipe {
                bytes: VecDeque::new(),
                capacity,
                readers: 1,
                writers: 1,
            },
        );
        let reader = match self.allocate(OpenDescription::PipeReader(pipe)) {
            Ok(reader) => reader,
            Err(error) => {
                self.pipes.remove(&pipe);
                return Err(error);
            }
        };
        let writer = match self.allocate(OpenDescription::PipeWriter(pipe)) {
            Ok(writer) => writer,
            Err(error) => {
                self.descriptions.remove(&reader);
                self.pipes.remove(&pipe);
                return Err(error);
            }
        };
        Ok((reader, writer))
    }

    fn allocate(&mut self, description: OpenDescription) -> Result<DescriptionId, DescriptorError> {
        if self.descriptions.len() >= MAX_OPEN_DESCRIPTIONS {
            return Err(DescriptorError::DescriptionLimit);
        }
        let id = self.next_description;
        self.next_description = self
            .next_description
            .checked_add(1)
            .ok_or(DescriptorError::DescriptionLimit)?;
        self.descriptions.insert(
            id,
            DescriptionEntry {
                description,
                references: 0,
            },
        );
        Ok(id)
    }

    fn retain(&mut self, id: DescriptionId) -> Result<(), DescriptorError> {
        let entry = self
            .descriptions
            .get_mut(&id)
            .ok_or(DescriptorError::InvalidFd)?;
        entry.references = entry
            .references
            .checked_add(1)
            .ok_or(DescriptorError::ReferenceOverflow)?;
        Ok(())
    }

    fn release(&mut self, id: DescriptionId) -> Result<(), DescriptorError> {
        let entry = self
            .descriptions
            .get_mut(&id)
            .ok_or(DescriptorError::InvalidFd)?;
        entry.references = entry
            .references
            .checked_sub(1)
            .ok_or(DescriptorError::InvalidFd)?;
        if entry.references != 0 {
            return Ok(());
        }
        let entry = self
            .descriptions
            .remove(&id)
            .ok_or(DescriptorError::InvalidFd)?;
        let pipe = match entry.description {
            OpenDescription::PipeReader(pipe) => {
                if let Some(state) = self.pipes.get_mut(&pipe) {
                    state.readers = state.readers.saturating_sub(1);
                }
                Some(pipe)
            }
            OpenDescription::PipeWriter(pipe) => {
                if let Some(state) = self.pipes.get_mut(&pipe) {
                    state.writers = state.writers.saturating_sub(1);
                }
                Some(pipe)
            }
            _ => None,
        };
        if let Some(pipe) = pipe {
            if self
                .pipes
                .get(&pipe)
                .is_some_and(|state| state.readers == 0 && state.writers == 0)
            {
                self.pipes.remove(&pipe);
            }
        }
        Ok(())
    }

    pub fn read(
        &mut self,
        id: DescriptionId,
        maximum: usize,
    ) -> Result<IoPoll<Vec<u8>>, DescriptorError> {
        let kind = self
            .descriptions
            .get(&id)
            .ok_or(DescriptorError::InvalidFd)?
            .description
            .clone();
        match kind {
            OpenDescription::Input { .. } => {
                let entry = self.descriptions.get_mut(&id).expect("description exists");
                let OpenDescription::Input { bytes, cursor } = &mut entry.description else {
                    unreachable!()
                };
                let end = cursor.saturating_add(maximum).min(bytes.len());
                let result = bytes[*cursor..end].to_vec();
                *cursor = end;
                Ok(IoPoll::Ready(result))
            }
            OpenDescription::Null => Ok(IoPoll::Ready(Vec::new())),
            OpenDescription::PipeReader(pipe) => {
                let pipe = self
                    .pipes
                    .get_mut(&pipe)
                    .ok_or(DescriptorError::InvalidFd)?;
                if pipe.bytes.is_empty() && pipe.writers > 0 {
                    return Ok(IoPoll::Blocked);
                }
                let count = maximum.min(pipe.bytes.len());
                Ok(IoPoll::Ready(pipe.bytes.drain(..count).collect()))
            }
            _ => Err(DescriptorError::WrongAccess),
        }
    }

    pub fn write(
        &mut self,
        id: DescriptionId,
        bytes: &[u8],
    ) -> Result<IoPoll<usize>, DescriptorError> {
        let kind = self
            .descriptions
            .get(&id)
            .ok_or(DescriptorError::InvalidFd)?
            .description
            .clone();
        match kind {
            OpenDescription::Capture { .. } => {
                let entry = self.descriptions.get_mut(&id).expect("description exists");
                let OpenDescription::Capture { bytes: output } = &mut entry.description else {
                    unreachable!()
                };
                if output.len().saturating_add(bytes.len()) > MAX_CAPTURE_BYTES {
                    return Err(DescriptorError::OutputLimit);
                }
                output.extend_from_slice(bytes);
                Ok(IoPoll::Ready(bytes.len()))
            }
            OpenDescription::Null => Ok(IoPoll::Ready(bytes.len())),
            OpenDescription::PipeWriter(pipe) => {
                let pipe = self
                    .pipes
                    .get_mut(&pipe)
                    .ok_or(DescriptorError::InvalidFd)?;
                if pipe.readers == 0 {
                    return Err(DescriptorError::BrokenPipe);
                }
                let available = pipe.capacity.saturating_sub(pipe.bytes.len());
                if available == 0 {
                    return Ok(IoPoll::Blocked);
                }
                let count = available.min(bytes.len());
                pipe.bytes.extend(&bytes[..count]);
                Ok(IoPoll::Ready(count))
            }
            _ => Err(DescriptorError::WrongAccess),
        }
    }

    pub fn capture(&self, id: DescriptionId) -> Result<&[u8], DescriptorError> {
        match &self
            .descriptions
            .get(&id)
            .ok_or(DescriptorError::InvalidFd)?
            .description
        {
            OpenDescription::Capture { bytes } => Ok(bytes),
            _ => Err(DescriptorError::WrongAccess),
        }
    }

    /// Stable synthetic target used by `/proc/PID/fd` without exposing implementation details.
    pub fn label(&self, id: DescriptionId) -> Result<String, DescriptorError> {
        let description = &self
            .descriptions
            .get(&id)
            .ok_or(DescriptorError::InvalidFd)?
            .description;
        Ok(match description {
            OpenDescription::Input { .. } | OpenDescription::Capture { .. } => {
                format!("pipe:[{id}]")
            }
            OpenDescription::Null => "/dev/null".to_string(),
            OpenDescription::PipeReader(pipe) | OpenDescription::PipeWriter(pipe) => {
                format!("pipe:[{pipe}]")
            }
        })
    }
}

/// Per-process mapping from small integers to shared machine descriptions.
#[derive(Clone, Default)]
pub struct FdTable {
    entries: BTreeMap<Fd, DescriptionId>,
}

impl FdTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, fd: Fd) -> Result<DescriptionId, DescriptorError> {
        self.entries
            .get(&fd)
            .copied()
            .ok_or(DescriptorError::InvalidFd)
    }

    pub fn install(
        &mut self,
        fd: Fd,
        description: DescriptionId,
        arena: &mut DescriptorArena,
    ) -> Result<(), DescriptorError> {
        if fd < 0 {
            return Err(DescriptorError::InvalidFd);
        }
        if !self.entries.contains_key(&fd) && self.entries.len() >= MAX_FDS_PER_PROCESS {
            return Err(DescriptorError::DescriptorLimit);
        }
        arena.retain(description)?;
        if let Some(previous) = self.entries.insert(fd, description) {
            arena.release(previous)?;
        }
        Ok(())
    }

    pub fn duplicate(
        &mut self,
        source: Fd,
        destination: Fd,
        arena: &mut DescriptorArena,
    ) -> Result<(), DescriptorError> {
        let description = self.get(source)?;
        self.install(destination, description, arena)
    }

    pub fn close(&mut self, fd: Fd, arena: &mut DescriptorArena) -> Result<(), DescriptorError> {
        let description = self.entries.remove(&fd).ok_or(DescriptorError::InvalidFd)?;
        arena.release(description)
    }

    /// Fork a descriptor map while retaining each shared open description atomically.
    pub fn fork(&self, arena: &mut DescriptorArena) -> Result<Self, DescriptorError> {
        let mut retained = Vec::new();
        for description in self.entries.values().copied() {
            if let Err(error) = arena.retain(description) {
                for retained in retained {
                    let _ = arena.release(retained);
                }
                return Err(error);
            }
            retained.push(description);
        }
        Ok(self.clone())
    }

    pub fn close_all(&mut self, arena: &mut DescriptorArena) {
        let descriptions = std::mem::take(&mut self.entries);
        for description in descriptions.into_values() {
            let _ = arena.release(description);
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (Fd, DescriptionId)> + '_ {
        self.entries
            .iter()
            .map(|(fd, description)| (*fd, *description))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicated_input_descriptors_share_a_cursor() {
        let mut arena = DescriptorArena::new();
        let input = arena.open_input(b"abcdef".to_vec()).unwrap();
        let mut fds = FdTable::new();
        fds.install(0, input, &mut arena).unwrap();
        fds.duplicate(0, 3, &mut arena).unwrap();
        assert_eq!(
            arena.read(fds.get(0).unwrap(), 2).unwrap(),
            IoPoll::Ready(b"ab".to_vec())
        );
        assert_eq!(
            arena.read(fds.get(3).unwrap(), 2).unwrap(),
            IoPoll::Ready(b"cd".to_vec())
        );
    }

    #[test]
    fn bounded_pipe_blocks_and_reaches_eof_after_last_writer_closes() {
        let mut arena = DescriptorArena::new();
        let (reader, writer) = arena.open_pipe(3).unwrap();
        let mut fds = FdTable::new();
        fds.install(0, reader, &mut arena).unwrap();
        fds.install(1, writer, &mut arena).unwrap();
        assert_eq!(arena.read(reader, 8).unwrap(), IoPoll::Blocked);
        assert_eq!(arena.write(writer, b"abcde").unwrap(), IoPoll::Ready(3));
        assert_eq!(arena.write(writer, b"de").unwrap(), IoPoll::Blocked);
        assert_eq!(
            arena.read(reader, 2).unwrap(),
            IoPoll::Ready(b"ab".to_vec())
        );
        assert_eq!(arena.write(writer, b"de").unwrap(), IoPoll::Ready(2));
        fds.close(1, &mut arena).unwrap();
        assert_eq!(
            arena.read(reader, 8).unwrap(),
            IoPoll::Ready(b"cde".to_vec())
        );
        assert_eq!(arena.read(reader, 8).unwrap(), IoPoll::Ready(Vec::new()));
    }

    #[test]
    fn fork_keeps_pipe_endpoints_alive_until_every_copy_closes() {
        let mut arena = DescriptorArena::new();
        let (reader, writer) = arena.open_pipe(8).unwrap();
        let mut parent = FdTable::new();
        parent.install(0, reader, &mut arena).unwrap();
        parent.install(1, writer, &mut arena).unwrap();
        let mut child = parent.fork(&mut arena).unwrap();
        parent.close(1, &mut arena).unwrap();
        assert_eq!(arena.read(reader, 1).unwrap(), IoPoll::Blocked);
        child.close(1, &mut arena).unwrap();
        assert_eq!(arena.read(reader, 1).unwrap(), IoPoll::Ready(Vec::new()));
        parent.close_all(&mut arena);
        child.close_all(&mut arena);
    }
}
