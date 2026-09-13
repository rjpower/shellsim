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
    Blocked(IoWait),
}

/// Exact pipe condition needed to resume a blocked descriptor operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IoWait {
    InputReadable(DescriptionId),
    PipeReadable(PipeId),
    PipeWritable(PipeId),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum OpenDescription {
    Input {
        bytes: Vec<u8>,
        cursor: usize,
        closed: bool,
    },
    Capture {
        bytes: Vec<u8>,
        delivered: usize,
    },
    Null,
    File {
        path: String,
        cursor: u64,
        readable: bool,
        writable: bool,
        remove_on_first_write_error: bool,
    },
    PipeReader(PipeId),
    PipeWriter(PipeId),
}

#[derive(Clone)]
struct DescriptionEntry {
    description: OpenDescription,
    references: u32,
}

#[derive(Clone)]
struct Pipe {
    bytes: VecDeque<u8>,
    capacity: usize,
    readers: u32,
    writers: u32,
}

/// VFS-facing state stored in one shared file description.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct FileState {
    pub path: String,
    pub cursor: u64,
    pub readable: bool,
    pub writable: bool,
    pub remove_on_first_write_error: bool,
}

/// Machine-owned descriptions and pipes shared by all process descriptor tables.
#[derive(Clone)]
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
        self.allocate(OpenDescription::Input {
            bytes,
            cursor: 0,
            closed: true,
        })
    }

    /// Create a bounded host-fed input channel that initially has no bytes and remains open.
    pub fn open_stream_input(&mut self) -> Result<DescriptionId, DescriptorError> {
        self.allocate(OpenDescription::Input {
            bytes: Vec::new(),
            cursor: 0,
            closed: false,
        })
    }

    /// Append explicitly supplied bytes to a streaming input description.
    pub fn append_input(&mut self, id: DescriptionId, input: &[u8]) -> Result<(), DescriptorError> {
        let entry = self
            .descriptions
            .get_mut(&id)
            .ok_or(DescriptorError::InvalidFd)?;
        let OpenDescription::Input { bytes, closed, .. } = &mut entry.description else {
            return Err(DescriptorError::WrongAccess);
        };
        if *closed {
            return Err(DescriptorError::WrongAccess);
        }
        if bytes.len().saturating_add(input.len()) > MAX_CAPTURE_BYTES {
            return Err(DescriptorError::OutputLimit);
        }
        bytes.extend_from_slice(input);
        Ok(())
    }

    /// Close a streaming input description so a drained reader observes EOF.
    pub fn close_input(&mut self, id: DescriptionId) -> Result<(), DescriptorError> {
        let entry = self
            .descriptions
            .get_mut(&id)
            .ok_or(DescriptorError::InvalidFd)?;
        let OpenDescription::Input { closed, .. } = &mut entry.description else {
            return Err(DescriptorError::WrongAccess);
        };
        *closed = true;
        Ok(())
    }

    pub fn open_capture(&mut self) -> Result<DescriptionId, DescriptorError> {
        self.allocate(OpenDescription::Capture {
            bytes: Vec::new(),
            delivered: 0,
        })
    }

    pub fn open_null(&mut self) -> Result<DescriptionId, DescriptorError> {
        self.allocate(OpenDescription::Null)
    }

    /// Open a VFS path description. The caller performs creation/truncation and all VFS I/O;
    /// this arena owns only shared cursor and access state.
    pub fn open_file(
        &mut self,
        path: String,
        cursor: u64,
        readable: bool,
        writable: bool,
        remove_on_first_write_error: bool,
    ) -> Result<DescriptionId, DescriptorError> {
        self.allocate(OpenDescription::File {
            path,
            cursor,
            readable,
            writable,
            remove_on_first_write_error,
        })
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

    /// Discard a description that could not be installed into a descriptor table.
    ///
    /// Allocation intentionally starts with no references so callers can construct an open
    /// description before selecting its descriptor number. Only that unowned state is removable
    /// through this method; installed descriptions must be released through [`FdTable`].
    pub(crate) fn discard_unreferenced(
        &mut self,
        id: DescriptionId,
    ) -> Result<(), DescriptorError> {
        let entry = self
            .descriptions
            .get(&id)
            .ok_or(DescriptorError::InvalidFd)?;
        if entry.references != 0 {
            return Err(DescriptorError::WrongAccess);
        }
        let description = self
            .descriptions
            .remove(&id)
            .expect("description was checked above")
            .description;
        if let OpenDescription::PipeReader(pipe) | OpenDescription::PipeWriter(pipe) = description {
            self.pipes.remove(&pipe);
        }
        Ok(())
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

    /// Retain an open description for machine-owned continuation state outside an FD table.
    pub(crate) fn retain_handle(&mut self, id: DescriptionId) -> Result<(), DescriptorError> {
        self.retain(id)
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

    /// Release a machine-owned continuation reference created by [`Self::retain_handle`].
    pub(crate) fn release_handle(&mut self, id: DescriptionId) -> Result<(), DescriptorError> {
        self.release(id)
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
                let OpenDescription::Input {
                    bytes,
                    cursor,
                    closed,
                } = &mut entry.description
                else {
                    unreachable!()
                };
                if *cursor == bytes.len() && !*closed {
                    return Ok(IoPoll::Blocked(IoWait::InputReadable(id)));
                }
                let end = cursor.saturating_add(maximum).min(bytes.len());
                let result = bytes[*cursor..end].to_vec();
                *cursor = end;
                Ok(IoPoll::Ready(result))
            }
            OpenDescription::Null => Ok(IoPoll::Ready(Vec::new())),
            OpenDescription::File { .. } => Err(DescriptorError::WrongAccess),
            OpenDescription::PipeReader(pipe) => {
                let state = self
                    .pipes
                    .get_mut(&pipe)
                    .ok_or(DescriptorError::InvalidFd)?;
                if state.bytes.is_empty() && state.writers > 0 {
                    return Ok(IoPoll::Blocked(IoWait::PipeReadable(pipe)));
                }
                let count = maximum.min(state.bytes.len());
                Ok(IoPoll::Ready(state.bytes.drain(..count).collect()))
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
                let OpenDescription::Capture { bytes: output, .. } = &mut entry.description else {
                    unreachable!()
                };
                if output.len().saturating_add(bytes.len()) > MAX_CAPTURE_BYTES {
                    return Err(DescriptorError::OutputLimit);
                }
                output.extend_from_slice(bytes);
                Ok(IoPoll::Ready(bytes.len()))
            }
            OpenDescription::Null => Ok(IoPoll::Ready(bytes.len())),
            OpenDescription::File { .. } => Err(DescriptorError::WrongAccess),
            OpenDescription::PipeWriter(pipe) => {
                let state = self
                    .pipes
                    .get_mut(&pipe)
                    .ok_or(DescriptorError::InvalidFd)?;
                if state.readers == 0 {
                    return Err(DescriptorError::BrokenPipe);
                }
                let available = state.capacity.saturating_sub(state.bytes.len());
                if available == 0 {
                    return Ok(IoPoll::Blocked(IoWait::PipeWritable(pipe)));
                }
                let count = available.min(bytes.len());
                state.bytes.extend(&bytes[..count]);
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
            OpenDescription::Capture { bytes, .. } => Ok(bytes),
            _ => Err(DescriptorError::WrongAccess),
        }
    }

    /// Return capture bytes not previously delivered and advance its delivery cursor.
    pub fn drain_capture(&mut self, id: DescriptionId) -> Result<Vec<u8>, DescriptorError> {
        let entry = self
            .descriptions
            .get_mut(&id)
            .ok_or(DescriptorError::InvalidFd)?;
        let OpenDescription::Capture { bytes, delivered } = &mut entry.description else {
            return Err(DescriptorError::WrongAccess);
        };
        let output = bytes[*delivered..].to_vec();
        *delivered = bytes.len();
        Ok(output)
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
            OpenDescription::File { path, .. } => path.clone(),
            OpenDescription::PipeReader(pipe) | OpenDescription::PipeWriter(pipe) => {
                format!("pipe:[{pipe}]")
            }
        })
    }

    /// Return the VFS-specific state for a file description.
    pub(crate) fn file_state(
        &self,
        id: DescriptionId,
    ) -> Result<Option<FileState>, DescriptorError> {
        let description = &self
            .descriptions
            .get(&id)
            .ok_or(DescriptorError::InvalidFd)?
            .description;
        Ok(match description {
            OpenDescription::File {
                path,
                cursor,
                readable,
                writable,
                remove_on_first_write_error,
            } => Some(FileState {
                path: path.clone(),
                cursor: *cursor,
                readable: *readable,
                writable: *writable,
                remove_on_first_write_error: *remove_on_first_write_error,
            }),
            _ => None,
        })
    }

    /// Advance one shared file cursor after successful VFS I/O.
    pub(crate) fn advance_file(
        &mut self,
        id: DescriptionId,
        bytes: usize,
    ) -> Result<(), DescriptorError> {
        let entry = self
            .descriptions
            .get_mut(&id)
            .ok_or(DescriptorError::InvalidFd)?;
        let OpenDescription::File { cursor, .. } = &mut entry.description else {
            return Err(DescriptorError::WrongAccess);
        };
        *cursor = cursor
            .checked_add(u64::try_from(bytes).map_err(|_| DescriptorError::OutputLimit)?)
            .ok_or(DescriptorError::OutputLimit)?;
        Ok(())
    }

    /// Return the pipe and endpoint direction for scheduler readiness notifications.
    pub(crate) fn pipe_endpoint(
        &self,
        id: DescriptionId,
    ) -> Result<Option<(PipeId, bool)>, DescriptorError> {
        let description = &self
            .descriptions
            .get(&id)
            .ok_or(DescriptorError::InvalidFd)?
            .description;
        Ok(match description {
            OpenDescription::PipeReader(pipe) => Some((*pipe, true)),
            OpenDescription::PipeWriter(pipe) => Some((*pipe, false)),
            _ => None,
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
    fn streaming_input_blocks_until_append_and_reaches_eof_after_close() {
        let mut arena = DescriptorArena::new();
        let input = arena.open_stream_input().unwrap();
        arena.retain_handle(input).unwrap();
        assert_eq!(
            arena.read(input, 8).unwrap(),
            IoPoll::Blocked(IoWait::InputReadable(input))
        );
        arena.append_input(input, b"abc").unwrap();
        assert_eq!(arena.read(input, 2).unwrap(), IoPoll::Ready(b"ab".to_vec()));
        assert_eq!(arena.read(input, 8).unwrap(), IoPoll::Ready(b"c".to_vec()));
        arena.close_input(input).unwrap();
        assert_eq!(arena.read(input, 8).unwrap(), IoPoll::Ready(Vec::new()));
        arena.release_handle(input).unwrap();

        let oversized = arena.open_stream_input().unwrap();
        assert_eq!(
            arena.append_input(oversized, &vec![0; MAX_CAPTURE_BYTES + 1]),
            Err(DescriptorError::OutputLimit)
        );
        arena.discard_unreferenced(oversized).unwrap();
    }

    #[test]
    fn bounded_pipe_blocks_and_reaches_eof_after_last_writer_closes() {
        let mut arena = DescriptorArena::new();
        let (reader, writer) = arena.open_pipe(3).unwrap();
        let mut fds = FdTable::new();
        fds.install(0, reader, &mut arena).unwrap();
        fds.install(1, writer, &mut arena).unwrap();
        assert_eq!(
            arena.read(reader, 8).unwrap(),
            IoPoll::Blocked(IoWait::PipeReadable(1))
        );
        assert_eq!(arena.write(writer, b"abcde").unwrap(), IoPoll::Ready(3));
        assert_eq!(
            arena.write(writer, b"de").unwrap(),
            IoPoll::Blocked(IoWait::PipeWritable(1))
        );
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
        assert_eq!(
            arena.read(reader, 1).unwrap(),
            IoPoll::Blocked(IoWait::PipeReadable(1))
        );
        child.close(1, &mut arena).unwrap();
        assert_eq!(arena.read(reader, 1).unwrap(), IoPoll::Ready(Vec::new()));
        parent.close_all(&mut arena);
        child.close_all(&mut arena);
    }

    #[test]
    fn descriptor_limit_rejects_growth_without_leaking_the_new_description() {
        let mut arena = DescriptorArena::new();
        let mut fds = FdTable::new();
        for fd in 0..MAX_FDS_PER_PROCESS as Fd {
            let description = arena.open_null().unwrap();
            fds.install(fd, description, &mut arena).unwrap();
        }
        let rejected = arena.open_null().unwrap();
        assert_eq!(
            fds.install(MAX_FDS_PER_PROCESS as Fd, rejected, &mut arena),
            Err(DescriptorError::DescriptorLimit)
        );
        arena.discard_unreferenced(rejected).unwrap();
        assert_eq!(arena.descriptions.len(), MAX_FDS_PER_PROCESS);
    }

    #[test]
    fn duplicated_file_descriptors_share_the_open_cursor() {
        let mut arena = DescriptorArena::new();
        let file = arena
            .open_file("/work/value".into(), 3, true, true, false)
            .unwrap();
        let mut fds = FdTable::new();
        fds.install(4, file, &mut arena).unwrap();
        fds.duplicate(4, 5, &mut arena).unwrap();
        arena.advance_file(fds.get(4).unwrap(), 2).unwrap();
        assert_eq!(
            arena.file_state(fds.get(5).unwrap()).unwrap(),
            Some(FileState {
                path: "/work/value".into(),
                cursor: 5,
                readable: true,
                writable: true,
                remove_on_first_write_error: false,
            })
        );
    }
}
