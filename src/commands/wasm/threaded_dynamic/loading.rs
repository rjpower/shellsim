//! Preflight a bounded VFS dependency graph, then allocate before taking the init gate.
//!
//! Only the loading owner calls guest malloc. Every reusable thread slot receives
//! storage before publication, so replay can interrupt an allocator critical section.

use super::{
    layout, replay, reserve_store, Initialization, Preparation, Process, Record, MAX_MODULES,
    MAX_TABLE_ELEMENTS,
};
use crate::commands::wasm::{compiled_command_module, fibers, memory, threads, Host};
use crate::vfs::{resolve_against, NodeKind, Vfs, VfsError, PATH_MAX};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use wasmtime::{AsContextMut, Caller, Error, Module};

const MAX_SOURCE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy)]
enum Reference {
    Existing(u32),
    Prepared(usize),
}

struct Pending {
    path: String,
    sha256: [u8; 32],
    module: Module,
    layout: layout::Layout,
    dependencies: Vec<Reference>,
    store_cost: u64,
}

#[derive(Default)]
struct Graph {
    nodes: Vec<Pending>,
    paths: BTreeMap<String, Reference>,
    visiting: BTreeSet<String>,
}

pub(super) fn reserve_process(caller: &Caller<'_, Host>, bytes: u64) -> Result<(), Error> {
    if !caller.data().machine.get().resources.reserve_memory(bytes) {
        return Err(super::super::exhausted());
    }
    if caller
        .data()
        .retained
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |owned| {
            owned.checked_add(bytes)
        })
        .is_err()
    {
        caller.data().machine.get().resources.release_memory(bytes);
        return Err(Error::msg("threaded process image accounting overflow"));
    }
    Ok(())
}

fn compile(
    caller: &mut Caller<'_, Host>,
    path: &str,
    process: &Process,
) -> Result<(Module, layout::Layout, [u8; 32], u64), Error> {
    process.admit(1)?;
    let length = {
        let machine = caller.data().machine.get();
        let node = machine
            .vfs
            .metadata_ref("/", path, true)
            .map_err(|error| Error::msg(error.to_string()))?;
        let NodeKind::File(bytes) = &node.kind else {
            return Err(Error::msg("threaded library must be a regular VFS file"));
        };
        bytes.len()
    };
    if length > MAX_SOURCE_BYTES {
        return Err(Error::msg("threaded library exceeds source limit"));
    }
    let scratch = (length as u64).saturating_mul(65).saturating_add(4096);
    if !caller
        .data()
        .machine
        .get()
        .resources
        .reserve_memory(scratch)
    {
        return Err(super::super::exhausted());
    }
    let compiled = (|| {
        if !caller
            .data()
            .machine
            .get()
            .resources
            .charge_cpu((length as u64).saturating_mul(10))
        {
            return Err(super::super::exhausted());
        }
        let source = caller
            .data()
            .machine
            .get()
            .vfs
            .read_limited("/", path, MAX_SOURCE_BYTES)
            .map_err(|error| Error::msg(error.to_string()))?;
        threads::reject_raw_waits(&source)?;
        let layout = layout::parse(&source)?;
        let module = compiled_command_module(&source)?;
        if module
            .exports()
            .any(|export| super::super::dynamic::canonical_runtime_symbol(export.name()))
        {
            return Err(Error::msg(
                "threaded side runtime symbols belong to the main executable",
            ));
        }
        if module.resources_required().num_memories != 0
            || module.resources_required().num_tables != 0
        {
            return Err(Error::msg("threaded sides must import memory and tables"));
        }
        let metadata = module
            .imports()
            .count()
            .saturating_add(module.exports().count()) as u64;
        let store_cost = metadata
            .saturating_mul(256)
            .saturating_add((length as u64).saturating_mul(4))
            .saturating_add(4096);
        let image = module.image_range();
        let retained = (image.end as usize).saturating_sub(image.start as usize) as u64;
        reserve_process(
            caller,
            retained
                .saturating_add((length as u64).saturating_mul(8))
                .saturating_add(65536),
        )?;
        Ok((module, layout, Sha256::digest(&source).into(), store_cost))
    })();
    caller
        .data()
        .machine
        .get()
        .resources
        .release_memory(scratch);
    compiled
}

fn append_path(output: &mut String, part: &str) -> Result<(), Error> {
    if part.len() >= PATH_MAX.saturating_sub(output.len()) {
        return Err(Error::msg("dylink runtime path exceeds VFS limit"));
    }
    output.push_str(part);
    Ok(())
}

/// Expand only ORIGIN, checking the VFS limit before allocating each output segment.
fn runtime_directory(directory: &str, origin: &str) -> Result<String, Error> {
    let mut remaining = directory;
    let mut output = String::new();
    while let Some(index) = remaining.find('$') {
        append_path(&mut output, &remaining[..index])?;
        remaining = &remaining[index..];
        remaining = if let Some(suffix) = remaining.strip_prefix("${ORIGIN}") {
            suffix
        } else if let Some(suffix) = remaining.strip_prefix("$ORIGIN") {
            if !suffix.is_empty() && !suffix.starts_with('/') {
                return Err(Error::msg("unsupported dylink runtime path"));
            }
            suffix
        } else {
            return Err(Error::msg("unsupported dylink runtime path"));
        };
        append_path(&mut output, origin)?;
    }
    append_path(&mut output, remaining)?;
    if !output.starts_with('/') {
        return Err(Error::msg("unsupported dylink runtime path"));
    }
    Ok(output)
}

/// Search only the guest VFS; emitted host build paths never grant host access.
/// Relative paths and variables other than ORIGIN are explicit unsupported frontiers.
fn dependency_path(
    vfs: &Vfs,
    importer: &str,
    paths: &[String],
    name: &str,
) -> Result<String, Error> {
    let origin = importer
        .rsplit_once('/')
        .map(|(parent, _)| if parent.is_empty() { "/" } else { parent })
        .unwrap_or("/");
    for directory in paths {
        let directory = runtime_directory(directory, origin)?;
        if directory.len().saturating_add(name.len()).saturating_add(1) >= PATH_MAX {
            return Err(Error::msg("dylink dependency path exceeds VFS limit"));
        }
        let candidate = resolve_against(&directory, name);
        match vfs.metadata_ref("/", &candidate, true) {
            Ok(node) if matches!(node.kind, NodeKind::File(_)) => return Ok(candidate),
            Ok(_) => return Err(Error::msg("threaded dependency must be a regular VFS file")),
            Err(VfsError::NotFound(_) | VfsError::NotADir(_)) => {}
            Err(error) => return Err(Error::msg(error.to_string())),
        }
    }
    Ok(format!("/lib/{name}"))
}

impl Graph {
    fn discover(
        &mut self,
        caller: &mut Caller<'_, Host>,
        path: String,
        process: &Process,
    ) -> Result<Reference, Error> {
        if self.nodes.len() >= MAX_MODULES || self.visiting.len() >= MAX_MODULES {
            return Err(Error::msg("threaded dependency graph exceeds limit"));
        }
        let path = resolve_against(&caller.data().cwd, &path);
        let path = caller
            .data()
            .machine
            .get()
            .vfs
            .realpath(&path, true)
            .map_err(|error| Error::msg(error.to_string()))?;
        if let Some(reference) = self.paths.get(&path) {
            return Ok(*reference);
        }
        if let Some((handle, expected)) = process.known(&path) {
            let length = {
                let machine = caller.data().machine.get();
                let node = machine
                    .vfs
                    .metadata_ref("/", &path, true)
                    .map_err(|error| Error::msg(error.to_string()))?;
                let NodeKind::File(data) = &node.kind else {
                    return Err(Error::msg("threaded library must be a file"));
                };
                data.len()
            };
            if length > MAX_SOURCE_BYTES {
                return Err(Error::msg("threaded library exceeds source limit"));
            }
            let scratch = (length as u64).saturating_add(4096);
            if !caller
                .data()
                .machine
                .get()
                .resources
                .reserve_memory(scratch)
            {
                return Err(super::super::exhausted());
            }
            let result = (|| {
                if !caller
                    .data()
                    .machine
                    .get()
                    .resources
                    .charge_cpu(length as u64)
                {
                    return Err(super::super::exhausted());
                }
                let source = caller
                    .data()
                    .machine
                    .get()
                    .vfs
                    .read_limited("/", &path, MAX_SOURCE_BYTES)
                    .map_err(|error| Error::msg(error.to_string()))?;
                let digest: [u8; 32] = Sha256::digest(source).into();
                if digest != expected {
                    return Err(Error::msg("threaded dynamic source identity changed"));
                }
                Ok(())
            })();
            caller
                .data()
                .machine
                .get()
                .resources
                .release_memory(scratch);
            result?;
            let reference = Reference::Existing(handle);
            self.paths.insert(path, reference);
            return Ok(reference);
        }

        if !self.visiting.insert(path.clone()) {
            return Err(Error::msg("cyclic threaded dependency graph"));
        }
        let (module, layout, digest, store_cost) = compile(caller, &path, process)?;
        if let Some(handle) = process.existing(&path, digest)? {
            self.visiting.remove(&path);
            let reference = Reference::Existing(handle);
            self.paths.insert(path, reference);
            return Ok(reference);
        }
        let mut dependencies = Vec::with_capacity(layout.needed.len());
        for name in &layout.needed {
            // Prepay each bounded path expansion and VFS lookup, including fallback.
            let search_cost = (layout.runtime_paths.len() as u64 + 1)
                .saturating_mul(PATH_MAX as u64)
                .saturating_mul(16);
            if !caller
                .data()
                .machine
                .get()
                .resources
                .charge_cpu(search_cost)
            {
                return Err(super::super::exhausted());
            }
            let dependency = dependency_path(
                &caller.data().machine.get().vfs,
                &path,
                &layout.runtime_paths,
                name,
            )?;
            dependencies.push(self.discover(caller, dependency, process)?);
        }
        self.visiting.remove(&path);
        let index = self.nodes.len();
        self.nodes.push(Pending {
            path: path.clone(),
            sha256: digest,
            module,
            layout,
            dependencies,
            store_cost,
        });
        let reference = Reference::Prepared(index);
        self.paths.insert(path, reference);
        Ok(reference)
    }
}

fn align(value: u32, alignment: u32) -> Result<u32, Error> {
    let mask = alignment
        .checked_sub(1)
        .ok_or_else(|| Error::msg("invalid loader alignment"))?;
    value
        .checked_add(mask)
        .map(|value| value & !mask)
        .ok_or_else(|| Error::msg("threaded allocation overflow"))
}

async fn allocate(
    caller: &mut Caller<'_, Host>,
    pending: Pending,
    first_handle: u32,
    process: &Process,
    global: bool,
) -> Result<Arc<Record>, Error> {
    let data_alignment = 1u32
        .checked_shl(pending.layout.memory_align)
        .ok_or_else(|| Error::msg("invalid data alignment"))?;
    let tls_alignment = pending.layout.tls_align;
    let stride = align(pending.layout.tls_size, tls_alignment)?;
    let total = pending
        .layout
        .memory_size
        .checked_add(data_alignment - 1)
        .and_then(|value| value.checked_add(tls_alignment - 1))
        .and_then(|value| value.checked_add(stride.checked_mul(threads::MAX_THREADS as u32)?))
        .ok_or_else(|| Error::msg("threaded TLS allocation overflow"))?
        .max(1);
    let main = caller
        .data()
        .threaded_dynamic
        .main
        .ok_or_else(|| Error::msg("threaded main unavailable"))?;
    let malloc = main.get_typed_func::<u32, u32>(&mut *caller, "malloc")?;
    let pointer = {
        let _fiber = fibers::begin(&mut *caller)?;
        malloc.call_async(&mut *caller, total).await?
    };
    if pointer == 0 {
        return Err(Error::msg("threaded guest allocation failed"));
    }
    let guest_memory =
        memory(caller).ok_or_else(|| Error::msg("threaded process memory unavailable"))?;
    guest_memory.range(&*caller, pointer as usize, total as usize)?;
    if !caller
        .data()
        .machine
        .get()
        .resources
        .charge_cpu(u64::from(total))
    {
        return Err(super::super::exhausted());
    }
    let zeros = [0u8; 4096];
    for offset in (0..total as usize).step_by(zeros.len()) {
        let length = zeros.len().min(total as usize - offset);
        guest_memory.write(&mut *caller, pointer as usize + offset, &zeros[..length])?;
    }
    let memory_base = align(pointer, data_alignment)?;
    let tls_base = align(
        memory_base
            .checked_add(pending.layout.memory_size)
            .ok_or_else(|| Error::msg("threaded data overflow"))?,
        tls_alignment,
    )?;
    let mut tls = [0; threads::MAX_THREADS];
    for (slot, address) in tls.iter_mut().enumerate() {
        *address = tls_base
            .checked_add(
                stride
                    .checked_mul(slot as u32)
                    .ok_or_else(|| Error::msg("threaded TLS overflow"))?,
            )
            .ok_or_else(|| Error::msg("threaded TLS overflow"))?;
    }
    let table_base =
        process.allocate_table(pending.layout.table_size, pending.layout.table_align)?;
    let mut indices = BTreeMap::new();
    let mut function_slots = BTreeMap::new();
    for (name, index) in &pending.layout.functions {
        let address = if let Some(address) = indices.get(index) {
            *address
        } else if let Some(offset) = pending.layout.element_functions.get(index) {
            let address = table_base
                .checked_add(*offset)
                .ok_or_else(|| Error::msg("table address overflow"))?;
            indices.insert(*index, address);
            address
        } else {
            let address = process.allocate_table(1, 0)?;
            indices.insert(*index, address);
            address
        };
        if address as usize >= MAX_TABLE_ELEMENTS {
            return Err(Error::msg("threaded table limit exceeded"));
        }
        function_slots.insert(name.clone(), address);
    }
    let dependencies = pending
        .dependencies
        .into_iter()
        .map(|reference| match reference {
            Reference::Existing(handle) => handle,
            Reference::Prepared(index) => first_handle + index as u32,
        })
        .collect();
    Ok(Arc::new(Record {
        path: pending.path,
        sha256: pending.sha256,
        module: pending.module,
        memory_base,
        table_base,
        tls,
        dependencies,
        global: AtomicBool::new(global),
        layout: pending.layout,
        function_slots,
        store_cost: pending.store_cost,
    }))
}

pub(super) async fn load(
    caller: &mut Caller<'_, Host>,
    path: String,
    flags: u32,
) -> Result<u32, Error> {
    if flags & !(1 | 2 | 8 | 256 | 4096) != 0
        || flags & 3 == 0
        || flags & 3 == 3
        || flags & (8 | 256) == (8 | 256)
    {
        return Err(Error::msg("unsupported dlopen flags"));
    }
    if !caller.data().threaded_dynamic.ready || caller.data().threaded_dynamic.initializing {
        return Err(Error::msg(
            "threaded loader reentry before readiness is unsupported",
        ));
    }
    let thread = caller.data().thread.as_ref().expect("thread host").clone();
    let process = thread.dynamic_process();
    let machine = caller.data().machine.clone();
    std::future::poll_fn(|_| {
        if !machine.get().resources.charge_cpu(32) {
            return std::task::Poll::Ready(Err(super::super::exhausted()));
        }
        match process.try_prepare(thread.id()) {
            Ok(true) => std::task::Poll::Ready(Ok(())),
            Ok(false) => {
                machine.signals().suspension = Some(super::super::Suspension::Yielded);
                std::task::Poll::Pending
            }
            Err(error) => std::task::Poll::Ready(Err(error)),
        }
    })
    .await?;
    let _preparation = Preparation {
        process: process.clone(),
        tid: thread.id(),
    };
    reserve_store(&mut caller.as_context_mut(), MAX_MODULES as u64 * 512)?;
    let mut graph = Graph::default();
    let root = graph.discover(caller, path, &process)?;
    if let Reference::Existing(handle) = root {
        if flags & 256 != 0 {
            process.promote(handle)?;
        }
        return Ok(handle);
    }
    let first_handle = process.generation().1 as u32 + 1;
    let mut records = Vec::with_capacity(graph.nodes.len());
    for pending in graph.nodes {
        records.push(allocate(caller, pending, first_handle, &process, flags & 256 != 0).await?);
    }
    // Preparation may yield in malloc while another Store reconstructs.
    // Acquire publication only after that Store releases its replay gate.
    std::future::poll_fn(|_| match process.resume_allowed(thread.id()) {
        Ok(true) => std::task::Poll::Ready(Ok(())),
        Ok(false) => {
            if !machine.get().resources.charge_cpu(32) {
                return std::task::Poll::Ready(Err(super::super::exhausted()));
            }
            machine.signals().suspension = Some(super::super::Suspension::Yielded);
            std::task::Poll::Pending
        }
        Err(error) => std::task::Poll::Ready(Err(error)),
    })
    .await?;
    let mut initialization: Initialization = process.begin_admitted(thread.id(), records.len())?;
    initialization.started();
    for record in &records {
        replay::install(caller.as_context_mut(), record.clone(), true).await?;
    }
    let handles = initialization.publish(records)?;
    caller.data_mut().threaded_dynamic.generation = process.generation().0;
    let Reference::Prepared(index) = root else {
        unreachable!()
    };
    Ok(handles[index])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_expansion_is_bounded_before_allocation() {
        assert!(runtime_directory(&"${ORIGIN}".repeat(32), &"/a".repeat(128)).is_err());
        assert!(runtime_directory(&"/a".repeat(PATH_MAX), "/pkg").is_err());
        assert_eq!(runtime_directory("$ORIGIN", "/").unwrap(), "/");
    }

    #[test]
    fn dependency_search_uses_importer_origin_order_and_guest_vfs_only() {
        let mut vfs = Vfs::new();
        for directory in ["/pkg/lib", "/other", "/lib"] {
            vfs.mkdir_all("/", directory).unwrap();
            vfs.write("/", &format!("{directory}/provider.so"), b"wasm", 0o644)
                .unwrap();
        }
        vfs.mkdir_all("/", "/pkg/modules").unwrap();
        vfs.write("/", "/pkg/modules/extension.so", b"wasm", 0o644)
            .unwrap();
        vfs.symlink("/", "/pkg/modules/extension.so", "/alias.so")
            .unwrap();
        // Graph::discover canonicalizes the importer before resolving its dependencies.
        let importer = vfs.realpath("/alias.so", true).unwrap();
        assert_eq!(
            dependency_path(&vfs, &importer, &["$ORIGIN/../lib".into()], "provider.so").unwrap(),
            "/pkg/lib/provider.so"
        );
        let paths = vec![
            "/host/build/absent".into(),
            "$ORIGIN/../lib".into(),
            "/other".into(),
        ];
        assert_eq!(
            dependency_path(&vfs, "/pkg/modules/extension.so", &paths, "provider.so").unwrap(),
            "/pkg/lib/provider.so"
        );
        assert_eq!(
            dependency_path(
                &vfs,
                "/elsewhere/modules/extension.so",
                &paths,
                "provider.so"
            )
            .unwrap(),
            "/other/provider.so"
        );
        assert_eq!(
            dependency_path(
                &vfs,
                "/pkg/extension.so",
                &["/host/build/absent".into()],
                "provider.so"
            )
            .unwrap(),
            "/lib/provider.so"
        );
        assert_eq!(
            dependency_path(
                &vfs,
                "/pkg/modules/extension.so",
                &["${ORIGIN}/../lib".into()],
                "provider.so"
            )
            .unwrap(),
            "/pkg/lib/provider.so"
        );
        for unsupported in ["relative", "$LIB", "$ORIGIN_SUFFIX"] {
            assert!(dependency_path(
                &vfs,
                "/pkg/extension.so",
                &[unsupported.into()],
                "provider.so"
            )
            .is_err());
        }
    }
}
