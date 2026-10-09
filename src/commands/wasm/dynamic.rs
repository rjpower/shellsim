//! LLVM wasm32 `dylink.0` loading within one process Store.
//!
//! Libraries come only from the VFS and share the executable's memory, C stack and function
//! table. Guest malloc owns their data regions for the process lifetime. The versioned bridge
//! and exact ABI marker make this an explicit opt-in profile, rather than a host dlopen fallback.
//! V2 loads declared dependencies only from `/lib` in the VFS, with bounded acyclic graphs.
//! TLS, unloading and guest-initiated nested loads remain unsupported. Side exception tags
//! are imported from the main runtime. Compilation scratch is charged before cache lookup;
//! retained images and table growth are released with the owning process.

use super::{
    build_linker, command_engine, compiled_command_module, ffi, fibers, memory, Host,
    MAX_WASM_BYTES,
};
use crate::vfs::resolve_against;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use wasmtime::{
    Caller, Error, Extern, ExternType, Global, GlobalType, Instance, Linker, Module, Mutability,
    Ref, Val, ValType,
};

pub(super) const NAMESPACE: &str = "shellsim_dylink_v1";
pub(super) const NAMESPACE_V2: &str = "shellsim_dylink_v2";
pub(super) const MEMORY_RESERVATION: u64 = 32 * 1024 * 1024;
const ABI_SECTION: &str = "shellsim.abi";
const ABI: &[u8] = b"shellsim-wasi-sdk24-cpython3137-v1";
const ABI_V2: &[u8] = b"shellsim-wasi-sdk34-cpython3137-v2";
pub(super) const MAX_LOADS: usize = 32;
const MAX_LIBRARY_BYTES: usize = 4 * 1024 * 1024;
const MAX_METADATA_BYTES: usize = 1024 * 1024;
const MAX_STRING_BYTES: usize = 4096;

/// Exact compiler/runtime contracts; a process never mixes their libraries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Abi {
    V1,
    V2,
}

impl Abi {
    pub(super) fn namespace(self) -> &'static str {
        match self {
            Self::V1 => NAMESPACE,
            Self::V2 => NAMESPACE_V2,
        }
    }

    pub(super) fn from_namespace(namespace: &str) -> Option<Self> {
        match namespace {
            NAMESPACE => Some(Self::V1),
            NAMESPACE_V2 => Some(Self::V2),
            _ => None,
        }
    }

    pub(super) fn fixed_reservation(self) -> u64 {
        match self {
            Self::V1 => MEMORY_RESERVATION,
            Self::V2 => 0,
        }
    }

    fn marker(self) -> &'static [u8] {
        match self {
            Self::V1 => ABI,
            Self::V2 => ABI_V2,
        }
    }

    fn library_limit(self) -> usize {
        match self {
            Self::V1 => MAX_LIBRARY_BYTES,
            Self::V2 => 16 * 1024 * 1024,
        }
    }
}

#[derive(Default)]
pub(super) struct Dynamic {
    pub(super) main: Option<Instance>,
    pub(super) shared_memory: Option<wasmtime::Memory>,
    abi: Option<Abi>,
    retained: Arc<AtomicU64>,
    libraries: Vec<Library>,
    paths: BTreeMap<String, u32>,
    attempts: usize,
    loading: bool,
    pending: BTreeSet<String>,
    failed: bool,
    error: Option<String>,
    pub(super) reserved: u64,
}

struct Library {
    instance: Instance,
    memory_base: u32,
    global: bool,
    dependencies: Vec<u32>,
}

impl Dynamic {
    pub(super) fn new(abi: Option<Abi>, retained: Arc<AtomicU64>) -> Self {
        Self {
            abi,
            retained,
            ..Default::default()
        }
    }
}

#[derive(Default, Debug)]
struct Layout {
    memory_size: u32,
    memory_align: u32,
    table_size: u32,
    table_align: u32,
    needed: Vec<String>,
}

/// A bounded binary cursor: section lengths and LEB values are checked before slicing.
struct Cursor<'a>(&'a [u8]);

impl<'a> Cursor<'a> {
    fn byte(&mut self) -> Result<u8, Error> {
        let value = *self
            .0
            .first()
            .ok_or_else(|| Error::msg("truncated dylink metadata"))?;
        self.0 = &self.0[1..];
        Ok(value)
    }

    fn number(&mut self) -> Result<u32, Error> {
        let mut value = 0;
        for shift in (0..35).step_by(7) {
            let byte = self.byte()?;
            if shift == 28 && byte > 15 {
                return Err(Error::msg("overflow in dylink metadata"));
            }
            value |= u32::from(byte & 127) << shift;
            if byte & 128 == 0 {
                return Ok(value);
            }
        }
        Err(Error::msg("overflow in dylink metadata"))
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], Error> {
        let data = self
            .0
            .get(..len)
            .ok_or_else(|| Error::msg("truncated dylink metadata"))?;
        self.0 = &self.0[len..];
        Ok(data)
    }

    fn string(&mut self) -> Result<&'a str, Error> {
        let len = self.number()? as usize;
        if len > MAX_STRING_BYTES {
            return Err(Error::msg("dylink string exceeds limit"));
        }
        Ok(std::str::from_utf8(self.take(len)?)?)
    }
}

fn sections(source: &[u8], abi: Abi) -> Result<Vec<(&str, &[u8])>, Error> {
    if source.get(..8) != Some(b"\0asm\x01\0\0\0") {
        return Err(Error::msg("expected wasm32 core module"));
    }
    let mut cursor = Cursor(&source[8..]);
    let mut sections = Vec::new();
    while !cursor.0.is_empty() {
        let kind = cursor.byte()?;
        let len = cursor.number()? as usize;
        let mut payload = Cursor(cursor.take(len)?);
        if kind == 8 && abi == Abi::V1 {
            return Err(Error::msg(
                "module start sections are unsupported by the dynamic ABI",
            ));
        }
        if kind == 0 {
            let name = payload.string()?;
            if matches!(name, "dylink.0" | ABI_SECTION) {
                if payload.0.len() > MAX_METADATA_BYTES
                    || sections.iter().any(|(old, _)| *old == name)
                {
                    return Err(Error::msg("duplicate or oversized dylink metadata"));
                }
                sections.push((name, payload.0));
            }
        }
    }
    Ok(sections)
}

pub(super) fn compatible_main(source: &[u8], abi: Abi) -> Result<(), Error> {
    if sections(source, abi)?
        .iter()
        .any(|(name, data)| *name == ABI_SECTION && *data == abi.marker())
    {
        return Ok(());
    }
    Err(Error::msg(format!(
        "dynamic loading ABI mismatch: expected {}",
        std::str::from_utf8(abi.marker()).expect("ASCII ABI marker")
    )))
}

fn layout(source: &[u8], abi: Abi) -> Result<Layout, Error> {
    compatible_main(source, abi)?;
    if abi == Abi::V2 {
        // A stripped private tag has no reliable symbol name. This initial ABI therefore
        // requires every side-library exception tag to be imported from the main runtime.
        let mut cursor = Cursor(&source[8..]);
        while !cursor.0.is_empty() {
            let kind = cursor.byte()?;
            let len = cursor.number()? as usize;
            let payload = cursor.take(len)?;
            if kind == 13 && Cursor(payload).number()? != 0 {
                return Err(Error::msg(
                    "dynamic ABI v2 requires side-library exception tags to be imported",
                ));
            }
        }
    }
    let sections = sections(source, abi)?;
    let data = sections
        .iter()
        .find(|(name, _)| *name == "dylink.0")
        .ok_or_else(|| Error::msg("library is missing dylink.0"))?
        .1;
    let mut cursor = Cursor(data);
    let mut result = None;
    let mut needed = Vec::new();
    while !cursor.0.is_empty() {
        let kind = cursor.byte()?;
        let len = cursor.number()? as usize;
        let mut payload = Cursor(cursor.take(len)?);
        match kind {
            1 => {
                if result.is_some() {
                    return Err(Error::msg("duplicate dylink memory layout"));
                }
                result = Some(Layout {
                    memory_size: payload.number()?,
                    memory_align: payload.number()?,
                    table_size: payload.number()?,
                    table_align: payload.number()?,
                    needed: Vec::new(),
                });
            }
            2 => {
                let count = payload.number()? as usize;
                if count > MAX_LOADS || (abi == Abi::V1 && count != 0) {
                    return Err(Error::msg(
                        "dylink needed libraries are unsupported or exceed limit",
                    ));
                }
                for _ in 0..count {
                    let name = payload.string()?;
                    if name.is_empty()
                        || name.contains('/')
                        || name.contains('\\')
                        || name.contains('\0')
                        || matches!(name, "." | "..")
                        || needed.iter().any(|old| old == name)
                    {
                        return Err(Error::msg(
                            "invalid or duplicate dylink dependency filename",
                        ));
                    }
                    needed.push(name.to_owned());
                }
            }
            3 | 4 if payload.number()? == 0 => {}
            3 | 4 => return Err(Error::msg("dylink symbol flags and TLS are unsupported")),
            _ => return Err(Error::msg("unsupported dylink metadata subsection")),
        }
        if !payload.0.is_empty() {
            return Err(Error::msg("invalid dylink metadata length"));
        }
    }
    let mut result = result.ok_or_else(|| Error::msg("library is missing dylink memory layout"))?;
    result.needed = needed;
    if result.memory_align > 16
        || result.table_align > 14
        || result.memory_size as usize > super::MAX_WASM_MEMORY
        || result.table_size as usize > super::LARGE_TABLE_ELEMENTS
    {
        return Err(Error::msg("dylink memory or table layout exceeds limit"));
    }
    Ok(result)
}

fn align(value: u32, exponent: u32) -> Result<u32, Error> {
    let mask = (1u32
        .checked_shl(exponent)
        .ok_or_else(|| Error::msg("invalid dylink alignment"))?)
    .saturating_sub(1);
    value
        .checked_add(mask)
        .map(|v| v & !mask)
        .ok_or_else(|| Error::msg("dylink address overflow"))
}

pub(super) fn reserve(caller: &mut Caller<'_, Host>, bytes: u64) -> Result<(), Error> {
    let total = caller
        .data()
        .dynamic
        .reserved
        .checked_add(bytes)
        .ok_or_else(|| Error::msg("dynamic loading memory accounting overflow"))?;
    match caller.data().dynamic.abi {
        Some(Abi::V1) if total > MEMORY_RESERVATION => {
            return Err(Error::msg("dynamic loading memory budget exhausted"))
        }
        Some(Abi::V2) => {
            if !caller.data().machine.get().resources.reserve_memory(bytes) {
                return Err(Error::msg("dynamic loading memory budget exhausted"));
            }
            caller
                .data()
                .dynamic
                .retained
                .fetch_add(bytes, Ordering::Relaxed);
        }
        _ => {}
    }
    caller.data_mut().dynamic.reserved = total;
    Ok(())
}

/// Undo a v2 reservation only when the operation it preflighted made no Store change.
pub(super) fn unreserve(caller: &mut Caller<'_, Host>, bytes: u64) {
    debug_assert_eq!(caller.data().dynamic.abi, Some(Abi::V2));
    caller.data_mut().dynamic.reserved -= bytes;
    caller
        .data()
        .dynamic
        .retained
        .fetch_sub(bytes, Ordering::Relaxed);
    caller.data().machine.get().resources.release_memory(bytes);
}

fn reserve_table(caller: &mut Caller<'_, Host>, entries: u64) -> Result<(), Error> {
    // V2 charges all table growth, including guest table.grow, in the Store limiter.
    if caller.data().dynamic.abi == Some(Abi::V2) {
        return Ok(());
    }
    reserve(caller, entries.saturating_mul(16))
}

fn canonical_runtime_symbol(name: &str) -> bool {
    name.starts_with("__cxa_")
        || name.starts_with("_Unwind_")
        || matches!(
            name,
            "__cpp_exception" | "__c_longjmp" | "__wasm_lpad_context"
        )
}

fn import_symbol(
    caller: &mut Caller<'_, Host>,
    name: &str,
    tag: bool,
    dependencies: &[u32],
) -> Option<(Extern, u32)> {
    if caller.data().dynamic.abi == Some(Abi::V2) && (tag || canonical_runtime_symbol(name)) {
        let main = caller.data().dynamic.main?;
        return main
            .get_export(&mut *caller, name)
            .map(|export| (export, 0));
    }
    lookup(caller, name).or_else(|| lookup_dependencies(caller, name, dependencies))
}

fn compile_library(
    caller: &mut Caller<'_, Host>,
    path: &str,
    bytes: usize,
    abi: Abi,
) -> Result<(Module, Layout), Error> {
    let bound = (bytes as u64).saturating_mul(65).saturating_add(4096);
    if abi == Abi::V1 {
        reserve(caller, bound)?;
    } else if !caller.data().machine.get().resources.reserve_memory(bound) {
        return Err(Error::msg("dynamic compilation memory budget exhausted"));
    }
    // The same temporary bound and work charge apply on cache hits. Release the compilation
    // scratch before retaining the resulting image, including on malformed-source failures.
    let result = (|| {
        if !caller
            .data()
            .machine
            .get()
            .resources
            .charge_cpu((bytes as u64).saturating_mul(10))
        {
            return Err(Error::msg("dynamic compilation budget exhausted"));
        }
        let source = caller
            .data()
            .machine
            .get()
            .vfs
            .read_limited("/", path, abi.library_limit().min(MAX_WASM_BYTES))
            .map_err(|error| Error::msg(error.to_string()))?;
        let layout = layout(&source, abi)?;
        let module = compiled_command_module(&source)?;
        Ok((module, layout))
    })();
    if abi == Abi::V2 {
        caller.data().machine.get().resources.release_memory(bound);
    }
    let (module, layout) = result?;
    if abi == Abi::V2 {
        let image = module.image_range();
        let image_bytes = (image.end as usize).saturating_sub(image.start as usize) as u64;
        // The image contains compiled code and serialized module data. Allow another source
        // length for instance/import metadata beyond the retained source and image itself.
        reserve(
            caller,
            image_bytes
                .saturating_add((bytes as u64).saturating_mul(2))
                .saturating_add(4096),
        )?;
    }
    Ok((module, layout))
}

fn guest_string(caller: &mut Caller<'_, Host>, pointer: u32, len: u32) -> Result<String, Error> {
    if len == 0 || len as usize > MAX_STRING_BYTES {
        return Err(Error::msg("invalid dynamic loading string length"));
    }
    let mut bytes = vec![0; len as usize];
    memory(caller)
        .ok_or_else(|| Error::msg("dynamic loading requires shared memory"))?
        .read(&*caller, pointer as usize, &mut bytes)?;
    if bytes.contains(&0) {
        return Err(Error::msg("dynamic loading string contains NUL"));
    }
    Ok(String::from_utf8(bytes)?)
}

fn lookup(caller: &mut Caller<'_, Host>, name: &str) -> Option<(Extern, u32)> {
    let main = caller.data().dynamic.main?;
    if let Some(export) = main.get_export(&mut *caller, name) {
        return Some((export, 0));
    }
    for index in 0..caller.data().dynamic.libraries.len() {
        let library = &caller.data().dynamic.libraries[index];
        if !library.global {
            continue;
        }
        let (instance, base) = (library.instance, library.memory_base);
        if let Some(export) = instance.get_export(&mut *caller, name) {
            return Some((export, base));
        }
    }
    None
}

fn lookup_dependencies(
    caller: &mut Caller<'_, Host>,
    name: &str,
    handles: &[u32],
) -> Option<(Extern, u32)> {
    let mut pending = handles.to_vec();
    let mut visited = BTreeSet::new();
    // At most MAX_LOADS live handles exist. The visited set bounds diamond graph traversal.
    for index in 0..MAX_LOADS {
        let handle = *pending.get(index)?;
        if !visited.insert(handle) {
            continue;
        }
        let library = caller.data().dynamic.libraries.get(handle as usize - 1)?;
        let (instance, base) = (library.instance, library.memory_base);
        let children = library.dependencies.clone();
        if let Some(export) = instance.get_export(&mut *caller, name) {
            return Some((export, base));
        }
        for child in children {
            if !pending.contains(&child) {
                pending.push(child);
            }
        }
    }
    None
}

fn address(caller: &mut Caller<'_, Host>, symbol: Extern, base: u32) -> Result<u32, Error> {
    match symbol {
        Extern::Global(global) => {
            let value = global
                .get(&mut *caller)
                .i32()
                .ok_or_else(|| Error::msg("dynamic symbol is not wasm32 data"))?
                as u32;
            // SDK34's start initializer relocates mutable exported address globals. Older
            // immutable exports still contain offsets from the side library's memory base.
            let base = if caller.data().dynamic.abi == Some(Abi::V2)
                && global.ty(&*caller).mutability() == Mutability::Var
            {
                0
            } else {
                base
            };
            base.checked_add(value)
                .ok_or_else(|| Error::msg("dynamic symbol address overflow"))
        }
        Extern::Func(function) => {
            let main = caller
                .data()
                .dynamic
                .main
                .ok_or_else(|| Error::msg("dynamic main unavailable"))?;
            let table = main
                .get_table(&mut *caller, "__indirect_function_table")
                .ok_or_else(|| Error::msg("dynamic loading requires exported function table"))?;
            let size = table.size(&*caller);
            if !caller.data().machine.get().resources.charge_cpu(size) {
                return Err(Error::msg("dynamic symbol budget exhausted"));
            }
            let target = function.to_raw(&mut *caller);
            for index in 0..size {
                if let Some(Ref::Func(Some(candidate))) = table.get(&mut *caller, index) {
                    if candidate.to_raw(&mut *caller) == target {
                        return Ok(index as u32);
                    }
                }
            }
            reserve_table(caller, 1)?;
            Ok(table.grow(&mut *caller, 1, Ref::Func(Some(function)))? as u32)
        }
        _ => Err(Error::msg("unsupported dynamic symbol kind")),
    }
}

async fn load(caller: &mut Caller<'_, Host>, path: String, flags: u32) -> Result<u32, Error> {
    let path = resolve_against(&caller.data().cwd, &path);
    let path = if caller.data().dynamic.abi == Some(Abi::V2) {
        caller
            .data()
            .machine
            .get()
            .vfs
            .realpath(&path, true)
            .map_err(|error| Error::msg(error.to_string()))?
    } else {
        path
    };
    if !caller.data_mut().dynamic.pending.insert(path.clone()) {
        return Err(Error::msg("cyclic dylink dependency graph"));
    }
    let result = load_inner(caller, path.clone(), flags).await;
    caller.data_mut().dynamic.pending.remove(&path);
    result
}

async fn load_inner(caller: &mut Caller<'_, Host>, path: String, flags: u32) -> Result<u32, Error> {
    let abi = caller
        .data()
        .dynamic
        .abi
        .ok_or_else(|| Error::msg("dynamic loading ABI is unavailable"))?;
    if flags & !(1 | 2 | 8 | 256 | 4096) != 0
        || flags & 3 == 0
        || flags & 3 == 3
        || flags & (8 | 256) == (8 | 256)
    {
        return Err(Error::msg("unsupported dlopen flags"));
    }
    let path = resolve_against(&caller.data().cwd, &path);
    if let Some(handle) = caller.data().dynamic.paths.get(&path).copied() {
        if flags & 256 != 0 {
            caller.data_mut().dynamic.libraries[handle as usize - 1].global = true;
        }
        return Ok(handle);
    }
    if caller.data().dynamic.attempts >= MAX_LOADS {
        return Err(Error::msg("dynamic library count exceeds limit"));
    }
    caller.data_mut().dynamic.attempts += 1;
    let bytes = {
        let machine = caller.data().machine.get();
        let node = machine
            .vfs
            .metadata_ref("/", &path, true)
            .map_err(|error| Error::msg(error.to_string()))?;
        match &node.kind {
            crate::vfs::NodeKind::File(data) => data.len(),
            _ => return Err(Error::msg("dynamic library must be a regular VFS file")),
        }
    };
    if bytes > abi.library_limit() {
        return Err(Error::msg("dynamic library exceeds size limit"));
    }
    let (module, layout) = compile_library(caller, &path, bytes, abi)?;
    if abi == Abi::V2
        && module.exports().any(|export| {
            canonical_runtime_symbol(export.name()) && !matches!(export.ty(), ExternType::Tag(_))
        })
    {
        return Err(Error::msg(
            "dynamic ABI v2 runtime symbols must be owned by the main executable",
        ));
    }
    let mut dependencies = Vec::new();
    for name in &layout.needed {
        let dependency = Box::pin(load(caller, format!("/lib/{name}"), 2)).await?;
        dependencies.push(dependency);
    }
    let required = module.resources_required();
    if required.num_memories != 0 || required.num_tables != 0 {
        return Err(Error::msg(
            "dynamic libraries must import their memory and tables",
        ));
    }
    let main = caller
        .data()
        .dynamic
        .main
        .ok_or_else(|| Error::msg("dynamic main unavailable"))?;
    let shared_memory = caller
        .data()
        .dynamic
        .shared_memory
        .ok_or_else(|| Error::msg("dynamic loading requires exported memory"))?;
    if shared_memory.ty(&*caller).is_64() || shared_memory.ty(&*caller).is_shared() {
        return Err(Error::msg("dynamic ABI requires unshared wasm32 memory"));
    }
    let table = main
        .get_table(&mut *caller, "__indirect_function_table")
        .ok_or_else(|| Error::msg("dynamic loading requires exported function table"))?;
    let stack = main
        .get_global(&mut *caller, "__stack_pointer")
        .ok_or_else(|| Error::msg("dynamic loading requires exported C stack pointer"))?;
    if stack.ty(&*caller).mutability() != Mutability::Var || stack.get(&mut *caller).i32().is_none()
    {
        return Err(Error::msg(
            "dynamic ABI requires a mutable wasm32 stack pointer",
        ));
    }
    if table.size(&*caller) == 0 {
        reserve_table(caller, 1)?;
        table.grow(&mut *caller, 1, Ref::Func(None))?;
    }
    if !matches!(table.get(&mut *caller, 0), Some(Ref::Func(None))) {
        return Err(Error::msg(
            "dynamic ABI requires a null function pointer at table index zero",
        ));
    }
    // malloc prevents late loading from overwriting a live interpreter's heap or stack.
    let allocation = layout
        .memory_size
        .checked_add((1u32 << layout.memory_align).saturating_sub(1))
        .ok_or_else(|| Error::msg("dylink allocation overflow"))?
        .max(1);
    let malloc = main.get_typed_func::<u32, u32>(&mut *caller, "malloc")?;
    let raw_base = {
        let _fiber = fibers::begin(&mut *caller)?;
        malloc.call_async(&mut *caller, allocation).await?
    };
    if raw_base == 0 {
        return Err(Error::msg("dynamic library data allocation failed"));
    }
    let memory_base = align(raw_base, layout.memory_align)?;
    let memory_end = memory_base
        .checked_add(layout.memory_size)
        .ok_or_else(|| Error::msg("dylink memory overflow"))? as usize;
    if memory_end > shared_memory.data_size(&*caller) {
        return Err(Error::msg("dynamic library data exceeds shared memory"));
    }
    if !caller
        .data()
        .machine
        .get()
        .resources
        .charge_cpu(layout.memory_size as u64)
    {
        return Err(Error::msg("dynamic initialization budget exhausted"));
    }
    shared_memory.data_mut(&mut *caller)[memory_base as usize..memory_end].fill(0);
    let table_base = align(u32::try_from(table.size(&*caller))?, layout.table_align)?;
    let table_end = table_base
        .checked_add(layout.table_size)
        .ok_or_else(|| Error::msg("dylink table overflow"))?;
    let growth = u64::from(table_end).saturating_sub(table.size(&*caller));
    reserve_table(caller, growth)?;
    table.grow(&mut *caller, growth, Ref::Func(None))?;
    let mut linker = build_linker(command_engine());
    let mut got = Vec::new();
    let mut imports_memory = false;
    for import in module.imports() {
        let namespace = import.module();
        let name = import.name();
        let value = match (namespace, name) {
            ("wasi_snapshot_preview1", _) => continue,
            (namespace, name)
                if abi == Abi::V2 && namespace == ffi::NAMESPACE && ffi::supported(name) =>
            {
                continue;
            }
            ("env", "memory") => {
                imports_memory = true;
                Extern::Memory(shared_memory)
            }
            ("env", "__indirect_function_table") => Extern::Table(table),
            ("env", "__stack_pointer") => Extern::Global(stack),
            ("env", "__memory_base") => Extern::Global(Global::new(
                &mut *caller,
                GlobalType::new(ValType::I32, Mutability::Const),
                Val::I32(memory_base as i32),
            )?),
            ("env", "__table_base") => Extern::Global(Global::new(
                &mut *caller,
                GlobalType::new(ValType::I32, Mutability::Const),
                Val::I32(table_base as i32),
            )?),
            ("GOT.mem" | "GOT.func", _) => {
                let initial = if abi == Abi::V2 {
                    let symbol =
                        import_symbol(caller, name, false, &dependencies).ok_or_else(|| {
                            Error::msg(format!("unresolved pre-start dynamic symbol: {name}"))
                        })?;
                    if (namespace == "GOT.mem" && !matches!(symbol.0, Extern::Global(_)))
                        || (namespace == "GOT.func" && !matches!(symbol.0, Extern::Func(_)))
                    {
                        return Err(Error::msg(format!("dynamic symbol kind mismatch: {name}")));
                    }
                    address(caller, symbol.0, symbol.1)?
                } else {
                    0
                };
                let global = Global::new(
                    &mut *caller,
                    GlobalType::new(ValType::I32, Mutability::Var),
                    Val::I32(initial as i32),
                )?;
                got.push((namespace.to_owned(), name.to_owned(), global));
                Extern::Global(global)
            }
            ("env", _) => {
                import_symbol(
                    caller,
                    name,
                    matches!(import.ty(), ExternType::Tag(_)),
                    &dependencies,
                )
                .ok_or_else(|| Error::msg(format!("missing dynamic symbol: {name}")))?
                .0
            }
            _ => {
                return Err(Error::msg(format!(
                    "unsupported dynamic import: {namespace}.{name}"
                )))
            }
        };
        linker.define(&*caller, namespace, name, value)?;
    }
    if !imports_memory {
        return Err(Error::msg("dynamic library must import shared memory"));
    }
    // Defined extra memories are rejected by the Store's one-memory limit; unavailable WASI
    // operations retain the main executable's explicit trap boundary.
    linker.define_unknown_imports_as_traps(&module)?;
    let instance = {
        let _fiber = fibers::begin(&mut *caller)?;
        linker.instantiate_async(&mut *caller, &module).await?
    };
    for (namespace, name, global) in got {
        let symbol = import_symbol(caller, &name, false, &dependencies)
            .or_else(|| {
                if abi == Abi::V2 && canonical_runtime_symbol(&name) {
                    return None;
                }
                instance
                    .get_export(&mut *caller, &name)
                    .map(|export| (export, memory_base))
            })
            .ok_or_else(|| Error::msg(format!("missing dynamic symbol: {name}")))?;
        if (namespace == "GOT.mem" && !matches!(symbol.0, Extern::Global(_)))
            || (namespace == "GOT.func" && !matches!(symbol.0, Extern::Func(_)))
        {
            return Err(Error::msg(format!("dynamic symbol kind mismatch: {name}")));
        }
        let value = address(caller, symbol.0, symbol.1)?;
        global.set(&mut *caller, Val::I32(value as i32))?;
    }
    for name in ["__wasm_apply_data_relocs", "__wasm_call_ctors"] {
        if let Some(function) = instance.get_func(&mut *caller, name) {
            let _fiber = fibers::begin(&mut *caller)?;
            function
                .typed::<(), ()>(&*caller)?
                .call_async(&mut *caller, ())
                .await?;
        }
    }
    let handle = caller.data().dynamic.libraries.len() as u32 + 1;
    caller.data_mut().dynamic.libraries.push(Library {
        instance,
        memory_base,
        global: flags & 256 != 0,
        dependencies,
    });
    caller.data_mut().dynamic.paths.insert(path, handle);
    Ok(handle)
}

pub(super) fn register(linker: &mut Linker<Host>) {
    for abi in [Abi::V1, Abi::V2] {
        linker
            .func_wrap_async(
                abi.namespace(),
                "open",
                |mut caller: Caller<'_, Host>, (pointer, length, flags): (u32, u32, u32)| {
                    Box::new(async move {
                        let path = guest_string(&mut caller, pointer, length)?;
                        if caller.data().dynamic.loading || caller.data().dynamic.failed {
                            caller.data_mut().dynamic.error = Some(
                                "nested loading or loading after failure is unsupported".to_owned(),
                            );
                            return Ok(0u32);
                        }
                        caller.data_mut().dynamic.loading = true;
                        let result = load(&mut caller, path, flags).await;
                        caller.data_mut().dynamic.loading = false;
                        match result {
                            Ok(handle) => Ok(handle),
                            Err(error) => {
                                caller.data_mut().dynamic.failed = true;
                                if error.is::<super::GuestExit>()
                                    || caller
                                        .data()
                                        .machine
                                        .get()
                                        .resources
                                        .stop_reason()
                                        .is_some()
                                    || caller.get_fuel()? == 0
                                {
                                    return Err(error);
                                }
                                caller.data_mut().dynamic.error =
                                    Some(format!("{error:#}").chars().take(2047).collect());
                                Ok(0)
                            }
                        }
                    })
                },
            )
            .expect("valid dynamic loader signature");
        linker
            .func_wrap(
                abi.namespace(),
                "symbol",
                |mut caller: Caller<'_, Host>,
                 handle: u32,
                 pointer: u32,
                 length: u32|
                 -> Result<u32, Error> {
                    let name = guest_string(&mut caller, pointer, length)?;
                    let result = if handle == 0 {
                        lookup(&mut caller, &name)
                    } else {
                        let library = caller
                            .data()
                            .dynamic
                            .libraries
                            .get(handle as usize - 1)
                            .map(|lib| (lib.instance, lib.memory_base, lib.dependencies.clone()));
                        library.and_then(|(instance, base, dependencies)| {
                            instance
                                .get_export(&mut caller, &name)
                                .map(|symbol| (symbol, base))
                                .or_else(|| lookup_dependencies(&mut caller, &name, &dependencies))
                        })
                    };
                    match result {
                        Some((symbol, base)) => address(&mut caller, symbol, base),
                        None => {
                            caller.data_mut().dynamic.error =
                                Some(format!("missing dynamic symbol: {name}"));
                            Ok(0)
                        }
                    }
                },
            )
            .expect("valid dynamic symbol signature");
        linker
            .func_wrap(
                abi.namespace(),
                "error",
                |mut caller: Caller<'_, Host>, pointer: u32, capacity: u32| -> Result<u32, Error> {
                    if capacity == 0 || capacity > 4096 {
                        return Err(Error::msg("invalid dlerror buffer"));
                    }
                    let Some(error) = caller.data_mut().dynamic.error.take() else {
                        return Ok(0);
                    };
                    let bytes = error.as_bytes();
                    let len = bytes.len().min(capacity as usize - 1);
                    let shared_memory = memory(&mut caller)
                        .ok_or_else(|| Error::msg("dynamic loading requires memory"))?;
                    shared_memory.write(&mut caller, pointer as usize, &bytes[..len])?;
                    shared_memory.write(&mut caller, pointer as usize + len, &[0])?;
                    Ok(len as u32)
                },
            )
            .expect("valid dynamic error signature");
    }
}

#[cfg(test)]
mod tests {
    use super::{align, Cursor};

    #[test]
    fn dylink_alignment_and_leb_boundaries_are_checked() {
        assert_eq!(align(17, 4).unwrap(), 32);
        assert!(align(u32::MAX, 4).is_err());
        assert!(Cursor(&[255, 255, 255, 255, 16]).number().is_err());
        assert!(Cursor(&[128]).number().is_err());
        assert_eq!(
            Cursor(&[255, 255, 255, 255, 15]).number().unwrap(),
            u32::MAX
        );
    }
}
