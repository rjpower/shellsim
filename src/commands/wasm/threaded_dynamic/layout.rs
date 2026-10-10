//! Bounded LLVM dylink metadata for the exact threaded v3 cohort.

use super::{MARKER, MAX_MEMORY_PAGES, MAX_MODULES, MAX_TABLE_ELEMENTS};
use std::collections::{BTreeMap, BTreeSet};
use wasmtime::Error;

pub(super) struct Layout {
    pub(super) memory_size: u32,
    pub(super) memory_align: u32,
    pub(super) table_size: u32,
    pub(super) table_align: u32,
    pub(super) needed: Vec<String>,
    pub(super) runtime_paths: Vec<String>,
    pub(super) tls_exports: BTreeSet<String>,
    pub(super) weak_imports: BTreeSet<(String, String)>,
    pub(super) has_start: bool,
    pub(super) tls_size: u32,
    pub(super) tls_align: u32,
    pub(super) functions: Vec<(String, u32)>,
    pub(super) element_functions: BTreeMap<u32, u32>,
    pub(super) relative_data: BTreeMap<String, u32>,
    pub(super) forwarded_exports: BTreeSet<String>,
}

struct Cursor<'a>(&'a [u8]);

impl<'a> Cursor<'a> {
    fn number(&mut self) -> Result<u32, Error> {
        let mut value = 0;
        for shift in (0..35).step_by(7) {
            let byte = *self
                .0
                .first()
                .ok_or_else(|| Error::msg("truncated dylink metadata"))?;
            self.0 = &self.0[1..];
            if shift == 28 && byte > 15 {
                return Err(Error::msg("dylink integer overflow"));
            }
            value |= u32::from(byte & 127) << shift;
            if byte & 128 == 0 {
                return Ok(value);
            }
        }
        Err(Error::msg("dylink integer overflow"))
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], Error> {
        let value = self
            .0
            .get(..count)
            .ok_or_else(|| Error::msg("truncated dylink metadata"))?;
        self.0 = &self.0[count..];
        Ok(value)
    }

    fn string(&mut self) -> Result<String, Error> {
        let count = self.number()? as usize;
        if count == 0 || count > 4096 {
            return Err(Error::msg("invalid dylink symbol length"));
        }
        let value = std::str::from_utf8(self.take(count)?)?;
        if value.contains('\0') {
            return Err(Error::msg("invalid dylink symbol"));
        }
        Ok(value.to_owned())
    }
}

/// The caller scans raw atomics and prepays compilation before this metadata
/// reaches an instantiated module. Private exception tags cannot join the cohort.
pub(super) fn parse(source: &[u8]) -> Result<Layout, Error> {
    let mut marker = false;
    let mut metadata = None;
    let mut has_start = false;
    let mut table_base_global = None;
    let mut active_elements = 0u32;
    let mut element_count = 0u32;
    let mut imported_globals = 0u32;
    let mut imported_functions = 0u32;
    let mut forwarded_exports = BTreeSet::new();
    let mut globals = Vec::new();
    let mut tls_size_index = None;
    let mut tls_align_index = None;
    let mut functions = Vec::new();
    let mut element_functions = BTreeMap::new();
    let mut data_exports = BTreeMap::new();
    for payload in wasmparser::Parser::new(0).parse_all(source) {
        match payload? {
            wasmparser::Payload::CustomSection(section) if section.name() == "shellsim.abi" => {
                if marker || section.data() != MARKER {
                    return Err(Error::msg("threaded dynamic ABI mismatch"));
                }
                marker = true;
            }
            wasmparser::Payload::CustomSection(section) if section.name() == "dylink.0" => {
                if metadata.is_some() || section.data().len() > 1024 * 1024 {
                    return Err(Error::msg("duplicate or oversized dylink metadata"));
                }
                metadata = Some(section.data());
            }
            wasmparser::Payload::StartSection { .. } => has_start = true,
            wasmparser::Payload::ImportSection(imports) => {
                for import in imports.into_imports() {
                    let import = import?;
                    if matches!(import.ty, wasmparser::TypeRef::Func(_)) {
                        imported_functions = imported_functions
                            .checked_add(1)
                            .ok_or_else(|| Error::msg("function index overflow"))?;
                    }
                    if matches!(import.ty, wasmparser::TypeRef::Global(_)) {
                        if import.module == "env" && import.name == "__table_base" {
                            table_base_global = Some(imported_globals);
                        }
                        imported_globals = imported_globals
                            .checked_add(1)
                            .ok_or_else(|| Error::msg("global index overflow"))?;
                    }
                }
            }
            wasmparser::Payload::TableSection(tables) if tables.count() != 0 => {
                return Err(Error::msg("threaded side table must be imported"));
            }
            wasmparser::Payload::ElementSection(elements) => {
                for element in elements {
                    let element = element?;
                    let wasmparser::ElementKind::Active {
                        table_index,
                        offset_expr,
                    } = element.kind
                    else {
                        return Err(Error::msg(
                            "threaded side element segments must use their assigned table base",
                        ));
                    };
                    let mut reader = offset_expr.get_operators_reader();
                    if table_index.unwrap_or(0) != 0
                        || active_elements != 0
                        || !matches!(reader.read()?, wasmparser::Operator::GlobalGet { global_index } if Some(global_index) == table_base_global)
                        || !matches!(reader.read()?, wasmparser::Operator::End)
                        || !reader.eof()
                    {
                        return Err(Error::msg(
                            "threaded side element segment exceeds assigned table placement",
                        ));
                    }
                    active_elements += 1;
                    element_count = match element.items {
                        wasmparser::ElementItems::Functions(items) => {
                            let count = items.count();
                            for (offset, index) in items.into_iter().enumerate() {
                                element_functions.entry(index?).or_insert(offset as u32);
                            }
                            count
                        }
                        wasmparser::ElementItems::Expressions(_, _) => {
                            return Err(Error::msg(
                                "threaded side element expressions are unsupported",
                            ))
                        }
                    };
                }
            }
            wasmparser::Payload::CodeSectionEntry(body) => {
                for operator in body.get_operators_reader()? {
                    if matches!(
                        operator?,
                        wasmparser::Operator::TableSet { .. }
                            | wasmparser::Operator::TableGrow { .. }
                            | wasmparser::Operator::TableCopy { .. }
                            | wasmparser::Operator::TableInit { .. }
                            | wasmparser::Operator::TableFill { .. }
                            | wasmparser::Operator::ElemDrop { .. }
                    ) {
                        return Err(Error::msg(
                            "side table mutation requires process registry operations",
                        ));
                    }
                }
            }
            wasmparser::Payload::GlobalSection(section) => {
                for global in section {
                    let mut reader = global?.init_expr.get_operators_reader();
                    let mut value = match reader.read()? {
                        wasmparser::Operator::I32Const { value } => Some(value as u32),
                        _ => None,
                    };
                    if !matches!(reader.read()?, wasmparser::Operator::End) || !reader.eof() {
                        value = None;
                    }
                    globals.push(value);
                }
            }
            wasmparser::Payload::ExportSection(exports) => {
                for export in exports {
                    let export = export?;
                    if export.name.len() > 4096 {
                        return Err(Error::msg("oversized side export name"));
                    }
                    if (export.kind == wasmparser::ExternalKind::Func
                        && export.index < imported_functions)
                        || (export.kind == wasmparser::ExternalKind::Global
                            && export.index < imported_globals)
                    {
                        forwarded_exports.insert(export.name.to_owned());
                        continue;
                    }
                    if export.kind == wasmparser::ExternalKind::Func {
                        functions.push((export.name.to_owned(), export.index));
                    }
                    if export.kind == wasmparser::ExternalKind::Global {
                        data_exports.insert(export.name.to_owned(), export.index);
                        if export.name == "__tls_size" {
                            tls_size_index = Some(export.index);
                        }
                        if export.name == "__tls_align" {
                            tls_align_index = Some(export.index);
                        }
                    }
                }
            }
            wasmparser::Payload::TagSection(tags) if tags.count() != 0 => {
                return Err(Error::msg(
                    "threaded side exception tags must import the canonical main tag",
                ))
            }
            wasmparser::Payload::MemorySection(memories) if memories.count() != 0 => {
                return Err(Error::msg("threaded side memory must be imported"))
            }
            wasmparser::Payload::DataSection(segments) => {
                for segment in segments {
                    if matches!(segment?.kind, wasmparser::DataKind::Active { .. }) {
                        return Err(Error::msg(
                            "threaded side data requires guarded passive initialization",
                        ));
                    }
                }
            }
            _ => {}
        }
    }
    if has_start {
        return Err(Error::msg(
            "threaded sides require explicit deferred initialization, without a module start",
        ));
    }
    if !marker {
        return Err(Error::msg("threaded dynamic ABI mismatch"));
    }
    let mut runtime_paths = Vec::new();
    let mut cursor = Cursor(metadata.ok_or_else(|| Error::msg("missing dylink.0"))?);
    let mut dimensions = None;
    let mut deferred = false;
    let mut needed = Vec::new();
    let mut flags_seen = BTreeSet::new();
    let mut tls_exports = BTreeSet::new();
    let mut weak_imports = BTreeSet::new();
    while !cursor.0.is_empty() {
        let kind = cursor.take(1)?[0];
        let length = cursor.number()? as usize;
        let mut payload = Cursor(cursor.take(length)?);
        match kind {
            128 => {
                if deferred
                    || payload.string()? != "shellsim.deferred-init"
                    || payload.number()? != 1
                {
                    return Err(Error::msg("unsupported deferred initialization protocol"));
                }
                deferred = true;
            }
            1 => {
                if dimensions.is_some() {
                    return Err(Error::msg("duplicate dylink layout"));
                }
                dimensions = Some((
                    payload.number()?,
                    payload.number()?,
                    payload.number()?,
                    payload.number()?,
                ));
            }
            2 => {
                if !flags_seen.insert(kind) {
                    return Err(Error::msg("duplicate dylink dependencies"));
                }
                let count = payload.number()? as usize;
                if count > MAX_MODULES {
                    return Err(Error::msg("dylink dependency limit exceeded"));
                }
                for _ in 0..count {
                    let name = payload.string()?;
                    if name.contains('/')
                        || name.contains('\\')
                        || matches!(name.as_str(), "." | "..")
                        || needed.contains(&name)
                    {
                        return Err(Error::msg("invalid dylink dependency"));
                    }
                    needed.push(name);
                }
            }
            5 => {
                if !flags_seen.insert(kind) {
                    return Err(Error::msg("duplicate dylink runtime paths"));
                }
                let count = payload.number()?;
                if count as usize > MAX_MODULES {
                    return Err(Error::msg("dylink runtime path limit exceeded"));
                }
                for _ in 0..count {
                    runtime_paths.push(payload.string()?);
                }
            }
            3 | 4 => {
                if !flags_seen.insert(kind) {
                    return Err(Error::msg("duplicate dylink symbol flags"));
                }
                let count = payload.number()?;
                if count > 65_536 {
                    return Err(Error::msg("dylink symbol limit exceeded"));
                }
                let mut symbols = BTreeMap::new();
                for _ in 0..count {
                    let namespace = if kind == 4 {
                        payload.string()?
                    } else {
                        String::new()
                    };
                    let name = payload.string()?;
                    let flags = payload.number()?;
                    if flags & !0x3ff != 0
                        || symbols
                            .insert((namespace.clone(), name.clone()), flags)
                            .is_some()
                    {
                        return Err(Error::msg("invalid dylink symbol flags"));
                    }
                    if kind == 3 && flags & 0x100 != 0 {
                        tls_exports.insert(name);
                    } else if kind == 4 && flags & 3 == 1 {
                        weak_imports.insert((namespace, name));
                    }
                }
            }
            _ => return Err(Error::msg("unsupported dylink metadata subsection")),
        }
        if !payload.0.is_empty() {
            return Err(Error::msg("invalid dylink metadata length"));
        }
    }
    if !deferred {
        return Err(Error::msg(
            "threaded side lacks shellsim.deferred-init revision 1",
        ));
    }
    let (memory_size, memory_align, table_size, table_align) =
        dimensions.ok_or_else(|| Error::msg("missing dylink layout"))?;
    if u64::from(memory_size) > MAX_MEMORY_PAGES * 65_536
        || memory_align > 16
        || table_align > 16
        || table_size as usize > MAX_TABLE_ELEMENTS
        || element_count > table_size
    {
        return Err(Error::msg("dylink allocation exceeds cohort limits"));
    }
    let constant = |index: Option<u32>| -> Result<u32, Error> {
        let Some(index) = index else {
            return Ok(0);
        };
        index
            .checked_sub(imported_globals)
            .and_then(|index| globals.get(index as usize))
            .copied()
            .flatten()
            .ok_or_else(|| Error::msg("TLS dimensions require constant exported globals"))
    };
    let tls_size = constant(tls_size_index)?;
    let tls_align = constant(tls_align_index)?;
    if u64::from(tls_size) > MAX_MEMORY_PAGES * 65_536
        || (tls_size != 0 && (!tls_align.is_power_of_two() || tls_align > 65_536))
    {
        return Err(Error::msg("invalid TLS dimensions"));
    }
    let relative_data = data_exports
        .into_iter()
        .filter_map(|(name, index)| {
            index
                .checked_sub(imported_globals)
                .and_then(|index| globals.get(index as usize))
                .copied()
                .flatten()
                .map(|offset| (name, offset))
        })
        .collect();
    Ok(Layout {
        memory_size,
        memory_align,
        table_size,
        table_align,
        needed,
        runtime_paths,
        tls_exports,
        weak_imports,
        has_start,
        tls_size,
        tls_align: tls_align.max(1),
        functions,
        element_functions,
        relative_data,
        forwarded_exports,
    })
}

/// Bounded executable linkage metadata. Imported function and global reexports
/// must never claim ownership of a symbol supplied by a dependency.
#[derive(Default)]
pub(in crate::commands::wasm) struct Executable {
    pub(in crate::commands::wasm) tls_exports: BTreeSet<String>,
    pub(in crate::commands::wasm) needed: Vec<String>,
    pub(super) runtime_paths: Vec<String>,
    pub(super) weak_imports: BTreeSet<(String, String)>,
    pub(in crate::commands::wasm) forwarded_exports: BTreeSet<String>,
    pub(super) start: Option<u32>,
}

pub(in crate::commands::wasm) const START_EXPORT: &str = "__shellsim_executable_start";

/// Executables own their memory/table placement; only dependency and symbol
/// metadata affect startup. Reject malformed metadata before guest execution.
pub(in crate::commands::wasm) fn executable(bytes: &[u8]) -> Result<Executable, Error> {
    let mut result = Executable::default();
    let mut metadata = None;
    let mut imported_functions = 0u32;
    let mut imported_globals = 0u32;
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        match payload? {
            wasmparser::Payload::CustomSection(section) if section.name() == "dylink.0" => {
                if metadata.is_some() || section.data().len() > 1024 * 1024 {
                    return Err(Error::msg(
                        "duplicate or oversized executable dylink metadata",
                    ));
                }
                metadata = Some(section.data());
            }
            wasmparser::Payload::StartSection { func, .. } => result.start = Some(func),
            wasmparser::Payload::ImportSection(imports) => {
                for import in imports.into_imports() {
                    match import?.ty {
                        wasmparser::TypeRef::Func(_) => {
                            imported_functions = imported_functions
                                .checked_add(1)
                                .ok_or_else(|| Error::msg("function index overflow"))?;
                        }
                        wasmparser::TypeRef::Global(_) => {
                            imported_globals = imported_globals
                                .checked_add(1)
                                .ok_or_else(|| Error::msg("global index overflow"))?;
                        }
                        _ => {}
                    }
                }
            }
            wasmparser::Payload::ExportSection(exports) => {
                for export in exports {
                    let export = export?;
                    if export.name == START_EXPORT {
                        return Err(Error::msg("reserved executable start export"));
                    }
                    if (export.kind == wasmparser::ExternalKind::Func
                        && export.index < imported_functions)
                        || (export.kind == wasmparser::ExternalKind::Global
                            && export.index < imported_globals)
                    {
                        result.forwarded_exports.insert(export.name.to_owned());
                    }
                }
            }
            _ => {}
        }
    }
    let mut declared = false;
    let mut seen = BTreeSet::new();
    let mut cursor = Cursor(metadata.ok_or_else(|| Error::msg("missing executable dylink.0"))?);
    while !cursor.0.is_empty() {
        let kind = cursor.take(1)?[0];
        let length = cursor.number()? as usize;
        let mut payload = Cursor(cursor.take(length)?);
        if !seen.insert(kind) {
            return Err(Error::msg("duplicate executable dylink subsection"));
        }
        match kind {
            129 => {
                if payload.string()? != "shellsim.main-tls" || payload.number()? != 1 {
                    return Err(Error::msg("invalid main TLS classification protocol"));
                }
                declared = true;
            }
            1 => {
                let memory = payload.number()?;
                let memory_align = payload.number()?;
                let table = payload.number()?;
                let table_align = payload.number()?;
                if u64::from(memory) > MAX_MEMORY_PAGES * 65_536
                    || memory_align > 16
                    || table_align > 16
                    || table as usize > MAX_TABLE_ELEMENTS
                {
                    return Err(Error::msg("executable dylink layout exceeds limits"));
                }
            }
            2 | 5 => {
                let count = payload.number()? as usize;
                if count > MAX_MODULES {
                    return Err(Error::msg("executable dependency/path limit exceeded"));
                }
                let values = if kind == 2 {
                    &mut result.needed
                } else {
                    &mut result.runtime_paths
                };
                for _ in 0..count {
                    let name = payload.string()?;
                    if kind == 2
                        && (name.contains('/')
                            || name.contains('\\')
                            || matches!(name.as_str(), "." | "..")
                            || values.contains(&name))
                    {
                        return Err(Error::msg("invalid executable dependency"));
                    }
                    values.push(name);
                }
            }
            3 | 4 => {
                let count = payload.number()?;
                if count > 65_536 {
                    return Err(Error::msg("executable symbol limit exceeded"));
                }
                let mut symbols = BTreeSet::new();
                for _ in 0..count {
                    let namespace = if kind == 4 {
                        payload.string()?
                    } else {
                        String::new()
                    };
                    let name = payload.string()?;
                    let flags = payload.number()?;
                    if flags & !0x3ff != 0 || !symbols.insert((namespace.clone(), name.clone())) {
                        return Err(Error::msg("invalid executable symbol flags"));
                    }
                    if kind == 3 && flags & 0x100 != 0 {
                        result.tls_exports.insert(name);
                    } else if kind == 4 && flags & 3 == 1 {
                        result.weak_imports.insert((namespace, name));
                    }
                }
            }
            _ => return Err(Error::msg("unsupported executable dylink subsection")),
        }
        if !payload.0.is_empty() {
            return Err(Error::msg("invalid executable dylink metadata length"));
        }
    }
    if !declared {
        return Err(Error::msg("main TLS exports require a versioned protocol"));
    }
    Ok(result)
}

/// Preserve the original start function as a private host-controlled export.
/// Its memory initialization is called separately before guest malloc; its
/// remaining relocations run only after strong imports and GOT addresses bind.
pub(in crate::commands::wasm) fn defer_start(
    bytes: &[u8],
    main: &Executable,
) -> Result<Vec<u8>, Error> {
    fn leb(output: &mut Vec<u8>, mut value: u32) {
        loop {
            let byte = (value & 127) as u8;
            value >>= 7;
            output.push(byte | if value != 0 { 128 } else { 0 });
            if value == 0 {
                break;
            }
        }
    }
    let Some(start) = main.start else {
        return Ok(bytes.to_vec());
    };
    let mut output = bytes[..8].to_vec();
    let mut exported = false;
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        let payload = payload?;
        if matches!(payload, wasmparser::Payload::StartSection { .. }) {
            continue;
        }
        let Some((id, range)) = payload.as_section() else {
            continue;
        };
        let range = usize::try_from(range.start)?..usize::try_from(range.end)?;
        if let wasmparser::Payload::ExportSection(exports) = payload {
            exported = true;
            let mut body = Vec::new();
            leb(
                &mut body,
                exports
                    .count()
                    .checked_add(1)
                    .ok_or_else(|| Error::msg("export count overflow"))?,
            );
            let mut cursor = Cursor(&bytes[range.clone()]);
            cursor.number()?;
            body.extend_from_slice(cursor.0);
            leb(&mut body, START_EXPORT.len() as u32);
            body.extend_from_slice(START_EXPORT.as_bytes());
            body.push(0);
            leb(&mut body, start);
            output.push(id);
            leb(&mut output, body.len() as u32);
            output.extend(body);
        } else {
            output.push(id);
            leb(&mut output, range.len() as u32);
            output.extend_from_slice(&bytes[range]);
        }
    }
    if !exported {
        return Err(Error::msg(
            "dynamic executable startup requires an export section",
        ));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn executable_metadata(body: &str, extra: &[u8]) -> Vec<u8> {
        let mut module = wat::parse_str(format!("(module {body})")).unwrap();
        let mut metadata = b"\x81\x13\x11shellsim.main-tls\x01".to_vec();
        metadata.extend_from_slice(extra);
        custom(&mut module, "dylink.0", &metadata);
        module
    }

    #[test]
    fn executable_ownership_excludes_function_and_global_import_reexports() {
        let source = executable_metadata(
            r#"
            (import "env" "function" (func $function))
            (import "GOT.mem" "data" (global $data (mut i32)))
            (export "function_alias" (func $function))
            (export "data_alias" (global $data))
            (func (export "owned"))"#,
            &[],
        );
        let layout = executable(&source).unwrap();
        assert_eq!(
            layout.forwarded_exports,
            BTreeSet::from(["function_alias".to_owned(), "data_alias".to_owned()])
        );
    }

    #[test]
    fn executable_dependencies_reject_truncation_duplicates_and_excessive_counts() {
        let needed = b"\x02\x09\x01\x07root.so";
        assert_eq!(
            executable(&executable_metadata("", needed)).unwrap().needed,
            ["root.so"]
        );
        assert!(executable(&executable_metadata("", &needed[..needed.len() - 1])).is_err());
        let mut duplicate = needed.to_vec();
        duplicate.extend_from_slice(needed);
        assert!(executable(&executable_metadata("", &duplicate)).is_err());
        assert!(executable(&executable_metadata("", b"\x02\x03\x01\x01/")).is_err());
        let count = leb(MAX_MODULES + 1);
        let mut excessive = vec![2];
        excessive.extend(leb(count.len()));
        excessive.extend(count);
        assert!(executable(&executable_metadata("", &excessive)).is_err());
    }

    #[test]
    fn deferred_executable_start_is_exported_once_and_cannot_be_spoofed() {
        let source = executable_metadata("(func $start (export \"_start\")) (start $start)", &[]);
        let layout = executable(&source).unwrap();
        let rewritten = defer_start(&source, &layout).unwrap();
        let mut entry = None;
        for payload in wasmparser::Parser::new(0).parse_all(&rewritten) {
            match payload.unwrap() {
                wasmparser::Payload::StartSection { .. } => panic!("start was not deferred"),
                wasmparser::Payload::ExportSection(exports) => {
                    for export in exports {
                        let export = export.unwrap();
                        if export.name == START_EXPORT {
                            entry = Some(export.index);
                        }
                    }
                }
                _ => {}
            }
        }
        assert_eq!(entry, layout.start);
        let source = executable_metadata("(func $start) (start $start)", &[]);
        assert!(defer_start(&source, &executable(&source).unwrap()).is_err());
        let source = executable_metadata("(func (export \"__shellsim_executable_start\"))", &[]);
        assert!(executable(&source).is_err());
    }

    fn leb(mut value: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        loop {
            let mut byte = (value & 127) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 128;
            }
            bytes.push(byte);
            if value == 0 {
                return bytes;
            }
        }
    }

    fn custom(module: &mut Vec<u8>, name: &str, data: &[u8]) {
        let mut body = leb(name.len());
        body.extend_from_slice(name.as_bytes());
        body.extend_from_slice(data);
        module.push(0);
        module.extend(leb(body.len()));
        module.extend(body);
    }

    fn side(body: &str) -> Vec<u8> {
        side_metadata(body, &[])
    }

    fn side_metadata(body: &str, extra: &[u8]) -> Vec<u8> {
        let mut module = wat::parse_str(format!(
            "(module (import \"env\" \"memory\" (memory 1 2 shared)) {body})"
        ))
        .unwrap();
        custom(&mut module, "shellsim.abi", MARKER);
        let mut metadata = vec![1, 4, 0, 0, 0, 0, 128, 24, 22];
        metadata.extend_from_slice(b"shellsim.deferred-init");
        metadata.push(1);
        metadata.extend_from_slice(extra);
        custom(&mut module, "dylink.0", &metadata);
        module
    }

    // LLD DylinkSection::writeBody emits subsection 5 as count + length-prefixed paths.
    #[test]
    fn runtime_path_vector_matches_lld_encoding_and_is_bounded() {
        let paths = ["/host/build/lib", "$ORIGIN/../lib"];
        let mut payload = vec![paths.len() as u8];
        for path in paths {
            payload.extend(leb(path.len()));
            payload.extend(path.as_bytes());
        }
        let mut subsection = vec![5];
        subsection.extend(leb(payload.len()));
        subsection.extend(&payload);
        assert_eq!(
            parse(&side_metadata("", &subsection))
                .unwrap()
                .runtime_paths,
            paths
        );
        let mut duplicate = subsection.clone();
        duplicate.extend(&subsection);
        assert!(parse(&side_metadata("", &duplicate)).is_err());
        subsection.pop();
        assert!(parse(&side_metadata("", &subsection)).is_err());
        let count = leb(MAX_MODULES + 1);
        let mut excessive = vec![5];
        excessive.extend(leb(count.len()));
        excessive.extend(count);
        assert!(parse(&side_metadata("", &excessive)).is_err());
        assert!(parse(&side_metadata("", &[5, 2, 1, 0])).is_err());
    }

    #[test]
    fn explicit_passive_initialization_accepts_standard_uint8_vendor_metadata() {
        assert!(parse(&side("(data \"retained template\")")).is_ok());
    }

    #[test]
    fn instantiation_cannot_reset_process_memory_or_run_guest_code() {
        assert!(parse(&side("(data (i32.const 0) \"overwrite\")")).is_err());
        assert!(parse(&side("(func $start) (start $start)")).is_err());
    }

    #[test]
    fn instantiation_cannot_overwrite_existing_function_pointer_slots() {
        assert!(parse(&side("(import \"env\" \"__indirect_function_table\" (table 1 funcref)) (func $f) (elem (i32.const 0) $f)")).is_err());
    }

    #[test]
    #[ignore = "requires freshly emitted pinned LLD side artifact"]
    fn emitted_side_has_replay_safe_layout_and_data_pointer_metadata() {
        let source = std::fs::read(std::env::var("SHELLSIM_DEFERRED_SIDE").unwrap()).unwrap();
        let layout = parse(&source).unwrap();
        assert!(layout.tls_size > 0);
        assert!(layout.functions.iter().any(|(name, _)| name == "increment"));
        assert!(layout.relative_data.contains_key("shared_value"));
    }
}
