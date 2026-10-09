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
    pub(super) tls_exports: BTreeSet<String>,
    pub(super) weak_imports: BTreeSet<(String, String)>,
    pub(super) has_start: bool,
    pub(super) tls_size: u32,
    pub(super) tls_align: u32,
    pub(super) functions: Vec<(String, u32)>,
    pub(super) element_functions: BTreeMap<u32, u32>,
    pub(super) relative_data: BTreeMap<String, u32>,
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
        tls_exports,
        weak_imports,
        has_start,
        tls_size,
        tls_align: tls_align.max(1),
        functions,
        element_functions,
        relative_data,
    })
}

/// Read explicit executable TLS classifications; ordinary exports remain data.
pub(in crate::commands::wasm) fn main_tls_exports(bytes: &[u8]) -> Result<BTreeSet<String>, Error> {
    let mut exports = BTreeSet::new();
    let mut declared = false;
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::CustomSection(section) = payload? {
            if section.name() != "dylink.0" {
                continue;
            }
            let mut cursor = Cursor(section.data());
            while !cursor.0.is_empty() {
                let kind = cursor.take(1)?[0];
                let length = cursor.number()? as usize;
                let mut payload = Cursor(cursor.take(length)?);
                if kind == 129 {
                    if declared
                        || payload.string()? != "shellsim.main-tls"
                        || payload.number()? != 1
                        || !payload.0.is_empty()
                    {
                        return Err(Error::msg("invalid main TLS classification protocol"));
                    }
                    declared = true;
                } else if kind == 3 {
                    let count = payload.number()?;
                    for _ in 0..count {
                        let name = payload.string()?;
                        let flags = payload.number()?;
                        if flags & !0x3ff != 0 {
                            return Err(Error::msg("invalid main symbol flags"));
                        }
                        if flags & 0x100 != 0 && !exports.insert(name) {
                            return Err(Error::msg("duplicate main TLS export"));
                        }
                    }
                    if !payload.0.is_empty() {
                        return Err(Error::msg("invalid main TLS export payload"));
                    }
                }
            }
        }
    }
    if !declared {
        return Err(Error::msg("main TLS exports require a versioned protocol"));
    }
    Ok(exports)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut module = wat::parse_str(format!(
            "(module (import \"env\" \"memory\" (memory 1 2 shared)) {body})"
        ))
        .unwrap();
        custom(&mut module, "shellsim.abi", MARKER);
        let mut metadata = vec![1, 4, 0, 0, 0, 0, 128, 24, 22];
        metadata.extend_from_slice(b"shellsim.deferred-init");
        metadata.push(1);
        custom(&mut module, "dylink.0", &metadata);
        module
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
