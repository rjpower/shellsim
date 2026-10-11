//! Adapt linked Wasm atomic waits to Shellsim's virtual scheduler imports.
//!
//! Translation runs on build workers before publication. Raw waits remain
//! forbidden in guests; translated calls use the metered virtual scheduler.
use std::{env, fs};
use wasm_encoder::reencode::{Error, Reencode, utils};
use wasm_encoder::{
    BlockType, CodeSection, EntityType, Function, FunctionSection, ImportSection, Instruction,
    Module, SectionId, TypeSection, ValType,
};
use wasmparser::{Operator, Parser, Payload, TypeRef};

struct Scheduler {
    imports: u32,
    types: u32,
    inserted_imports: bool,
    defined: u32,
    wrappers: Vec<(u32, bool)>,
    namespace: &'static str,
}
impl Scheduler {
    fn add_imports(&self, section: &mut ImportSection) {
        section.import(self.namespace, "wait32", EntityType::Function(self.types));
        section.import(
            self.namespace,
            "notify",
            EntityType::Function(self.types + 1),
        );
    }
}
impl Reencode for Scheduler {
    type Error = &'static str;
    fn function_index(&mut self, index: u32) -> Result<u32, Error<Self::Error>> {
        if index < self.imports {
            Ok(index)
        } else {
            index
                .checked_add(2)
                .ok_or(Error::UserError("function index overflow"))
        }
    }
    fn parse_type_section(
        &mut self,
        types: &mut TypeSection,
        section: wasmparser::TypeSectionReader<'_>,
    ) -> Result<(), Error<Self::Error>> {
        utils::parse_type_section(self, types, section)?;
        types
            .ty()
            .function([ValType::I32, ValType::I32, ValType::I64], [ValType::I32]);
        types
            .ty()
            .function([ValType::I32, ValType::I32], [ValType::I32]);
        Ok(())
    }
    fn parse_import_section(
        &mut self,
        imports: &mut ImportSection,
        section: wasmparser::ImportSectionReader<'_>,
    ) -> Result<(), Error<Self::Error>> {
        utils::parse_import_section(self, imports, section)?;
        self.add_imports(imports);
        self.inserted_imports = true;
        Ok(())
    }
    fn intersperse_section_hook(
        &mut self,
        module: &mut Module,
        after: Option<SectionId>,
        before: Option<SectionId>,
    ) -> Result<(), Error<Self::Error>> {
        if !self.inserted_imports
            && after == Some(SectionId::Type)
            && before != Some(SectionId::Import)
        {
            let mut imports = ImportSection::new();
            self.add_imports(&mut imports);
            module.section(&imports);
            self.inserted_imports = true;
        }
        Ok(())
    }
    fn parse_function_section(
        &mut self,
        functions: &mut FunctionSection,
        section: wasmparser::FunctionSectionReader<'_>,
    ) -> Result<(), Error<Self::Error>> {
        utils::parse_function_section(self, functions, section)?;
        for &(_, notify) in &self.wrappers {
            functions.function(self.types + u32::from(notify));
        }
        Ok(())
    }
    fn parse_code_section(
        &mut self,
        code: &mut CodeSection,
        section: wasmparser::CodeSectionReader<'_>,
    ) -> Result<(), Error<Self::Error>> {
        utils::parse_code_section(self, code, section)?;
        for &(offset, notify) in &self.wrappers {
            let mut function = Function::new([]);
            for op in [
                Instruction::LocalGet(0),
                Instruction::I64ExtendI32U,
                Instruction::I64Const(i64::from(offset)),
                Instruction::I64Add,
                Instruction::I64Const(i64::from(u32::MAX)),
                Instruction::I64GtU,
                Instruction::If(BlockType::Empty),
                Instruction::Unreachable,
                Instruction::End,
                Instruction::LocalGet(0),
                Instruction::I32Const(offset as i32),
                Instruction::I32Add,
                Instruction::LocalGet(1),
            ] {
                function.instruction(&op);
            }
            if !notify {
                function.instruction(&Instruction::LocalGet(2));
            }
            function
                .instruction(&Instruction::Call(self.imports + u32::from(notify)))
                .instruction(&Instruction::End);
            code.function(&function);
        }
        Ok(())
    }
    fn instruction<'a>(&mut self, op: Operator<'a>) -> Result<Instruction<'a>, Error<Self::Error>> {
        match op {
            Operator::MemoryAtomicWait32 { memarg } | Operator::MemoryAtomicNotify { memarg } => {
                if memarg.memory != 0 {
                    return Err(Error::UserError(
                        "scheduler translation requires memory zero",
                    ));
                }
                let notify = matches!(op, Operator::MemoryAtomicNotify { .. });
                let offset = u32::try_from(memarg.offset)
                    .map_err(|_| Error::UserError("offset outside wasm32"))?;
                if offset == 0 {
                    return Ok(Instruction::Call(self.imports + u32::from(notify)));
                }
                let position = self
                    .wrappers
                    .iter()
                    .position(|&key| key == (offset, notify))
                    .ok_or(Error::UserError("missing offset wrapper"))?;
                Ok(Instruction::Call(
                    self.imports + 2 + self.defined + position as u32,
                ))
            }
            Operator::MemoryAtomicWait64 { .. } => {
                Err(Error::UserError("wait64 is outside the scheduler ABI"))
            }
            _ => utils::instruction(self, op),
        }
    }
}
fn translate(bytes: &[u8], namespace: &'static str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    wasmparser::Validator::new().validate_all(bytes)?;
    let (mut imports, mut types, mut waits, mut defined) = (0u32, 0u32, 0u32, 0u32);
    let mut wrappers = Vec::new();
    for p in Parser::new(0).parse_all(bytes) {
        match p? {
            Payload::TypeSection(s) => {
                for group in s {
                    types = types
                        .checked_add(group?.types().len().try_into()?)
                        .ok_or("type count overflow")?;
                }
            }
            Payload::ImportSection(s) => {
                for import in s.into_imports() {
                    if matches!(import?.ty, TypeRef::Func(_) | TypeRef::FuncExact(_)) {
                        imports = imports.checked_add(1).ok_or("import count overflow")?;
                    }
                }
            }
            Payload::FunctionSection(s) => defined = s.count(),
            Payload::CodeSectionEntry(body) => {
                for op in body.get_operators_reader()? {
                    match op? {
                        op @ (Operator::MemoryAtomicWait32 { memarg }
                        | Operator::MemoryAtomicNotify { memarg }) => {
                            waits += 1;
                            let key = (
                                u32::try_from(memarg.offset)?,
                                matches!(op, Operator::MemoryAtomicNotify { .. }),
                            );
                            if key.0 != 0 && !wrappers.contains(&key) {
                                wrappers.push(key);
                            }
                        }
                        Operator::MemoryAtomicWait64 { .. } => {
                            return Err("wait64 is outside the scheduler ABI".into());
                        }
                        _ => {}
                    }
                }
            }
            Payload::CustomSection(s)
                if s.name() == "linking" || s.name().starts_with("reloc.") =>
            {
                return Err("translate only fully linked modules".into());
            }
            _ => {}
        }
    }
    if waits == 0 {
        return Ok(bytes.to_vec());
    }
    types.checked_add(2).ok_or("type index overflow")?;
    imports.checked_add(2).ok_or("function index overflow")?;
    let mut module = Module::new();
    Scheduler {
        imports,
        types,
        inserted_imports: false,
        defined,
        wrappers,
        namespace,
    }
    .parse_core_module(&mut module, Parser::new(0), bytes)
    .map_err(|error| format!("{error:?}"))?;
    let result = module.finish();
    wasmparser::Validator::new().validate_all(&result)?;
    Ok(result)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args_os().collect();
    if args.len() != 3 && args.len() != 4 {
        return Err(
            "usage: shellsim-wasm-scheduler INPUT OUTPUT [shellsim_threads_v1|shellsim_threads_v2]"
                .into(),
        );
    }
    let namespace = match args.get(3).map(|arg| arg.to_str()) {
        None | Some(Some("shellsim_threads_v1")) => "shellsim_threads_v1",
        Some(Some("shellsim_threads_v2")) => "shellsim_threads_v2",
        _ => return Err("unsupported scheduler namespace".into()),
    };
    fs::write(&args[2], translate(&fs::read(&args[1])?, namespace)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module(wat: &str) -> Vec<u8> {
        wat::parse_str(wat).unwrap()
    }

    #[test]
    fn preserve_modules_without_waits() {
        let bytes = module("(module (func (export \"value\") (result i32) i32.const 7))");
        assert_eq!(translate(&bytes, "shellsim_threads_v2").unwrap(), bytes);
    }

    #[test]
    fn translate_waits_and_remap_defined_function_references() {
        let bytes = module(
            r#"(module
            (import "env" "memory" (memory 1 1 shared))
            (import "env" "existing" (func))
            (table 1 funcref)
            (elem (i32.const 0) $wait)
            (func $wait (export "wait") (param i32 i32 i64) (result i32)
                local.get 0 local.get 1 local.get 2 memory.atomic.wait32)
            (func (export "caller") (param i32 i32 i64) (result i32)
                call 0 local.get 0 local.get 1 local.get 2 call $wait))"#,
        );
        let output = translate(&bytes, "shellsim_threads_v2").unwrap();
        let mut calls = Vec::new();
        let mut export = None;
        let mut elements = Vec::new();
        let mut imports = Vec::new();
        for payload in Parser::new(0).parse_all(&output) {
            match payload.unwrap() {
                Payload::ImportSection(section) => {
                    for entry in section.into_imports() {
                        let entry = entry.unwrap();
                        imports.push((entry.module.to_owned(), entry.name.to_owned()));
                    }
                }
                Payload::ExportSection(section) => {
                    for entry in section {
                        let entry = entry.unwrap();
                        if entry.name == "wait" {
                            export = Some(entry.index);
                        }
                    }
                }
                Payload::ElementSection(section) => {
                    for entry in section {
                        if let wasmparser::ElementItems::Functions(items) = entry.unwrap().items {
                            elements.extend(items.into_iter().map(Result::unwrap));
                        }
                    }
                }
                Payload::CodeSectionEntry(body) => {
                    for op in body.get_operators_reader().unwrap() {
                        match op.unwrap() {
                            Operator::Call { function_index } => calls.push(function_index),
                            Operator::MemoryAtomicWait32 { .. }
                            | Operator::MemoryAtomicNotify { .. } => {
                                panic!("raw scheduler operation remains")
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        assert_eq!(export, Some(3));
        assert_eq!(elements, vec![3]);
        assert_eq!(calls, vec![1, 0, 3]);
        assert!(imports.contains(&("shellsim_threads_v2".into(), "wait32".into())));
        assert!(imports.contains(&("shellsim_threads_v2".into(), "notify".into())));
    }

    #[test]
    fn offset_wrappers_guard_address_overflow() {
        let bytes = module(
            r#"(module (memory 1 1 shared)
            (func (param i32 i32 i64) (result i32)
                local.get 0 local.get 1 local.get 2 memory.atomic.wait32 offset=8)
            (func (param i32 i32) (result i32)
                local.get 0 local.get 1 memory.atomic.notify offset=8))"#,
        );
        let output = translate(&bytes, "shellsim_threads_v1").unwrap();
        let bodies: Vec<_> = Parser::new(0)
            .parse_all(&output)
            .filter_map(|p| match p.unwrap() {
                Payload::CodeSectionEntry(body) => Some(body),
                _ => None,
            })
            .collect();
        assert_eq!(bodies.len(), 4);
        for body in &bodies[2..] {
            let operators: Vec<_> = body
                .get_operators_reader()
                .unwrap()
                .into_iter()
                .map(Result::unwrap)
                .collect();
            assert!(operators.iter().any(|op| matches!(op, Operator::I64GtU)));
            assert!(
                operators
                    .iter()
                    .any(|op| matches!(op, Operator::Unreachable))
            );
            assert!(
                operators.iter().any(
                    |op| matches!(op,Operator::I64Const{value} if *value==i64::from(u32::MAX))
                )
            );
        }
    }

    #[test]
    fn reject_wait64_and_unlinked_modules() {
        let bytes = module(
            "(module (memory 1 1 shared) (func (param i32 i64 i64) (result i32) local.get 0 local.get 1 local.get 2 memory.atomic.wait64))",
        );
        assert!(translate(&bytes, "shellsim_threads_v1").is_err());
        let mut bytes = module("(module)");
        let mut module = Module::new();
        module.section(&wasm_encoder::CustomSection {
            name: "linking".into(),
            data: Vec::new().into(),
        });
        bytes.extend_from_slice(&module.finish()[8..]);
        assert!(translate(&bytes, "shellsim_threads_v1").is_err());
        assert!(translate(b"invalid", "shellsim_threads_v1").is_err());
    }
}
