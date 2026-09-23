//! Stable shellsim bytecode.
//!
//! The compiler works with descriptive [`Operation`] values. [`Code`] lowers them to compact,
//! copyable [`Opcode`] values plus immutable side tables. The VM therefore dispatches without
//! cloning data-bearing operations or retaining borrows across runtime mutation.

use super::ast::{BinaryOperator, ComparisonOperator, Constant, UnaryOperator};
use super::source::Span;
use std::collections::HashMap;
use std::sync::Arc;

/// Shared immutable executable code.
///
/// The dispatcher retains one handle for a whole execution quantum and refreshes it only when the
/// active frame changes. Frames and callable objects therefore remain safely owned without
/// reference-count traffic on each opcode.
pub type CodeRef = Arc<Code>;

macro_rules! side_table_id {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name(u32);

        impl $name {
            pub(super) fn new(index: usize) -> Self {
                Self(u32::try_from(index).expect("bounded source produced an oversized side table"))
            }

            pub(super) fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

side_table_id!(NameId);
side_table_id!(ConstantId);
side_table_id!(CallId);
side_table_id!(FunctionId);
side_table_id!(ClassId);
side_table_id!(FormatId);
side_table_id!(DictId);
side_table_id!(ErrorId);

#[derive(Clone, Debug, PartialEq)]
pub struct Code {
    pub instructions: Box<[Instruction]>,
    pub spans: Box<[Span]>,
    pub parameters: Box<[Parameter]>,
    /// Call-shape facts derived once by the compiler and shared by every invocation.
    pub call_signature: CallSignature,
    /// Stable slot names for locals owned by this code object.
    pub local_names: Arc<[String]>,
    names: Box<[Arc<str>]>,
    constants: Box<[Constant]>,
    calls: Box<[CallSpec]>,
    functions: Box<[FunctionSpec]>,
    classes: Box<[ClassSpec]>,
    formats: Box<[FormatSpec]>,
    dicts: Box<[Box<[bool]>]>,
    errors: Box<[String]>,
}

/// Immutable argument-binding metadata for one code object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallSignature {
    pub is_generator: bool,
    pub is_coroutine: bool,
    pub positional_count: usize,
    pub variadic_slot: Option<usize>,
    pub keyword_variadic_slot: Option<usize>,
    pub default_slots: Box<[usize]>,
}

impl Code {
    pub fn name_count(&self) -> usize {
        self.names.len()
    }

    pub fn name(&self, id: NameId) -> &str {
        &self.names[id.index()]
    }

    pub fn constant(&self, id: ConstantId) -> &Constant {
        &self.constants[id.index()]
    }

    pub fn call(&self, id: CallId) -> &CallSpec {
        &self.calls[id.index()]
    }

    pub fn function(&self, id: FunctionId) -> &FunctionSpec {
        &self.functions[id.index()]
    }

    pub fn class(&self, id: ClassId) -> &ClassSpec {
        &self.classes[id.index()]
    }

    pub fn format(&self, id: FormatId) -> &FormatSpec {
        &self.formats[id.index()]
    }

    pub fn dict_entries(&self, id: DictId) -> &[bool] {
        &self.dicts[id.index()]
    }

    pub fn error(&self, id: ErrorId) -> &str {
        &self.errors[id.index()]
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub has_default: bool,
    pub kind: ParameterKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParameterKind {
    Positional,
    Variadic,
    KeywordOnly,
    KeywordVariadic,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClassField {
    pub name: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Instruction {
    pub opcode: Opcode,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CallSpec {
    pub positional: usize,
    pub keywords: Box<[Option<NameId>]>,
    pub starred: Box<[bool]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunctionSpec {
    pub name: NameId,
    pub code: CodeRef,
    pub defaults: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClassSpec {
    pub name: NameId,
    pub code: CodeRef,
    pub bases: usize,
    pub has_metaclass: bool,
    pub fields: Box<[ClassField]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FormatSpec {
    pub conversion: Option<char>,
    pub format_spec: String,
}

/// Fixed-width executable operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Opcode {
    LoadConstant(ConstantId),
    LoadName(NameId),
    LoadGlobal(NameId),
    StoreName(NameId),
    LoadLocal(usize),
    StoreLocal(usize),
    StoreEnclosing {
        name: NameId,
        scope_hops: usize,
    },
    StoreNonlocal(NameId),
    StoreGlobal(NameId),
    StoreAttribute(NameId),
    StoreSubscript,
    DeleteName(NameId),
    DeleteLocal(usize),
    DeleteGlobal(NameId),
    DeleteSubscript,
    Import {
        name: NameId,
        bind_root: bool,
    },
    LoadAttribute(NameId),
    LoadSubscript,
    BuildSlice {
        has_start: bool,
        has_stop: bool,
        has_step: bool,
    },
    BuildList(usize),
    BuildTuple(usize),
    BuildDict(DictId),
    BuildSet(usize),
    UnpackSequence {
        count: usize,
        star_index: Option<usize>,
    },
    MakeFunction(FunctionId),
    MakeClass(ClassId),
    GetIterator,
    ForIterator(usize),
    Unary(UnaryOperator),
    Binary(BinaryOperator),
    FormatValue(FormatId),
    Compare(ComparisonOperator),
    Call(CallId),
    Copy(usize),
    Swap(usize),
    PopTop,
    Jump(usize),
    JumpIfFalseOrPop(usize),
    JumpIfTrueOrPop(usize),
    PopJumpIfFalse(usize),
    Return,
    RuntimeError(ErrorId),
    Assert,
    TryBegin(usize),
    TryEnd,
    MatchException {
        typed: bool,
    },
    ClearException,
    Reraise,
    Raise(bool),
    Yield,
    AwaitResult,
    WithEnter,
    WithExit,
    WithExitException,
    AsyncWithExitException,
    AsyncWithFinishException,
    PopExpression,
    Halt,
}

/// Descriptive compiler operation lowered to [`Opcode`] when a code object is finished.
#[derive(Clone, Debug, PartialEq)]
pub enum Operation {
    LoadConstant(Constant),
    LoadName(String),
    LoadGlobal(String),
    StoreName(String),
    LoadLocal(usize),
    StoreLocal(usize),
    /// Store a named-expression result outside synthetic comprehension scopes.
    StoreEnclosing {
        name: String,
        scope_hops: usize,
    },
    StoreNonlocal(String),
    StoreGlobal(String),
    StoreAttribute(String),
    StoreSubscript,
    DeleteName(String),
    DeleteLocal(usize),
    DeleteGlobal(String),
    DeleteSubscript,
    Import {
        name: String,
        bind_root: bool,
    },
    LoadAttribute(String),
    LoadSubscript,
    BuildSlice {
        has_start: bool,
        has_stop: bool,
        has_step: bool,
    },
    BuildList(usize),
    BuildTuple(usize),
    /// Build a dictionary from source-ordered entries. `true` consumes one mapping while `false`
    /// consumes one key/value pair.
    BuildDict(Vec<bool>),
    BuildSet(usize),
    UnpackSequence {
        count: usize,
        star_index: Option<usize>,
    },
    MakeFunction {
        name: String,
        code: CodeRef,
        defaults: usize,
    },
    MakeClass {
        name: String,
        code: CodeRef,
        bases: usize,
        has_metaclass: bool,
        fields: Vec<ClassField>,
    },
    GetIterator,
    ForIterator(usize),
    Unary(UnaryOperator),
    Binary(BinaryOperator),
    FormatValue {
        conversion: Option<char>,
        format_spec: String,
    },
    Compare(ComparisonOperator),
    Call {
        positional: usize,
        keywords: Vec<Option<String>>,
        starred: Vec<bool>,
    },
    Copy(usize),
    Swap(usize),
    PopTop,
    Jump(usize),
    JumpIfFalseOrPop(usize),
    JumpIfTrueOrPop(usize),
    PopJumpIfFalse(usize),
    Return,
    RuntimeError(String),
    Assert,
    TryBegin(usize),
    TryEnd,
    /// Match the active exception. A typed handler leaves its exception-type expression on the
    /// stack immediately above the raised value.
    MatchException {
        typed: bool,
    },
    ClearException,
    Reraise,
    Raise(bool),
    /// Suspend a generator frame and return the value on top of the stack.
    Yield,
    /// Unwrap the scheduler outcome sent into a suspended coroutine.
    AwaitResult,
    WithEnter,
    WithExit,
    WithExitException,
    AsyncWithExitException,
    AsyncWithFinishException,
    PopExpression,
    Halt,
}

#[derive(Default)]
pub struct CodeBuilder {
    names: Vec<Arc<str>>,
    name_ids: HashMap<Arc<str>, NameId>,
    constants: Vec<Constant>,
    calls: Vec<CallSpec>,
    functions: Vec<FunctionSpec>,
    classes: Vec<ClassSpec>,
    formats: Vec<FormatSpec>,
    dicts: Vec<Box<[bool]>>,
    errors: Vec<String>,
}

impl CodeBuilder {
    fn name(&mut self, name: String) -> NameId {
        if let Some(id) = self.name_ids.get(name.as_str()) {
            return *id;
        }
        let id = NameId::new(self.names.len());
        let name: Arc<str> = name.into();
        self.name_ids.insert(name.clone(), id);
        self.names.push(name);
        id
    }

    pub fn lower(&mut self, operation: Operation) -> Opcode {
        match operation {
            Operation::LoadConstant(value) => {
                let id = ConstantId::new(self.constants.len());
                self.constants.push(value);
                Opcode::LoadConstant(id)
            }
            Operation::LoadName(name) => Opcode::LoadName(self.name(name)),
            Operation::LoadGlobal(name) => Opcode::LoadGlobal(self.name(name)),
            Operation::StoreName(name) => Opcode::StoreName(self.name(name)),
            Operation::LoadLocal(slot) => Opcode::LoadLocal(slot),
            Operation::StoreLocal(slot) => Opcode::StoreLocal(slot),
            Operation::StoreEnclosing { name, scope_hops } => Opcode::StoreEnclosing {
                name: self.name(name),
                scope_hops,
            },
            Operation::StoreNonlocal(name) => Opcode::StoreNonlocal(self.name(name)),
            Operation::StoreGlobal(name) => Opcode::StoreGlobal(self.name(name)),
            Operation::StoreAttribute(name) => Opcode::StoreAttribute(self.name(name)),
            Operation::StoreSubscript => Opcode::StoreSubscript,
            Operation::DeleteName(name) => Opcode::DeleteName(self.name(name)),
            Operation::DeleteLocal(slot) => Opcode::DeleteLocal(slot),
            Operation::DeleteGlobal(name) => Opcode::DeleteGlobal(self.name(name)),
            Operation::DeleteSubscript => Opcode::DeleteSubscript,
            Operation::Import { name, bind_root } => Opcode::Import {
                name: self.name(name),
                bind_root,
            },
            Operation::LoadAttribute(name) => Opcode::LoadAttribute(self.name(name)),
            Operation::LoadSubscript => Opcode::LoadSubscript,
            Operation::BuildSlice {
                has_start,
                has_stop,
                has_step,
            } => Opcode::BuildSlice {
                has_start,
                has_stop,
                has_step,
            },
            Operation::BuildList(count) => Opcode::BuildList(count),
            Operation::BuildTuple(count) => Opcode::BuildTuple(count),
            Operation::BuildDict(entries) => {
                let id = DictId::new(self.dicts.len());
                self.dicts.push(entries.into_boxed_slice());
                Opcode::BuildDict(id)
            }
            Operation::BuildSet(count) => Opcode::BuildSet(count),
            Operation::UnpackSequence { count, star_index } => {
                Opcode::UnpackSequence { count, star_index }
            }
            Operation::MakeFunction {
                name,
                code,
                defaults,
            } => {
                let name = self.name(name);
                let id = FunctionId::new(self.functions.len());
                self.functions.push(FunctionSpec {
                    name,
                    code,
                    defaults,
                });
                Opcode::MakeFunction(id)
            }
            Operation::MakeClass {
                name,
                code,
                bases,
                has_metaclass,
                fields,
            } => {
                let name = self.name(name);
                let id = ClassId::new(self.classes.len());
                self.classes.push(ClassSpec {
                    name,
                    code,
                    bases,
                    has_metaclass,
                    fields: fields.into_boxed_slice(),
                });
                Opcode::MakeClass(id)
            }
            Operation::GetIterator => Opcode::GetIterator,
            Operation::ForIterator(target) => Opcode::ForIterator(target),
            Operation::Unary(operator) => Opcode::Unary(operator),
            Operation::Binary(operator) => Opcode::Binary(operator),
            Operation::FormatValue {
                conversion,
                format_spec,
            } => {
                let id = FormatId::new(self.formats.len());
                self.formats.push(FormatSpec {
                    conversion,
                    format_spec,
                });
                Opcode::FormatValue(id)
            }
            Operation::Compare(operator) => Opcode::Compare(operator),
            Operation::Call {
                positional,
                keywords,
                starred,
            } => {
                let keywords = keywords
                    .into_iter()
                    .map(|keyword| keyword.map(|keyword| self.name(keyword)))
                    .collect();
                let id = CallId::new(self.calls.len());
                self.calls.push(CallSpec {
                    positional,
                    keywords,
                    starred: starred.into_boxed_slice(),
                });
                Opcode::Call(id)
            }
            Operation::Copy(depth) => Opcode::Copy(depth),
            Operation::Swap(depth) => Opcode::Swap(depth),
            Operation::PopTop => Opcode::PopTop,
            Operation::Jump(target) => Opcode::Jump(target),
            Operation::JumpIfFalseOrPop(target) => Opcode::JumpIfFalseOrPop(target),
            Operation::JumpIfTrueOrPop(target) => Opcode::JumpIfTrueOrPop(target),
            Operation::PopJumpIfFalse(target) => Opcode::PopJumpIfFalse(target),
            Operation::Return => Opcode::Return,
            Operation::RuntimeError(error) => {
                let id = ErrorId::new(self.errors.len());
                self.errors.push(error);
                Opcode::RuntimeError(id)
            }
            Operation::Assert => Opcode::Assert,
            Operation::TryBegin(target) => Opcode::TryBegin(target),
            Operation::TryEnd => Opcode::TryEnd,
            Operation::MatchException { typed } => Opcode::MatchException { typed },
            Operation::ClearException => Opcode::ClearException,
            Operation::Reraise => Opcode::Reraise,
            Operation::Raise(cause) => Opcode::Raise(cause),
            Operation::Yield => Opcode::Yield,
            Operation::AwaitResult => Opcode::AwaitResult,
            Operation::WithEnter => Opcode::WithEnter,
            Operation::WithExit => Opcode::WithExit,
            Operation::WithExitException => Opcode::WithExitException,
            Operation::AsyncWithExitException => Opcode::AsyncWithExitException,
            Operation::AsyncWithFinishException => Opcode::AsyncWithFinishException,
            Operation::PopExpression => Opcode::PopExpression,
            Operation::Halt => Opcode::Halt,
        }
    }

    pub fn finish(
        self,
        instructions: Vec<Instruction>,
        spans: Vec<Span>,
        parameters: Vec<Parameter>,
        local_names: Vec<String>,
        is_coroutine: bool,
    ) -> CodeRef {
        let positional_count = parameters
            .iter()
            .take_while(|parameter| parameter.kind == ParameterKind::Positional)
            .count();
        let call_signature = CallSignature {
            is_generator: instructions
                .iter()
                .any(|instruction| matches!(instruction.opcode, Opcode::Yield)),
            is_coroutine,
            positional_count,
            variadic_slot: parameters
                .iter()
                .position(|parameter| parameter.kind == ParameterKind::Variadic),
            keyword_variadic_slot: parameters
                .iter()
                .position(|parameter| parameter.kind == ParameterKind::KeywordVariadic),
            default_slots: parameters
                .iter()
                .enumerate()
                .filter_map(|(slot, parameter)| parameter.has_default.then_some(slot))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        };
        Arc::new(Code {
            instructions: instructions.into_boxed_slice(),
            spans: spans.into_boxed_slice(),
            parameters: parameters.into_boxed_slice(),
            call_signature,
            local_names: local_names.into(),
            names: self.names.into_boxed_slice(),
            constants: self.constants.into_boxed_slice(),
            calls: self.calls.into_boxed_slice(),
            functions: self.functions.into_boxed_slice(),
            classes: self.classes.into_boxed_slice(),
            formats: self.formats.into_boxed_slice(),
            dicts: self.dicts.into_boxed_slice(),
            errors: self.errors.into_boxed_slice(),
        })
    }
}
