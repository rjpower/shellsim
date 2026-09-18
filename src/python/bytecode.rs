//! Stable shellsim bytecode. This is semantic bytecode, not CPython's release-specific format.

use super::ast::{BinaryOperator, ComparisonOperator, Constant, UnaryOperator};
use super::source::Span;
use std::sync::Arc;

/// Shared executable code. Functions, generators, and active frames retain this handle instead of
/// copying an immutable instruction stream.
pub type CodeRef = Arc<Code>;

#[derive(Clone, Debug, PartialEq)]
pub struct Code {
    pub instructions: Arc<[Instruction]>,
    pub parameters: Arc<[Parameter]>,
    /// Stable slot names for locals owned by this code object.
    pub local_names: Arc<[String]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub has_default: bool,
    pub variadic: bool,
    pub keyword_only: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClassField {
    pub name: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Instruction {
    pub operation: Operation,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Operation {
    LoadConstant(Constant),
    LoadName(String),
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
    /// Build a dictionary from source-ordered entries. `true` consumes one
    /// mapping while `false` consumes one key/value pair.
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
        keywords: Vec<String>,
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
    /// Match the active exception. A typed handler leaves its exception-type
    /// expression on the stack immediately above the raised value.
    MatchException {
        typed: bool,
    },
    ClearException,
    Reraise,
    Raise(bool),
    /// Suspend a generator frame and return the value on top of the stack.
    Yield,
    WithEnter,
    WithExit,
    WithExitException,
    PopExpression,
    Halt,
}
