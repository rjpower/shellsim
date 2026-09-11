//! Stable shellsim bytecode. This is semantic bytecode, not CPython's release-specific format.

use super::ast::{BinaryOperator, ComparisonOperator, Constant, UnaryOperator};
use super::source::Span;

#[derive(Clone, Debug, PartialEq)]
pub struct Code {
    pub instructions: Vec<Instruction>,
    pub parameters: Vec<Parameter>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub has_default: bool,
    pub variadic: bool,
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
    StoreNonlocal(String),
    StoreAttribute(String),
    StoreSubscript,
    DeleteName(String),
    Import(String),
    LoadAttribute(String),
    LoadSubscript,
    LoadSlice {
        has_start: bool,
        has_stop: bool,
        has_step: bool,
    },
    BuildList(usize),
    BuildTuple(usize),
    BuildDict(usize),
    BuildSet(usize),
    UnpackSequence {
        count: usize,
        star_index: Option<usize>,
    },
    MakeFunction {
        name: String,
        code: Box<Code>,
        defaults: usize,
    },
    MakeClass {
        name: String,
        code: Box<Code>,
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
    MatchException(Option<String>),
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
