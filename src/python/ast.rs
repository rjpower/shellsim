//! Minimal AST. Spans survive parsing so later diagnostics never need to re-scan source text.

use super::source::Span;

#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    pub statements: Vec<Statement>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Statement {
    pub kind: StatementKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum StatementKind {
    Import {
        module: String,
        binding: String,
    },
    ImportFrom {
        module: String,
        names: Vec<(String, String)>,
    },
    Assign {
        target: AssignmentTarget,
        value: Expression,
    },
    AugmentedAssign {
        target: AssignmentTarget,
        operator: BinaryOperator,
        value: Expression,
    },
    Delete(String),
    Expression(Expression),
    If {
        test: Expression,
        body: Vec<Statement>,
        otherwise: Vec<Statement>,
    },
    While {
        test: Expression,
        body: Vec<Statement>,
        otherwise: Vec<Statement>,
    },
    For {
        target: AssignmentTarget,
        iterable: Expression,
        body: Vec<Statement>,
        otherwise: Vec<Statement>,
    },
    Function {
        name: String,
        parameters: Vec<Parameter>,
        body: Vec<Statement>,
    },
    Class {
        name: String,
        bases: Vec<Expression>,
        metaclass: Option<Expression>,
        body: Vec<Statement>,
    },
    Decorated {
        decorators: Vec<Expression>,
        statement: Box<StatementKind>,
    },
    AnnotatedAssign {
        target: AssignmentTarget,
        value: Option<Expression>,
    },
    Return(Option<Expression>),
    Break,
    Continue,
    Nonlocal(Vec<String>),
    Pass,
    Assert {
        test: Expression,
        message: Option<Expression>,
    },
    Try {
        body: Vec<Statement>,
        handlers: Vec<ExceptHandler>,
        otherwise: Vec<Statement>,
        finalbody: Vec<Statement>,
    },
    Raise(Option<Expression>),
    With {
        context: Expression,
        target: Option<AssignmentTarget>,
        body: Vec<Statement>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExceptHandler {
    pub kind: Option<Expression>,
    pub name: Option<String>,
    pub body: Vec<Statement>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AssignmentTarget {
    Name(String),
    Star(Box<AssignmentTarget>),
    Sequence(Vec<AssignmentTarget>),
    Attribute {
        value: Expression,
        name: String,
    },
    Subscript {
        value: Expression,
        index: Expression,
    },
}

impl StatementKind {
    pub fn is_compound(&self) -> bool {
        matches!(
            self,
            Self::If { .. }
                | Self::While { .. }
                | Self::For { .. }
                | Self::Function { .. }
                | Self::Class { .. }
                | Self::Decorated { .. }
                | Self::Try { .. }
                | Self::With { .. }
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Expression {
    pub kind: ExpressionKind,
    pub span: Span,
}

/// One ``for`` clause in a comprehension.  Keeping clauses in source order makes
/// the compiler's nesting (and therefore evaluation order) explicit.
#[derive(Clone, Debug, PartialEq)]
pub struct ComprehensionClause {
    pub target: AssignmentTarget,
    pub iterable: Expression,
    pub conditions: Vec<Expression>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExpressionKind {
    Constant(Constant),
    FString(Vec<FStringPart>),
    Starred(Box<Expression>),
    Name(String),
    List(Vec<Expression>),
    Tuple(Vec<Expression>),
    Dict(Vec<(Expression, Expression)>),
    Set(Vec<Expression>),
    ListComprehension {
        element: Box<Expression>,
        clauses: Vec<ComprehensionClause>,
    },
    SetComprehension {
        element: Box<Expression>,
        clauses: Vec<ComprehensionClause>,
    },
    DictComprehension {
        key: Box<Expression>,
        value: Box<Expression>,
        clauses: Vec<ComprehensionClause>,
    },
    /// The VM deliberately materializes this as a list for now.  This gives
    /// generator expressions an ordinary iterable and isolated comprehension
    /// scope while keeping the bounded-resource contract simple and explicit.
    GeneratorExpression {
        element: Box<Expression>,
        clauses: Vec<ComprehensionClause>,
    },
    Yield(Option<Box<Expression>>),
    Attribute {
        value: Box<Expression>,
        name: String,
    },
    Subscript {
        value: Box<Expression>,
        index: Box<Expression>,
    },
    Call {
        function: Box<Expression>,
        arguments: Vec<CallArgument>,
    },
    Lambda {
        parameters: Vec<Parameter>,
        body: Box<Expression>,
    },
    Unary {
        operator: UnaryOperator,
        operand: Box<Expression>,
    },
    Binary {
        left: Box<Expression>,
        operator: BinaryOperator,
        right: Box<Expression>,
    },
    Boolean {
        left: Box<Expression>,
        operator: BooleanOperator,
        right: Box<Expression>,
    },
    Comparison {
        left: Box<Expression>,
        comparisons: Vec<(ComparisonOperator, Expression)>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct CallArgument {
    pub name: Option<String>,
    pub value: Expression,
    pub starred: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FStringPart {
    Text(String),
    Expression(Expression),
}

/// A formal parameter and its optional definition-time default expression.
///
/// Defaults remain in the AST until compilation so the enclosing scope can
/// evaluate them immediately before creating the function object.  This is
/// important for Python's definition-time semantics (and for closures).
#[derive(Clone, Debug, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub default: Option<Expression>,
    pub variadic: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Constant {
    None,
    Bool(bool),
    Integer(i64),
    BigInteger(String),
    Float(f64),
    String(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOperator {
    Positive,
    Negative,
    Not,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BooleanOperator {
    And,
    Or,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComparisonOperator {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    In,
    NotIn,
    Is,
    IsNot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    FloorDivide,
    Remainder,
}
