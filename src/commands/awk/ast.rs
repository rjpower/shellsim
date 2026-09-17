//! Typed syntax tree for the supported awk language slice.

#[derive(Clone, Debug)]
pub(super) struct Program {
    pub rules: Vec<Rule>,
}

#[derive(Clone, Debug)]
pub(super) struct Rule {
    pub pattern: Pattern,
    pub body: Vec<Stmt>,
}

#[derive(Clone, Debug)]
pub(super) enum Pattern {
    Begin,
    End,
    Always,
    Expr(Expr),
}

#[derive(Clone, Debug)]
pub(super) enum Stmt {
    Block(Vec<Stmt>),
    If {
        condition: Expr,
        then_branch: Box<Stmt>,
        else_branch: Option<Box<Stmt>>,
    },
    While {
        condition: Expr,
        body: Box<Stmt>,
    },
    For {
        init: Option<Expr>,
        condition: Option<Expr>,
        update: Option<Expr>,
        body: Box<Stmt>,
    },
    ForIn {
        name: String,
        array: String,
        body: Box<Stmt>,
    },
    Break,
    Continue,
    Delete(LValue),
    Next,
    NextFile,
    Exit(Option<Expr>),
    Print(Vec<Expr>),
    Printf(Vec<Expr>),
    Expr(Expr),
}

#[derive(Clone, Debug)]
pub(super) enum Expr {
    String(String),
    Number(f64),
    Regex(String),
    Variable(String),
    Field(Box<Expr>),
    Array {
        name: String,
        indices: Vec<Expr>,
    },
    Call {
        name: String,
        args: Vec<Expr>,
    },
    Assign {
        target: LValue,
        op: AssignOp,
        value: Box<Expr>,
    },
    Binary {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },
    Unary {
        op: UnaryOp,
        value: Box<Expr>,
    },
    Increment {
        target: LValue,
        delta: i8,
        prefix: bool,
    },
}

#[derive(Clone, Debug)]
pub(super) enum LValue {
    Variable(String),
    Field(Box<Expr>),
    Array { name: String, indices: Vec<Expr> },
}

#[derive(Clone, Copy, Debug)]
pub(super) enum AssignOp {
    Set,
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum BinaryOp {
    Or,
    And,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Match,
    NotMatch,
    In,
    Concat,
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum UnaryOp {
    Not,
    Positive,
    Negative,
}

impl Expr {
    pub fn into_lvalue(self) -> Option<LValue> {
        match self {
            Self::Variable(name) => Some(LValue::Variable(name)),
            Self::Field(index) => Some(LValue::Field(index)),
            Self::Array { name, indices } => Some(LValue::Array { name, indices }),
            _ => None,
        }
    }
}
