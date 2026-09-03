//! Source locations used by every front-end stage.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub line: usize,
    pub column: usize,
}

impl Span {
    pub const fn new(start: usize, end: usize, line: usize, column: usize) -> Self {
        Self {
            start,
            end,
            line,
            column,
        }
    }

    pub const fn through(self, end: Self) -> Self {
        Self {
            start: self.start,
            end: end.end,
            line: self.line,
            column: self.column,
        }
    }
}
