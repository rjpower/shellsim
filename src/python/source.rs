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

/// Host bytes the front end holds per source byte while a program is lexed, parsed and
/// compiled. Tokens, syntax tree nodes and pending instructions are each far larger than the
/// text they come from, so the reservation is made from the source length before any of them
/// exist; a token count refines it once lexing is done.
pub const FRONT_END_BYTES_PER_SOURCE_BYTE: usize = 64;

/// Host bytes the parser and compiler hold per token beyond the source-length reservation.
pub const FRONT_END_BYTES_PER_TOKEN: usize = 192;

/// Memory to reserve before lexing `source_len` bytes, or `None` when the product overflows.
pub fn front_end_memory(source_len: usize) -> Option<usize> {
    source_len.checked_mul(FRONT_END_BYTES_PER_SOURCE_BYTE)
}

/// Memory to reserve for parsing and compiling `tokens` tokens, or `None` on overflow.
pub fn token_memory(tokens: usize) -> Option<usize> {
    tokens.checked_mul(FRONT_END_BYTES_PER_TOKEN)
}
