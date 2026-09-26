//! Bash shell arithmetic: `$(( ))`, `(( ))`, `let`, arithmetic `for`, and slice offsets.
//!
//! The evaluator follows Bash's grammar and semantics. That covers the full operator precedence
//! table, right-associative `**`, `?:`, and assignments, and the comma operator. Integers are
//! 64-bit with wraparound, and constants may be decimal, octal (`010`), hexadecimal (`0x1f`),
//! or `base#digits` for bases 2 through 64. A variable whose value is not a plain integer is
//! evaluated as an expression in turn, up to a recursion limit. `&&`, `||`, and `?:` evaluate
//! only the operand they select, so an assignment in a skipped operand has no effect.
//!
//! Errors carry Bash's wording and its "error token", which is the rest of the expression from
//! the last token read. Callers decide whether an error aborts the shell (`$(( ))`) or only
//! fails the command (`(( ))` and `let`).
//!
//! Variable access goes through [`ArithVars`], so this module has no dependency on the
//! interpreter.
//!
//! ```text
//! evaluate(vars, "x = 2 ** 3, x << 1")  => Ok(16), with x set to 8
//! evaluate(vars, "1 / 0")               => Err("division by 0 (error token is \"0\")")
//! ```

/// Variable storage used by arithmetic evaluation.
pub(crate) trait ArithVars {
    /// Value of a scalar variable, or of element 0 of an array. `Ok(None)` means unset. An
    /// `Err` carries a diagnostic such as an unbound-variable error under `set -u`.
    fn arith_get(&mut self, name: &str) -> Result<Option<String>, String>;
    /// Value of an indexed array element.
    fn arith_get_element(&mut self, name: &str, index: i64) -> Result<Option<String>, String>;
    /// Assign a scalar variable. An `Err` carries a diagnostic such as a readonly violation.
    fn arith_set(&mut self, name: &str, value: i64) -> Result<(), String>;
    /// Assign an indexed array element.
    fn arith_set_element(&mut self, name: &str, index: i64, value: i64) -> Result<(), String>;
}

/// A failed arithmetic evaluation, formatted the way Bash reports it after the expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ArithError {
    pub(crate) message: String,
    /// Variable errors (readonly, unbound) are reported without the expression, as in Bash.
    variable: bool,
}

impl ArithError {
    fn expression(message: String) -> Self {
        Self {
            message,
            variable: false,
        }
    }

    fn variable(message: String) -> Self {
        Self {
            message,
            variable: true,
        }
    }

    /// Bash's diagnostic body: `EXPR: MESSAGE`, where EXPR has its leading blanks removed, or
    /// just the message for a variable error.
    pub(crate) fn describe(&self, expression: &str) -> String {
        if self.variable {
            return self.message.clone();
        }
        format!("{}: {}", expression.trim_start(), self.message)
    }
}

/// Nesting limit for variables whose values are themselves expressions. It is lower than
/// Bash's, because each level recurses on the host stack.
const MAX_RECURSION: usize = 128;
/// Bound on the parser's syntactic nesting (parentheses and unary chains) for the same reason.
const MAX_NESTING: usize = 256;

/// Evaluate a complete arithmetic expression. An empty or all-blank expression is 0.
pub(crate) fn evaluate(vars: &mut dyn ArithVars, expression: &str) -> Result<i64, ArithError> {
    evaluate_at_depth(vars, expression, 0)
}

fn evaluate_at_depth(
    vars: &mut dyn ArithVars,
    expression: &str,
    depth: usize,
) -> Result<i64, ArithError> {
    let mut parser = Parser {
        vars,
        chars: expression.chars().collect(),
        position: 0,
        token_start: 0,
        depth,
        nesting: 0,
    };
    parser.skip_blanks();
    if parser.at_end() {
        return Ok(0);
    }
    let value = parser.comma(false)?.value;
    parser.skip_blanks();
    if !parser.at_end() {
        parser.token_start = parser.position;
        return Err(parser.error("arithmetic syntax error in expression"));
    }
    Ok(value)
}

/// An operand's value and, when it names a variable, the place an assignment would store to.
struct Operand {
    value: i64,
    place: Option<Place>,
}

impl Operand {
    fn rvalue(value: i64) -> Self {
        Self { value, place: None }
    }
}

#[derive(Clone)]
enum Place {
    Scalar(String),
    Element(String, i64),
}

struct Parser<'a> {
    vars: &'a mut dyn ArithVars,
    chars: Vec<char>,
    position: usize,
    /// Start of the most recently read token, which Bash reports as the error token.
    token_start: usize,
    depth: usize,
    nesting: usize,
}

const ASSIGNMENT_OPERATORS: [&str; 11] = [
    "<<=", ">>=", "*=", "/=", "%=", "+=", "-=", "&=", "^=", "|=", "=",
];

impl Parser<'_> {
    fn at_end(&self) -> bool {
        self.position >= self.chars.len()
    }

    fn skip_blanks(&mut self) {
        while self
            .chars
            .get(self.position)
            .is_some_and(|c| c.is_whitespace())
        {
            self.position += 1;
        }
    }

    fn peek_char(&mut self) -> Option<char> {
        self.skip_blanks();
        self.chars.get(self.position).copied()
    }

    fn looking_at(&mut self, text: &str) -> bool {
        self.skip_blanks();
        text.chars()
            .enumerate()
            .all(|(offset, c)| self.chars.get(self.position + offset) == Some(&c))
    }

    /// Consume `text` as the next token when it is present.
    fn eat(&mut self, text: &str) -> bool {
        if !self.looking_at(text) {
            return false;
        }
        self.token_start = self.position;
        self.position += text.chars().count();
        true
    }

    /// The binary operator at the cursor, if it is one of `candidates` and is not the prefix of
    /// a longer operator (so `<` does not match `<<` or `<=`, and `&` does not match `&&`).
    fn binary_operator(&mut self, candidates: &[&'static str]) -> Option<&'static str> {
        self.skip_blanks();
        let longer = [
            "<<=", ">>=", "**", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||", "*=", "/=", "%=",
            "+=", "-=", "&=", "^=", "|=",
        ];
        for &candidate in candidates {
            if !self.looking_at(candidate) {
                continue;
            }
            let shadowed = longer.iter().any(|operator| {
                operator.len() > candidate.len()
                    && operator.starts_with(candidate)
                    && !candidates.contains(operator)
                    && self.looking_at(operator)
            });
            if !shadowed {
                return Some(candidate);
            }
        }
        None
    }

    fn error(&self, message: &str) -> ArithError {
        let token: String = self.chars[self.token_start.min(self.chars.len())..]
            .iter()
            .collect();
        ArithError::expression(format!("{message} (error token is \"{token}\")"))
    }

    fn nest(&mut self) -> Result<(), ArithError> {
        self.nesting += 1;
        if self.nesting > MAX_NESTING {
            return Err(self.error("expression nesting too deep"));
        }
        Ok(())
    }

    fn comma(&mut self, skip: bool) -> Result<Operand, ArithError> {
        let mut operand = self.assignment(skip)?;
        while self.eat(",") {
            operand = self.assignment(skip)?;
        }
        Ok(operand)
    }

    fn assignment(&mut self, skip: bool) -> Result<Operand, ArithError> {
        let target = self.conditional(skip)?;
        let Some(operator) = self.binary_operator(&ASSIGNMENT_OPERATORS) else {
            return Ok(target);
        };
        self.eat(operator);
        let Some(place) = target.place else {
            return Err(self.error("attempted assignment to non-variable"));
        };
        let right = self.assignment(skip)?.value;
        if skip {
            return Ok(Operand::rvalue(0));
        }
        let value = if operator == "=" {
            right
        } else {
            let binary = &operator[..operator.len() - 1];
            self.apply(binary, target.value, right)?
        };
        self.store(&place, value)?;
        Ok(Operand::rvalue(value))
    }

    fn conditional(&mut self, skip: bool) -> Result<Operand, ArithError> {
        let condition = self.logical_or(skip)?;
        if !self.eat("?") {
            return Ok(condition);
        }
        let take_first = condition.value != 0;
        self.nest()?;
        let first = self.comma(skip || !take_first)?;
        if !self.eat(":") {
            self.token_start = self.position;
            return Err(self.error("`:' expected for conditional expression"));
        }
        let second = self.conditional(skip || take_first)?;
        self.nesting -= 1;
        Ok(Operand::rvalue(if take_first {
            first.value
        } else {
            second.value
        }))
    }

    fn logical_or(&mut self, skip: bool) -> Result<Operand, ArithError> {
        let mut left = self.logical_and(skip)?;
        while self.eat("||") {
            let decided = left.value != 0;
            let right = self.logical_and(skip || decided)?;
            left = Operand::rvalue(i64::from(decided || right.value != 0));
        }
        Ok(left)
    }

    fn logical_and(&mut self, skip: bool) -> Result<Operand, ArithError> {
        let mut left = self.binary_level(0, skip)?;
        while self.eat("&&") {
            let decided = left.value == 0;
            let right = self.binary_level(0, skip || decided)?;
            left = Operand::rvalue(i64::from(!decided && right.value != 0));
        }
        Ok(left)
    }

    /// Left-associative binary operators from `|` (level 0) down to `* / %`.
    fn binary_level(&mut self, level: usize, skip: bool) -> Result<Operand, ArithError> {
        const LEVELS: [&[&str]; 8] = [
            &["|"],
            &["^"],
            &["&"],
            &["==", "!="],
            &["<=", ">=", "<", ">"],
            &["<<", ">>"],
            &["+", "-"],
            &["*", "/", "%"],
        ];
        let Some(operators) = LEVELS.get(level) else {
            return self.power(skip);
        };
        let mut left = self.binary_level(level + 1, skip)?;
        while let Some(operator) = self.binary_operator(operators) {
            self.eat(operator);
            let right = self.binary_level(level + 1, skip)?;
            let value = if skip {
                0
            } else {
                self.apply(operator, left.value, right.value)?
            };
            left = Operand::rvalue(value);
        }
        Ok(left)
    }

    /// `**` is right-associative and binds tighter than `*` but looser than unary operators.
    fn power(&mut self, skip: bool) -> Result<Operand, ArithError> {
        let base = self.unary(skip)?;
        if self.binary_operator(&["**"]).is_none() {
            return Ok(base);
        }
        self.eat("**");
        self.nest()?;
        let exponent = self.power(skip)?;
        self.nesting -= 1;
        if skip {
            return Ok(Operand::rvalue(0));
        }
        Ok(Operand::rvalue(self.apply(
            "**",
            base.value,
            exponent.value,
        )?))
    }

    fn unary(&mut self, skip: bool) -> Result<Operand, ArithError> {
        self.skip_blanks();
        for (operator, delta) in [("++", 1i64), ("--", -1i64)] {
            if self.looking_at(operator) && self.identifier_after(operator.len()) {
                self.eat(operator);
                let target = self.postfix(skip)?;
                let Some(place) = target.place else {
                    return Err(self.error("attempted assignment to non-variable"));
                };
                if skip {
                    return Ok(Operand::rvalue(0));
                }
                let value = target.value.wrapping_add(delta);
                self.store(&place, value)?;
                return Ok(Operand::rvalue(value));
            }
        }
        let Some(operator) = ['-', '+', '!', '~']
            .into_iter()
            .find(|operator| self.peek_char() == Some(*operator))
        else {
            return self.postfix(skip);
        };
        self.eat(&operator.to_string());
        self.nest()?;
        let operand = self.unary(skip)?.value;
        self.nesting -= 1;
        Ok(Operand::rvalue(match operator {
            '-' => operand.wrapping_neg(),
            '+' => operand,
            '!' => i64::from(operand == 0),
            _ => !operand,
        }))
    }

    fn identifier_after(&self, offset: usize) -> bool {
        let mut index = self.position + offset;
        while self.chars.get(index).is_some_and(|c| c.is_whitespace()) {
            index += 1;
        }
        self.chars
            .get(index)
            .is_some_and(|c| c.is_ascii_alphabetic() || *c == '_')
    }

    fn postfix(&mut self, skip: bool) -> Result<Operand, ArithError> {
        let operand = self.primary(skip)?;
        let Some(place) = operand.place.clone() else {
            return Ok(operand);
        };
        for (operator, delta) in [("++", 1i64), ("--", -1i64)] {
            if self.eat(operator) {
                if !skip {
                    self.store(&place, operand.value.wrapping_add(delta))?;
                }
                return Ok(Operand::rvalue(operand.value));
            }
        }
        Ok(operand)
    }

    fn primary(&mut self, skip: bool) -> Result<Operand, ArithError> {
        let Some(first) = self.peek_char() else {
            return Err(self.error("arithmetic syntax error: operand expected"));
        };
        if first == '(' {
            self.eat("(");
            self.nest()?;
            let value = self.comma(skip)?.value;
            self.nesting -= 1;
            if !self.eat(")") {
                self.token_start = self.position;
                return Err(self.error("missing `)'"));
            }
            return Ok(Operand::rvalue(value));
        }
        if first.is_ascii_digit() {
            return self.constant().map(Operand::rvalue);
        }
        if first.is_ascii_alphabetic() || first == '_' {
            return self.variable(skip);
        }
        self.token_start = self.position;
        Err(self.error("arithmetic syntax error: operand expected"))
    }

    fn constant(&mut self) -> Result<i64, ArithError> {
        self.token_start = self.position;
        let start = self.position;
        while self
            .chars
            .get(self.position)
            .is_some_and(|c| c.is_ascii_alphanumeric() || matches!(c, '#' | '@' | '_'))
        {
            self.position += 1;
        }
        let text: String = self.chars[start..self.position].iter().collect();
        parse_constant(&text).map_err(|message| self.error(message))
    }

    fn variable(&mut self, skip: bool) -> Result<Operand, ArithError> {
        self.token_start = self.position;
        let start = self.position;
        while self
            .chars
            .get(self.position)
            .is_some_and(|c| c.is_ascii_alphanumeric() || *c == '_')
        {
            self.position += 1;
        }
        let name: String = self.chars[start..self.position].iter().collect();
        let place = if self.chars.get(self.position) == Some(&'[') {
            self.position += 1;
            self.nest()?;
            let index = self.comma(skip)?.value;
            self.nesting -= 1;
            if !self.eat("]") {
                self.token_start = self.position;
                return Err(self.error("missing `]'"));
            }
            Place::Element(name, index)
        } else {
            Place::Scalar(name)
        };
        if skip {
            return Ok(Operand {
                value: 0,
                place: Some(place),
            });
        }
        let stored = match &place {
            Place::Scalar(name) => self.vars.arith_get(name),
            Place::Element(name, index) => self.vars.arith_get_element(name, *index),
        }
        .map_err(ArithError::variable)?;
        let value = self.variable_value(&place, stored)?;
        Ok(Operand {
            value,
            place: Some(place),
        })
    }

    /// Bash evaluates a variable's value as an expression when it is not a plain integer.
    fn variable_value(&mut self, place: &Place, stored: Option<String>) -> Result<i64, ArithError> {
        let Some(text) = stored else {
            return Ok(0);
        };
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Ok(0);
        }
        if let Ok(value) = trimmed.parse::<i64>() {
            return Ok(value);
        }
        if self.depth >= MAX_RECURSION {
            let name = match place {
                Place::Scalar(name) | Place::Element(name, _) => name,
            };
            return Err(ArithError::expression(format!(
                "expression recursion level exceeded (error token is \"{name}\")"
            )));
        }
        evaluate_at_depth(&mut *self.vars, trimmed, self.depth + 1)
    }

    fn store(&mut self, place: &Place, value: i64) -> Result<(), ArithError> {
        match place {
            Place::Scalar(name) => self.vars.arith_set(name, value),
            Place::Element(name, index) => self.vars.arith_set_element(name, *index, value),
        }
        .map_err(ArithError::variable)
    }

    fn apply(&self, operator: &str, left: i64, right: i64) -> Result<i64, ArithError> {
        Ok(match operator {
            "|" => left | right,
            "^" => left ^ right,
            "&" => left & right,
            "==" => i64::from(left == right),
            "!=" => i64::from(left != right),
            "<=" => i64::from(left <= right),
            ">=" => i64::from(left >= right),
            "<" => i64::from(left < right),
            ">" => i64::from(left > right),
            // Shift counts wrap modulo 64, as x86 Bash builds do.
            "<<" => left.wrapping_shl(right as u32),
            ">>" => left.wrapping_shr(right as u32),
            "+" => left.wrapping_add(right),
            "-" => left.wrapping_sub(right),
            "*" => left.wrapping_mul(right),
            "/" | "%" if right == 0 => return Err(self.error("division by 0")),
            "/" => left.wrapping_div(right),
            "%" => left.wrapping_rem(right),
            "**" if right < 0 => return Err(self.error("exponent less than 0")),
            "**" => wrapping_power(left, right),
            _ => unreachable!("arithmetic operator table and apply() disagree on {operator}"),
        })
    }
}

fn wrapping_power(mut base: i64, mut exponent: i64) -> i64 {
    let mut result = 1i64;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = result.wrapping_mul(base);
        }
        base = base.wrapping_mul(base);
        exponent >>= 1;
    }
    result
}

/// Parse an integer constant: decimal, `0` octal, `0x` hexadecimal, or `BASE#DIGITS`.
fn parse_constant(text: &str) -> Result<i64, &'static str> {
    let (base, digits) = if let Some((base, digits)) = text.split_once('#') {
        let base: u32 = base.parse().map_err(|_| "invalid arithmetic base")?;
        if !(2..=64).contains(&base) {
            return Err("invalid arithmetic base");
        }
        if digits.is_empty() {
            return Err("invalid integer constant");
        }
        (base, digits)
    } else if let Some(digits) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        (16, digits)
    } else if text.len() > 1 && text.starts_with('0') {
        (8, &text[1..])
    } else {
        (10, text)
    };
    let mut value = 0i64;
    for c in digits.chars() {
        let digit = match c {
            '0'..='9' => c as u32 - '0' as u32,
            'a'..='z' => c as u32 - 'a' as u32 + 10,
            'A'..='Z' if base <= 36 => c as u32 - 'A' as u32 + 10,
            'A'..='Z' => c as u32 - 'A' as u32 + 36,
            '@' => 62,
            '_' => 63,
            _ => return Err("invalid integer constant"),
        };
        if digit >= base {
            return Err("value too great for base");
        }
        value = value
            .wrapping_mul(i64::from(base))
            .wrapping_add(i64::from(digit));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{evaluate, ArithVars};

    #[derive(Default)]
    struct Vars {
        scalars: BTreeMap<String, String>,
        elements: BTreeMap<(String, i64), String>,
    }

    impl ArithVars for Vars {
        fn arith_get(&mut self, name: &str) -> Result<Option<String>, String> {
            Ok(self.scalars.get(name).cloned())
        }
        fn arith_get_element(&mut self, name: &str, index: i64) -> Result<Option<String>, String> {
            Ok(self.elements.get(&(name.to_string(), index)).cloned())
        }
        fn arith_set(&mut self, name: &str, value: i64) -> Result<(), String> {
            self.scalars.insert(name.to_string(), value.to_string());
            Ok(())
        }
        fn arith_set_element(&mut self, name: &str, index: i64, value: i64) -> Result<(), String> {
            self.elements
                .insert((name.to_string(), index), value.to_string());
            Ok(())
        }
    }

    fn eval(expression: &str) -> Result<i64, String> {
        let mut vars = Vars::default();
        vars.scalars.insert("x".into(), "2".into());
        vars.scalars.insert("e".into(), "3+4".into());
        vars.scalars.insert("self".into(), "self".into());
        evaluate(&mut vars, expression).map_err(|error| error.message)
    }

    // Expected values come from GNU Bash 5.2.
    #[test]
    fn operators_follow_bash_precedence_and_associativity() {
        for (expression, expected) in [
            ("2**10", 1024),
            ("2**3**2", 512),
            ("-2**2", 4),
            ("(1<<4) | 3", 19),
            ("5&3", 1),
            ("5^3", 6),
            ("~5", -6),
            ("!!5", 1),
            ("7>3 ? 10 : 20", 10),
            ("3 > 2 > 1", 0),
            ("5 == 5 == 1", 1),
            ("10/3*3", 9),
            ("-7 / 2", -3),
            ("-7 % 3", -1),
            ("1 << 65", 2),
            ("0x1f + 010 + 2#101", 44),
            ("64#_ + 36#z", 98),
            ("e*2", 14),
            ("++5", 5),
            ("", 0),
        ] {
            assert_eq!(eval(expression), Ok(expected), "{expression}");
        }
    }

    #[test]
    fn assignments_and_increments_update_variables_in_order() {
        let mut vars = Vars::default();
        vars.scalars.insert("y".into(), "1".into());
        assert_eq!(evaluate(&mut vars, "y++ + ++y"), Ok(4));
        assert_eq!(evaluate(&mut vars, "z=5, z+=2, z<<=1, z"), Ok(14));
        assert_eq!(evaluate(&mut vars, "a[1]=4, a[1]*2"), Ok(8));
        assert_eq!(vars.scalars["z"], "14");
        assert_eq!(vars.elements[&("a".to_string(), 1)], "4");
    }

    #[test]
    fn short_circuit_operands_have_no_side_effects() {
        let mut vars = Vars::default();
        assert_eq!(
            evaluate(&mut vars, "0 && (y=9), 1 || (y=8), 1 ? 2 : (y=7), y"),
            Ok(0)
        );
        assert!(!vars.scalars.contains_key("y"));
    }

    #[test]
    fn errors_use_bash_wording_and_error_tokens() {
        for (expression, expected) in [
            ("1/0", "division by 0 (error token is \"0\")"),
            ("2**-1", "exponent less than 0 (error token is \"1\")"),
            (
                "1=2",
                "attempted assignment to non-variable (error token is \"=2\")",
            ),
            ("08", "value too great for base (error token is \"08\")"),
            ("65#1", "invalid arithmetic base (error token is \"65#1\")"),
            ("2#", "invalid integer constant (error token is \"2#\")"),
            (
                "1 + ",
                "arithmetic syntax error: operand expected (error token is \"+ \")",
            ),
            (
                "5++ ",
                "arithmetic syntax error: operand expected (error token is \"+ \")",
            ),
            (
                "1 ? 2 : x=3 ",
                "attempted assignment to non-variable (error token is \"=3 \")",
            ),
            (
                "self",
                "expression recursion level exceeded (error token is \"self\")",
            ),
        ] {
            assert_eq!(eval(expression), Err(expected.to_string()), "{expression}");
        }
    }

    #[test]
    fn deep_nesting_is_rejected_without_exhausting_the_stack() {
        let expression = format!("{}1{}", "(".repeat(10_000), ")".repeat(10_000));
        assert!(eval(&expression).is_err());
        assert!(eval(&"-".repeat(10_000)).is_err());
    }
}
