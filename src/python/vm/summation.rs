//! The `sum()` builtin, following CPython 3.14's `builtin_sum_impl`.
//!
//! CPython keeps running totals in C while every item has the start value's exact type: machine
//! integers add directly, and floats and complex numbers use Neumaier's compensated summation,
//! so `sum([0.1] * 10) == 1.0`. The first item that does not fit ends the fast path; from there
//! on the total is a Python object and each item is added with `+`. Results therefore depend on
//! the order of types in the iterable exactly as they do in CPython.

use num_traits::ToPrimitive;

use super::super::number::{self, NumberRef};
use super::super::string;
use super::{Object, Value, Vm};
use crate::python::error::{PyError, PyResult};

/// Neumaier's running sum: `high` is the ordinary float total and `low` the rounding error it
/// has dropped so far.
#[derive(Clone, Copy)]
struct CompensatedSum {
    high: f64,
    low: f64,
}

impl CompensatedSum {
    fn new(value: f64) -> Self {
        Self {
            high: value,
            low: 0.0,
        }
    }

    fn add(self, value: f64) -> Self {
        let total = self.high + value;
        let error = if self.high.abs() >= value.abs() {
            (self.high - total) + value
        } else {
            (value - total) + self.high
        };
        Self {
            high: total,
            low: self.low + error,
        }
    }

    /// Adding a zero or non-finite compensation would lose a negative zero's sign or turn an
    /// overflowed total into NaN.
    fn value(self) -> f64 {
        if self.low != 0.0 && self.low.is_finite() {
            self.high + self.low
        } else {
            self.high
        }
    }
}

/// How the fast paths classify one item, by CPython's exact-type checks.
enum Item {
    /// An `int`, `bool`, or `int` subclass instance that fits a machine word.
    Word(i64),
    /// A larger `int` as a float, or `None` when it overflows (`PyLong_AsDouble`).
    Integer(Option<f64>),
    Float(f64),
    Complex(f64, f64),
    Other,
}

impl<'s> Vm<'s> {
    /// `sum(iterable, start=0)`.
    pub(super) fn builtin_sum(&mut self, values: Vec<Value>, start: Value) -> PyResult<Value> {
        self.reject_sequence_start(&start)?;
        let mut items = values.into_iter();
        let mut total = start;
        // `immediate_int` also accepts `bool`, which CPython adds here but never starts from.
        if let (Some(mut word), None) = (total.immediate_int(), total.bool_value()) {
            total = loop {
                let Some(item) = items.next() else {
                    return Ok(Value::Int(word));
                };
                self.charge_cpu(1)?;
                if let Some(next) = item
                    .immediate_int()
                    .and_then(|value| word.checked_add(value))
                {
                    word = next;
                    continue;
                }
                break self.add_numbers(Value::Int(word), item)?;
            };
        }
        if let Some(value) = total.float_value() {
            let mut sum = CompensatedSum::new(value);
            total = loop {
                let Some(item) = items.next() else {
                    return Ok(Value::Float(sum.value()));
                };
                self.charge_cpu(1)?;
                match self.sum_item(&item)? {
                    Item::Float(value) => sum = sum.add(value),
                    Item::Word(value) => sum = sum.add(value as f64),
                    Item::Integer(value) => sum = sum.add(self.integer_as_float(value)?),
                    Item::Complex(..) | Item::Other => {
                        break self.add_numbers(Value::Float(sum.value()), item)?
                    }
                }
            };
        }
        if let Item::Complex(real, imag) = self.sum_item(&total)? {
            let (mut real, mut imag) = (CompensatedSum::new(real), CompensatedSum::new(imag));
            total = loop {
                let Some(item) = items.next() else {
                    return self.allocate_object(Object::Complex {
                        real: real.value(),
                        imag: imag.value(),
                    });
                };
                self.charge_cpu(1)?;
                match self.sum_item(&item)? {
                    Item::Complex(re, im) => (real, imag) = (real.add(re), imag.add(im)),
                    Item::Word(value) => real = real.add(value as f64),
                    Item::Integer(value) => real = real.add(self.integer_as_float(value)?),
                    Item::Float(value) => real = real.add(value),
                    Item::Other => {
                        let partial = self.allocate_object(Object::Complex {
                            real: real.value(),
                            imag: imag.value(),
                        })?;
                        break self.add_numbers(partial, item)?;
                    }
                }
            };
        }
        for item in items {
            self.charge_cpu(1)?;
            total = self.add_numbers(total, item)?;
        }
        Ok(total)
    }

    fn reject_sequence_start(&mut self, start: &Value) -> PyResult<()> {
        let kind = if string::string_ref(self.heap(), *start)?.is_some() {
            "strings [use ''.join(seq) instead]"
        } else {
            match start.is_object().then(|| self.get(*start)).transpose()? {
                Some(Object::Bytes(_)) => "bytes [use b''.join(seq) instead]",
                Some(Object::ByteArray(_)) => "bytearray [use b''.join(seq) instead]",
                _ => return Ok(()),
            }
        };
        Err(PyError::exception(
            "TypeError",
            format!("sum() can't sum {kind}"),
        ))
    }

    /// Classify `value` for the fast paths. Registered numbers such as NumPy scalars are not
    /// Python `int`, `float`, or `complex` objects, so they take the generic path.
    fn sum_item(&self, value: &Value) -> PyResult<Item> {
        if number::registered_number(self.heap(), value).is_some() {
            return Ok(Item::Other);
        }
        Ok(match number::view(self.heap(), value) {
            Some(NumberRef::Int(value)) => Item::Word(value),
            Some(NumberRef::UInt(value)) => Item::Integer(Some(value as f64)),
            Some(NumberRef::BigInt(value)) => {
                Item::Integer(value.to_f64().filter(|value| value.is_finite()))
            }
            Some(NumberRef::Float(value)) => Item::Float(value),
            Some(NumberRef::Complex(real, imag)) => {
                if value.is_object() && matches!(self.get(*value)?, Object::Complex { .. }) {
                    Item::Complex(real, imag)
                } else {
                    Item::Other
                }
            }
            None => Item::Other,
        })
    }

    fn integer_as_float(&mut self, value: Option<f64>) -> PyResult<f64> {
        value
            .ok_or_else(|| PyError::exception("OverflowError", "int too large to convert to float"))
    }
}
