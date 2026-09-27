//! Argument binding shared by NumPy functions and methods.
//!
//! [`Signature`] binds positional and keyword arguments to named parameters with CPython's
//! error text, so each function declares its parameters once:
//!
//! ```ignore
//! static SIGNATURE: Signature = Signature::new("sum", &["a", "axis", "dtype", "out", "keepdims"], 1);
//! let bound = SIGNATURE.bind(&args)?;
//! let axis = axes(runtime, bound.get("axis"), array.ndim())?;
//! ```
//!
//! The extractors convert common argument kinds: axes, dtypes, shapes, and optional ints.

use super::super::super::native::{
    CallArgs, PyError, PyKind, PyResult, PyRuntime, PyTypeObject, PyValue,
};
use super::array::normalize_axis;
use super::dtype::DType;

/// Parameters of one function: positional-or-keyword names, then keyword-only names.
pub(in crate::python) struct Signature {
    function: &'static str,
    parameters: &'static [&'static str],
    keyword_only: &'static [&'static str],
    required: usize,
}

impl Signature {
    /// `required` counts the leading parameters without defaults.
    pub(in crate::python) const fn new(
        function: &'static str,
        parameters: &'static [&'static str],
        required: usize,
    ) -> Self {
        Self {
            function,
            parameters,
            keyword_only: &[],
            required,
        }
    }

    /// Add keyword-only parameters, all optional.
    pub(in crate::python) const fn keyword_only(mut self, names: &'static [&'static str]) -> Self {
        self.keyword_only = names;
        self
    }

    pub(in crate::python) fn bind(&'static self, args: &CallArgs) -> PyResult<Bound> {
        let positional = args.positional();
        let function = self.function;
        if positional.len() > self.parameters.len() {
            let expected = if self.required == self.parameters.len() {
                self.required.to_string()
            } else {
                format!("from {} to {}", self.required, self.parameters.len())
            };
            return Err(PyError::type_error(format!(
                "{function}() takes {expected} positional arguments but {} were given",
                positional.len()
            )));
        }
        let mut values = vec![None; self.parameters.len() + self.keyword_only.len()];
        for (slot, value) in values.iter_mut().zip(positional) {
            *slot = Some(*value);
        }
        for (name, value) in args.keywords() {
            let Some(index) = self
                .parameters
                .iter()
                .chain(self.keyword_only)
                .position(|parameter| parameter == name)
            else {
                return Err(PyError::type_error(format!(
                    "{function}() got an unexpected keyword argument '{name}'"
                )));
            };
            if values[index].is_some() {
                return Err(PyError::type_error(format!(
                    "argument for {function}() given by name ('{name}') and position ({})",
                    index + 1
                )));
            }
            values[index] = Some(*value);
        }
        if let Some(index) = values[..self.required].iter().position(Option::is_none) {
            return Err(PyError::type_error(format!(
                "{function}() missing required argument '{}' (pos {})",
                self.parameters[index],
                index + 1
            )));
        }
        Ok(Bound {
            signature: self,
            values,
        })
    }
}

/// Arguments bound to a [`Signature`].
pub(in crate::python) struct Bound {
    signature: &'static Signature,
    values: Vec<Option<PyValue>>,
}

impl Bound {
    fn index(&self, name: &str) -> usize {
        self.signature
            .parameters
            .iter()
            .chain(self.signature.keyword_only)
            .position(|parameter| *parameter == name)
            .unwrap_or_else(|| panic!("{name} is not a parameter of {}", self.signature.function))
    }

    /// The argument, if one was passed. An explicit `None` is returned as `Some(None)`.
    pub(in crate::python) fn get(&self, name: &str) -> Option<PyValue> {
        self.values[self.index(name)]
    }

    /// The argument unless it was omitted or passed as `None`.
    pub(in crate::python) fn value(&self, name: &str) -> Option<PyValue> {
        self.get(name).filter(|value| !value.is_none())
    }

    /// A required argument.
    pub(in crate::python) fn required(&self, name: &str) -> PyValue {
        self.get(name)
            .expect("required arguments are checked when binding")
    }
}

/// Which axes a reduction or shape operation applies to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::python) enum Axes {
    All,
    /// Distinct, normalized axes in ascending order.
    Some(Vec<usize>),
}

impl Axes {
    /// Whether `axis` is selected.
    pub(in crate::python) fn contains(&self, axis: usize) -> bool {
        match self {
            Self::All => true,
            Self::Some(axes) => axes.contains(&axis),
        }
    }
}

/// Parse `axis=None | int | tuple[int, ...]` for an array of rank `ndim`.
pub(in crate::python) fn axes(
    runtime: &mut dyn PyRuntime,
    value: Option<PyValue>,
    ndim: usize,
) -> PyResult<Axes> {
    let Some(value) = value.filter(|value| !value.is_none()) else {
        return Ok(Axes::All);
    };
    let raw = if runtime.kind(&value)? == PyKind::Tuple {
        let tuple = value.cast(runtime)?;
        runtime
            .tuple_items(tuple)?
            .iter()
            .map(|item| index_int(runtime, item))
            .collect::<PyResult<Vec<_>>>()?
    } else {
        vec![index_int(runtime, &value)?]
    };
    let mut axes = raw
        .into_iter()
        .map(|axis| normalize_axis(axis, ndim))
        .collect::<PyResult<Vec<_>>>()?;
    axes.sort_unstable();
    if axes.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(PyError::value_error("duplicate value in 'axis'"));
    }
    Ok(Axes::Some(axes))
}

/// A single optional axis, normalized.
pub(in crate::python) fn axis(
    runtime: &mut dyn PyRuntime,
    value: Option<PyValue>,
    ndim: usize,
) -> PyResult<Option<usize>> {
    value
        .filter(|value| !value.is_none())
        .map(|value| normalize_axis(index_int(runtime, &value)?, ndim))
        .transpose()
}

use super::super::super::native::PyValueCast;

/// An integer argument accepting Python ints, bools, and NumPy integer scalars.
pub(in crate::python) fn index_int(runtime: &mut dyn PyRuntime, value: &PyValue) -> PyResult<i64> {
    if let Some(value) = runtime.int_value(value) {
        return Ok(value);
    }
    if let Some((dtype, number)) = super::scalar::unbox_number(runtime, value) {
        if super::scalar::is_index_dtype(dtype) {
            return Ok(number.wrapping_i64());
        }
    }
    Err(PyError::type_error(format!(
        "'{}' object cannot be interpreted as an integer",
        runtime.type_name(value)?
    )))
}

/// An optional integer; `None` counts as omitted.
pub(in crate::python) fn optional_int(
    runtime: &mut dyn PyRuntime,
    value: Option<PyValue>,
) -> PyResult<Option<i64>> {
    value
        .filter(|value| !value.is_none())
        .map(|value| index_int(runtime, &value))
        .transpose()
}

/// A float argument accepting any real Python or NumPy number.
pub(in crate::python) fn float_arg(runtime: &mut dyn PyRuntime, value: &PyValue) -> PyResult<f64> {
    use super::super::super::number::NumberRef;
    if let Some((_, number)) = super::scalar::unbox_number(runtime, value) {
        return Ok(number.as_f64());
    }
    match runtime.number(value) {
        Some(NumberRef::Int(value)) => Ok(value as f64),
        Some(NumberRef::BigInt(value)) => {
            Ok(num_traits::ToPrimitive::to_f64(value).unwrap_or(f64::INFINITY))
        }
        Some(NumberRef::Float(value)) => Ok(value),
        _ => Err(PyError::type_error(format!(
            "must be real number, not {}",
            runtime.type_name(value)?
        ))),
    }
}

/// Truth of a flag argument such as `keepdims`.
pub(in crate::python) fn flag(
    runtime: &mut dyn PyRuntime,
    value: Option<PyValue>,
    default: bool,
) -> PyResult<bool> {
    match value {
        Some(value) => runtime.truth(&value),
        None => Ok(default),
    }
}

/// A shape given as an int or a sequence of ints, with NumPy's negative-dimension error.
pub(in crate::python) fn shape(
    runtime: &mut dyn PyRuntime,
    value: PyValue,
) -> PyResult<Vec<usize>> {
    let dimensions = match runtime.kind(&value)? {
        PyKind::Tuple => {
            let tuple = value.cast(runtime)?;
            runtime.tuple_items(tuple)?
        }
        PyKind::List => {
            let list = value.cast(runtime)?;
            runtime.list_items(list)?
        }
        _ => vec![value],
    };
    let shape = dimensions
        .iter()
        .map(|dimension| {
            let dimension = index_int(runtime, dimension)?;
            usize::try_from(dimension)
                .map_err(|_| PyError::value_error("negative dimensions are not allowed"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    super::array::element_count(&shape)?;
    Ok(shape)
}

/// A `dtype=` argument: a dtype, a dtype string, a Python type, or a NumPy scalar type.
pub(in crate::python) fn dtype(runtime: &mut dyn PyRuntime, value: PyValue) -> PyResult<DType> {
    if let Some(dtype) = super::dtype_object::unpack(runtime, &value) {
        return Ok(dtype);
    }
    match runtime.kind(&value)? {
        PyKind::String => {
            let text = runtime.string_value(&value)?.unwrap_or_default();
            return DType::parse(&text);
        }
        // Field lists and dicts describe structured dtypes; a `(base, shape)` tuple describes a
        // subarray dtype.
        PyKind::List | PyKind::Dict | PyKind::Tuple => {
            let repr = runtime.repr(&value)?;
            return Err(PyError::unsupported(format!(
                "NumPy dtype {repr} is not supported"
            )));
        }
        _ => {}
    }
    match runtime.type_object(&value) {
        Some(PyTypeObject::Builtin(name)) => match name {
            "int" => Ok(DType::INT64),
            "float" => Ok(DType::FLOAT64),
            "complex" => Ok(DType::COMPLEX128),
            "bool" => Ok(DType::BOOL),
            "str" => DType::str(0),
            "object" => Ok(DType::OBJECT),
            _ => Err(PyError::type_error(format!(
                "Cannot interpret '<class '{name}'>' as a data type"
            ))),
        },
        Some(PyTypeObject::Kind(kind)) => super::scalar::kind_dtype(kind).ok_or_else(|| {
            PyError::type_error(format!(
                "Converting '{}' to a dtype is not supported",
                kind.name
            ))
        }),
        None => {
            let repr = runtime.repr(&value)?;
            Err(PyError::type_error(format!(
                "Cannot interpret '{repr}' as a data type"
            )))
        }
    }
}

/// An optional `dtype=` argument; `None` counts as omitted.
pub(in crate::python) fn optional_dtype(
    runtime: &mut dyn PyRuntime,
    value: Option<PyValue>,
) -> PyResult<Option<DType>> {
    value
        .filter(|value| !value.is_none())
        .map(|value| dtype(runtime, value))
        .transpose()
}
