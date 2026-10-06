//! The interpreter's one error type.
//!
//! Every fallible operation in the Python implementation, from the heap and the object
//! protocol up through the VM and the native modules, returns [`PyResult`]. A [`PyError`] is one
//! of three things:
//!
//! - A pending exception: the VM already holds the exception object as its pending exception,
//!   and the error is only the signal to unwind. This is the common case once an exception is
//!   in flight, and it allocates nothing.
//! - A builtin exception still to be raised, described by its class name and message (or an
//!   `OSError` by its errno). The heap, native kernels and most VM code raise this way, without
//!   allocating; the VM creates the exception object only when the error unwinds a frame or
//!   code takes the exception, and `Vm::catch` matches it by class name before that.
//! - A stop that is not a Python exception: an unsupported feature or broken invariant, a
//!   resource limit, or a scheduler request to exit or suspend. `except` never catches these.
//!
//! The error is one pointer wide, so `PyResult<Value>` comes back in registers, and everything
//! but a pending exception lives behind that pointer.

use std::fmt;

use crate::scheduler::WaitReason;

/// The result of any fallible Python operation; most return a value.
pub(in crate::python) type PyResult<T = super::Value> = Result<T, PyError>;

/// A scheduler request that travels as an error out of a native call, split off with
/// [`PyError::into_control`].
pub(in crate::python) enum Control {
    Exit(i32),
    Suspend(WaitReason),
}

/// Why a Python operation stopped. See the module documentation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::python) struct PyError(Option<Box<ErrorDetail>>);

#[derive(Clone, Debug, PartialEq, Eq)]
struct ErrorDetail {
    kind: PyErrorKind,
    message: String,
}

/// What a non-pending [`PyError`] asks for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::python) enum PyErrorKind {
    /// Raise an instance of the builtin exception class with this name, with the message as its
    /// only argument, or with no arguments when the message is empty.
    Exception(&'static str),
    /// Raise an `OSError` for an errno and an optional operand path. The VM picks the subclass
    /// from the errno as CPython does and stores `(errno, strerror, filename)` as the args; the
    /// message holds the `strerror` text.
    OsError {
        errno: i32,
        filename: Option<String>,
    },
    /// A valid Python operation that shellsim does not model, or a broken interpreter
    /// invariant. The program stops with the minimal-shim diagnostic.
    Unsupported,
    /// A resource limit stopped the process; the meter records which one.
    Resource,
    /// The program asked to exit with this status.
    Exit(i32),
    /// Internal cooperative control flow: the scheduler must resume the call when its resource
    /// is ready. The bytecode VM consumes it; it never becomes a Python exception.
    Suspend(WaitReason),
}

impl PyError {
    /// The signal for an exception the VM already holds as its pending exception.
    #[inline(always)]
    pub const fn pending() -> Self {
        Self(None)
    }

    #[cold]
    pub fn new(kind: PyErrorKind, message: impl Into<String>) -> Self {
        Self(Some(Box::new(ErrorDetail {
            kind,
            message: message.into(),
        })))
    }

    /// Raise the builtin exception class named `kind`.
    pub fn exception(kind: &'static str, message: impl Into<String>) -> Self {
        Self::new(PyErrorKind::Exception(kind), message)
    }

    pub fn type_error(message: impl Into<String>) -> Self {
        Self::exception("TypeError", message)
    }

    pub fn value_error(message: impl Into<String>) -> Self {
        Self::exception("ValueError", message)
    }

    pub fn overflow_error(message: impl Into<String>) -> Self {
        Self::exception("OverflowError", message)
    }

    pub fn zero_division_error(message: impl Into<String>) -> Self {
        Self::exception("ZeroDivisionError", message)
    }

    pub fn runtime_error(message: impl Into<String>) -> Self {
        Self::exception("RuntimeError", message)
    }

    /// A catchable `NotImplementedError` for a library feature outside shellsim's subset, such
    /// as an unsupported dtype or option in NumPy or SciPy.
    pub fn not_implemented_error(message: impl Into<String>) -> Self {
        Self::exception("NotImplementedError", message)
    }

    /// An `OSError` for `errno` with CPython's `[Errno N] strerror: 'filename'` form; the VM
    /// raises the errno's subclass, such as `FileNotFoundError` for `ENOENT`.
    pub fn os_error(errno: i32, strerror: impl Into<String>, filename: Option<String>) -> Self {
        Self::new(PyErrorKind::OsError { errno, filename }, strerror)
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::new(PyErrorKind::Unsupported, message)
    }

    pub fn resource_error(message: impl Into<String>) -> Self {
        Self::new(PyErrorKind::Resource, message)
    }

    pub fn exit(status: i32) -> Self {
        Self::new(PyErrorKind::Exit(status), "Python callable requested exit")
    }

    /// Suspend a scheduler-owned native call until its modeled resource becomes ready.
    pub fn suspend(reason: WaitReason) -> Self {
        Self::new(PyErrorKind::Suspend(reason), "Python native call suspended")
    }

    #[inline(always)]
    pub fn is_pending(&self) -> bool {
        self.0.is_none()
    }

    /// What the error asks for, or `None` for a pending exception.
    pub fn kind(&self) -> Option<&PyErrorKind> {
        self.0.as_ref().map(|detail| &detail.kind)
    }

    /// The message, empty for a pending exception, whose message lives in its object.
    pub fn message(&self) -> &str {
        self.0.as_ref().map_or("", |detail| detail.message.as_str())
    }

    /// Whether this is a builtin exception still to be raised whose class is `kind`.
    pub fn is_exception(&self, kind: &str) -> bool {
        matches!(self.kind(), Some(PyErrorKind::Exception(name)) if *name == kind)
    }

    /// Separate a scheduler exit or suspend request from every other error.
    pub fn into_control(self) -> Result<Control, Self> {
        if !matches!(
            self.kind(),
            Some(PyErrorKind::Exit(_) | PyErrorKind::Suspend(_))
        ) {
            return Err(self);
        }
        match self.into_parts() {
            Some((PyErrorKind::Exit(status), _)) => Ok(Control::Exit(status)),
            Some((PyErrorKind::Suspend(reason), _)) => Ok(Control::Suspend(reason)),
            _ => unreachable!("checked above"),
        }
    }

    /// The kind and message of a non-pending error.
    pub fn into_parts(self) -> Option<(PyErrorKind, String)> {
        self.0.map(|detail| (detail.kind, detail.message))
    }

    /// Add where a stop that is not a Python exception happened to its message. Exceptions
    /// record locations in their traceback instead.
    pub fn located(self, location: impl FnOnce() -> String) -> Self {
        match self.0 {
            Some(mut detail)
                if !matches!(
                    detail.kind,
                    PyErrorKind::Exception(_) | PyErrorKind::OsError { .. }
                ) =>
            {
                detail.message.push_str(&location());
                Self(Some(detail))
            }
            other => Self(other),
        }
    }
}

/// A bare message is an unsupported operation or broken invariant.
impl From<String> for PyError {
    #[cold]
    fn from(message: String) -> Self {
        Self::unsupported(message)
    }
}

impl From<&str> for PyError {
    #[cold]
    fn from(message: &str) -> Self {
        Self::unsupported(message)
    }
}

impl fmt::Display for PyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            None => formatter.write_str("pending Python exception"),
            Some(detail) => formatter.write_str(&detail.message),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_value_result_stays_small() {
        assert_eq!(std::mem::size_of::<PyError>(), 8);
        assert!(std::mem::size_of::<PyResult>() <= 24);
    }

    #[test]
    fn locations_are_added_to_stops_but_not_exceptions() {
        let stop = PyError::unsupported("boom").located(|| " in f at line 2, column 1".into());
        assert_eq!(stop.message(), "boom in f at line 2, column 1");
        let exception = PyError::type_error("bad").located(|| unreachable!());
        assert_eq!(exception.message(), "bad");
        assert!(PyError::pending().located(|| unreachable!()).is_pending());
    }
}
