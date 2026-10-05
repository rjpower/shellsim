//! The closed table of exception classes that the VM and native modules can name, with CPython's
//! class hierarchy, and the questions the runtime asks of an exception instance.
//!
//! Builtin exceptions, shellsim's pytest outcomes and exceptions raised by native stdlib modules
//! share one table. The type registry linearizes their bases with C3, including NumPy's AxisError
//! with its two parents. A user class records an exception ancestor for its instance layout.
//! An exception instance is an ordinary object whose type is registered here and whose payload
//! is its `args`; the helpers at the end classify one and read those arguments.

use super::heap::Object;
use super::{ReplState, Value};

/// One exception class: its name, its parent, whether `builtins` binds the name, and the
/// `__module__` CPython reports for it.
pub(super) struct ExceptionTypeDef {
    pub(super) name: &'static str,
    /// `None` only for `BaseException`.
    pub(super) parent: Option<&'static str>,
    pub(super) secondary_parent: Option<&'static str>,
    pub(super) builtin: bool,
    pub(super) module: &'static str,
}

const fn builtin(name: &'static str, parent: &'static str) -> ExceptionTypeDef {
    ExceptionTypeDef {
        name,
        parent: Some(parent),
        secondary_parent: None,
        builtin: true,
        module: "builtins",
    }
}

const fn native(
    name: &'static str,
    parent: &'static str,
    module: &'static str,
) -> ExceptionTypeDef {
    ExceptionTypeDef {
        name,
        parent: Some(parent),
        secondary_parent: None,
        builtin: false,
        module,
    }
}

/// CPython 3.14's builtin hierarchy, followed by the non-builtin classes native code raises.
pub(super) const EXCEPTION_TYPES: &[ExceptionTypeDef] = &[
    ExceptionTypeDef {
        name: "BaseException",
        parent: None,
        secondary_parent: None,
        builtin: true,
        module: "builtins",
    },
    builtin("GeneratorExit", "BaseException"),
    builtin("KeyboardInterrupt", "BaseException"),
    builtin("SystemExit", "BaseException"),
    builtin("Exception", "BaseException"),
    builtin("ArithmeticError", "Exception"),
    builtin("FloatingPointError", "ArithmeticError"),
    builtin("OverflowError", "ArithmeticError"),
    builtin("ZeroDivisionError", "ArithmeticError"),
    builtin("AssertionError", "Exception"),
    builtin("AttributeError", "Exception"),
    builtin("BufferError", "Exception"),
    builtin("EOFError", "Exception"),
    builtin("ImportError", "Exception"),
    builtin("ModuleNotFoundError", "ImportError"),
    builtin("LookupError", "Exception"),
    builtin("IndexError", "LookupError"),
    builtin("KeyError", "LookupError"),
    builtin("MemoryError", "Exception"),
    builtin("NameError", "Exception"),
    builtin("UnboundLocalError", "NameError"),
    builtin("OSError", "Exception"),
    builtin("BlockingIOError", "OSError"),
    builtin("ChildProcessError", "OSError"),
    builtin("ConnectionError", "OSError"),
    builtin("BrokenPipeError", "ConnectionError"),
    builtin("ConnectionAbortedError", "ConnectionError"),
    builtin("ConnectionRefusedError", "ConnectionError"),
    builtin("ConnectionResetError", "ConnectionError"),
    builtin("FileExistsError", "OSError"),
    builtin("FileNotFoundError", "OSError"),
    builtin("InterruptedError", "OSError"),
    builtin("IsADirectoryError", "OSError"),
    builtin("NotADirectoryError", "OSError"),
    builtin("PermissionError", "OSError"),
    builtin("ProcessLookupError", "OSError"),
    builtin("TimeoutError", "OSError"),
    builtin("ReferenceError", "Exception"),
    builtin("RuntimeError", "Exception"),
    builtin("NotImplementedError", "RuntimeError"),
    builtin("RecursionError", "RuntimeError"),
    builtin("StopAsyncIteration", "Exception"),
    builtin("StopIteration", "Exception"),
    builtin("SyntaxError", "Exception"),
    builtin("IndentationError", "SyntaxError"),
    builtin("TabError", "IndentationError"),
    builtin("SystemError", "Exception"),
    builtin("TypeError", "Exception"),
    builtin("ValueError", "Exception"),
    builtin("UnicodeError", "ValueError"),
    builtin("UnicodeDecodeError", "UnicodeError"),
    builtin("UnicodeEncodeError", "UnicodeError"),
    builtin("UnicodeTranslateError", "UnicodeError"),
    builtin("Warning", "Exception"),
    builtin("BytesWarning", "Warning"),
    builtin("DeprecationWarning", "Warning"),
    builtin("EncodingWarning", "Warning"),
    builtin("FutureWarning", "Warning"),
    builtin("ImportWarning", "Warning"),
    builtin("PendingDeprecationWarning", "Warning"),
    builtin("ResourceWarning", "Warning"),
    builtin("RuntimeWarning", "Warning"),
    builtin("SyntaxWarning", "Warning"),
    builtin("UnicodeWarning", "Warning"),
    builtin("UserWarning", "Warning"),
    // The pytest runner's wrapper catches `Skipped` by name and reports `Failed` like any other
    // `Exception`, so both stay below `Exception` rather than pytest's `BaseException`.
    builtin("Skipped", "Exception"),
    native("Failed", "Exception", "builtins"),
    native("SubprocessError", "Exception", "subprocess"),
    native("CalledProcessError", "SubprocessError", "subprocess"),
    native("TimeoutExpired", "SubprocessError", "subprocess"),
    ExceptionTypeDef {
        name: "AxisError",
        parent: Some("ValueError"),
        secondary_parent: Some("IndexError"),
        builtin: false,
        module: "numpy.exceptions",
    },
    native("LinAlgError", "ValueError", "numpy.linalg"),
    native("ComplexWarning", "RuntimeWarning", "numpy.exceptions"),
];

/// Look up a modeled exception class by name.
pub(super) fn exception_type(name: &str) -> Option<&'static ExceptionTypeDef> {
    EXCEPTION_TYPES
        .iter()
        .find(|definition| definition.name == name)
}

/// Whether the class named `kind` is `base` or one of its subclasses.
///
/// A kind missing from the table (an exception raised under a name the table does not model)
/// is treated as a direct subclass of `Exception`, which is how CPython would classify any
/// ordinary exception class.
///
/// ```text
/// exception_is_subclass("KeyError", "LookupError") == true
/// exception_is_subclass("SystemExit", "Exception") == false
/// ```
pub(super) fn exception_is_subclass(kind: &str, base: &str) -> bool {
    if kind == base {
        return true;
    }
    if let Some(definition) = exception_type(kind) {
        return definition
            .parent
            .into_iter()
            .chain(definition.secondary_parent)
            .any(|parent| exception_is_subclass(parent, base));
    }
    exception_is_subclass("Exception", base)
}

/// The `OSError` subclass CPython raises for `errno`, or `OSError` itself when the errno has no
/// dedicated subclass. Linux errno values.
pub(super) fn os_error_subclass(errno: i32) -> &'static str {
    match errno {
        1 | 13 => "PermissionError",
        2 => "FileNotFoundError",
        3 => "ProcessLookupError",
        4 => "InterruptedError",
        10 => "ChildProcessError",
        11 | 114 | 115 => "BlockingIOError",
        17 => "FileExistsError",
        20 => "NotADirectoryError",
        21 => "IsADirectoryError",
        32 | 108 => "BrokenPipeError",
        103 => "ConnectionAbortedError",
        104 => "ConnectionResetError",
        110 => "TimeoutError",
        111 => "ConnectionRefusedError",
        _ => "OSError",
    }
}

/// The class name of an exception instance, or `None` when `value` is not one. Rendering the
/// message can fail on self-referential arguments, so callers that only classify use this.
pub(super) fn exception_type_name(
    state: &ReplState,
    value: Value<'_>,
) -> Result<Option<String>, String> {
    if !value.is_object() {
        return Ok(None);
    }
    let type_id = state.heap.type_id(value)?;
    if !state.types.is_exception_type(type_id)? {
        return Ok(None);
    }
    Ok(Some(state.types.get(type_id)?.name.clone()))
}

/// The closest builtin exception class an exception instance derives from, or `None` when
/// `value` is not an exception.
pub(super) fn exception_base(
    state: &ReplState,
    value: Value<'_>,
) -> Result<Option<&'static str>, String> {
    if !value.is_object() {
        return Ok(None);
    }
    state.types.exception_base(state.heap.type_id(value)?)
}

/// The closest builtin exception class and the constructor arguments of an exception instance,
/// which its payload holds as `BaseException.args`.
pub(super) fn exception_args<'s>(
    state: &ReplState,
    value: Value<'_>,
) -> Result<Option<(&'static str, Vec<Value<'s>>)>, String> {
    let Some(base) = exception_base(state, value)? else {
        return Ok(None);
    };
    let heap = &state.heap;
    let Object::Exception(args) = heap.get(value)? else {
        return Err("exception instance has a non-exception layout".into());
    };
    Ok(Some((base, heap.handles(args))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_parent_is_a_modeled_class() {
        for definition in EXCEPTION_TYPES {
            if let Some(parent) = definition.parent {
                assert!(exception_type(parent).is_some(), "{}", definition.name);
            }
            if let Some(parent) = definition.secondary_parent {
                assert!(exception_type(parent).is_some(), "{}", definition.name);
            }
        }
    }

    #[test]
    fn subclass_checks_follow_the_cpython_hierarchy() {
        assert!(exception_is_subclass("KeyError", "LookupError"));
        assert!(exception_is_subclass("FileNotFoundError", "OSError"));
        assert!(exception_is_subclass(
            "ZeroDivisionError",
            "ArithmeticError"
        ));
        assert!(exception_is_subclass("RuntimeWarning", "Exception"));
        assert!(exception_is_subclass("TabError", "SyntaxError"));
        assert!(!exception_is_subclass("SystemExit", "Exception"));
        assert!(exception_is_subclass("SystemExit", "BaseException"));
        assert!(!exception_is_subclass("ValueError", "LookupError"));
        assert!(exception_is_subclass("NotModeled", "Exception"));
        assert!(!exception_is_subclass("Exception", "ValueError"));
    }
}
