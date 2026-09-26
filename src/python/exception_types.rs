//! The closed table of exception classes that the VM and native modules can name, with CPython's
//! class hierarchy.
//!
//! Builtin exceptions, shellsim's pytest outcomes and exceptions raised by native stdlib modules
//! share one table, so `except`, `isinstance` and `issubclass` all follow the same parent chain.
//! Each class has exactly one parent here; CPython's multiple-inheritance exception classes are
//! not modeled. A user class that derives from one of these records it as its exception base, and
//! its subclass checks start from that base.

/// One exception class: its name, its parent, and whether `builtins` binds the name.
pub(super) struct ExceptionTypeDef {
    pub(super) name: &'static str,
    /// `None` only for `BaseException`.
    pub(super) parent: Option<&'static str>,
    pub(super) builtin: bool,
}

const fn builtin(name: &'static str, parent: &'static str) -> ExceptionTypeDef {
    ExceptionTypeDef {
        name,
        parent: Some(parent),
        builtin: true,
    }
}

const fn native(name: &'static str, parent: &'static str) -> ExceptionTypeDef {
    ExceptionTypeDef {
        name,
        parent: Some(parent),
        builtin: false,
    }
}

/// CPython 3.14's builtin hierarchy, followed by the non-builtin classes native code raises.
pub(super) const EXCEPTION_TYPES: &[ExceptionTypeDef] = &[
    ExceptionTypeDef {
        name: "BaseException",
        parent: None,
        builtin: true,
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
    native("Failed", "Exception"),
    native("SubprocessError", "Exception"),
    native("CalledProcessError", "SubprocessError"),
    native("TimeoutExpired", "SubprocessError"),
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
    let mut parent = match exception_type(kind) {
        Some(definition) => definition.parent,
        None => Some("Exception"),
    };
    while let Some(name) = parent {
        if name == base {
            return true;
        }
        parent = exception_type(name).and_then(|definition| definition.parent);
    }
    false
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
