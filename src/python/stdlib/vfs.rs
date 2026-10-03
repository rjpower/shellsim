//! Private VFS primitives for frozen pure-Python stdlib facades.

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, OwnedPyString, PyBytes, PyError, PyFileKind, PyResult,
    PyRuntime, PyValueCast,
};
use super::super::Value;

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "_shellsim_vfs",
    functions: &[
        FunctionDef {
            module: "_shellsim_vfs",
            name: "read_text",
            call: read_text,
        },
        FunctionDef {
            module: "_shellsim_vfs",
            name: "write_text",
            call: write_text,
        },
        FunctionDef {
            module: "_shellsim_vfs",
            name: "append_text",
            call: append_text,
        },
        FunctionDef {
            module: "_shellsim_vfs",
            name: "read_bytes",
            call: read_bytes,
        },
        FunctionDef {
            module: "_shellsim_vfs",
            name: "write_bytes",
            call: write_bytes,
        },
        FunctionDef {
            module: "_shellsim_vfs",
            name: "append_bytes",
            call: append_bytes,
        },
        FunctionDef {
            module: "_shellsim_vfs",
            name: "exists",
            call: exists,
        },
        FunctionDef {
            module: "_shellsim_vfs",
            name: "disk_usage",
            call: disk_usage,
        },
        FunctionDef {
            module: "_shellsim_vfs",
            name: "is_file",
            call: is_file,
        },
        FunctionDef {
            module: "_shellsim_vfs",
            name: "is_dir",
            call: is_dir,
        },
        FunctionDef {
            module: "_shellsim_vfs",
            name: "is_symlink",
            call: is_symlink,
        },
        FunctionDef {
            module: "_shellsim_vfs",
            name: "list_dir",
            call: list_dir,
        },
        FunctionDef {
            module: "_shellsim_vfs",
            name: "stat",
            call: stat,
        },
        FunctionDef {
            module: "_shellsim_vfs",
            name: "mkdir",
            call: mkdir,
        },
        FunctionDef {
            module: "_shellsim_vfs",
            name: "rmdir",
            call: rmdir,
        },
        FunctionDef {
            module: "_shellsim_vfs",
            name: "chmod",
            call: chmod,
        },
        FunctionDef {
            module: "_shellsim_vfs",
            name: "symlink",
            call: symlink,
        },
        FunctionDef {
            module: "_shellsim_vfs",
            name: "readlink",
            call: readlink,
        },
        FunctionDef {
            module: "_shellsim_vfs",
            name: "utime",
            call: utime,
        },
        FunctionDef {
            module: "_shellsim_vfs",
            name: "glob",
            call: glob_paths,
        },
        FunctionDef {
            module: "_shellsim_vfs",
            name: "remove_file",
            call: remove_file,
        },
        FunctionDef {
            module: "_shellsim_vfs",
            name: "remove_tree",
            call: remove_tree,
        },
        FunctionDef {
            module: "_shellsim_vfs",
            name: "rename",
            call: rename,
        },
    ],
    values: &[],
};

fn read_text<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("_shellsim_vfs.read_text", 1, 1)?;
    args.reject_keywords("_shellsim_vfs.read_text")?;
    let OwnedPyString(path) = args.positional()[0].cast(runtime)?;
    runtime
        .filesystem()
        .read_text(&path)
        .and_then(|text| runtime.new_string(text))
}

fn write_text<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("_shellsim_vfs.write_text", 2, 2)?;
    args.reject_keywords("_shellsim_vfs.write_text")?;
    let OwnedPyString(path) = args.positional()[0].cast(runtime)?;
    let OwnedPyString(contents) = args.positional()[1].cast(runtime)?;
    runtime.filesystem().write_text(&path, &contents)?;
    Ok(Value::None)
}

fn append_text<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("_shellsim_vfs.append_text", 2, 2)?;
    args.reject_keywords("_shellsim_vfs.append_text")?;
    let OwnedPyString(path) = args.positional()[0].cast(runtime)?;
    let OwnedPyString(contents) = args.positional()[1].cast(runtime)?;
    let position = runtime.filesystem().append_text(&path, &contents)?;
    i64::try_from(position)
        .map(Value::Int)
        .map_err(|_| PyError::overflow_error("file position exceeds Python int range"))
}

fn read_bytes<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("_shellsim_vfs.read_bytes", 1, 1)?;
    args.reject_keywords("_shellsim_vfs.read_bytes")?;
    let OwnedPyString(path) = args.positional()[0].cast(runtime)?;
    runtime
        .filesystem()
        .read_bytes(&path)
        .and_then(|bytes| runtime.new_bytes(bytes))
}

fn write_bytes<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("_shellsim_vfs.write_bytes", 2, 2)?;
    args.reject_keywords("_shellsim_vfs.write_bytes")?;
    let OwnedPyString(path) = args.positional()[0].cast(runtime)?;
    let PyBytes(contents) = args.positional()[1].cast(runtime)?;
    runtime.filesystem().write_bytes(&path, &contents)?;
    Ok(Value::None)
}

fn append_bytes<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("_shellsim_vfs.append_bytes", 2, 2)?;
    args.reject_keywords("_shellsim_vfs.append_bytes")?;
    let OwnedPyString(path) = args.positional()[0].cast(runtime)?;
    let PyBytes(contents) = args.positional()[1].cast(runtime)?;
    let position = runtime.filesystem().append_bytes(&path, &contents)?;
    i64::try_from(position)
        .map(Value::Int)
        .map_err(|_| PyError::overflow_error("file position exceeds Python int range"))
}

/// `(used, limit)` bytes of the simulated disk. An unlimited disk reports `i64::MAX` as its
/// limit so callers can still compute a finite free space.
fn disk_usage<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("_shellsim_vfs.disk_usage", 0, 0)?;
    args.reject_keywords("_shellsim_vfs.disk_usage")?;
    let (used, limit) = runtime.filesystem().disk_usage();
    let used = Value::Int(i64::try_from(used).unwrap_or(i64::MAX));
    let limit = Value::Int(i64::try_from(limit).unwrap_or(i64::MAX));
    runtime.new_tuple(vec![used, limit])
}

fn exists<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("_shellsim_vfs.exists", 1, 1)?;
    args.reject_keywords("_shellsim_vfs.exists")?;
    let OwnedPyString(path) = args.positional()[0].cast(runtime)?;
    Ok(Value::Bool(runtime.filesystem().exists(&path)))
}

fn is_file<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("_shellsim_vfs.is_file", 1, 1)?;
    args.reject_keywords("_shellsim_vfs.is_file")?;
    let OwnedPyString(path) = args.positional()[0].cast(runtime)?;
    Ok(Value::Bool(runtime.filesystem().is_file(&path)))
}

fn is_dir<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("_shellsim_vfs.is_dir", 1, 1)?;
    args.reject_keywords("_shellsim_vfs.is_dir")?;
    let OwnedPyString(path) = args.positional()[0].cast(runtime)?;
    Ok(Value::Bool(runtime.filesystem().is_dir(&path)))
}

fn is_symlink<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("_shellsim_vfs.is_symlink", 1, 1)?;
    args.reject_keywords("_shellsim_vfs.is_symlink")?;
    let OwnedPyString(path) = args.positional()[0].cast(runtime)?;
    Ok(Value::Bool(runtime.filesystem().is_symlink(&path)))
}

fn list_dir<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("_shellsim_vfs.list_dir", 1, 1)?;
    args.reject_keywords("_shellsim_vfs.list_dir")?;
    let OwnedPyString(path) = args.positional()[0].cast(runtime)?;
    let entries = runtime.filesystem().list_dir(&path)?;
    let mut values = Vec::with_capacity(entries.len());
    for entry in entries {
        values.push(runtime.new_string(entry)?);
    }
    runtime.new_list(values)
}

/// `stat(path, follow)`: `(mode, size, mtime_ms, kind)` where kind is 0 for a regular file,
/// 1 for a directory, 2 for a symlink and 3 for anything else.
fn stat<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("_shellsim_vfs.stat", 2, 2)?;
    args.reject_keywords("_shellsim_vfs.stat")?;
    let OwnedPyString(path) = args.positional()[0].cast(runtime)?;
    let follow = runtime.truth(&args.positional()[1])?;
    let metadata = runtime.filesystem().metadata(&path, follow)?;
    let mode = Value::Int(i64::from(metadata.mode));
    let size = i64::try_from(metadata.size)
        .map(Value::Int)
        .map_err(|_| PyError::overflow_error("file size exceeds Python int range"))?;
    let mtime = i64::try_from(metadata.mtime_ms)
        .map(Value::Int)
        .map_err(|_| PyError::overflow_error("file time exceeds Python int range"))?;
    let kind = Value::Int(match metadata.kind {
        PyFileKind::File => 0,
        PyFileKind::Dir => 1,
        PyFileKind::Symlink => 2,
        PyFileKind::Other => 3,
    });
    runtime.new_tuple(vec![mode, size, mtime, kind])
}

fn rmdir<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("_shellsim_vfs.rmdir", 1, 1)?;
    args.reject_keywords("_shellsim_vfs.rmdir")?;
    let OwnedPyString(path) = args.positional()[0].cast(runtime)?;
    runtime.filesystem().rmdir(&path)?;
    Ok(Value::None)
}

fn chmod<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("_shellsim_vfs.chmod", 2, 2)?;
    args.reject_keywords("_shellsim_vfs.chmod")?;
    let OwnedPyString(path) = args.positional()[0].cast(runtime)?;
    let mode = runtime
        .int_value(&args.positional()[1])
        .and_then(|mode| u32::try_from(mode).ok())
        .ok_or_else(|| PyError::value_error("mode must be a non-negative integer"))?;
    runtime.filesystem().chmod(&path, mode & 0o7777)?;
    Ok(Value::None)
}

fn symlink<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("_shellsim_vfs.symlink", 2, 2)?;
    args.reject_keywords("_shellsim_vfs.symlink")?;
    let OwnedPyString(target) = args.positional()[0].cast(runtime)?;
    let OwnedPyString(link_path) = args.positional()[1].cast(runtime)?;
    runtime.filesystem().symlink(&target, &link_path)?;
    Ok(Value::None)
}

fn readlink<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("_shellsim_vfs.readlink", 1, 1)?;
    args.reject_keywords("_shellsim_vfs.readlink")?;
    let OwnedPyString(path) = args.positional()[0].cast(runtime)?;
    let target = runtime.filesystem().read_link(&path)?;
    runtime.new_string(target)
}

fn utime<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("_shellsim_vfs.utime", 1, 1)?;
    args.reject_keywords("_shellsim_vfs.utime")?;
    let OwnedPyString(path) = args.positional()[0].cast(runtime)?;
    runtime.filesystem().touch(&path)?;
    Ok(Value::None)
}

fn mkdir<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("_shellsim_vfs.mkdir", 3, 3)?;
    args.reject_keywords("_shellsim_vfs.mkdir")?;
    let OwnedPyString(path) = args.positional()[0].cast(runtime)?;
    let parents = runtime.truth(&args.positional()[1])?;
    let exist_ok = runtime.truth(&args.positional()[2])?;
    runtime.filesystem().mkdir(&path, parents, exist_ok)?;
    Ok(Value::None)
}

fn glob_paths<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("_shellsim_vfs.glob", 1, 1)?;
    args.reject_keywords("_shellsim_vfs.glob")?;
    let OwnedPyString(pattern) = args.positional()[0].cast(runtime)?;
    let paths = runtime.filesystem().glob(&pattern)?;
    let mut values = Vec::with_capacity(paths.len());
    for path in paths {
        values.push(runtime.new_string(path)?);
    }
    runtime.new_list(values)
}

fn remove_file<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("_shellsim_vfs.remove_file", 1, 1)?;
    args.reject_keywords("_shellsim_vfs.remove_file")?;
    let OwnedPyString(path) = args.positional()[0].cast(runtime)?;
    runtime.filesystem().remove_file(&path)?;
    Ok(Value::None)
}

fn remove_tree<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("_shellsim_vfs.remove_tree", 1, 1)?;
    args.reject_keywords("_shellsim_vfs.remove_tree")?;
    let OwnedPyString(path) = args.positional()[0].cast(runtime)?;
    runtime.filesystem().remove_tree(&path)?;
    Ok(Value::None)
}

fn rename<'s>(runtime: &mut dyn PyRuntime<'s>, args: CallArgs<'s>) -> PyResult<'s> {
    args.expect_positional("_shellsim_vfs.rename", 2, 2)?;
    args.reject_keywords("_shellsim_vfs.rename")?;
    let OwnedPyString(source) = args.positional()[0].cast(runtime)?;
    let OwnedPyString(destination) = args.positional()[1].cast(runtime)?;
    runtime.filesystem().rename(&source, &destination)?;
    Ok(Value::None)
}
