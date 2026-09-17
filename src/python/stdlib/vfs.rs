//! Private VFS primitives for frozen pure-Python stdlib facades.

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyBytes, PyError, PyResult, PyRuntime, PyString, PyValueCast,
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

fn read_text(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_shellsim_vfs.read_text", 1, 1)?;
    args.reject_keywords("_shellsim_vfs.read_text")?;
    let PyString(path) = args.positional()[0].cast(runtime)?;
    runtime
        .filesystem()
        .read_text(&path)
        .and_then(|text| runtime.new_string(text))
}

fn write_text(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_shellsim_vfs.write_text", 2, 2)?;
    args.reject_keywords("_shellsim_vfs.write_text")?;
    let PyString(path) = args.positional()[0].cast(runtime)?;
    let PyString(contents) = args.positional()[1].cast(runtime)?;
    runtime.filesystem().write_text(&path, &contents)?;
    Ok(Value::None)
}

fn append_text(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_shellsim_vfs.append_text", 2, 2)?;
    args.reject_keywords("_shellsim_vfs.append_text")?;
    let PyString(path) = args.positional()[0].cast(runtime)?;
    let PyString(contents) = args.positional()[1].cast(runtime)?;
    let position = runtime.filesystem().append_text(&path, &contents)?;
    i64::try_from(position)
        .map(Value::Int)
        .map_err(|_| PyError::overflow_error("file position exceeds Python int range"))
}

fn read_bytes(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_shellsim_vfs.read_bytes", 1, 1)?;
    args.reject_keywords("_shellsim_vfs.read_bytes")?;
    let PyString(path) = args.positional()[0].cast(runtime)?;
    runtime
        .filesystem()
        .read_bytes(&path)
        .and_then(|bytes| runtime.new_bytes(bytes))
}

fn write_bytes(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_shellsim_vfs.write_bytes", 2, 2)?;
    args.reject_keywords("_shellsim_vfs.write_bytes")?;
    let PyString(path) = args.positional()[0].cast(runtime)?;
    let PyBytes(contents) = args.positional()[1].cast(runtime)?;
    runtime.filesystem().write_bytes(&path, &contents)?;
    Ok(Value::None)
}

fn append_bytes(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_shellsim_vfs.append_bytes", 2, 2)?;
    args.reject_keywords("_shellsim_vfs.append_bytes")?;
    let PyString(path) = args.positional()[0].cast(runtime)?;
    let PyBytes(contents) = args.positional()[1].cast(runtime)?;
    let position = runtime.filesystem().append_bytes(&path, &contents)?;
    i64::try_from(position)
        .map(Value::Int)
        .map_err(|_| PyError::overflow_error("file position exceeds Python int range"))
}

fn exists(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_shellsim_vfs.exists", 1, 1)?;
    args.reject_keywords("_shellsim_vfs.exists")?;
    let PyString(path) = args.positional()[0].cast(runtime)?;
    Ok(Value::Bool(runtime.filesystem().exists(&path)))
}

fn is_file(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_shellsim_vfs.is_file", 1, 1)?;
    args.reject_keywords("_shellsim_vfs.is_file")?;
    let PyString(path) = args.positional()[0].cast(runtime)?;
    Ok(Value::Bool(runtime.filesystem().is_file(&path)))
}

fn is_dir(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_shellsim_vfs.is_dir", 1, 1)?;
    args.reject_keywords("_shellsim_vfs.is_dir")?;
    let PyString(path) = args.positional()[0].cast(runtime)?;
    Ok(Value::Bool(runtime.filesystem().is_dir(&path)))
}

fn stat(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_shellsim_vfs.stat", 1, 1)?;
    args.reject_keywords("_shellsim_vfs.stat")?;
    let PyString(path) = args.positional()[0].cast(runtime)?;
    let metadata = runtime.filesystem().metadata(&path)?;
    let mode = Value::Int(i64::from(metadata.mode));
    let size = i64::try_from(metadata.size)
        .map(Value::Int)
        .map_err(|_| PyError::overflow_error("file size exceeds Python int range"))?;
    runtime.new_tuple(vec![mode, size])
}

fn mkdir(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_shellsim_vfs.mkdir", 3, 3)?;
    args.reject_keywords("_shellsim_vfs.mkdir")?;
    let PyString(path) = args.positional()[0].cast(runtime)?;
    let parents = runtime.truth(&args.positional()[1])?;
    let exist_ok = runtime.truth(&args.positional()[2])?;
    runtime.filesystem().mkdir(&path, parents, exist_ok)?;
    Ok(Value::None)
}

fn glob_paths(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_shellsim_vfs.glob", 1, 1)?;
    args.reject_keywords("_shellsim_vfs.glob")?;
    let PyString(pattern) = args.positional()[0].cast(runtime)?;
    let paths = runtime.filesystem().glob(&pattern)?;
    let mut values = Vec::with_capacity(paths.len());
    for path in paths {
        values.push(runtime.new_string(path)?);
    }
    runtime.new_list(values)
}

fn remove_file(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_shellsim_vfs.remove_file", 1, 1)?;
    args.reject_keywords("_shellsim_vfs.remove_file")?;
    let PyString(path) = args.positional()[0].cast(runtime)?;
    runtime.filesystem().remove_file(&path)?;
    Ok(Value::None)
}

fn remove_tree(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_shellsim_vfs.remove_tree", 1, 1)?;
    args.reject_keywords("_shellsim_vfs.remove_tree")?;
    let PyString(path) = args.positional()[0].cast(runtime)?;
    runtime.filesystem().remove_tree(&path)?;
    Ok(Value::None)
}

fn rename(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_shellsim_vfs.rename", 2, 2)?;
    args.reject_keywords("_shellsim_vfs.rename")?;
    let PyString(source) = args.positional()[0].cast(runtime)?;
    let PyString(destination) = args.positional()[1].cast(runtime)?;
    runtime.filesystem().rename(&source, &destination)?;
    Ok(Value::None)
}
