//! The registered Enum class for the bounded :mod:`enum` compatibility surface.
//!
//! An enum member is an ordinary instance of its enum class. Its header carries the class's type
//! id, its payload is a copy of the member value when the class mixes in a builtin such as `str`
//! or `int`, and the member name and value live in the instance attributes `_name_` and
//! `_value_`, as CPython stores them. The `Enum` type contributes the `name` and `value` getters
//! and the repr, str and hash slots that read those attributes.

use super::super::native::PyValue as Value;
use super::super::native::{
    GetterDef, ModuleDef, NativeTypeDef, PyError, PyMarker, PyResult, PyRuntime, ValueDef,
};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "enum",
    functions: &[],
    values: &[ValueDef::Factory {
        name: "Enum",
        get: enum_base,
    }],
};

pub(crate) static ENUM_TYPE: NativeTypeDef = NativeTypeDef {
    name: "enum.Enum",
    methods: &[],
    getters: &[
        GetterDef {
            owner: "enum.Enum",
            name: "name",
            get: member_name,
        },
        GetterDef {
            owner: "enum.Enum",
            name: "value",
            get: member_value,
        },
    ],
};

fn enum_base<'s>(runtime: &mut dyn PyRuntime<'s>) -> PyResult<'s> {
    Ok(runtime.marker(PyMarker::EnumType))
}

/// The instance attribute a member was given when its class was created.
fn member_attribute<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    member: Value<'s>,
    attribute: &str,
) -> PyResult<'s> {
    runtime
        .get_attribute_default(member, attribute)?
        .ok_or_else(|| {
            PyError::exception("AttributeError", format!("enum member has no {attribute}"))
        })
}

fn member_name<'s>(runtime: &mut dyn PyRuntime<'s>, member: Value<'s>) -> PyResult<'s> {
    member_attribute(runtime, member, "_name_")
}

fn member_value<'s>(runtime: &mut dyn PyRuntime<'s>, member: Value<'s>) -> PyResult<'s> {
    member_attribute(runtime, member, "_value_")
}

fn member_name_text<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    member: Value<'s>,
) -> PyResult<'s, String> {
    let name = member_name(runtime, member)?;
    runtime
        .string_value(&name)?
        .ok_or_else(|| PyError::type_error("enum member name must be a string"))
}

/// `Enum.__repr__`: `<Color.RED: 1>`.
pub(crate) fn slot_repr<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    member: Value<'s>,
) -> PyResult<'s, Option<Value<'s>>> {
    let class_name = runtime.type_name(&member)?;
    let name = member_name_text(runtime, member)?;
    let value = member_value(runtime, member)?;
    let value = runtime.repr(&value)?;
    runtime
        .new_string(format!("<{class_name}.{name}: {value}>"))
        .map(Some)
}

/// `Enum.__str__`: `Color.RED`.
pub(crate) fn slot_str<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    member: Value<'s>,
) -> PyResult<'s, Option<Value<'s>>> {
    let class_name = runtime.type_name(&member)?;
    let name = member_name_text(runtime, member)?;
    runtime.new_string(format!("{class_name}.{name}")).map(Some)
}

/// `Enum.__hash__` hashes the member name, as CPython does.
pub(crate) fn slot_hash<'s>(
    runtime: &mut dyn PyRuntime<'s>,
    member: Value<'s>,
) -> PyResult<'s, Option<Value<'s>>> {
    let name = member_name_text(runtime, member)?;
    Ok(Some(Value::Int(super::super::hash::string(&name))))
}
