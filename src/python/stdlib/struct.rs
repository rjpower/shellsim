//! Deterministic packing of common fixed-width binary records.
//!
//! The implementation intentionally supports standard sizes and explicit byte order. Native
//! alignment (`@`) is rejected because shellsim must not inherit the host machine ABI.

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, PyBytes, PyError, PyResult, PyRuntime, PyString, PyValue,
    PyValueCast,
};
use super::super::number::PyNumber;

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "_struct",
    functions: &[
        FunctionDef {
            module: "_struct",
            name: "calcsize",
            call: calcsize,
        },
        FunctionDef {
            module: "_struct",
            name: "pack",
            call: pack,
        },
        FunctionDef {
            module: "_struct",
            name: "unpack",
            call: unpack,
        },
    ],
    values: &[],
};

const MAX_STRUCT_SIZE: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy)]
enum Endian {
    Little,
    Big,
}

#[derive(Clone, Copy)]
struct Field {
    code: u8,
    count: usize,
}

struct Format {
    endian: Endian,
    fields: Vec<Field>,
    size: usize,
    values: usize,
}

fn parse_format(text: &str) -> PyResult<Format> {
    let bytes = text.as_bytes();
    let (endian, mut position) = match bytes.first() {
        Some(b'<') | Some(b'=') => (Endian::Little, 1),
        Some(b'>') | Some(b'!') => (Endian::Big, 1),
        Some(b'@') => {
            return Err(PyError::value_error(
                "native-aligned struct formats are not deterministic",
            ))
        }
        _ => (Endian::Little, 0),
    };
    let mut fields = Vec::new();
    let mut size = 0usize;
    let mut values = 0usize;
    while position < bytes.len() {
        if bytes[position].is_ascii_whitespace() {
            position += 1;
            continue;
        }
        let mut count = 0usize;
        while position < bytes.len() && bytes[position].is_ascii_digit() {
            count = count
                .checked_mul(10)
                .and_then(|value| value.checked_add(usize::from(bytes[position] - b'0')))
                .ok_or_else(|| PyError::resource_error("struct repeat count is too large"))?;
            position += 1;
        }
        if count == 0 {
            count = 1;
        }
        let code = *bytes
            .get(position)
            .ok_or_else(|| PyError::value_error("incomplete struct format"))?;
        position += 1;
        let unit: usize = match code {
            b'x' | b'c' | b'b' | b'B' | b'?' | b's' | b'p' => 1,
            b'h' | b'H' => 2,
            b'i' | b'I' | b'l' | b'L' | b'f' => 4,
            b'q' | b'Q' | b'd' => 8,
            _ => return Err(PyError::value_error("unsupported struct format character")),
        };
        let field_size = unit
            .checked_mul(count)
            .ok_or_else(|| PyError::resource_error("struct result is too large"))?;
        size = size
            .checked_add(field_size)
            .ok_or_else(|| PyError::resource_error("struct result is too large"))?;
        if size > MAX_STRUCT_SIZE {
            return Err(PyError::resource_error(
                "struct result exceeds the 4 MiB limit",
            ));
        }
        values = values
            .checked_add(match code {
                b'x' => 0,
                b's' | b'p' => 1,
                _ => count,
            })
            .ok_or_else(|| PyError::resource_error("too many struct fields"))?;
        fields.push(Field { code, count });
    }
    Ok(Format {
        endian,
        fields,
        size,
        values,
    })
}

fn calcsize(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("struct.calcsize", 1, 1)?;
    args.reject_keywords("struct.calcsize")?;
    let PyString(format) = args.positional()[0].cast(runtime)?;
    let format = parse_format(&format)?;
    Ok(PyValue::Int(
        i64::try_from(format.size).expect("bounded size fits i64"),
    ))
}

fn pack(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    if args.positional().is_empty() {
        return Err(PyError::type_error("struct.pack requires a format"));
    }
    args.reject_keywords("struct.pack")?;
    let PyString(format_text) = args.positional()[0].cast(runtime)?;
    let format = parse_format(&format_text)?;
    let values = &args.positional()[1..];
    if values.len() != format.values {
        return Err(PyError::type_error(format!(
            "pack expected {} items for packing (got {})",
            format.values,
            values.len()
        )));
    }
    runtime.reserve_memory(format.size)?;
    runtime.charge_cpu(u64::try_from(format.size).unwrap_or(u64::MAX))?;
    let mut output = Vec::with_capacity(format.size);
    let mut value_index = 0usize;
    for field in format.fields {
        match field.code {
            b'x' => output.resize(output.len() + field.count, 0),
            b's' => {
                let PyBytes(value) = values[value_index].cast(runtime)?;
                value_index += 1;
                let copied = value.len().min(field.count);
                output.extend_from_slice(&value[..copied]);
                output.resize(output.len() + field.count - copied, 0);
            }
            b'p' => {
                let PyBytes(value) = values[value_index].cast(runtime)?;
                value_index += 1;
                if field.count != 0 {
                    let copied = value.len().min(field.count.saturating_sub(1)).min(255);
                    output.push(u8::try_from(copied).expect("limited to 255"));
                    output.extend_from_slice(&value[..copied]);
                    output.resize(output.len() + field.count - copied - 1, 0);
                }
            }
            code => {
                for _ in 0..field.count {
                    pack_value(
                        runtime,
                        code,
                        format.endian,
                        values[value_index],
                        &mut output,
                    )?;
                    value_index += 1;
                }
            }
        }
    }
    runtime.new_bytes(output)
}

fn pack_value(
    runtime: &mut dyn PyRuntime,
    code: u8,
    endian: Endian,
    value: PyValue,
    output: &mut Vec<u8>,
) -> PyResult<()> {
    macro_rules! integer {
        ($type:ty) => {{
            let value = integer_value(runtime, value)?;
            let value = <$type>::try_from(value)
                .map_err(|_| PyError::exception("StructError", "integer out of range"))?;
            let bytes = match endian {
                Endian::Little => value.to_le_bytes(),
                Endian::Big => value.to_be_bytes(),
            };
            output.extend_from_slice(&bytes);
        }};
    }
    match code {
        b'c' => {
            let PyBytes(bytes) = value.cast(runtime)?;
            if bytes.len() != 1 {
                return Err(PyError::exception(
                    "StructError",
                    "char format requires a bytes object of length 1",
                ));
            }
            output.push(bytes[0]);
        }
        b'?' => output.push(u8::from(runtime.truth(&value)?)),
        b'b' => integer!(i8),
        b'B' => integer!(u8),
        b'h' => integer!(i16),
        b'H' => integer!(u16),
        b'i' | b'l' => integer!(i32),
        b'I' | b'L' => integer!(u32),
        b'q' => integer!(i64),
        b'Q' => integer!(u64),
        b'f' => {
            let value = value.cast::<PyNumber>(runtime)?.into_f64()? as f32;
            let bytes = match endian {
                Endian::Little => value.to_le_bytes(),
                Endian::Big => value.to_be_bytes(),
            };
            output.extend_from_slice(&bytes);
        }
        b'd' => {
            let value = value.cast::<PyNumber>(runtime)?.into_f64()?;
            let bytes = match endian {
                Endian::Little => value.to_le_bytes(),
                Endian::Big => value.to_be_bytes(),
            };
            output.extend_from_slice(&bytes);
        }
        _ => unreachable!("format parser validated code"),
    }
    Ok(())
}

fn integer_value(runtime: &dyn PyRuntime, value: PyValue) -> PyResult<i128> {
    if let Some(value) = runtime.int_value(&value) {
        return Ok(i128::from(value));
    }
    runtime
        .integer_text(&value)?
        .and_then(|value| value.parse::<i128>().ok())
        .ok_or_else(|| PyError::type_error("required argument is not an integer"))
}

fn unpack(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("struct.unpack", 2, 2)?;
    args.reject_keywords("struct.unpack")?;
    let PyString(format_text) = args.positional()[0].cast(runtime)?;
    let PyBytes(input) = args.positional()[1].cast(runtime)?;
    let format = parse_format(&format_text)?;
    if input.len() != format.size {
        return Err(PyError::exception(
            "StructError",
            format!("unpack requires a buffer of {} bytes", format.size),
        ));
    }
    runtime.reserve_memory(format.values.saturating_mul(std::mem::size_of::<PyValue>()))?;
    runtime.charge_cpu(u64::try_from(format.size).unwrap_or(u64::MAX))?;
    let mut offset = 0usize;
    let mut values = Vec::with_capacity(format.values);
    for field in format.fields {
        match field.code {
            b'x' => offset += field.count,
            b's' => {
                values.push(runtime.new_bytes(input[offset..offset + field.count].to_vec())?);
                offset += field.count;
            }
            b'p' => {
                let length = usize::from(input[offset]).min(field.count.saturating_sub(1));
                values.push(runtime.new_bytes(input[offset + 1..offset + 1 + length].to_vec())?);
                offset += field.count;
            }
            code => {
                let size = field_size(code);
                for _ in 0..field.count {
                    values.push(unpack_value(
                        runtime,
                        code,
                        format.endian,
                        &input[offset..offset + size],
                    )?);
                    offset += size;
                }
            }
        }
    }
    runtime.new_tuple(values)
}

fn field_size(code: u8) -> usize {
    match code {
        b'c' | b'b' | b'B' | b'?' => 1,
        b'h' | b'H' => 2,
        b'i' | b'I' | b'l' | b'L' | b'f' => 4,
        b'q' | b'Q' | b'd' => 8,
        _ => unreachable!("only scalar codes use field_size"),
    }
}

fn unpack_value(
    runtime: &mut dyn PyRuntime,
    code: u8,
    endian: Endian,
    input: &[u8],
) -> PyResult<PyValue> {
    macro_rules! integer {
        ($type:ty, $size:expr) => {{
            let bytes: [u8; $size] = input.try_into().expect("format size selected input width");
            let value = match endian {
                Endian::Little => <$type>::from_le_bytes(bytes),
                Endian::Big => <$type>::from_be_bytes(bytes),
            };
            PyValue::Int(value as i64)
        }};
    }
    Ok(match code {
        b'c' => return runtime.new_bytes(vec![input[0]]),
        b'?' => PyValue::Bool(input[0] != 0),
        b'b' => PyValue::Int(i64::from(input[0] as i8)),
        b'B' => PyValue::Int(i64::from(input[0])),
        b'h' => integer!(i16, 2),
        b'H' => integer!(u16, 2),
        b'i' | b'l' => integer!(i32, 4),
        b'I' | b'L' => integer!(u32, 4),
        b'q' => integer!(i64, 8),
        b'Q' => {
            let bytes: [u8; 8] = input.try_into().expect("format size selected input width");
            let value = match endian {
                Endian::Little => u64::from_le_bytes(bytes),
                Endian::Big => u64::from_be_bytes(bytes),
            };
            return runtime.new_integer(&value.to_string());
        }
        b'f' => {
            let bytes: [u8; 4] = input.try_into().expect("format size selected input width");
            PyValue::Float(f64::from(match endian {
                Endian::Little => f32::from_le_bytes(bytes),
                Endian::Big => f32::from_be_bytes(bytes),
            }))
        }
        b'd' => {
            let bytes: [u8; 8] = input.try_into().expect("format size selected input width");
            PyValue::Float(match endian {
                Endian::Little => f64::from_le_bytes(bytes),
                Endian::Big => f64::from_be_bytes(bytes),
            })
        }
        _ => unreachable!("format parser validated code"),
    })
}
