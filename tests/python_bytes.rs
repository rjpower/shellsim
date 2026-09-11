//! Differential behavior for immutable byte sequences and their text conversion boundary.

use shellsim::{python, Environment};

fn run(source: &str) -> (i32, String, String) {
    let mut environment = Environment::default();
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status = python::run_python(
        &mut environment,
        &["python3.14".into(), "-c".into(), source.into()],
        Vec::new(),
        &mut stdout,
        &mut stderr,
    );
    (
        status,
        String::from_utf8(stdout).expect("UTF-8 stdout"),
        String::from_utf8(stderr).expect("UTF-8 stderr"),
    )
}

#[test]
fn bytes_are_distinct_byte_preserving_values() {
    let source = r#"
value = b'A\x00\xff\n'
print(type(value) is bytes, repr(value), len(value), value[0], value[-1])
print(value[1:3], list(value), value.hex())
print(b'ab' + b'cd', b'xy' * 2, 255 in value, b'a' < b'b')
print('café'.encode().decode())
print(bytes([0, 127, 255]), bytes(3), bytes('ok', 'utf-8'))
print(rb'\xff', br'\n')
print(value.find(b'\xff'), b'caf\xe9'.decode('latin-1'), 'ÿ'.encode('latin-1'))
"#;
    assert_eq!(
        run(source),
        (
            0,
            "True b'A\\x00\\xff\\n' 4 65 10\nb'\\x00\\xff' [65, 0, 255, 10] 4100ff0a\nb'abcd' b'xyxy' True True\ncafé\nb'\\x00\\x7f\\xff' b'\\x00\\x00\\x00' b'ok'\nb'\\\\xff' b'\\\\n'\n2 café b'\\xff'\n".into(),
            String::new(),
        )
    );
}

#[test]
fn bytes_reject_non_ascii_literals_and_invalid_utf8_decoding() {
    let (status, _, error) = run("b'café'");
    assert_eq!(status, 2);
    assert!(error.contains("ASCII"));

    let (status, _, error) = run("b'\\xff'.decode()");
    assert_eq!(status, 2);
    assert!(error.contains("invalid UTF-8"));
}

#[test]
fn binary_files_and_pathlib_preserve_arbitrary_octets() {
    let source = r#"
from pathlib import Path

path = Path('/tmp/data.bin')
print(path.write_bytes(b'\x00\xff\n'))
print(path.read_bytes())
with open(path, 'ab') as stream:
    print(stream.tell(), stream.write(b'X'))
with open(path, 'rb') as stream:
    print(stream.readline(), stream.read(), stream.seek(1), stream.read(1))
"#;
    assert_eq!(
        run(source),
        (
            0,
            "3\nb'\\x00\\xff\\n'\n3 1\nb'\\x00\\xff\\n' b'X' 1 b'\\xff'\n".into(),
            String::new(),
        )
    );
}

#[test]
fn bytearray_mutates_through_normal_sequence_slots() {
    let source = r#"
value = bytearray(b'ab')
value.append(255)
value.extend([0, 1])
value[0] = 90
print(value, list(value), len(value), value.hex())
print(value[1:4], type(value[1:4]) is bytearray)
print(bytes(value))
print(bytearray(b'ab') + b'cd', 2 * bytearray(b'x'))
"#;
    assert_eq!(
        run(source),
        (
            0,
            "bytearray(b'Zb\\xff\\x00\\x01') [90, 98, 255, 0, 1] 5 5a62ff0001\nbytearray(b'b\\xff\\x00') True\nb'Zb\\xff\\x00\\x01'\nbytearray(b'abcd') bytearray(b'xx')\n".into(),
            String::new(),
        )
    );
}
