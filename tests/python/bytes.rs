//! Parser and decoding failures for immutable byte sequences.

use super::support::run_python_text;

#[test]
fn bytes_reject_non_ascii_literals_and_invalid_utf8_decoding() {
    let (status, _, error) = run_python_text("b'café'");
    assert_eq!(status, 2);
    assert!(error.contains("ASCII"));

    let (status, _, error) = run_python_text("b'\\xff'.decode()");
    assert_eq!(status, 1);
    assert!(error.contains("invalid UTF-8"));
}

#[test]
fn binary_files_and_pathlib_preserve_arbitrary_octets_in_the_vfs() {
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
        run_python_text(source),
        (
            0,
            "3\nb'\\x00\\xff\\n'\n3 1\nb'\\x00\\xff\\n' b'X' 1 b'\\xff'\n".into(),
            String::new(),
        )
    );
}

#[test]
fn update_modes_share_one_seekable_read_write_cursor() {
    let source = r#"
with open('/tmp/value.txt', 'w+') as stream:
    stream.write('abcdef')
    stream.seek(2)
    stream.write('XY')
    stream.seek(0)
    print(stream.read())
with open('/tmp/value.txt', 'a+') as stream:
    stream.seek(0)
    print(stream.read())
    stream.seek(0)
    stream.write('!')
    stream.seek(0)
    print(stream.read())
with open('/tmp/sparse.bin', 'w+b') as stream:
    stream.seek(2)
    stream.write(b'x')
    stream.seek(0)
    print(stream.read())
"#;
    assert_eq!(
        run_python_text(source),
        (
            0,
            "abXYef\nabXYef\nabXYef!\nb'\\x00\\x00x'\n".into(),
            String::new(),
        )
    );
}

#[test]
fn bytes_and_bytearray_support_common_search_and_layout_methods() {
    let source = r#"
for value in (b'ababa', bytearray(b'ababa')):
    print(value.count(b'ab'), value.count(b'', 1, 3))
    print(value.partition(b'ba'), value.rpartition(b'ba'))
    missing = value.partition(b'x')
    print(missing, type(missing[0]) is type(value))
    print(value.center(8, b'.'), type(value.center(8)) is type(value))
print(bytes.count(b'aaaa', b'aa'))
mutable = bytearray(b'abcd')
print(mutable.reverse(), mutable)
"#;
    assert_eq!(
        run_python_text(source),
        (
            0,
            concat!(
                "2 3\n",
                "(b'a', b'ba', b'ba') (b'aba', b'ba', b'')\n",
                "(b'ababa', b'', b'') True\n",
                "b'.ababa..' True\n",
                "2 3\n",
                "(bytearray(b'a'), bytearray(b'ba'), bytearray(b'ba')) (bytearray(b'aba'), bytearray(b'ba'), bytearray(b''))\n",
                "(bytearray(b'ababa'), bytearray(b''), bytearray(b'')) True\n",
                "bytearray(b'.ababa..') True\n",
                "2\n",
                "None bytearray(b'dcba')\n",
            )
            .into(),
            String::new(),
        )
    );
}

#[test]
fn three_argument_pow_uses_bounded_integer_modular_exponentiation() {
    let source = r#"
print(pow(2, 10, 17), pow(-2, 3, 5), pow(2, 3, -5))
try:
    pow(2, 3, 0)
except ZeroDivisionError:
    print("zero modulus")
for arguments in (("x", 2, 3), (2, "x", 3), (2, 3, "x")):
    try:
        pow(*arguments)
    except TypeError:
        print("integer arguments")
"#;
    assert_eq!(
        run_python_text(source),
        (
            0,
            "4 2 -2\nzero modulus\ninteger arguments\ninteger arguments\ninteger arguments\n"
                .into(),
            String::new()
        )
    );
}
