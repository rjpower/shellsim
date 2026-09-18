//! Parser and decoding failures for immutable byte sequences.

use super::support::run_python_text;

#[test]
fn bytes_reject_non_ascii_literals_and_invalid_utf8_decoding() {
    let (status, _, error) = run_python_text("b'café'");
    assert_eq!(status, 2);
    assert!(error.contains("ASCII"));

    let (status, _, error) = run_python_text("b'\\xff'.decode()");
    assert_eq!(status, 2);
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
