//! Compatibility tests for stdlib modules bundled as Python source and executed by the VM.

use shellsim::{python, Environment, Limits};

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
fn csv_reader_and_dict_reader_handle_quoted_records() {
    let source = r#"
import csv
rows = list(csv.reader(['name,note\n', 'Ada,"one, two"\n', 'Lin,"line 1\n', 'line 2"\n']))
print(rows)
print(list(csv.DictReader(['name,value\n', 'a,1\n'])))
"#;
    assert_eq!(
        run(source),
        (
            0,
            "[['name', 'note'], ['Ada', 'one, two'], ['Lin', 'line 1\\nline 2']]\n[{'name': 'a', 'value': '1'}]\n".into(),
            String::new(),
        )
    );
}

#[test]
fn csv_writer_uses_an_ordinary_python_file_protocol() {
    let source = r#"
import csv

class Sink:
    def __init__(self):
        self.value = ''

    def write(self, value):
        self.value += value
        return len(value)

sink = Sink()
output = csv.DictWriter(sink, ['name', 'note'], lineterminator='\n')
output.writeheader()
output.writerow({'name': 'Ada', 'note': 'one, two'})
print(sink.value)
"#;
    assert_eq!(
        run(source),
        (0, "name,note\nAda,\"one, two\"\n\n".into(), String::new())
    );
}

#[test]
fn builtin_open_and_csv_share_the_modeled_vfs() {
    let source = r#"
import csv

with open('/tmp/items.csv', 'w', newline='') as stream:
    output = csv.writer(stream, lineterminator='\n')
    output.writerow(['name', 'note'])
    output.writerow(['Ada', 'one, two'])

with open('/tmp/items.csv', 'r') as stream:
    print(list(csv.reader(stream)))
"#;
    assert_eq!(
        run(source),
        (
            0,
            "[['name', 'note'], ['Ada', 'one, two']]\n".into(),
            String::new(),
        )
    );
}

#[test]
fn frozen_pathlib_uses_paths_without_host_filesystem_access() {
    let source = r#"
from pathlib import Path

root = Path('/tmp')
path = root / 'sample.txt'
print(path, path.parent, path.name, path.stem, path.suffix)
print(path.exists(), path.write_text('hello'), path.is_file(), path.read_text())
with path.open('a') as stream:
    stream.write('!')
print(path.read_text())
"#;
    assert_eq!(
        run(source),
        (
            0,
            "/tmp/sample.txt /tmp sample.txt sample .txt\nFalse 5 True hello\nhello!\n".into(),
            String::new(),
        )
    );
}

#[test]
fn frozen_glob_and_pathlib_mkdir_observe_only_vfs_entries() {
    let source = r#"
import glob
from pathlib import Path

directory = Path('/tmp/data')
directory.mkdir(parents=True)
(directory / 'a.txt').write_text('a')
(directory / 'b.json').write_text('b')
print(glob.glob('/tmp/data/*.txt'))
print([path.name for path in directory.glob('*.json')])
"#;
    assert_eq!(
        run(source),
        (0, "['/tmp/data/a.txt']\n['b.json']\n".into(), String::new())
    );
}

#[test]
fn frozen_abc_and_logging_cover_capability_free_setup_code() {
    let source = r#"
from abc import ABC, abstractmethod
from typing import Any, Optional
import logging

class Example(ABC):
    @abstractmethod
    def value(self):
        return 42

logging.basicConfig(level=logging.INFO)
logging.getLogger(__name__).info("loaded")
print(Example().value(), Any, Optional)
"#;
    assert_eq!(
        run(source),
        (0, "42 typing.Any typing.Optional\n".into(), String::new())
    );
}

#[test]
fn frozen_hashlib_and_uuid_use_small_native_or_deterministic_cores() {
    let source = r#"
import hashlib
import uuid

digest = hashlib.sha256("ab".encode())
digest.update(b"c")
print(digest.hexdigest())
first = uuid.uuid4()
second = uuid.uuid4()
print(str(first), first.hex, first != second)
"#;
    assert_eq!(
        run(source),
        (
            0,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\n00000000-0000-4000-8000-000000000001 00000000000040008000000000000001 True\n".into(),
            String::new(),
        )
    );
}

#[test]
fn python_filesystem_writes_obey_vfs_quota_atomically() {
    let mut environment = Environment::with_limits(Limits {
        disk: 1024,
        ..Limits::unlimited()
    });
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status = python::run_python(
        &mut environment,
        &[
            "python3.14".into(),
            "-c".into(),
            "open('/tmp/limited.txt', 'w').write('x' * 2000)".into(),
        ],
        Vec::new(),
        &mut stdout,
        &mut stderr,
    );

    assert_ne!(status, 0);
    assert!(stdout.is_empty());
    let stderr = String::from_utf8(stderr).expect("UTF-8 stderr");
    assert!(
        stderr.contains("No space left on device"),
        "unexpected stderr: {stderr:?}"
    );
    assert_eq!(
        environment
            .vfs
            .read_string("/", "/tmp/limited.txt")
            .expect("empty file created before failed replacement"),
        ""
    );
    assert!(environment.vfs.disk_used() <= 1024);
}
