//! Compatibility tests for stdlib modules bundled as Python source and executed by the VM.

use shellsim::{python, Environment, Limits};

use super::support::{run_python_text as run, run_python_text_in as run_in};

#[test]
fn urllib_request_uses_typed_routes_and_file_like_responses() {
    let mut environment = Environment::new();
    environment
        .net
        .route(
            "https://api.test/items",
            Some("GET"),
            shellsim::net::HttpResponse {
                status: 200,
                headers: vec![("Content-Type".into(), "application/json".into())],
                body: b"line one\nline two\n".to_vec(),
            },
        )
        .unwrap();
    let source = r#"
from urllib.request import Request, urlopen

request = Request('https://api.test/items', headers={'X-Test': 'yes'})
with urlopen(request) as response:
    print(response.status, response.getcode(), response.geturl())
    print(response.headers['content-type'], response.getheader('CONTENT-TYPE'))
    print(response.read(5), response.readline())
    buffer = bytearray(b'........')
    print(response.readinto(buffer), bytes(buffer))
    print(response.readinto(buffer), bytes(buffer))
    print(response.readinto(buffer), bytes(buffer))
    try:
        response.readinto(b'')
    except TypeError as error:
        print(error)
"#;

    assert_eq!(
        run_in(&mut environment, source),
        (
            0,
            "200 200 https://api.test/items\napplication/json application/json\nb'line ' b'one\\n'\n8 b'line two'\n1 b'\\nine two'\n0 b'\\nine two'\nreadinto() requires a bytearray\n".into(),
            String::new(),
        )
    );
    assert_eq!(environment.net.log.len(), 1);
    assert_eq!(environment.net.log[0].method, "GET");
    assert_eq!(
        environment.net.log[0].headers,
        [("X-Test".into(), "yes".into())]
    );
}

#[test]
fn copy_textwrap_pathlib_and_shutil_cover_common_file_recipes() {
    let mut environment = Environment::new();
    environment
        .vfs
        .mkdir_all("/", "/work/source/nested")
        .unwrap();
    environment
        .vfs
        .put_file("/work/source/value.txt", b"value".to_vec(), 0o644)
        .unwrap();
    environment
        .vfs
        .put_file("/work/source/nested/leaf.txt", b"leaf".to_vec(), 0o644)
        .unwrap();
    let source = r#"import copy
import shutil
import textwrap
from pathlib import Path

original = {"items": [[1], [2]]}
cloned = copy.deepcopy(original)
cloned["items"][0].append(3)
print(original, cloned)
print(repr(textwrap.dedent("    one\n      two\n")))
print(textwrap.fill("one two three four", width=9))
print(repr(textwrap.indent("a\n\nb\n", "> ")))
print(textwrap.shorten("one two three four", 13))

path = Path("archive.tar.gz")
print(path.suffixes, path.with_name("other.txt"), path.as_posix())
created = Path("/work/touched")
created.touch()
print(created.exists(), created.read_bytes())

print(shutil.copy("/work/source/value.txt", "/work/copied.txt"))
print(shutil.copytree("/work/source", "/work/tree"))
print(Path("/work/copied.txt").read_text(), Path("/work/tree/nested/leaf.txt").read_text())
shutil.rmtree("/work/tree")
print(Path("/work/tree").exists())
"#;
    assert_eq!(
        run_in(&mut environment, source),
        (
            0,
            "{'items': [[1], [2]]} {'items': [[1, 3], [2]]}\n'one\\n  two\\n'\none two\nthree\nfour\n'> a\\n\\n> b\\n'\none two [...]\n['.tar', '.gz'] other.txt archive.tar.gz\nTrue b''\n/work/copied.txt\n/work/tree\nvalue leaf\nFalse\n".into(),
            String::new()
        )
    );
}

#[test]
fn urllib_post_redirects_and_errors_preserve_http_semantics() {
    let mut environment = Environment::new();
    environment
        .net
        .route(
            "https://api.test/start",
            Some("POST"),
            shellsim::net::HttpResponse {
                status: 302,
                headers: vec![("Location".into(), "/done".into())],
                body: Vec::new(),
            },
        )
        .unwrap();
    environment
        .net
        .route_static("https://api.test/done", 200, b"finished")
        .unwrap();
    environment
        .net
        .route_static("https://api.test/missing", 404, b"missing body")
        .unwrap();
    let source = r#"
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

print(urlopen(Request('https://api.test/start', data=b'name=test')).read())
try:
    urlopen('https://api.test/missing')
except URLError as error:
    print(isinstance(error, HTTPError), error.code, error.reason, error.read())
try:
    urlopen('https://unmatched.test/')
except OSError as error:
    print(error.reason)
"#;

    assert_eq!(
        run_in(&mut environment, source),
        (
            0,
            "b'finished'\nTrue 404 Not Found b'missing body'\nno matching virtual HTTP route for https://unmatched.test/\n".into(),
            String::new(),
        )
    );
    assert_eq!(
        environment
            .net
            .log
            .iter()
            .map(|request| (request.method.as_str(), request.body_bytes, request.matched))
            .collect::<Vec<_>>(),
        [
            ("POST", 9, true),
            ("GET", 0, true),
            ("GET", 0, true),
            ("GET", 0, false),
        ]
    );
    assert!(environment.net.log[0].headers.contains(&(
        "Content-Type".into(),
        "application/x-www-form-urlencoded".into()
    )));
    assert!(environment.net.log[0]
        .headers
        .contains(&("Content-Length".into(), "9".into())));
}

#[test]
fn http_client_builds_urls_and_rejects_unmodeled_connection_features() {
    let mut environment = Environment::new();
    environment
        .net
        .route_static("http://service.test/ping", 200, b"pong")
        .unwrap();
    environment
        .net
        .route_static("https://secure.test:8443/data", 201, b"created")
        .unwrap();
    let source = r#"
from http.client import HTTPConnection, HTTPSConnection

connection = HTTPConnection('service.test')
connection.request('GET', '/ping', headers={'X-Test': 'one'})
response = connection.getresponse()
print(response.status, response.reason, response.read())

secure = HTTPSConnection('secure.test', 8443)
secure.request('POST', 'data', body='value')
print(secure.getresponse().status, secure.getresponse().read())
try:
    connection.set_tunnel('proxy.test')
except ValueError as error:
    print(error)
"#;

    assert_eq!(
        run_in(&mut environment, source),
        (
            0,
            "200 OK b'pong'\n201 b'created'\nHTTP CONNECT tunnels are not supported by shellsim\n"
                .into(),
            String::new(),
        )
    );
    assert_eq!(environment.net.log[1].method, "POST");
    assert_eq!(environment.net.log[1].body_bytes, 5);
}

#[test]
fn urllib_rejects_non_http_urls_and_custom_handlers_explicitly() {
    let source = r#"
from urllib.request import build_opener, urlopen

for operation in [lambda: urlopen('file:///etc/passwd'), lambda: build_opener(1)]:
    try:
        operation()
    except ValueError as error:
        print(error)
"#;
    assert_eq!(
        run(source),
        (
            0,
            "only http:// and https:// URLs are supported\ncustom urllib handlers are not supported by shellsim\n".into(),
            String::new(),
        )
    );
}

#[test]
fn csv_reader_and_dict_reader_handle_quoted_records() {
    let source = r#"
import csv
rows = list(csv.reader(['name,note\n', 'Ada,"one, two"\n', 'Lin,"line 1\n', 'line 2"\n', '\n']))
print(rows)
print(list(csv.DictReader(['name,value\n', 'a,1\n', '\n'])))
"#;
    assert_eq!(
        run(source),
        (
            0,
            "[['name', 'note'], ['Ada', 'one, two'], ['Lin', 'line 1\\nline 2'], []]\n[{'name': 'a', 'value': '1'}]\n".into(),
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
print(path.relative_to(root), path.stat().st_size, path.stat().st_mode)
"#;
    assert_eq!(
        run(source),
        (
            0,
            "/tmp/sample.txt /tmp sample.txt sample .txt\nFalse 5 True hello\nhello!\nsample.txt 6 420\n".into(),
            String::new(),
        )
    );
}

#[test]
fn pathlib_relative_to_rejects_paths_outside_the_base() {
    let source = r#"
from pathlib import Path

try:
    Path('/tmp/other').relative_to('/work')
except ValueError as error:
    print('ValueError')
"#;
    assert_eq!(run(source), (0, "ValueError\n".into(), String::new()));
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
(directory / 'nested').mkdir()
(directory / 'nested' / 'c.txt').write_text('c')
print(glob.glob('/tmp/data/*.txt'))
print([path.name for path in directory.glob('*.json')])
print([str(path) for path in directory.rglob('*.txt')])
print([path.name for path in directory.iterdir()])
"#;
    assert_eq!(
        run(source),
        (
            0,
            "['/tmp/data/a.txt']\n['b.json']\n['/tmp/data/a.txt', '/tmp/data/nested/c.txt']\n['a.txt', 'b.json', 'nested']\n"
                .into(),
            String::new()
        )
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
fn frozen_random_is_deterministic_and_covers_common_sequence_helpers() {
    let source = r#"
import random

random.seed(7)
print(random.randint(1, 10), f'{random.random():.6f}', random.choice(['a', 'b', 'c']))
values = [1, 2, 3, 4]
random.shuffle(values)
print(values, random.sample(values, 2))
first = random.Random(11)
second = random.Random(11)
print(first.random() == second.random(), first.randrange(10, 20, 2))
try:
    random.randrange(0)
except ValueError:
    print('empty')
"#;
    assert_eq!(
        run(source),
        (
            0,
            "7 0.299265 b\n[3, 2, 1, 4] [1, 3]\nTrue 10\nempty\n".into(),
            String::new(),
        )
    );
}

#[test]
fn frozen_random_gauss_reuses_the_second_box_muller_sample() {
    let source = r#"import math
import random

first = random.Random(7)
reference = random.Random(7)
angle = math.tau * reference.random()
radius = math.sqrt(-2.0 * math.log(1.0 - reference.random()))
print(abs(first.gauss(3.0, 2.0) - (3.0 + 2.0 * math.cos(angle) * radius)) < 1e-12)
print(abs(first.gauss() - math.sin(angle) * radius) < 1e-12)
print(first.random() == reference.random())
first.seed(7)
print(abs(first.gauss() - math.cos(angle) * radius) < 1e-12)
random.seed(7)
print(isinstance(random.gauss(), float))
"#;
    assert_eq!(
        run(source),
        (0, "True\nTrue\nTrue\nTrue\nTrue\n".into(), String::new())
    );
}

#[test]
fn frozen_zipfile_reads_writes_and_extracts_vfs_archives() {
    let source = r#"import io
import zipfile
from pathlib import Path

buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w') as archive:
    archive.writestr('one.txt', b'first')
    archive.writestr('nested/two.bin', bytes([0, 1, 255]))

with zipfile.ZipFile(io.BytesIO(buffer.getvalue()), 'r') as archive:
    print(archive.namelist())
    print(archive.read('one.txt'), archive.open('nested/two.bin').read())
    print([(item.filename, item.file_size) for item in archive.infolist()])
    archive.extractall('/output')

print(Path('/output/one.txt').read_text(), Path('/output/nested/two.bin').read_bytes())
with zipfile.ZipFile('/work/items.zip', 'w') as archive:
    archive.write('/output/one.txt', 'renamed.txt')
print(zipfile.is_zipfile('/work/items.zip'))
with zipfile.ZipFile('/work/items.zip') as archive:
    print(archive.read('renamed.txt'))
"#;
    assert_eq!(
        run(source),
        (
            0,
            concat!(
                "['one.txt', 'nested/two.bin']\n",
                "b'first' b'\\x00\\x01\\xff'\n",
                "[('one.txt', 5), ('nested/two.bin', 3)]\n",
                "first b'\\x00\\x01\\xff'\n",
                "True\n",
                "b'first'\n",
            )
            .into(),
            String::new(),
        )
    );
}

#[test]
fn frozen_statistics_path_parts_json_error_and_access_cover_common_calls() {
    let source = r#"import json
import os
import statistics
from pathlib import Path

print(statistics.fmean([1, 2, 6]))
print(statistics.median([4, 1, 3, 2]), statistics.pstdev([2, 2]))
print(statistics.multimode([1, 2, 1, 2, 3]))
print(statistics.quantiles([0, 10, 20, 30, 40], n=4, method="inclusive"))
print(Path("/work/item.txt").parts, Path("a/b").parts)
print(Path("/work/item.txt").is_absolute(), Path("a/b").is_absolute())
Path("/work/tool").write_text("x")
print(os.access("/work/tool", os.F_OK), os.access("/work/tool", os.R_OK), os.access("/missing", os.F_OK))
try:
    json.loads("{")
except json.JSONDecodeError:
    print("json-error")
"#;
    let (status, stdout, stderr) = run(source);
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(
        stdout,
        "3.0\n2.5 0.0\n[1, 2]\n[10.0, 20.0, 30.0]\n('/', 'work', 'item.txt') ('a', 'b')\nTrue False\nTrue True False\njson-error\n"
    );
    assert!(stderr.is_empty());
}

#[test]
fn frozen_collections_use_generic_container_protocols() {
    let source = r#"
from collections import Counter, deque, defaultdict

counts = Counter("abracadabra")
print(counts["a"], counts["z"], counts.most_common(2), counts.total())
counts.subtract("ab")
print(list(counts.elements()))

values = deque([1, 2, 3], 3)
values.append(4)
values.appendleft(0)
values.rotate(-1)
print(len(values), values[0], list(values), values.popleft())
print(defaultdict(list)["missing"])
"#;
    assert_eq!(
        run(source),
        (
            0,
            "5 0 [('a', 5), ('b', 2)] 11\n['a', 'a', 'a', 'a', 'b', 'r', 'r', 'c', 'd']\n3 2 [2, 3, 0] 2\n[]\n".into(),
            String::new(),
        )
    );
}

#[test]
fn frozen_json_stream_helpers_wrap_the_bounded_codec() {
    let source = r#"
import json

class Buffer:
    def __init__(self):
        self.value = ""
    def write(self, value):
        self.value += value
    def read(self):
        return self.value

stream = Buffer()
json.dump({'b': 2, 'a': 1}, stream, sort_keys=True)
print(stream.value)
print(json.load(stream))
print(json.dumps({'a': [1, 2]}, indent=2))
"#;
    assert_eq!(
        run(source),
        (
            0,
            "{\"a\": 1, \"b\": 2}\n{'a': 1, 'b': 2}\n{\n  \"a\": [\n    1,\n    2\n  ]\n}\n".into(),
            String::new()
        )
    );
}

#[test]
fn frozen_os_path_helpers_use_only_modeled_state() {
    let source = r#"
import os

print(os.getcwd(), os.getenv('HOME'))
print(os.path.join('/tmp', 'a', 'b.txt'))
print(os.path.basename('/tmp/a/b.txt'), os.path.dirname('/tmp/a/b.txt'))
print(os.path.splitext('/tmp/a/b.txt'))
print(os.path.normpath('/tmp/a/../b'), os.path.abspath('work'))
os.makedirs('/tmp/tree/leaf', exist_ok=True)
print(os.path.exists('/tmp/tree'), os.path.isdir('/tmp/tree/leaf'))
"#;
    assert_eq!(
        run(source),
        (
            0,
            "/ /root\n/tmp/a/b.txt\nb.txt /tmp/a\n('/tmp/a/b', '.txt')\n/tmp/b /work\nTrue True\n"
                .into(),
            String::new()
        )
    );
}

#[test]
fn frozen_datetime_uses_the_virtual_clock_and_gregorian_arithmetic() {
    let source = r#"
from datetime import UTC, date, datetime, timedelta, timezone

print(datetime.now(timezone.utc).isoformat())
print(datetime.fromtimestamp(0, UTC).strftime('%Y-%m-%d %H:%M:%S'))
start = datetime.strptime('2024-02-28', '%Y-%m-%d')
end = start + timedelta(days=2)
print(end.strftime('%Y-%m-%d'), (end - start).days)
print(datetime.fromisoformat('2024-01-02T03:04:05Z').timestamp())
print(datetime(2024, 1, 2, 3, 4).replace(minute=9, tzinfo=UTC).isoformat())
print(date.today())
"#;
    assert_eq!(
        run(source),
        (
            0,
            "2025-01-01T00:00:00+00:00\n1970-01-01 00:00:00\n2024-03-01 2\n1704164645.0\n2024-01-02T03:09:00+00:00\n2025-01-01\n".into(),
            String::new()
        )
    );
}

#[test]
fn frozen_base64_and_zlib_preserve_arbitrary_bytes() {
    let source = r#"
import base64
import zlib

value = b'\x00\xffhello'
encoded = base64.b64encode(value)
print(encoded, base64.b64decode(encoded))
compressed = zlib.compress(value)
print(zlib.decompress(compressed), zlib.crc32(b'123456789'))
"#;
    assert_eq!(
        run(source),
        (
            0,
            "b'AP9oZWxsbw==' b'\\x00\\xffhello'\nb'\\x00\\xffhello' 3421780262\n".into(),
            String::new(),
        )
    );
}

#[test]
fn frozen_struct_packs_standard_width_binary_records() {
    let source = r#"
import struct

packed = struct.pack('>Hif4s', 513, -7, 1.5, b'xy')
print(struct.calcsize('>Hif4s'), packed)
print(struct.unpack('>Hif4s', packed))
print(struct.unpack('<Q', struct.pack('<Q', 18446744073709551615)))
"#;
    assert_eq!(
        run(source),
        (
            0,
            "14 b'\\x02\\x01\\xff\\xff\\xff\\xf9?\\xc0\\x00\\x00xy\\x00\\x00'\n(513, -7, 1.5, b'xy\\x00\\x00')\n(18446744073709551615,)\n".into(),
            String::new(),
        )
    );
}

#[test]
fn source_codecs_tempfiles_and_path_replacement_use_runtime_protocols() {
    let source = r#"
import codecs
import io
import logging
import tempfile
from pathlib import Path

print(codecs.encode('Hello, World!', 'rot_13'))
print(codecs.decode('Uryyb, Jbeyq!', 'rot_13'))
text_buffer = io.StringIO()
text_buffer.write('text')
byte_buffer = io.BytesIO(b'ab')
byte_buffer.seek(2)
byte_buffer.write(b'\xff')
print(text_buffer.getvalue(), byte_buffer.getvalue())
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger().addHandler(handler)
with tempfile.TemporaryDirectory(prefix='case-', dir='/tmp') as directory:
    source = Path(directory) / 'old.bin'
    target = Path(directory) / 'new.bin'
    source.write_bytes(b'\x00\xff')
    print(source.replace(target), target.read_bytes())
    print(source.exists(), target.exists())
print(Path(directory).exists())
with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.dat') as stream:
    stream.write(b'ok')
print(stream.name, Path(stream.name).read_bytes())
"#;
    assert_eq!(
        run(source),
        (
            0,
            "Uryyb, Jbeyq!\nHello, World!\ntext b'ab\\xff'\n/tmp/case-00000000000040008000000000000001/new.bin b'\\x00\\xff'\nFalse True\nFalse\n/tmp/tmp00000000000040008000000000000002.dat b'ok'\n".into(),
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
