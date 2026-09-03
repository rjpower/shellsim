use std::process::Command;

use shellsim::Environment;

fn run(source: &str) -> (i32, Vec<u8>, Vec<u8>) {
    let mut environment = Environment::new();
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status = shellsim::python::run_python(
        &mut environment,
        &["python3.14".into(), "-c".into(), source.into()],
        Vec::new(),
        &mut stdout,
        &mut stderr,
    );
    (status, stdout, stderr)
}

#[test]
fn math_and_string_constants_match_cpython() {
    let source = r#"import math
import string
prefix = string.ascii_lowercase[0] + string.ascii_lowercase[1] + string.ascii_lowercase[2]
print(math.sqrt(9), math.ceil(3 / 2), math.isinf(math.inf), prefix)
print(math.log(8, 2), math.sin(0))"#;
    let simulated = run(source);
    assert_eq!(
        simulated,
        (0, b"3.0 2 True abc\n3.0 0.0\n".to_vec(), Vec::new())
    );

    let reference = Command::new("python3.14").arg("-c").arg(source).output();
    if let Ok(reference) = reference {
        assert_eq!(simulated.0, reference.status.code().unwrap_or(1));
        assert_eq!(simulated.1, reference.stdout);
        assert_eq!(simulated.2, reference.stderr);
    }
}

#[test]
fn finite_iterator_heap_bisect_reduce_and_safe_imports_match_cpython() {
    let source = r#"import itertools
import heapq
import bisect
import functools
import typing
import subprocess

counter = itertools.count(3)
print(list(itertools.islice(counter, 3)))
values = [4, 1, 3]
heapq.heapify(values)
print(heapq.heappop(values), bisect.bisect_left([1, 3, 5], 4))
print(functools.reduce(lambda a, b: a + b, [1, 2, 3]))
print(typing.List[int])
print(next(counter))
"#;
    let simulated = run(source);
    let reference = Command::new("python3.14").arg("-c").arg(source).output();
    if let Ok(reference) = reference {
        assert_eq!(simulated.0, reference.status.code().unwrap_or(1));
        assert_eq!(simulated.1, reference.stdout);
        assert_eq!(simulated.2, reference.stderr);
    }
    assert_eq!(
        simulated,
        (
            0,
            b"[3, 4, 5]\n1 2\n6\ntyping.List[int]\n6\n".to_vec(),
            Vec::new()
        )
    );
}

#[test]
fn infinite_count_cannot_be_materialized_without_a_bound() {
    let simulated = run("import itertools\nprint(list(itertools.count()))");
    assert_ne!(simulated.0, 0);
    assert!(simulated.2.windows(8).any(|window| window == b"infinite"));
}

#[test]
fn regex_basic_operations_match_cpython() {
    let source = r##"import re
text = "A12 b34"
m = re.search(r"([A-Z])(\d+)", text)
print(m.group(0), m.group(1), m.group(2), m.start(), m.end())
print(re.match(r"[A-Z]", text).group())
print(re.findall(r"\d+", text))
print(re.sub(r"\d+", "#", text))
print(re.escape("a+b? c"))
print(re.search(r"b", text, flags=re.IGNORECASE).group())
"##;
    let simulated = run(source);
    let reference = Command::new("python3.14").arg("-c").arg(source).output();
    if let Ok(reference) = reference {
        assert_eq!(simulated.0, reference.status.code().unwrap_or(1));
        assert_eq!(simulated.1, reference.stdout);
        assert_eq!(simulated.2, reference.stderr);
    }
    assert_eq!(
        simulated,
        (
            0,
            b"A12 A 12 0 3\nA\n['12', '34']\nA# b#\na\\+b\\?\\ c\nb\n".to_vec(),
            Vec::new()
        )
    );
}

#[test]
fn argparse_parser_and_namespace_are_capability_free() {
    let source = r#"import argparse
parser = argparse.ArgumentParser(prog="demo")
parser.add_argument("--count", type=int, default=2)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args(["--count", "7", "--verbose"])
print(parser.prog, args.count, args.verbose)
"#;
    let simulated = run(source);
    assert_eq!(simulated, (0, b"demo 7 True\n".to_vec(), Vec::new()));
    let reference = Command::new("python3.14").arg("-c").arg(source).output();
    if let Ok(reference) = reference {
        assert_eq!(simulated.0, reference.status.code().unwrap_or(1));
        assert_eq!(simulated.1, reference.stdout);
        assert_eq!(simulated.2, reference.stderr);
    }
}

#[test]
fn regex_frontier_constructs_fail_closed() {
    for source in [
        r#"import re
print(re.search(r"(?=a)", "a"))
"#,
        r#"import re
print(re.search(r"(a)\1", "aa"))
"#,
    ] {
        let simulated = run(source);
        assert_ne!(simulated.0, 0, "unsupported regex syntax was accepted");
        assert!(
            !simulated.2.is_empty(),
            "unsupported regex syntax failed silently"
        );
    }
}
