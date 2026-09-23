//! Standard-library behavior requiring VFS setup, exact diagnostics, or CPython comparison.

use std::process::Command;

use shellsim::Environment;

use super::support::{run_python as run, run_python_in as run_in};

#[test]
fn os_environ_supports_required_key_lookup() {
    let mut environment = Environment::new();
    environment.set_var("SHELLSIM_REQUIRED", "present");

    assert_eq!(
        run_in(
            &mut environment,
            "import os\nprint(os.environ['SHELLSIM_REQUIRED'])"
        ),
        (0, b"present\n".to_vec(), Vec::new())
    );
}

#[test]
fn print_accepts_common_output_keywords() {
    assert_eq!(
        run("print('a', 'b', sep=':', end='!')"),
        (0, b"a:b!".to_vec(), Vec::new())
    );
}

#[test]
fn os_chdir_changes_only_the_simulated_process_directory() {
    let mut environment = Environment::new();
    environment.set_var("PWD", "/work");
    environment.vfs.mkdir_all("/", "/work/project").unwrap();
    environment
        .vfs
        .put_file("/work/project/value.txt", b"inside\n".to_vec(), 0o644)
        .unwrap();
    let source = r#"import os
print(os.getcwd())
os.chdir("project")
print(os.getcwd())
print(open("value.txt").read().strip())
"#;
    assert_eq!(
        run_in(&mut environment, source),
        (0, b"/work\n/work/project\ninside\n".to_vec(), Vec::new())
    );
    assert_eq!(environment.cwd, "/work");
}

#[test]
fn os_listdir_and_walk_traverse_only_the_modeled_vfs() {
    let mut environment = Environment::new();
    environment.vfs.mkdir_all("/", "/work/tree/nested").unwrap();
    environment
        .vfs
        .put_file("/work/tree/root.txt", b"root".to_vec(), 0o644)
        .unwrap();
    environment
        .vfs
        .put_file("/work/tree/nested/leaf.txt", b"leaf".to_vec(), 0o644)
        .unwrap();
    environment
        .vfs
        .symlink("/", "/work/tree/nested", "/work/tree/link")
        .unwrap();
    let source = r#"import os
print(os.listdir("/work/tree"))
for root, directories, files in os.walk("/work/tree"):
    print(root, directories, files)
"#;
    assert_eq!(
        run_in(&mut environment, source),
        (
            0,
            b"['link', 'nested', 'root.txt']\n/work/tree ['link', 'nested'] ['root.txt']\n/work/tree/nested [] ['leaf.txt']\n".to_vec(),
            Vec::new()
        )
    );
}

#[test]
fn os_paths_accept_the_path_like_protocol() {
    let mut environment = Environment::new();
    environment.vfs.mkdir_all("/", "/work/tree").unwrap();
    environment
        .vfs
        .put_file("/work/tree/value.txt", b"value".to_vec(), 0o644)
        .unwrap();
    let source = r#"from pathlib import Path
import os
root = Path("/work/tree")
print(os.fspath(root), os.listdir(root))
print(list(os.walk(root)))
"#;
    assert_eq!(
        run_in(&mut environment, source),
        (
            0,
            b"/work/tree ['value.txt']\n[('/work/tree', [], ['value.txt'])]\n".to_vec(),
            Vec::new()
        )
    );
}

#[test]
fn vfs_operations_raise_the_python_os_exception_family() {
    let mut environment = Environment::new();
    environment.set_var("PWD", "/work");
    environment.vfs.put_dir("/work/folder", 0o755).unwrap();
    environment
        .vfs
        .put_file("/work/file", b"value".to_vec(), 0o644)
        .unwrap();
    let source = r#"import os
for operation in [
    lambda: open("missing").read(),
    lambda: open("folder").read(),
    lambda: os.chdir("file"),
]:
    try:
        operation()
    except FileNotFoundError:
        print("missing")
    except IsADirectoryError:
        print("directory")
    except NotADirectoryError:
        print("not-directory")
try:
    open("missing").read()
except OSError:
    print("os-base")
"#;
    assert_eq!(
        run_in(&mut environment, source),
        (
            0,
            b"missing\ndirectory\nnot-directory\nos-base\n".to_vec(),
            Vec::new()
        )
    );
}

#[test]
fn sys_exit_returns_the_requested_process_status() {
    assert_eq!(run("import sys\nsys.exit(4)"), (4, Vec::new(), Vec::new()));
    assert_eq!(run("import sys\nsys.exit()"), (0, Vec::new(), Vec::new()));
}

#[test]
fn sys_path_is_a_mutable_vfs_import_search_list() {
    let mut environment = Environment::new();
    environment.vfs.mkdir_all("/", "/opt/modules").unwrap();
    environment
        .vfs
        .put_file("/opt/modules/value.py", b"answer = 42\n".to_vec(), 0o644)
        .unwrap();
    let source = r#"import sys
sys.path.insert(0, "/opt/modules")
import value
print(sys.path[0], value.answer)
"#;
    assert_eq!(
        run_in(&mut environment, source),
        (0, b"/opt/modules 42\n".to_vec(), Vec::new())
    );
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
fn expanded_math_and_complex_functions_cover_agent_numeric_checks() {
    let source = r#"import cmath
import math
print(math.radians(180), math.degrees(math.pi))
print(math.pow(2, 3), math.log10(100), math.hypot(3, 4))
print(round(math.atan(1), 6), round(math.atan2(1, 1), 6), round(math.asin(1), 6))
value = complex(1, 2)
print(value, value.real, value.imag, value.conjugate(), abs(complex(3, 4)))
print(cmath.sqrt(-1), cmath.exp(0), tuple(round(x, 6) for x in cmath.polar(complex(3, 4))))
print("radians" in dir(math), "sqrt" in dir(cmath))"#;
    assert_eq!(
        run(source),
        (
            0,
            b"3.141592653589793 180.0\n8.0 2.0 5.0\n0.785398 0.785398 1.570796\n(1+2j) 1.0 2.0 (1-2j) 5.0\n1j (1+0j) (5.0, 0.927295)\nTrue True\n".to_vec(),
            Vec::new()
        )
    );
}

#[test]
fn finite_itertools_bisect_and_integer_math_cover_common_recipes() {
    let source = r#"import bisect
import itertools
import math

values = [1, 3, 3, 7]
print(bisect.bisect_left(values, 3), bisect.bisect(values, 3))
bisect.insort(values, 4)
bisect.insort_left(values, 3)
print(values)
print(list(itertools.chain([1, 2], [3])))
print(list(itertools.product("ab", repeat=2)))
print(list(itertools.permutations([1, 2, 3], 2)))
print(list(itertools.combinations([1, 2, 3], 2)))
print(math.floor(-1.2), math.trunc(-1.8), math.fabs(-2))
print(math.isfinite(1.0), math.isfinite(math.inf))
print(math.gcd(18, 24), math.lcm(6, 8), math.factorial(6))
"#;
    assert_eq!(
        run(source),
        (
            0,
            b"1 3\n[1, 3, 3, 3, 4, 7]\n[1, 2, 3]\n[('a', 'a'), ('a', 'b'), ('b', 'a'), ('b', 'b')]\n[(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]\n[(1, 2), (1, 3), (2, 3)]\n-2 -1 2.0\nTrue False\n6 24 720\n".to_vec(),
            Vec::new()
        )
    );
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
plus_ten = functools.partial(lambda a, b: a + b, 10)
print(plus_ten(5), plus_ten.func(2, 3), plus_ten.args, plus_ten.keywords)
scaled = functools.partial(lambda value, scale=1: value * scale, scale=3)
print(scaled(4), scaled(4, scale=5), scaled.keywords)
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
            b"[3, 4, 5]\n1 2\n6\n15 5 (10,) {}\n12 20 {'scale': 3}\ntyping.List[int]\n6\n".to_vec(),
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
pattern = re.compile(r"([a-z])(\d)")
matched = pattern.search("a1 b2")
print(matched.group(1), pattern.findall("a1 b2"), pattern.sub("#", "a1 b2"))
named = re.search(r"(?P<word>[a-z]+)-(?P<number>\d+)", "abc-42")
print(named.group("word"), named.group("number"), named.groups())
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
            b"A12 A 12 0 3\nA\n['12', '34']\nA# b#\na\\+b\\?\\ c\nb\na [('a', '1'), ('b', '2')] # #\nabc 42 ('abc', '42')\n".to_vec(),
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
parser.add_argument("--mode", choices=["fast", "safe"], default="safe")
args = parser.parse_args(["--count", "7", "--verbose", "--mode", "fast"])
print(parser.prog, args.count, args.verbose, args.mode)
"#;
    let simulated = run(source);
    assert_eq!(simulated, (0, b"demo 7 True fast\n".to_vec(), Vec::new()));
    let reference = Command::new("python3.14").arg("-c").arg(source).output();
    if let Ok(reference) = reference {
        assert_eq!(simulated.0, reference.status.code().unwrap_or(1));
        assert_eq!(simulated.1, reference.stdout);
        assert_eq!(simulated.2, reference.stderr);
    }

    let invalid = run(
        "import argparse\np = argparse.ArgumentParser()\np.add_argument('--mode', choices=['a'])\np.parse_args(['--mode', 'b'])",
    );
    assert_ne!(invalid.0, 0);
    assert!(invalid
        .2
        .windows(14)
        .any(|window| window == b"invalid choice"));
}

#[test]
fn argparse_help_known_args_and_subcommands_cover_common_cli_shapes() {
    let subcommands = run(r#"import argparse
parser = argparse.ArgumentParser(prog="tool", description="example tool")
commands = parser.add_subparsers(dest="command", required=True, help="available commands")
run_parser = commands.add_parser("run", help="run the job")
run_parser.add_argument("path", help="input path")
run_parser.add_argument("--count", type=int, default=2, help="repeat count")
run_parser.add_argument("--enabled", action="store_false")
args = parser.parse_args(["run", "input.txt", "--count", "4", "--enabled"])
print(args.command, args.path, args.count, args.enabled)
"#);
    assert_eq!(
        subcommands,
        (0, b"run input.txt 4 False\n".to_vec(), Vec::new())
    );

    let known = run(r#"import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--seed", type=int, default=1)
args, rest = parser.parse_known_args(["--seed", "7", "--foreign", "value"])
print(args.seed, rest)
"#);
    assert_eq!(
        known,
        (0, b"7 ['--foreign', 'value']\n".to_vec(), Vec::new())
    );

    let help = run(r#"import argparse
parser = argparse.ArgumentParser(prog="tool", description="example tool")
parser.add_argument("--count", type=int, help="repeat count")
parser.parse_args(["--help"])
"#);
    assert_eq!(help.0, 0);
    let help = String::from_utf8(help.1).unwrap();
    assert!(help.contains("usage: tool [-h] [--count COUNT]"));
    assert!(help.contains("example tool"));
    assert!(help.contains("-h, --help"));
    assert!(help.contains("--count COUNT"));
    assert!(help.contains("repeat count"));
}

#[test]
fn importlib_util_executes_vfs_source_in_an_isolated_module() {
    let mut environment = Environment::new();
    environment.vfs.mkdir_all("/", "/work/pkg").unwrap();
    environment
        .vfs
        .put_file(
            "/work/pkg/helper.py",
            b"def double(value):\n    return value * 2\n".to_vec(),
            0o644,
        )
        .unwrap();
    environment
        .vfs
        .put_file(
            "/work/pkg/plugin.py",
            b"from helper import double\nvalue = double(21)\n".to_vec(),
            0o644,
        )
        .unwrap();
    let source = r#"import importlib.util
spec = importlib.util.spec_from_file_location("plugin_name", "/work/pkg/plugin.py")
module = importlib.util.module_from_spec(spec)
print(importlib.__name__, importlib.util.__name__)
print(spec.name, spec.origin, spec.loader is not None)
spec.loader.exec_module(module)
print(module.__name__, module.__file__, module.__package__, module.__spec__ is spec, module.value)
"#;
    assert_eq!(
        run_in(&mut environment, source),
        (
            0,
            b"importlib importlib.util\nplugin_name /work/pkg/plugin.py True\nplugin_name /work/pkg/plugin.py  True 42\n".to_vec(),
            Vec::new()
        )
    );
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
