//! Unittest entrypoint, lifecycle, assertion, and unsupported-discovery behavior.

use shellsim::Environment;

fn run_unittest(source: &str, command: &str) -> (i32, Vec<u8>, Vec<u8>) {
    let mut environment = Environment::new();
    environment
        .vfs
        .put_file("/test_sample.py", source.as_bytes().to_vec(), 0o644)
        .expect("install test module");
    let (outcome, stdout, stderr) = environment.run_script_capture(command);
    (outcome.exit_status, stdout, stderr)
}

#[test]
fn runs_testcase_methods_in_definition_order_with_fixtures_and_assertions() {
    let source = "import unittest\nclass Sample(unittest.TestCase):\n    def setUp(self):\n        self.value = 4\n    def tearDown(self):\n        self.value = None\n    def test_second(self):\n        self.assertEqual(self.value, 4)\n        self.assertTrue(self.value)\n    def test_first(self):\n        self.assertFalse(False)\n        self.assertIsNone(None)\n        with self.assertRaises(ValueError):\n            raise ValueError('bad')\n";
    let (status, stdout, stderr) = run_unittest(source, "python3.14 -m unittest /test_sample.py");
    assert_eq!(status, 0, "{stderr:?}");
    assert_eq!(
        stdout,
        b"/test_sample.py::Sample.test_second ok\n/test_sample.py::Sample.test_first ok\nRan 2 tests\nOK\n"
    );
    assert!(stderr.is_empty());
}

#[test]
fn exposes_testcase_name_and_rejects_unsupported_discovery() {
    let (status, stdout, stderr) = run_unittest(
        "import unittest\nprint(unittest.TestCase.__name__)\n",
        "python3.14 /test_sample.py",
    );
    assert_eq!(status, 0, "{stderr:?}");
    assert_eq!(stdout, b"TestCase\n");
    assert!(stderr.is_empty());

    let (status, _stdout, stderr) = run_unittest(
        "import unittest\nclass Sample(unittest.TestCase):\n    @classmethod\n    def setUpClass(cls):\n        pass\n    def test_ok(self):\n        pass\n",
        "python3.14 -m unittest /test_sample.py --verbose",
    );
    assert_eq!(status, 2);
    assert!(String::from_utf8_lossy(&stderr).contains("unittest option"));

    let (status, _stdout, stderr) = run_unittest(
        "import unittest\nclass Sample(unittest.TestCase):\n    def setUpClass(cls):\n        pass\n    def test_ok(self):\n        pass\n",
        "python3.14 -m unittest /test_sample.py",
    );
    assert_eq!(status, 2);
    assert!(String::from_utf8_lossy(&stderr).contains("class hook"));
}

#[test]
fn runs_a_test_class_without_teardown() {
    let source = "import unittest\nclass Sample(unittest.TestCase):\n    def test_ok(self):\n        self.assertEqual(2 + 2, 4)\n";
    let (status, stdout, stderr) = run_unittest(source, "python3.14 -m unittest /test_sample.py");

    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(
        stdout,
        b"/test_sample.py::Sample.test_ok ok\nRan 1 tests\nOK\n"
    );
}

#[test]
fn caps_generated_unittest_wrapper_before_execution() {
    let mut environment = Environment::new();
    let mut source = String::from("import unittest\nclass Many(unittest.TestCase):\n");
    source.push_str(&"    def test_ok(self):\n        pass\n".repeat(7_000));
    environment
        .vfs
        .put_file("/many_test.py", source.into_bytes(), 0o644)
        .expect("install many-test module");
    let (outcome, stdout, stderr) =
        environment.run_script_capture("python3.14 -m unittest /many_test.py");
    assert_eq!(outcome.exit_status, 2);
    assert!(stdout.is_empty());
    assert!(String::from_utf8_lossy(&stderr).contains("generated wrapper"));
}
