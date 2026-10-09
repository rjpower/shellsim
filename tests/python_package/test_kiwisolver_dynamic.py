"""Exercise the independently installed upstream Kiwi extension in guest CPython."""

import hashlib
import os

import pytest
import shellsim


def test_public_kiwisolver_install_and_constraint_exceptions():
    keys = ("SHELLSIM_KIWISOLVER_BUNDLE", "SHELLSIM_KIWISOLVER_UNIVERSE", "SHELLSIM_PATCHED_UV")
    if not all(os.environ.get(key) for key in keys):
        pytest.skip("set the fixed CPython bundle, Kiwi catalog and patched uv paths")
    runtime = shellsim.CPythonRuntime(os.environ[keys[0]], universe=os.environ[keys[1]], uv=os.environ[keys[2]])
    image = runtime.bundle / "rootfs/usr/bin/python3.wasm"
    before = hashlib.sha256(image.read_bytes()).hexdigest()
    env = shellsim.Environment(cpu=10_000_000_000, memory=1024**3, disk=128 * 1024**2)
    runtime.mount(env)
    guest_image_before = hashlib.sha256(env.read_file("/usr/bin/python3.wasm")).hexdigest()
    runtime.install_pypi(env, "kiwisolver==1.5.1")
    result = runtime.run(
        env,
        [
            "-c",
            """
import kiwisolver as k
assert k.__version__ == '1.5.1'
x, y = k.Variable('x'), k.Variable('y')
solver = k.Solver()
constraint = x + y == 100
solver.addConstraint(constraint)
solver.addConstraint(x == 40)
solver.updateVariables()
assert x.value() == 40 and y.value() == 60
assert x.name() == 'x' and x.context() is None
term = k.Term(y, 2)
assert term.variable() is y and term.coefficient() == 2 and term.value() == 120
expression = constraint.expression()
assert len(expression.terms()) == 2 and expression.constant() == -100 and expression.value() == 0
assert constraint.op() == '==' and constraint.strength() == k.strength.required and not constraint.violated()
assert 0 < k.strength.weak < k.strength.medium < k.strength.strong < k.strength.required
def raises(kind, action):
    try:
        action()
    except kind:
        return
    raise AssertionError(kind.__name__)
raises(k.UnsatisfiableConstraint, lambda: solver.addConstraint(x == 41))
raises(k.DuplicateConstraint, lambda: solver.addConstraint(constraint))
raises(k.UnknownConstraint, lambda: solver.removeConstraint(y == 999))
raises(TypeError, lambda: solver.addConstraint(None))
assert isinstance(solver.dumps(), str)
solver.dump()
solver.reset()
solver.addEditVariable(x, k.strength.strong)
solver.suggestValue(x, 17)
solver.updateVariables()
assert x.value() == 17
print('Kiwi constraints and C++ exception translation passed')
""",
        ],
    )
    assert result.returncode == 0, result.stderr
    assert b"Kiwi constraints and C++ exception translation passed" in result.stdout
    assert hashlib.sha256(image.read_bytes()).hexdigest() == before
    assert hashlib.sha256(env.read_file("/usr/bin/python3.wasm")).hexdigest() == guest_image_before
