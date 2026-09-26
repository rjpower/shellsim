//! Differential checks for the native `complex` type.
//!
//! Expected outputs were recorded by running the same source under CPython 3.14; the harness
//! never invokes a host interpreter. Cases cover arithmetic, parsing, `repr`, errors, and the
//! explicitly unsupported frontier.

use super::support::run_python_text as run;

#[test]
fn complex_semantics_match_cpython() {
    let source = r#"def show(label, thunk):
    try:
        print(label, repr(thunk()))
    except TypeError as error:
        print(label, "TypeError", error)
    except ValueError as error:
        print(label, "ValueError", error)
    except ZeroDivisionError as error:
        print(label, "ZeroDivisionError", error)
    except OverflowError as error:
        print(label, "OverflowError", error)
show("literals", lambda: (1j, 2j, 1 + 1j, 1j * 1j, -0j, complex(-0.0, -1)))
show("repr", lambda: [repr(complex(float("nan"), float("inf"))), repr(complex(1e16, 1e-7)), str(1 + 2j)])
show("arithmetic", lambda: [(1 + 2j) * (3 - 4j), (1 + 2j) / (3 - 4j), 1 / (0.0 + 1j), -(1 + 2j), +(1 + 2j), abs(3 + 4j)])
show("powers", lambda: [(1 + 2j) ** 2, (1 + 2j) ** -2, (1 + 2j) ** 0.5, 2 ** 1j, (1 + 2j) ** (1 + 1j), 1j ** 1000])
show("mixed", lambda: [1 + 2j, 2j - 1, 3 - 2j, 2.5 * 2j, 2j / 2, 2 / 2j, (2**100) * 1j, True + 1j])
show("signed zeros", lambda: [complex(1, -0.0) + 1.0, (1 + 2j) * float("inf")])
show("attributes", lambda: ((1 + 2j).real, (1 + 2j).imag, (1 + 2j).conjugate(), hasattr(1j, "imag")))
show("equality", lambda: (1 + 2j == 1 + 2j, 1 == 1 + 0j, 1.0 == complex(1), 1j != 1j, 2**53 + 1 == complex(2**53 + 1), 1j == "1j"))
show("containers", lambda: (1j in [0, 1j], {1j: "a"}[1j], len({1j, complex(0, 1)}), bool(0j), bool(1j)))
show("constructor", lambda: [complex(), complex(2), complex(1.5, 2), complex(real=1, imag=2), complex(imag=2), complex(True), complex(1e400)])
show("strings", lambda: [complex("1+2j"), complex(" ( 1.5-2e3J ) "), complex("-j"), complex("1_0+2_0j"), complex("inf-nanj")])
show("identity", lambda: (lambda value: complex(value) is value)(1 + 2j))
show("types", lambda: (isinstance(1j, complex), type(1j) is complex, type(1j)))
show("less", lambda: 1j < 2j)
show("sorted", lambda: sorted([1j, 2j]))
show("zero power", lambda: 0j ** -1)
show("complex zero power", lambda: 0j ** 1j)
show("power overflow", lambda: (10 + 0j) ** 400)
show("divide by zero", lambda: (1 + 1j) / 0)
show("complex divide by zero", lambda: (1 + 1j) / 0j)
show("floor divide", lambda: (1 + 1j) // 1)
show("modulo", lambda: 1 % (1 + 1j))
show("int", lambda: int(1j))
show("float", lambda: float(1j))
show("huge int", lambda: 2**2000 + 1j)
show("malformed", lambda: complex("1+2"))
show("string and imag", lambda: complex("1+2j", 1))
show("string imag", lambda: complex(1, "a"))
show("none", lambda: complex(None))
show("formatting", lambda: f"{1j}" + "%s" % (1 + 1j) + str(2j) + "{}".format(3j))
"#;
    let expected = "literals (1j, 2j, (1+1j), (-1+0j), (-0-0j), (-0-1j))\nrepr ['(nan+infj)', '(1e+16+1e-07j)', '(1+2j)']\narithmetic [(11+2j), (-0.2+0.4j), -1j, (-1-2j), (1+2j), 5.0]\npowers [(-3+4j), (-0.12-0.16j), (1.272019649514069+0.7861513777574233j), (0.7692389013639721+0.6389612763136348j), (-0.24720004426291722+0.6964504870825432j), (1-1.6070832296378168e-13j)]\nmixed [(1+2j), (-1+2j), (3-2j), 5j, 1j, -1j, 1.2676506002282294e+30j, (1+1j)]\nsigned zeros [(2-0j), (inf+infj)]\nattributes (1.0, 2.0, (1-2j), True)\nequality (True, True, True, False, False, False)\ncontainers (True, 'a', 1, False, True)\nconstructor [0j, (2+0j), (1.5+2j), (1+2j), 2j, (1+0j), (inf+0j)]\nstrings [(1+2j), (1.5-2000j), -1j, (10+20j), (inf+nanj)]\nidentity True\ntypes (True, True, <class 'complex'>)\nless TypeError '<' not supported between instances of 'complex' and 'complex'\nsorted TypeError '<' not supported between instances of 'complex' and 'complex'\nzero power ZeroDivisionError zero to a negative or complex power\ncomplex zero power ZeroDivisionError zero to a negative or complex power\npower overflow OverflowError complex exponentiation\ndivide by zero ZeroDivisionError division by zero\ncomplex divide by zero ZeroDivisionError division by zero\nfloor divide TypeError unsupported operand type(s) for //: 'complex' and 'int'\nmodulo TypeError unsupported operand type(s) for %: 'int' and 'complex'\nint TypeError int() argument must be a string, a bytes-like object or a real number, not 'complex'\nfloat TypeError float() argument must be a string or a real number, not 'complex'\nhuge int OverflowError int too large to convert to float\nmalformed ValueError complex() arg is a malformed string\nstring and imag TypeError complex() argument 'real' must be a real number, not str\nstring imag TypeError complex() argument 'imag' must be a real number, not str\nnone TypeError complex() argument must be a string or a number, not NoneType\nformatting '1j(1+1j)2j3j'\n";
    assert_eq!(run(source), (0, expected.into(), String::new()));
}

#[test]
fn numeric_tower_attributes_and_readonly_components() {
    let source = r#"x = 1 + 2j
try:
    x.real = 3
except AttributeError as error:
    print(error)
print(complex.real, getattr(x, "imag"), x.conjugate().conjugate() == x)
import cmath
print(cmath.sqrt(-1), cmath.exp(0), cmath.polar(1j), cmath.rect(1, 0))
"#;
    assert_eq!(
        run(source),
        (
            0,
            "attribute 'real' of 'complex' objects is not writable\n<attribute 'real' of 'complex' objects> 2.0 True\n1j (1+0j) (1.0, 1.5707963267948966) (1+0j)\n".into(),
            String::new()
        )
    );
}

#[test]
fn real_only_operations_reject_complex_explicitly() {
    for (source, expected) in [
        (
            "import math\nmath.sqrt(1j)",
            "expected a real number, got complex",
        ),
        ("round(1j)", "expected a real number, got complex"),
        (
            "max(1j, 2j)",
            "not supported between instances of 'complex' and 'complex'",
        ),
        (
            "1 < 1j",
            "not supported between instances of 'complex' and 'int'",
        ),
    ] {
        let (status, _, stderr) = run(source);
        assert_ne!(status, 0, "{source} unexpectedly succeeded");
        assert!(stderr.contains(expected), "{source}: {stderr}");
    }
}

#[test]
fn complex_values_support_string_format_specifications() {
    let source = r#"z = 3.14159 + 2.71828j
print(f"{z:.2f}")
print("{:.1f}".format(1 + 2j))
"#;
    assert_eq!(
        run(source),
        (0, "3.14+2.72j\n1.0+2.0j\n".into(), String::new())
    );
}

#[test]
fn complex_formats_preserve_default_repr_and_format_components() {
    // Checked against CPython: default presentation keeps parentheses and suppresses +0 real.
    let source = r#"z = 1234.5-5678.5j
print(f'{z:.2}', f'{z:.2g}', f'{z:+.2f}', f'{z:,.1f}')
print(f'{1+2j:>14.1f}', f'{2j:.1f}', f'{complex(-0.0, -0.0):.1f}')
print(f'{1+2j:10}', f'{2j:+}', f'{1+2j:.1e}')
for spec in ['010.2f', '.2%', 'd', '#.2f']:
    try:
        ('{:' + spec + '}').format(1j)
    except ValueError:
        print('ValueError')
"#;
    assert_eq!(run(source), (0, "(1.2e+03-5.7e+03j) 1.2e+03-5.7e+03j +1234.50-5678.50j 1,234.5-5,678.5j\n      1.0+2.0j 0.0+2.0j -0.0-0.0j\n    (1+2j) +2j 1.0e+00+2.0e+00j\nValueError\nValueError\nValueError\nValueError\n".into(), String::new()));
}

#[test]
fn conj_alias_matches_conjugate_for_complex_and_numpy_scalars() {
    // Builtin complex.conj is a shellsim extension; NumPy scalars use the same alias.
    let source = r#"import numpy as np
for value in [1+2j, 2j, complex(1, -0.0), np.complex128(3+4j), np.float64(2), np.int32(3)]:
    assert repr(value.conj()) == repr(value.conjugate())
    assert type(value.conj()) is type(value.conjugate())
assert (1+2j).conj().conj() == 1+2j
for call in [lambda: (1j).conj(1), lambda: (1j).conj(x=1), lambda: np.float64(2).conj(1)]:
    try:
        call()
    except TypeError:
        pass
    else:
        assert False
print('ok')
"#;
    assert_eq!(run(source), (0, "ok\n".into(), String::new()));
}
