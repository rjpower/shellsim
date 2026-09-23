# Derived Python integration corpus

This corpus contains small contracts written for shellsim's compatibility work. The cases are
inspired by recurring patterns in the MicroPython, Exercism, TaskTrove, and ordinary Python
programs used during compatibility research; they do not copy those programs or their tests.

Each case names the capabilities it exercises. Passing cases establish checked behavior, while
frontier cases require an exact failure class, exit status, and unsupported-feature diagnostic.
This keeps missing behavior visible without allowing a parser error, crash, hang, or different
unsupported boundary to count as the expected result.

Keep cases focused enough that a failure identifies one small cluster of related behavior. Use a
few multi-file and cross-subsystem cases to verify composition, but prefer direct contracts over
application-shaped fixtures.
