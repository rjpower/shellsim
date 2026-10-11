# Math verification

This pure wheel and its parser dependencies support fixed-expression smoke
checks in the CPython WASI guest. The declared probe explicitly disables the
library's alarm deadlines for those small trusted expressions.

This does not qualify main-thread task grading. That workflow depends on a
working `signal.alarm` deadline, which needs separate guest capability and
interruption checks. Do not use the smoke probe as evidence that arbitrary
parsing or comparison has a five-second deadline.
