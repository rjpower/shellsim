# Type dispatch probe

Run the ignored `source_suites::type_dispatch_benchmark` test in release mode to measure a fixed workload for each lookup path. It reports host elapsed time and modeled CPU units. The test uses separate fresh environments for its cases and checks the result of each program.

The following single-run measurements were taken on 2026-09-28 from `4066eda` and this branch, each in an isolated release build directory. The benchmark function was copied into the historical checkout for the baseline run. Host times are rounded to milliseconds and are useful only as rough comparisons on this machine; modeled CPU is deterministic for each revision.

| Workload | `4066eda` host | Branch host | `4066eda` CPU | Branch CPU |
| --- | ---: | ---: | ---: | ---: |
| Exact sequence length, indexing, iteration | 42 ms | 41 ms | 760,616 | 760,616 |
| List subclass length and indexing | unsupported | 23 ms | unsupported | 280,606 |
| Bound method lookup and call | 37 ms | 41 ms | 220,609 | 260,609 |
| Deep MRO hit | 14 ms | 18 ms | 280,733 | 320,739 |
| Dynamic attribute miss | 28 ms | 26 ms | 300,593 | 360,593 |
| Metaclass data descriptor | unsupported | 33 ms | unsupported | 240,708 |
| Integer in-place arithmetic | 6 ms | 4 ms | 140,443 | 140,443 |
| NumPy array in-place arithmetic | 64 ms | 49 ms | 310,003 | 308,492 |

The two unsupported baseline programs are new compatibility paths. Bound methods, deep MRO hits, and dynamic misses use more modeled CPU after lookup was unified; the host-time differences are too small for a single run to establish a performance regression. The exact-sequence and arithmetic cases retain their modeled CPU costs.
