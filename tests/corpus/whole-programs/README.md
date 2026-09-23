# Whole-program compatibility corpus

This corpus checks whether supported features compose in maintained programs. It contains:

- twelve unchanged Modernish capability probes, sourced through a small status driver;
- zlib's `configure` help path and a bounded static-configuration attempt with modeled tool stubs;
- explicit skipped targets for full Modernish initialization and a ShellSpec smoke run.

The manifest pins repository revisions and hashes every executed fixture. The adjacent license
files come from those revisions. The zlib files are the vendored zlib 1.2.8 copy in the pinned Oils
tree; its `README` contains the upstream license notice.

Skipped cases describe broad composition frontiers. They stay visible in reports but do not copy a
large source tree into every test run. The stubbed zlib configuration is an executed frontier: it
must remain bounded and classified until shellsim has a coherent compiler and archive-tool model.
