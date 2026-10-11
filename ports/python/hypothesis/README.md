# Hypothesis

Use the upstream pure Hypothesis 6.151.9 wheel with SortedContainers. The newer
observed 6.168.5 release requires a Rust extension; this compatible version
keeps the first property-testing loop small. The guest probe uses bounded,
deterministic examples with deadlines and the persistent database disabled.
