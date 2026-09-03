# Reduced from the public Apache-2.0 TaskTrove/TBLite build-system-task-ordering task.
# This is a language/runtime milestone, not a task-specific implementation.

def normalized_targets(lines):
    targets = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "TARGET" and len(parts) == 2:
            targets.append(parts[1])
        elif parts[0] == "OVERRIDE" and len(parts) == 3:
            targets.append(parts[1])
        else:
            return "PARSE_ERROR"
    return sorted(targets)


print(normalized_targets([
    "# comment",
    "TARGET compile",
    "OVERRIDE test 20",
    "TARGET all",
]))
