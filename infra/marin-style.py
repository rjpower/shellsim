#!/usr/bin/env -S uv run --script --frozen
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "marin-style @ git+https://github.com/marin-community/marin-style@4b2fc2b0ec0bb9ad1304ef76936eb68e97f893cb",
# ]
# ///
"""Run pinned marin-style checks with shellsim's review catalog.

Marin-style's public configuration selects whole review lanes but cannot replace individual
rules. This exact-revision adapter keeps its file selection, Ruff runner, parallel review lanes,
read-only Claude restrictions, composer, and logs while loading repository-owned rules that are
language-neutral and specific to shellsim's simulation boundary.
"""

from pathlib import Path

from marin_style import lint_review, precommit

CATALOG_ROOT = Path(__file__).with_name("lint")
LANE_NAMES = ("design", "robustness", "cruft", "prose", "meta")


def local_catalog_text(name: str) -> str:
    """Load one checked-in catalog file by the name requested by marin-style."""

    return (CATALOG_ROOT / name).read_text(encoding="utf-8")


def configure_review() -> None:
    """Replace marin-style's fixed catalog with shellsim's curated lanes.

    The upstream revision is immutable because this adapter relies on two module-level extension
    points that are not yet public API. Updating the pin requires running the repository lint gate
    and a live ``infra/pre-commit.py --review`` pass to confirm the local catalog remains active.
    """

    if not hasattr(lint_review, "_catalog_text") or not hasattr(lint_review, "LINT_LANES"):
        raise RuntimeError("the pinned marin-style revision no longer exposes the expected catalog hooks")
    missing_catalogs = [name for name in ("shared", *LANE_NAMES) if not (CATALOG_ROOT / f"{name}.md").is_file()]
    if missing_catalogs:
        raise RuntimeError(f"missing shellsim lint catalogs: {', '.join(missing_catalogs)}")

    lint_review.LINT_LANES = tuple(
        lint_review.LintLane(name=name, include_complexity_leads=False) for name in LANE_NAMES[:-1]
    ) + (
        lint_review.LintLane(
            name="meta",
            include_complexity_leads=False,
            instructions=lint_review.META_LANE_INSTRUCTIONS,
            min_diff_lines=lint_review.META_LANE_MIN_DIFF_LINES,
        ),
    )
    lint_review._catalog_text = local_catalog_text


def main() -> None:
    """Configure the local review catalog and invoke marin-style's supported CLI."""

    configure_review()
    precommit.main()


if __name__ == "__main__":
    main()
