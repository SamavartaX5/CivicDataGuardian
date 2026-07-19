"""Command-line entry point for CivicData Guardian monitoring."""

from __future__ import annotations

import sys

from src.pipeline import print_final_summary, run_pipeline


def main() -> int:
    """Run monitoring and return a clear non-zero exit code on failure."""
    try:
        print_final_summary(run_pipeline())
    except Exception as error:
        print(f"Monitoring pipeline failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
