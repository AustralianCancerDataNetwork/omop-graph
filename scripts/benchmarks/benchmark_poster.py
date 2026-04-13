"""Grounding-focused benchmark tailored for poster/showcase outputs."""

from __future__ import annotations

from pathlib import Path

from benchmark_base import run_grounded_benchmark_cli


def main() -> None:
    run_grounded_benchmark_cli(
        description="Grounding benchmark for publication/poster showcases.",
        default_cases=Path(__file__).with_name("resolver_cases.json"),
        default_out=Path("/home/vscode/benchmark_poster.json"),
    )


if __name__ == "__main__":
    main()
