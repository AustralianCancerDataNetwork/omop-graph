"""Run the grounded benchmark with the cancer NSW case set by default."""

from __future__ import annotations

from pathlib import Path

from benchmark_base import run_grounded_benchmark_cli


def main() -> None:
    run_grounded_benchmark_cli(
        description="Grounding benchmark for the cancer NSW case set.",
        default_cases=Path(__file__).with_name("cancer_nsw_cases.json"),
        default_out=Path("/home/vscode/benchmark_cancer_nsw.json"),
    )


if __name__ == "__main__":
    main()
