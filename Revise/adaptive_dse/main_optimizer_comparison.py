"""PyCharm entry point for the NSGA-II/MOEA-D/MOPSO comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __name__ == "__main__" and not __package__:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    __package__ = "Revise.adaptive_dse"

from .optimizer_comparison import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate the model, data, algorithms and settings without optimizing.",
    )
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Rebuild metrics and figures from completed checkpoint fronts.",
    )
    arguments = parser.parse_args()
    run_benchmark(
        preflight=arguments.preflight,
        postprocess_only=arguments.postprocess_only,
    )


if __name__ == "__main__":
    main()
