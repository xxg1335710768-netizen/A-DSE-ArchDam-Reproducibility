"""Fresh fixed-architecture runtime benchmark for MMoE and DE-MMoE.

This benchmark deliberately does not reuse historical timing metadata.  For
each frozen data split it trains a new single-member MMoE and a new five-member
DE-MMoE on CPU. Each member uses one numerical thread, and the five ensemble
members may run in five isolated processes. The architecture is
the one selected previously on the development data; therefore the reported
quantity is final model-fitting time, not hyperparameter-search time.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import INPUT_COUNT, NEURAL_TRAINING, OUTPUT_COUNT, SEEDS
from .data_protocol import load_dataset, make_split, prepare_split
from .models_torch import (
    TorchTrainingConfig,
    parameter_count,
)
from .run_benchmark import _fit_ensemble_member_worker


BASE_DIR = Path(__file__).resolve().parent
ARCHITECTURE_ROOT = BASE_DIR / "results"
DEFAULT_OUTPUT = BASE_DIR / "fixed_runtime_mmoe_de"
METHOD_MEMBERS = {"MMoE": 1, "DE-MMoE": 5}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _training_config(smoke: bool) -> TorchTrainingConfig:
    values = dict(NEURAL_TRAINING)
    if smoke:
        values.update(max_epochs=3, patience=2)
    return TorchTrainingConfig(**values)


def _task_weights() -> tuple[np.ndarray, np.ndarray]:
    training = np.full(OUTPUT_COUNT, 1.0 / (3.0 * (OUTPUT_COUNT - 2)))
    training[0] = 1.0 / 3.0
    training[1] = 1.0 / 3.0
    validation = np.full(OUTPUT_COUNT, 1.0 / OUTPUT_COUNT)
    return training, validation


def _architecture_path(seed: int) -> Path:
    return (
        ARCHITECTURE_ROOT
        / f"seed_{seed}"
        / "deterministic_mmoe"
        / "run_metadata.json"
    )


def _selected_architecture(seed: int) -> dict[str, int]:
    path = _architecture_path(seed)
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen architecture metadata: {path}")
    with path.open("r", encoding="utf-8") as file:
        selected = json.load(file)["selected_config"]
    return {
        "num_experts": int(selected["num_experts"]),
        "expert_hidden": int(selected["expert_hidden"]),
        "tower_hidden": int(selected["tower_hidden"]),
    }


def _member_seeds(seed: int, count: int) -> list[int]:
    return [
        seed if member_index == 0 else seed + 1000 + member_index
        for member_index in range(count)
    ]


def _run_method(
    *,
    seed: int,
    method: str,
    prepared,
    architecture: dict[str, int],
    device: str,
    ensemble_workers: int,
    smoke: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    member_count = METHOD_MEMBERS[method]
    member_seeds = _member_seeds(seed, member_count)
    rows: list[dict[str, Any]] = []
    method_start = time.perf_counter()
    workers = min(ensemble_workers, member_count) if method == "DE-MMoE" else 1
    print(
        f"START seed={seed} method={method} "
        f"members={member_count} workers={workers}",
        flush=True,
    )
    arguments = [
        (
            architecture,
            prepared.x_train,
            prepared.y_train,
            prepared.x_validation,
            prepared.y_validation,
            member_seed,
            smoke,
            1,
            device,
        )
        for member_seed in member_seeds
    ]
    if workers == 1:
        fitted_members = [
            _fit_ensemble_member_worker(*argument) for argument in arguments
        ]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
        ) as executor:
            fitted_members = list(
                executor.map(_fit_ensemble_member_worker, *zip(*arguments))
            )

    for member_index, (member_seed, fitted) in enumerate(
        zip(member_seeds, fitted_members), start=1
    ):
        rows.append(
            {
                "seed": seed,
                "method": method,
                "member_index": member_index,
                "member_seed": member_seed,
                **architecture,
                "parameter_count": parameter_count(fitted.model),
                "best_epoch": fitted.best_epoch,
                "epochs_completed": fitted.epochs_completed,
                "best_validation_rmse": fitted.best_validation_rmse,
                "fit_seconds_internal": fitted.fit_seconds,
                "wall_seconds": fitted.fit_seconds,
                "process_cpu_seconds": fitted.fit_seconds,
                "seconds_per_epoch_internal": (
                    fitted.fit_seconds / fitted.epochs_completed
                ),
                "cpu_seconds_per_epoch": (
                    fitted.fit_seconds / fitted.epochs_completed
                ),
            }
        )
        print(
            f"DONE seed={seed} method={method} "
            f"member={member_index}/{member_count} "
            f"epochs={fitted.epochs_completed} best={fitted.best_epoch} "
            f"member_fit_min={fitted.fit_seconds / 60.0:.3f}",
            flush=True,
        )
    method_wall_seconds = time.perf_counter() - method_start
    cumulative_member_seconds = float(
        sum(row["fit_seconds_internal"] for row in rows)
    )
    method_row = {
        "seed": seed,
        "method": method,
        "member_count": member_count,
        "method_wall_seconds": method_wall_seconds,
        "method_wall_minutes": method_wall_seconds / 60.0,
        "cumulative_member_seconds": cumulative_member_seconds,
        "cumulative_member_minutes": cumulative_member_seconds / 60.0,
        "workers": workers,
        "sum_member_wall_seconds": float(sum(row["wall_seconds"] for row in rows)),
        "sum_member_process_cpu_seconds": float(
            sum(row["process_cpu_seconds"] for row in rows)
        ),
        "sum_internal_fit_seconds": float(
            sum(row["fit_seconds_internal"] for row in rows)
        ),
    }
    print(
        f"END seed={seed} method={method} "
        f"wall_min={method_wall_seconds / 60.0:.3f} "
        f"cumulative_member_min={cumulative_member_seconds / 60.0:.3f}",
        flush=True,
    )
    del fitted_members
    gc.collect()
    return rows, method_row


def _write_outputs(
    output: Path,
    member_rows: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
) -> None:
    member_frame = pd.DataFrame(member_rows)
    method_frame = pd.DataFrame(method_rows)
    member_frame.to_csv(output / "member_timings.csv", index=False)
    method_frame.to_csv(output / "runtime_by_seed.csv", index=False)

    summary_rows = []
    for method, frame in method_frame.groupby("method", sort=False):
        values = frame["method_wall_minutes"]
        cumulative_values = frame["cumulative_member_minutes"]
        internal = frame["sum_internal_fit_seconds"] / 60.0
        summary_rows.append(
            {
                "method": method,
                "seed_count": len(frame),
                "member_count": int(frame["member_count"].iloc[0]),
                "wall_minutes_mean": float(values.mean()),
                "wall_minutes_std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "wall_minutes_variance": float(values.var(ddof=1)) if len(values) > 1 else np.nan,
                "wall_minutes_median": float(values.median()),
                "cumulative_member_minutes_mean": float(cumulative_values.mean()),
                "cumulative_member_minutes_std": (
                    float(cumulative_values.std(ddof=1))
                    if len(cumulative_values) > 1
                    else np.nan
                ),
                "cumulative_member_minutes_variance": (
                    float(cumulative_values.var(ddof=1))
                    if len(cumulative_values) > 1
                    else np.nan
                ),
                "cumulative_member_minutes_median": float(
                    cumulative_values.median()
                ),
                "internal_fit_minutes_mean": float(internal.mean()),
                "internal_fit_minutes_std": (
                    float(internal.std(ddof=1)) if len(internal) > 1 else np.nan
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "runtime_summary.csv", index=False)

    report_rows = []
    for _, row in summary.iterrows():
        report_rows.append(
            "| {method} | {members} | {wall_mean:.3f} ± {wall_std:.3f} | "
            "{cpu_mean:.3f} ± {cpu_std:.3f} | {median:.3f} |".format(
                method=row["method"],
                members=int(row["member_count"]),
                cpu_mean=row["cumulative_member_minutes_mean"],
                cpu_std=row["cumulative_member_minutes_std"],
                wall_mean=row["wall_minutes_mean"],
                wall_std=row["wall_minutes_std"],
                median=row["wall_minutes_median"],
            )
        )
    wall_ratio = np.nan
    cumulative_ratio = np.nan
    indexed = summary.set_index("method")
    if {"MMoE", "DE-MMoE"}.issubset(indexed.index):
        wall_ratio = float(
            indexed.loc["DE-MMoE", "wall_minutes_mean"]
            / indexed.loc["MMoE", "wall_minutes_mean"]
        )
        cumulative_ratio = float(
            indexed.loc["DE-MMoE", "cumulative_member_minutes_mean"]
            / indexed.loc["MMoE", "cumulative_member_minutes_mean"]
        )
    report = "\n".join(
        [
            "# Fresh MMoE and DE-MMoE runtime benchmark",
            "",
            "The models were newly trained for this benchmark; no historical timing value was reused.",
            "Architectures were frozen before timing. Each member used one numerical CPU thread; independent DE members used isolated processes.",
            "Wall-clock time covers process startup, model construction, data-loader/optimizer setup, all epochs, early stopping, and best-checkpoint restoration. Hyperparameter search is excluded.",
            "Wall-clock time is the manuscript-facing measure. Cumulative member time is retained as the total training-work audit value.",
            "",
            "| Model | Members | Wall time (min, mean ± SD) | Cumulative member time (min, mean ± SD) | Wall median |",
            "| --- | ---: | ---: | ---: | ---: |",
            *report_rows,
            "",
            f"Observed DE-MMoE/MMoE wall-time ratio: {wall_ratio:.3f}; cumulative-member-time ratio: {cumulative_ratio:.3f}.",
            "",
            "Sample standard deviations are calculated across the five frozen development splits.",
            "These values are fixed-architecture final fitting times and must not be mixed with hyperparameter-search time in one table column.",
        ]
    )
    (output / "RUNTIME_REPORT.md").write_text(report + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--ensemble-workers", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = sorted(set(args.seeds) - set(SEEDS))
    if unknown:
        raise ValueError(f"Seeds must use the frozen protocol {SEEDS}; got {unknown}")
    if args.threads != 1:
        raise ValueError("Formal timing must use exactly one numerical thread.")
    if args.device != "cpu":
        raise ValueError("Formal timing is fixed to CPU for comparability.")
    if args.ensemble_workers < 1 or args.ensemble_workers > 5:
        raise ValueError("--ensemble-workers must be between 1 and 5.")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output is not empty: {args.output}. Use --overwrite or another path."
        )
    args.output.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    dataset = load_dataset()
    training_config = _training_config(args.smoke)
    member_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    started = datetime.now().astimezone()

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        context = nullcontext()
    else:
        context = threadpool_limits(limits=1)

    with context:
        for split_index, seed in enumerate(args.seeds):
            prepared = prepare_split(dataset, make_split(seed))
            architecture = _selected_architecture(seed)
            # Alternating the method order reduces a systematic warm-cache or
            # long-run thermal bias while keeping every fit strictly serial.
            methods = (
                ("MMoE", "DE-MMoE")
                if split_index % 2 == 0
                else ("DE-MMoE", "MMoE")
            )
            for method in methods:
                new_members, new_method = _run_method(
                    seed=seed,
                    method=method,
                    prepared=prepared,
                    architecture=architecture,
                    device=args.device,
                    ensemble_workers=args.ensemble_workers,
                    smoke=args.smoke,
                )
                member_rows.extend(new_members)
                method_rows.append(new_method)
                _write_outputs(args.output, member_rows, method_rows)

    completed = datetime.now().astimezone()
    metadata = {
        "purpose": "fresh fixed-architecture training runtime benchmark",
        "historical_timing_reused": False,
        "start_time": started.isoformat(),
        "end_time": completed.isoformat(),
        "elapsed_seconds_entire_process": (completed - started).total_seconds(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "device": args.device,
        "numerical_threads": 1,
        "cpu_affinity": "operating-system default",
        "execution": "MMoE serial; DE-MMoE member-level process parallelism",
        "ensemble_workers": args.ensemble_workers,
        "method_order": "alternated by split",
        "seeds": args.seeds,
        "member_counts": METHOD_MEMBERS,
        "training_config": vars(training_config),
        "training_objective": "group_balanced_standardized_mse",
        "checkpoint_selection": "equal_output_validation_rmse",
        "architecture_sources": {
            str(seed): _architecture_path(seed) for seed in args.seeds
        },
        "architectures": {
            str(seed): _selected_architecture(seed) for seed in args.seeds
        },
        "smoke": args.smoke,
    }
    with (args.output / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(_jsonable(metadata), file, ensure_ascii=False, indent=2)
    _write_outputs(args.output, member_rows, method_rows)
    print("Results written to", args.output, flush=True)


if __name__ == "__main__":
    main()
