"""Run the nested-split DE-MMoE sample-size sensitivity experiment.

The total FE development budget is increased from 25 to 400 in steps of 25.
Every budget preserves the 64%/16%/20% train/validation/calibration allocation,
uses nested role-specific prefixes within each seed, and is evaluated on the
same independent 80-sample test set.  Five mean-first MMoE members are trained
in isolated processes so that their random-number streams remain independent.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import (
    DEVELOPMENT_COUNT,
    INPUT_COUNT,
    NEURAL_TRAINING,
    OUTPUT_COUNT,
    SEEDS,
)
from .data_protocol import ArrayStandardizer, DatasetBundle, load_dataset, make_split
from .metrics import group_regression_metrics, per_output_regression_metrics
from .models_torch import (
    DeterministicMMoE,
    TorchTrainingConfig,
    fit_with_early_stopping,
    parameter_count,
    predict_deterministic_ensemble,
)


BASE_DIR = Path(__file__).resolve().parent
FORMAL_RESULTS = BASE_DIR / "results"
FORMAL_RUNTIME = BASE_DIR / "runtime_rerun_mmoe_de_formal_5workers"
DEFAULT_OUTPUT = BASE_DIR / "sample_size_sensitivity"
BUDGETS = tuple(range(25, DEVELOPMENT_COUNT + 1, 25))
MEMBER_COUNT = 5


@dataclass(frozen=True)
class BudgetIndices:
    seed: int
    sample_size: int
    train: np.ndarray
    validation: np.ndarray
    calibration: np.ndarray

    def validate(self) -> None:
        expected = budget_counts(self.sample_size)
        actual = (len(self.train), len(self.validation), len(self.calibration))
        if actual != expected:
            raise ValueError(f"Expected role counts {expected}, got {actual}.")
        combined = np.concatenate((self.train, self.validation, self.calibration))
        if len(np.unique(combined)) != self.sample_size:
            raise ValueError("Budget split contains overlapping development rows.")
        if np.any(combined < 0) or np.any(combined >= DEVELOPMENT_COUNT):
            raise ValueError("Budget split contains an out-of-range development index.")


@dataclass(frozen=True)
class PreparedBudget:
    indices: BudgetIndices
    x_scaler: ArrayStandardizer
    y_scaler: ArrayStandardizer
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def budget_counts(sample_size: int) -> tuple[int, int, int]:
    if sample_size not in BUDGETS:
        raise ValueError(f"Unsupported sample size: {sample_size}.")
    # All approved budgets are multiples of 25, so these counts are exact.
    return (
        16 * sample_size // 25,
        4 * sample_size // 25,
        5 * sample_size // 25,
    )


def make_budget_indices(seed: int, sample_size: int) -> BudgetIndices:
    train_count, validation_count, calibration_count = budget_counts(sample_size)
    permutation = np.random.default_rng(seed).permutation(DEVELOPMENT_COUNT)
    train_pool = permutation[:256]
    validation_pool = permutation[256:320]
    calibration_pool = permutation[320:]
    indices = BudgetIndices(
        seed=int(seed),
        sample_size=int(sample_size),
        train=np.sort(train_pool[:train_count]),
        validation=np.sort(validation_pool[:validation_count]),
        calibration=np.sort(calibration_pool[:calibration_count]),
    )
    indices.validate()
    return indices


def prepare_budget(dataset: DatasetBundle, indices: BudgetIndices) -> PreparedBudget:
    indices.validate()
    x_scaler = ArrayStandardizer.fit(dataset.development_x[indices.train])
    y_scaler = ArrayStandardizer.fit(dataset.development_y[indices.train])
    return PreparedBudget(
        indices=indices,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        x_train=x_scaler.transform(dataset.development_x[indices.train]),
        y_train=y_scaler.transform(dataset.development_y[indices.train]),
        x_validation=x_scaler.transform(dataset.development_x[indices.validation]),
        y_validation=y_scaler.transform(dataset.development_y[indices.validation]),
        x_test=x_scaler.transform(dataset.test_x),
        y_test=y_scaler.transform(dataset.test_y),
    )


def _group_balanced_task_weights() -> np.ndarray:
    weights = np.full(OUTPUT_COUNT, 1.0 / (3.0 * (OUTPUT_COUNT - 2)))
    weights[0] = 1.0 / 3.0
    weights[1] = 1.0 / 3.0
    return weights


def _training_config(smoke: bool) -> TorchTrainingConfig:
    values = dict(NEURAL_TRAINING)
    if smoke:
        values.update(max_epochs=3, patience=2)
    return TorchTrainingConfig(**values)


def _selected_architecture(seed: int) -> dict[str, int]:
    path = FORMAL_RESULTS / f"seed_{seed}" / "deterministic_mmoe" / "run_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen architecture metadata: {path}")
    with path.open("r", encoding="utf-8") as file:
        selected = json.load(file)["selected_config"]
    return {
        "num_experts": int(selected["num_experts"]),
        "expert_hidden": int(selected["expert_hidden"]),
        "tower_hidden": int(selected["tower_hidden"]),
    }


def _fit_member_worker(arguments: tuple[Any, ...]):
    (
        architecture,
        x_train,
        y_train,
        x_validation,
        y_validation,
        member_seed,
        smoke,
        threads,
    ) = arguments
    import torch

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        thread_context = nullcontext()
    else:
        thread_context = threadpool_limits(limits=threads)

    factory = lambda: DeterministicMMoE(
        d_in=INPUT_COUNT,
        num_tasks=OUTPUT_COUNT,
        activation_name="swish",
        **architecture,
    )
    validation_weights = np.full(OUTPUT_COUNT, 1.0 / OUTPUT_COUNT)
    with thread_context:
        return fit_with_early_stopping(
            factory,
            x_train,
            y_train,
            x_validation,
            y_validation,
            probabilistic=False,
            config=_training_config(smoke),
            seed=int(member_seed),
            device="cpu",
            selection_metric="rmse",
            task_weights=_group_balanced_task_weights(),
            validation_task_weights=validation_weights,
        )


def _fit_ensemble(
    prepared: PreparedBudget,
    architecture: dict[str, int],
    seed: int,
    *,
    smoke: bool,
    workers: int,
    threads: int,
):
    member_seeds = [seed if index == 0 else seed + 1000 + index for index in range(5)]
    arguments = [
        (
            architecture,
            prepared.x_train,
            prepared.y_train,
            prepared.x_validation,
            prepared.y_validation,
            member_seed,
            smoke,
            threads,
        )
        for member_seed in member_seeds
    ]
    start = time.perf_counter()
    if workers == 1:
        members = [_fit_member_worker(argument) for argument in arguments]
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, MEMBER_COUNT),
            mp_context=mp.get_context("spawn"),
        ) as executor:
            members = list(executor.map(_fit_member_worker, arguments))
    return members, time.perf_counter() - start, member_seeds


def _upsert_csv(
    path: Path,
    new_rows: pd.DataFrame,
    keys: list[str],
    sort_columns: list[str],
) -> None:
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat((existing, new_rows), ignore_index=True)
    else:
        combined = new_rows.copy()
    combined = (
        combined.drop_duplicates(subset=keys, keep="last")
        .sort_values(sort_columns)
        .reset_index(drop=True)
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_manifest(output_dir: Path, seeds: Iterable[int], budgets: Iterable[int]) -> None:
    rows: list[dict[str, int | str]] = []
    for seed in seeds:
        for sample_size in budgets:
            indices = make_budget_indices(seed, sample_size)
            for role, selected in (
                ("train", indices.train),
                ("validation", indices.validation),
                ("calibration", indices.calibration),
            ):
                rows.extend(
                    {
                        "seed": seed,
                        "sample_size": sample_size,
                        "role": role,
                        "development_index": int(index),
                    }
                    for index in selected
                )
    pd.DataFrame(rows).sort_values(
        ["seed", "sample_size", "role", "development_index"]
    ).to_csv(output_dir / "split_manifest.csv", index=False)


def _validate_nested_protocol(seeds: Iterable[int], budgets: Iterable[int]) -> None:
    ordered_budgets = sorted(budgets)
    for seed in seeds:
        previous: BudgetIndices | None = None
        for sample_size in ordered_budgets:
            current = make_budget_indices(seed, sample_size)
            if previous is not None:
                for role in ("train", "validation", "calibration"):
                    if not set(getattr(previous, role)).issubset(set(getattr(current, role))):
                        raise ValueError(
                            f"Non-nested {role} subset for seed {seed}, budget {sample_size}."
                        )
            previous = current
        if DEVELOPMENT_COUNT in ordered_budgets:
            formal = make_split(seed)
            full = make_budget_indices(seed, DEVELOPMENT_COUNT)
            for role in ("train", "validation", "calibration"):
                if not np.array_equal(getattr(formal, role), getattr(full, role)):
                    raise ValueError(f"N=400 {role} split mismatch for seed {seed}.")


def _formal_n400_rows(seed: int, dataset: DatasetBundle) -> tuple[pd.DataFrame, ...]:
    result_dir = FORMAL_RESULTS / f"seed_{seed}" / "ensemble_mmoe"
    per_output = pd.read_csv(result_dir / "per_output_metrics.csv")
    group = pd.read_csv(result_dir / "group_metrics.csv")
    all_41 = group.loc[group["group"] == "all_41"].iloc[0]
    if len(per_output) != OUTPUT_COUNT:
        raise ValueError(f"Expected 41 formal per-output rows for seed {seed}.")
    if per_output["output_name"].tolist() != list(dataset.output_names):
        raise ValueError(f"Formal output order mismatch for seed {seed}.")
    train_count, validation_count, calibration_count = budget_counts(DEVELOPMENT_COUNT)
    metrics = pd.DataFrame(
        [
            {
                "seed": seed,
                "sample_size": DEVELOPMENT_COUNT,
                "train_count": train_count,
                "validation_count": validation_count,
                "calibration_count": calibration_count,
                "r2_macro": float(all_41["r2_macro"]),
                "rmse_macro": float(all_41["rmse_macro_physical"]),
                "nrmse_macro": float(all_41["nrmse_macro"]),
                "case_wall_seconds": np.nan,
                "cumulative_member_seconds": np.nan,
                "source": "verified_formal_N400_reuse",
            }
        ]
    )
    per_output = per_output.assign(
        seed=seed,
        sample_size=DEVELOPMENT_COUNT,
        train_count=train_count,
        validation_count=validation_count,
        calibration_count=calibration_count,
        source="verified_formal_N400_reuse",
    )

    runtime_by_seed = pd.read_csv(FORMAL_RUNTIME / "runtime_by_seed.csv")
    runtime = runtime_by_seed.loc[
        (runtime_by_seed["seed"] == seed)
        & (runtime_by_seed["method"] == "DE-MMoE")
    ].iloc[0]
    metrics.loc[0, "case_wall_seconds"] = float(runtime["method_wall_seconds"])
    metrics.loc[0, "cumulative_member_seconds"] = float(
        runtime["cumulative_member_seconds"]
    )

    timing = pd.read_csv(FORMAL_RUNTIME / "member_timings.csv")
    timing = timing.loc[
        (timing["seed"] == seed) & (timing["method"] == "DE-MMoE")
    ].copy()
    diagnostics = pd.DataFrame(
        {
            "seed": seed,
            "sample_size": DEVELOPMENT_COUNT,
            "member_index": timing["member_index"].astype(int),
            "member_seed": timing["member_seed"].astype(int),
            "best_epoch": timing["best_epoch"].astype(int),
            "epochs_completed": timing["epochs_completed"].astype(int),
            "best_validation_rmse": timing["best_validation_rmse"].astype(float),
            "fit_seconds": timing["fit_seconds_internal"].astype(float),
            "parameter_count": timing["parameter_count"].astype(int),
            "source": "verified_formal_N400_reuse",
        }
    )
    return metrics, per_output, diagnostics


def _fresh_case_rows(
    dataset: DatasetBundle,
    seed: int,
    sample_size: int,
    *,
    smoke: bool,
    workers: int,
    threads: int,
) -> tuple[pd.DataFrame, ...]:
    indices = make_budget_indices(seed, sample_size)
    prepared = prepare_budget(dataset, indices)
    architecture = _selected_architecture(seed)
    members, wall_seconds, member_seeds = _fit_ensemble(
        prepared,
        architecture,
        seed,
        smoke=smoke,
        workers=workers,
        threads=threads,
    )
    prediction = predict_deterministic_ensemble(members, prepared.x_test, device="cpu")
    mean_physical = prepared.y_scaler.inverse_transform(prediction.mean)
    per_output = per_output_regression_metrics(
        dataset.test_y,
        mean_physical,
        dataset.output_names,
    )
    grouped = group_regression_metrics(per_output)
    all_41 = grouped.loc[grouped["group"] == "all_41"].iloc[0]
    train_count, validation_count, calibration_count = budget_counts(sample_size)
    source = "fresh_sensitivity_fit"
    metrics = pd.DataFrame(
        [
            {
                "seed": seed,
                "sample_size": sample_size,
                "train_count": train_count,
                "validation_count": validation_count,
                "calibration_count": calibration_count,
                "r2_macro": float(all_41["r2_macro"]),
                "rmse_macro": float(all_41["rmse_macro_physical"]),
                "nrmse_macro": float(all_41["nrmse_macro"]),
                "case_wall_seconds": float(wall_seconds),
                "cumulative_member_seconds": float(
                    sum(member.fit_seconds for member in members)
                ),
                "source": source,
            }
        ]
    )
    per_output = per_output.assign(
        seed=seed,
        sample_size=sample_size,
        train_count=train_count,
        validation_count=validation_count,
        calibration_count=calibration_count,
        source=source,
    )
    diagnostics = pd.DataFrame(
        [
            {
                "seed": seed,
                "sample_size": sample_size,
                "member_index": index + 1,
                "member_seed": member_seeds[index],
                "best_epoch": member.best_epoch,
                "epochs_completed": member.epochs_completed,
                "best_validation_rmse": member.best_validation_rmse,
                "fit_seconds": member.fit_seconds,
                "parameter_count": parameter_count(member.model),
                "source": source,
            }
            for index, member in enumerate(members)
        ]
    )
    return metrics, per_output, diagnostics


def _write_summary(output_dir: Path, expected_seed_count: int) -> pd.DataFrame:
    by_case = pd.read_csv(output_dir / "metrics_by_seed_budget.csv")
    rows: list[dict[str, float | int]] = []
    for sample_size, frame in by_case.groupby("sample_size", sort=True):
        row: dict[str, float | int] = {
            "sample_size": int(sample_size),
            "seed_count": int(len(frame)),
            "train_count": int(frame["train_count"].iloc[0]),
            "validation_count": int(frame["validation_count"].iloc[0]),
            "calibration_count": int(frame["calibration_count"].iloc[0]),
        }
        for metric in ("r2_macro", "rmse_macro", "nrmse_macro"):
            values = frame[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_variance"] = (
                float(np.var(values, ddof=1)) if len(values) > 1 else np.nan
            )
            row[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            )
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("sample_size")
    summary["complete"] = summary["seed_count"] == expected_seed_count
    summary.to_csv(output_dir / "summary_by_sample_size.csv", index=False)
    return summary


def _validate_outputs(
    output_dir: Path,
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
) -> None:
    expected_cases = len(seeds) * len(budgets)
    metrics = pd.read_csv(output_dir / "metrics_by_seed_budget.csv")
    per_output = pd.read_csv(output_dir / "per_output_metrics.csv")
    diagnostics = pd.read_csv(output_dir / "training_diagnostics.csv")
    summary = pd.read_csv(output_dir / "summary_by_sample_size.csv")
    expected_pairs = {(seed, budget) for seed in seeds for budget in budgets}
    actual_pairs = set(
        zip(metrics["seed"].astype(int), metrics["sample_size"].astype(int))
    )
    if len(metrics) != expected_cases or actual_pairs != expected_pairs:
        raise RuntimeError("Seed-by-budget metric coverage is incomplete or duplicated.")
    if len(per_output) != expected_cases * OUTPUT_COUNT:
        raise RuntimeError("Per-output result count is inconsistent with 41 outputs.")
    if not (per_output.groupby(["seed", "sample_size"]).size() == OUTPUT_COUNT).all():
        raise RuntimeError("At least one case lacks a complete 41-output metric set.")
    if len(diagnostics) != expected_cases * MEMBER_COUNT:
        raise RuntimeError("Training diagnostics do not contain five members per case.")
    if not (
        diagnostics.groupby(["seed", "sample_size"]).size() == MEMBER_COUNT
    ).all():
        raise RuntimeError("At least one case lacks five member diagnostics.")
    numeric_columns = [
        "r2_macro",
        "rmse_macro",
        "nrmse_macro",
        "case_wall_seconds",
        "cumulative_member_seconds",
    ]
    if not np.isfinite(metrics[numeric_columns].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Non-finite values occur in the case-level metrics.")
    if len(summary) != len(budgets) or not bool(summary["complete"].all()):
        raise RuntimeError("Sample-size summary is incomplete.")
    for sample_size, frame in metrics.groupby("sample_size"):
        summary_row = summary.loc[summary["sample_size"] == sample_size].iloc[0]
        for metric in ("r2_macro", "rmse_macro", "nrmse_macro"):
            values = frame[metric].to_numpy(dtype=np.float64)
            expected_mean = float(np.mean(values))
            expected_variance = float(np.var(values, ddof=1))
            if not np.isclose(summary_row[f"{metric}_mean"], expected_mean, atol=1e-12):
                raise RuntimeError(f"Mean mismatch for N={sample_size}, {metric}.")
            if not np.isclose(
                summary_row[f"{metric}_variance"], expected_variance, atol=1e-12
            ):
                raise RuntimeError(f"Variance mismatch for N={sample_size}, {metric}.")


def _parse_integer_list(text: str, allowed: Iterable[int], label: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError(f"At least one {label} is required.")
    invalid = sorted(set(values) - set(allowed))
    if invalid:
        raise ValueError(f"Unsupported {label}: {invalid}")
    return tuple(dict.fromkeys(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--budgets", default=",".join(map(str, BUDGETS)))
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--refit-400", action="store_true")
    args = parser.parse_args()

    seeds = _parse_integer_list(args.seeds, SEEDS, "seed")
    budgets = _parse_integer_list(args.budgets, BUDGETS, "budget")
    if args.workers < 1 or args.workers > MEMBER_COUNT:
        raise ValueError("--workers must be between 1 and 5.")
    if args.threads < 1:
        raise ValueError("--threads must be positive.")
    output_dir = args.output
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT.with_name(
            DEFAULT_OUTPUT.name + ("_smoke" if args.smoke else "")
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    _validate_nested_protocol(seeds, budgets)
    _write_manifest(output_dir, seeds, budgets)
    dataset = load_dataset()

    metadata = {
        "experiment": "DE-MMoE sample-size sensitivity",
        "seeds": list(seeds),
        "sample_sizes": list(budgets),
        "development_allocation": "64% train, 16% validation, 20% calibration",
        "independent_test_count": int(len(dataset.test_x)),
        "member_count": MEMBER_COUNT,
        "member_training": "group-balanced standardized MSE",
        "checkpoint": "equal-output validation standardized RMSE",
        "architecture": "frozen per seed from formal deterministic MMoE benchmark",
        "nested_subsets": True,
        "primary_metrics": "macro R2 and macro physical-scale RMSE across 41 outputs",
        "negative_r2_clipped": False,
        "reuse_formal_n400": not args.refit_400,
        "workers": args.workers,
        "threads_per_worker": args.threads,
        "smoke": args.smoke,
        "environment_thread_limits": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    with (output_dir / "experiment_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    metrics_path = output_dir / "metrics_by_seed_budget.csv"
    completed: set[tuple[int, int]] = set()
    if metrics_path.exists():
        prior = pd.read_csv(metrics_path)
        completed = set(
            zip(prior["seed"].astype(int), prior["sample_size"].astype(int))
        )

    total = len(seeds) * len(budgets)
    counter = 0
    for sample_size in sorted(budgets):
        for seed in seeds:
            counter += 1
            if (seed, sample_size) in completed:
                print(
                    f"SKIP {counter}/{total}: seed={seed}, N={sample_size} already complete.",
                    flush=True,
                )
                continue
            print(f"START {counter}/{total}: seed={seed}, N={sample_size}", flush=True)
            case_start = time.perf_counter()
            if sample_size == DEVELOPMENT_COUNT and not args.refit_400 and not args.smoke:
                metric_rows, output_rows, diagnostic_rows = _formal_n400_rows(seed, dataset)
            else:
                metric_rows, output_rows, diagnostic_rows = _fresh_case_rows(
                    dataset,
                    seed,
                    sample_size,
                    smoke=args.smoke,
                    workers=args.workers,
                    threads=args.threads,
                )
            _upsert_csv(
                metrics_path,
                metric_rows,
                ["seed", "sample_size"],
                ["sample_size", "seed"],
            )
            _upsert_csv(
                output_dir / "per_output_metrics.csv",
                output_rows,
                ["seed", "sample_size", "output_index"],
                ["sample_size", "seed", "output_index"],
            )
            _upsert_csv(
                output_dir / "training_diagnostics.csv",
                diagnostic_rows,
                ["seed", "sample_size", "member_index"],
                ["sample_size", "seed", "member_index"],
            )
            _write_summary(output_dir, expected_seed_count=len(seeds))
            elapsed = time.perf_counter() - case_start
            row = metric_rows.iloc[0]
            print(
                f"DONE  {counter}/{total}: seed={seed}, N={sample_size}, "
                f"R2={row['r2_macro']:.6f}, RMSE={row['rmse_macro']:.6f}, "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    summary = _write_summary(output_dir, expected_seed_count=len(seeds))
    if len(summary) != len(budgets) or not bool(summary["complete"].all()):
        raise RuntimeError("Experiment finished with incomplete seed coverage.")
    _validate_outputs(output_dir, seeds, budgets)
    print(f"COMPLETE: {output_dir}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
