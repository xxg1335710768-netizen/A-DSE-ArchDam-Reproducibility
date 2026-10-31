"""Fixed-configuration final-fitting runtime benchmark for baseline surrogates.

The benchmark reuses only the hyperparameters selected previously for each
frozen development split.  Hyperparameter search, data preparation, test-set
prediction, uncertainty sampling, metric calculation and file serialization
are outside the timed region.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
import warnings
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning

from .config import INPUT_COUNT, NEURAL_TRAINING, OUTPUT_COUNT, SEEDS
from .data_protocol import load_dataset, make_split, prepare_split
from .models_sklearn import IndependentGPR, IndependentXGBoost, PCALatentGPR
from .models_torch import (
    IndependentTaskMLP,
    MultiTaskMLP,
    TorchTrainingConfig,
    fit_with_early_stopping,
)


BASE_DIR = Path(__file__).resolve().parent
BASE_RESULTS = BASE_DIR / "results"
MC_RESULTS = BASE_RESULTS
MMOE_RUNTIME_DIR = BASE_DIR / "fixed_runtime_mmoe_de"
DEFAULT_OUTPUT = BASE_DIR / "fixed_runtime_all_models"

# Labels match the manuscript table.  MLP and MT-MLP use the positive-dropout
# configurations from the formal MC-Dropout experiment, but stochastic
# inference is deliberately excluded from final model fitting time.
METHODS: dict[str, tuple[Path, str, str]] = {
    "GPR": (BASE_RESULTS, "gpr", "gpr"),
    "XGB": (BASE_RESULTS, "xgboost", "xgboost"),
    "MLP": (MC_RESULTS, "single_mlp_mc_dropout", "single_mlp_mc_dropout"),
    "PCA-GPR": (BASE_RESULTS, "pca_gpr", "pca_gpr"),
    "MT-MLP": (
        MC_RESULTS,
        "multitask_mlp_mc_dropout",
        "multitask_mlp_mc_dropout",
    ),
}


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


def _metadata_path(method: str, seed: int) -> Path:
    root, stored_name, _ = METHODS[method]
    return root / f"seed_{seed}" / stored_name / "run_metadata.json"


def _selected_config(method: str, seed: int) -> dict[str, Any]:
    path = _metadata_path(method, seed)
    if not path.is_file():
        raise FileNotFoundError(f"Missing selected configuration: {path}")
    with path.open(encoding="utf-8") as file:
        return dict(json.load(file)["selected_config"])


def _training_config() -> TorchTrainingConfig:
    return TorchTrainingConfig(**dict(NEURAL_TRAINING))


def _fit_fixed(method: str, config: dict[str, Any], prepared, seed: int):
    """Fit one frozen model and return the model plus timing diagnostics."""
    _, _, implementation = METHODS[method]
    start = time.perf_counter()
    if implementation == "gpr":
        model = IndependentGPR(random_state=seed, **config).fit(
            prepared.x_train, prepared.y_train
        )
        diagnostics = {"fitted_outputs": len(model.models)}
    elif implementation == "pca_gpr":
        model = PCALatentGPR(random_state=seed, **config).fit(
            prepared.x_train, prepared.y_train
        )
        diagnostics = {"retained_pca_components": model.retained_components}
    elif implementation == "xgboost":
        model = IndependentXGBoost(params=config, random_state=seed).fit(
            prepared.x_train,
            prepared.y_train,
            prepared.x_validation,
            prepared.y_validation,
        )
        diagnostics = {
            "best_iteration_per_output": [
                int(item.best_iteration) if item.best_iteration is not None else None
                for item in model.models
            ]
        }
    elif implementation == "single_mlp_mc_dropout":
        factory = lambda: IndependentTaskMLP(
            INPUT_COUNT,
            OUTPUT_COUNT,
            tuple(config["hidden_dims"]),
            float(config["dropout"]),
        )
        model = fit_with_early_stopping(
            factory,
            prepared.x_train,
            prepared.y_train,
            prepared.x_validation,
            prepared.y_validation,
            probabilistic=False,
            config=_training_config(),
            seed=seed,
            device="cpu",
            selection_metric="loss",
        )
        diagnostics = {
            "best_epoch": model.best_epoch,
            "epochs_completed": model.epochs_completed,
            "internal_fit_seconds": model.fit_seconds,
        }
    elif implementation == "multitask_mlp_mc_dropout":
        factory = lambda: MultiTaskMLP(
            INPUT_COUNT,
            OUTPUT_COUNT,
            tuple(config["hidden_dims"]),
            float(config["dropout"]),
        )
        model = fit_with_early_stopping(
            factory,
            prepared.x_train,
            prepared.y_train,
            prepared.x_validation,
            prepared.y_validation,
            probabilistic=False,
            config=_training_config(),
            seed=seed,
            device="cpu",
            selection_metric="loss",
        )
        diagnostics = {
            "best_epoch": model.best_epoch,
            "epochs_completed": model.epochs_completed,
            "internal_fit_seconds": model.fit_seconds,
        }
    else:
        raise ValueError(f"Unsupported implementation: {implementation}")
    wall_seconds = time.perf_counter() - start
    return model, wall_seconds, diagnostics


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in frame.groupby("method", sort=False):
        values = group["wall_minutes"]
        rows.append(
            {
                "method": method,
                "seed_count": len(group),
                "wall_minutes_mean": float(values.mean()),
                "wall_minutes_std": (
                    float(values.std(ddof=1)) if len(values) > 1 else np.nan
                ),
                "wall_minutes_variance": (
                    float(values.var(ddof=1)) if len(values) > 1 else np.nan
                ),
                "wall_minutes_median": float(values.median()),
                "wall_minutes_total_five_seeds": float(values.sum()),
            }
        )
    return pd.DataFrame(rows)


def _combine_with_mmoe(baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = MMOE_RUNTIME_DIR / "runtime_by_seed.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing fresh MMoE timing result: {path}")
    source = pd.read_csv(path)
    mmoe = source[["seed", "method", "method_wall_seconds", "method_wall_minutes"]].rename(
        columns={
            "method_wall_seconds": "wall_seconds",
            "method_wall_minutes": "wall_minutes",
        }
    )
    mmoe["selected_config"] = "See fresh MMoE runtime metadata"
    mmoe["diagnostics"] = ""
    combined = pd.concat(
        [
            baseline[["seed", "method", "wall_seconds", "wall_minutes", "selected_config", "diagnostics"]],
            mmoe,
        ],
        ignore_index=True,
    )
    order = {name: index for index, name in enumerate([*METHODS, "MMoE", "DE-MMoE"])}
    combined["method_order"] = combined["method"].map(order)
    combined.sort_values(["method_order", "seed"], inplace=True)
    combined.drop(columns="method_order", inplace=True)
    return combined, _summary(combined)


def _write_report(output: Path, summary: pd.DataFrame) -> None:
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            f"| {row['method']} | {row['wall_minutes_mean']:.3f} ± "
            f"{row['wall_minutes_std']:.3f} | {row['wall_minutes_total_five_seeds']:.3f} |"
        )
    report = "\n".join(
        [
            "# Unified fixed-configuration runtime benchmark",
            "",
            "All entries are newly measured final-fitting wall-clock times after hyperparameters were frozen.",
            "The timed region excludes hyperparameter search, data preparation, test inference, MC-Dropout sampling, metric calculation and result serialization.",
            "Every individual model uses one numerical CPU thread. The five DE-MMoE members use five isolated one-thread CPU processes in parallel; process startup is included in its wall time.",
            "Values are mean ± sample standard deviation over the five frozen development partitions.",
            "",
            "| Model | Final fitting time (min) | Five-seed total (min) |",
            "| --- | ---: | ---: |",
            *rows,
        ]
    )
    (output / "RUNTIME_REPORT.md").write_text(report + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = sorted(set(args.seeds) - set(SEEDS))
    if unknown:
        raise ValueError(f"Seeds must use the frozen protocol {SEEDS}; got {unknown}")
    args.output.mkdir(parents=True, exist_ok=True)
    runtime_path = args.output / "baseline_runtime_by_seed.csv"
    if args.overwrite and runtime_path.exists():
        runtime_path.unlink()
    rows = (
        pd.read_csv(runtime_path).to_dict("records") if runtime_path.exists() else []
    )
    completed = {(int(row["seed"]), str(row["method"])) for row in rows}

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    # Resolve lazy imports before the first timed model.
    from xgboost import XGBRegressor  # noqa: F401

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    dataset = load_dataset()
    started = datetime.now().astimezone()
    method_names = list(METHODS)

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        thread_context = nullcontext()
    else:
        thread_context = threadpool_limits(limits=1)

    with thread_context:
        for split_index, seed in enumerate(args.seeds):
            prepared = prepare_split(dataset, make_split(seed))
            # Rotate order across splits to reduce systematic warm-up or thermal bias.
            order = method_names[split_index:] + method_names[:split_index]
            for method in order:
                if (seed, method) in completed:
                    print(f"SKIP seed={seed} method={method}", flush=True)
                    continue
                config = _selected_config(method, seed)
                gc.collect()
                print(f"START seed={seed} method={method}", flush=True)
                model, wall_seconds, diagnostics = _fit_fixed(
                    method, config, prepared, seed
                )
                row = {
                    "seed": seed,
                    "method": method,
                    "wall_seconds": wall_seconds,
                    "wall_minutes": wall_seconds / 60.0,
                    "selected_config": json.dumps(_jsonable(config), ensure_ascii=False),
                    "diagnostics": json.dumps(_jsonable(diagnostics), ensure_ascii=False),
                }
                rows.append(row)
                completed.add((seed, method))
                _atomic_csv(pd.DataFrame(rows), runtime_path)
                print(
                    f"END seed={seed} method={method} wall_min={wall_seconds / 60.0:.3f}",
                    flush=True,
                )
                del model
                gc.collect()

    baseline = pd.DataFrame(rows)
    baseline.sort_values(["method", "seed"], inplace=True)
    _atomic_csv(baseline, runtime_path)
    baseline_summary = _summary(baseline)
    baseline_summary.to_csv(args.output / "baseline_runtime_summary.csv", index=False)
    combined, combined_summary = _combine_with_mmoe(baseline)
    combined.to_csv(args.output / "all_models_runtime_by_seed.csv", index=False)
    combined_summary.to_csv(args.output / "all_models_runtime_summary.csv", index=False)
    _write_report(args.output, combined_summary)

    completed_at = datetime.now().astimezone()
    metadata = {
        "purpose": "unified fixed-configuration final-fitting wall-clock benchmark",
        "start_time": started.isoformat(),
        "end_time": completed_at.isoformat(),
        "elapsed_seconds_this_process": (completed_at - started).total_seconds(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "device": "cpu",
        "numerical_threads_per_individual_model": 1,
        "execution": "baseline methods sequential; DE-MMoE five-member process parallelism retained from the fresh benchmark",
        "timed_region": "model construction, final fitting, early stopping where applicable, and best-checkpoint restoration",
        "excluded": [
            "hyperparameter search",
            "data loading and split preparation",
            "test inference",
            "MC-Dropout uncertainty sampling",
            "metric calculation",
            "model and result serialization",
        ],
        "seeds": list(args.seeds),
        "selected_configuration_sources": {
            method: {str(seed): str(_metadata_path(method, seed)) for seed in args.seeds}
            for method in METHODS
        },
        "mmoe_runtime_source": str(MMOE_RUNTIME_DIR),
    }
    with (args.output / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(_jsonable(metadata), file, ensure_ascii=False, indent=2)
    print("RESULTS", args.output, flush=True)
    print(combined_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
