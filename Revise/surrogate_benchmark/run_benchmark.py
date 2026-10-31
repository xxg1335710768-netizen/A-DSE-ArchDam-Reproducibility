"""Run the leakage-safe surrogate-model benchmark.

The script keeps model selection on the 64-row validation set, variance
calibration on the 80-row calibration set and final evaluation on the fixed
80-row independent test set.  It writes one compact result directory per
model and seed so interrupted experiments can be resumed without mixing runs.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import (
    GPR_GRID,
    INPUT_COUNT,
    MC_DROPOUT_GRID,
    MC_DROPOUT_SAMPLES,
    MLP_GRID,
    MMOE_GRID,
    NEURAL_TRAINING,
    OUTPUT_COUNT,
    PCA_GPR_GRID,
    RESULTS_DIR,
    SEEDS,
    XGBOOST_GRID,
)
from .data_protocol import load_dataset, make_split, prepare_split, split_manifest
from .metrics import (
    control_stress_metrics,
    coverage_table,
    fit_residual_variance_floor,
    fit_variance_scale,
    gaussian_nll,
    group_regression_metrics,
    per_output_regression_metrics,
)


MODEL_NAMES = (
    "gpr",
    "xgboost",
    "single_mlp",
    "single_mlp_mc_dropout",
    "pca_gpr",
    "multitask_mlp",
    "multitask_mlp_mc_dropout",
    "deterministic_mmoe",
    "ensemble_mmoe",
)


@dataclass
class CandidateResult:
    config_index: int
    config: dict[str, Any]
    validation_rmse_standardized_macro: float
    validation_nll_macro: float | None
    fit_seconds: float
    model: Any | None


@dataclass
class PredictionBundle:
    mean: np.ndarray
    variance: np.ndarray | None = None
    model_variance: np.ndarray | None = None
    residual_variance: np.ndarray | None = None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _validation_standardized_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmse = np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2, axis=0))
    return float(np.mean(rmse))


def _group_balanced_task_weights() -> np.ndarray:
    weights = np.full(OUTPUT_COUNT, 1.0 / (3.0 * (OUTPUT_COUNT - 2)))
    weights[0] = 1.0 / 3.0
    weights[1] = 1.0 / 3.0
    return weights


def _candidate_key(candidate: CandidateResult) -> tuple[float, float]:
    secondary = (
        float(candidate.validation_nll_macro)
        if candidate.validation_nll_macro is not None
        else float("inf")
    )
    return candidate.validation_rmse_standardized_macro, secondary


def _grid_for(model_name: str, smoke: bool) -> tuple[dict[str, Any], ...]:
    grids = {
        "gpr": GPR_GRID,
        "xgboost": XGBOOST_GRID,
        "single_mlp": MLP_GRID,
        "single_mlp_mc_dropout": MC_DROPOUT_GRID,
        "pca_gpr": PCA_GPR_GRID,
        "multitask_mlp": MLP_GRID,
        "multitask_mlp_mc_dropout": MC_DROPOUT_GRID,
        "deterministic_mmoe": MMOE_GRID,
        "ensemble_mmoe": MMOE_GRID,
    }
    grid = tuple(dict(item) for item in grids[model_name])
    if smoke:
        grid = grid[:1]
        if model_name in {"gpr", "pca_gpr"}:
            grid[0]["n_restarts_optimizer"] = 0
        if model_name == "xgboost":
            grid[0]["n_estimators"] = 20
    return grid


def _torch_training_config(smoke: bool):
    try:
        from .models_torch import TorchTrainingConfig
    except ImportError as error:
        raise RuntimeError("PyTorch is required for the selected neural model.") from error
    values = dict(NEURAL_TRAINING)
    if smoke:
        values.update(max_epochs=3, patience=2)
    return TorchTrainingConfig(**values)


def _fit_candidate(
    model_name: str,
    config: dict[str, Any],
    prepared,
    *,
    seed: int,
    device: str,
    smoke: bool,
) -> tuple[Any, PredictionBundle, float]:
    start = time.perf_counter()
    if model_name in {"gpr", "pca_gpr", "xgboost"}:
        try:
            from .models_sklearn import IndependentGPR, IndependentXGBoost, PCALatentGPR
        except ImportError as error:
            raise RuntimeError("scikit-learn is required for this baseline.") from error
        if model_name == "gpr":
            model = IndependentGPR(random_state=seed, **config).fit(
                prepared.x_train, prepared.y_train
            )
            prediction = model.predict_distribution(prepared.x_validation)
            bundle = PredictionBundle(prediction.mean, prediction.variance)
        elif model_name == "pca_gpr":
            model = PCALatentGPR(random_state=seed, **config).fit(
                prepared.x_train, prepared.y_train
            )
            prediction = model.predict_distribution(prepared.x_validation)
            bundle = PredictionBundle(prediction.mean, prediction.variance)
        else:
            model = IndependentXGBoost(params=config, random_state=seed).fit(
                prepared.x_train,
                prepared.y_train,
                prepared.x_validation,
                prepared.y_validation,
            )
            bundle = PredictionBundle(model.predict(prepared.x_validation))
        return model, bundle, time.perf_counter() - start

    try:
        from .models_torch import (
            DeterministicMMoE,
            IndependentTaskMLP,
            MultiTaskMLP,
            fit_with_early_stopping,
            predict_deterministic,
            predict_mc_dropout,
            predict_probabilistic,
        )
    except ImportError as error:
        raise RuntimeError("PyTorch is required for the selected neural model.") from error

    training = _torch_training_config(smoke)
    if model_name in {"single_mlp", "single_mlp_mc_dropout"}:
        factory = lambda: IndependentTaskMLP(
            INPUT_COUNT, OUTPUT_COUNT, tuple(config["hidden_dims"]), config["dropout"]
        )
        probabilistic = False
    elif model_name in {"multitask_mlp", "multitask_mlp_mc_dropout"}:
        factory = lambda: MultiTaskMLP(
            INPUT_COUNT, OUTPUT_COUNT, tuple(config["hidden_dims"]), config["dropout"]
        )
        probabilistic = False
    elif model_name == "deterministic_mmoe":
        factory = lambda: DeterministicMMoE(
            d_in=INPUT_COUNT,
            num_tasks=OUTPUT_COUNT,
            activation_name="swish",
            **config,
        )
        probabilistic = False
    elif model_name == "ensemble_mmoe":
        # Architecture selection deliberately matches the deterministic-MMoE
        # benchmark.  The selected architecture is then refitted as a fresh
        # five-member group-balanced ensemble in _complete_ensemble().
        factory = lambda: DeterministicMMoE(
            d_in=INPUT_COUNT,
            num_tasks=OUTPUT_COUNT,
            activation_name="swish",
            **config,
        )
        probabilistic = False
    else:
        raise ValueError(f"Unknown model: {model_name}")

    fitted = fit_with_early_stopping(
        factory,
        prepared.x_train,
        prepared.y_train,
        prepared.x_validation,
        prepared.y_validation,
        probabilistic=probabilistic,
        config=training,
        seed=seed,
        device=device,
        selection_metric="loss",
    )
    if model_name in {"single_mlp_mc_dropout", "multitask_mlp_mc_dropout"}:
        moments = predict_mc_dropout(
            fitted,
            prepared.x_validation,
            mc_samples=MC_DROPOUT_SAMPLES,
            device=device,
        )
        bundle = PredictionBundle(
            moments.mean,
            moments.total_var,
            moments.epistemic_var,
            moments.aleatoric_var,
        )
    elif probabilistic:
        moments = predict_probabilistic(fitted, prepared.x_validation, device=device)
        bundle = PredictionBundle(
            moments.mean,
            moments.total_var,
            moments.epistemic_var,
            moments.aleatoric_var,
        )
    else:
        bundle = PredictionBundle(
            predict_deterministic(fitted, prepared.x_validation, device=device)
        )
    return fitted, bundle, time.perf_counter() - start


def _predict(model_name: str, model: Any, x: np.ndarray, device: str) -> PredictionBundle:
    if model_name in {"gpr", "pca_gpr"}:
        prediction = model.predict_distribution(x)
        return PredictionBundle(prediction.mean, prediction.variance)
    if model_name == "xgboost":
        return PredictionBundle(model.predict(x))
    from .models_torch import (
        predict_deterministic,
        predict_deterministic_ensemble,
        predict_ensemble,
        predict_mc_dropout,
        predict_probabilistic,
    )

    if model_name in {"single_mlp_mc_dropout", "multitask_mlp_mc_dropout"}:
        moments = predict_mc_dropout(
            model,
            x,
            mc_samples=MC_DROPOUT_SAMPLES,
            device=device,
        )
        return PredictionBundle(
            moments.mean,
            moments.total_var,
            moments.epistemic_var,
            moments.aleatoric_var,
        )

    if model_name == "ensemble_mmoe":
        fitted_members = model if isinstance(model, list) else [model]
        moments = predict_deterministic_ensemble(
            fitted_members,
            x,
            device=device,
        )
        return PredictionBundle(
            moments.mean,
            moments.epistemic_var,
            moments.epistemic_var,
            moments.aleatoric_var,
        )
    return PredictionBundle(predict_deterministic(model, x, device=device))


def _fit_ensemble_member_worker(
    config: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    member_seed: int,
    smoke: bool,
    threads: int,
    device: str,
):
    """Fit one independent member in an isolated CPU process.

    Process isolation is important here: ``fit_with_early_stopping`` seeds the
    PyTorch global RNG, so fitting members concurrently in Python threads would
    make their random streams interfere.  A spawned process preserves the same
    member seed and training trajectory as the serial implementation.
    """
    import torch

    from .models_torch import DeterministicMMoE, fit_with_early_stopping

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting this only before inter-op work starts.
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
        **config,
    )
    weights = _group_balanced_task_weights()
    validation_weights = np.full(OUTPUT_COUNT, 1.0 / OUTPUT_COUNT)
    with thread_context:
        return fit_with_early_stopping(
            factory,
            x_train,
            y_train,
            x_validation,
            y_validation,
            probabilistic=False,
            config=_torch_training_config(smoke),
            seed=member_seed,
            device=device,
            selection_metric="rmse",
            task_weights=weights,
            validation_task_weights=validation_weights,
        )


def _complete_ensemble(
    selected: CandidateResult,
    prepared,
    *,
    seed: int,
    device: str,
    smoke: bool,
    workers: int,
    threads: int,
) -> tuple[list[Any], float, float]:

    if selected.model is None:
        raise RuntimeError("The selected ensemble member was released unexpectedly.")
    member_count = 2 if smoke else 5
    member_seeds = [
        seed if member_index == 0 else seed + 1000 + member_index
        for member_index in range(member_count)
    ]
    workers = min(workers, member_count)
    if workers > 1 and device != "cpu":
        raise ValueError(
            "Parallel ensemble workers currently require --device cpu. "
            "Use one worker for a single GPU, or add explicit multi-GPU mapping."
        )

    arguments = [
        (
            selected.config,
            prepared.x_train,
            prepared.y_train,
            prepared.x_validation,
            prepared.y_validation,
            member_seed,
            smoke,
            threads,
            device,
        )
        for member_seed in member_seeds
    ]
    wall_start = time.perf_counter()
    if workers == 1:
        members = [_fit_ensemble_member_worker(*argument) for argument in arguments]
    else:
        # ``map`` preserves input order, so the ensemble aggregation is exactly
        # member 0..M-1 as in the former serial loop.
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
        ) as executor:
            members = list(executor.map(_fit_ensemble_member_worker, *zip(*arguments)))
    wall_seconds = time.perf_counter() - wall_start
    cumulative_member_seconds = float(sum(member.fit_seconds for member in members))
    return members, wall_seconds, cumulative_member_seconds


def _group_probability_metrics(
    output_names: tuple[str, ...],
    raw_nll: np.ndarray,
    calibrated_nll: np.ndarray,
) -> pd.DataFrame:
    rows = []
    groups = {
        "Volu": np.array([0]),
        "St": np.array([1]),
        "stress_39": np.arange(2, len(output_names)),
        "all_41": np.arange(len(output_names)),
    }
    for name, indices in groups.items():
        rows.append(
            {
                "group": name,
                "output_count": int(len(indices)),
                "nll_raw_macro": float(np.mean(raw_nll[indices])),
                "nll_calibrated_macro": float(np.mean(calibrated_nll[indices])),
            }
        )
    return pd.DataFrame(rows)


def _median_inference_seconds(function: Callable[[], Any], repeats: int = 7) -> float:
    function()
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        durations.append(time.perf_counter() - start)
    return float(np.median(durations))


def _model_diagnostics(model_name: str, model: Any) -> dict[str, Any]:
    if model_name in {
        "single_mlp",
        "single_mlp_mc_dropout",
        "multitask_mlp",
        "multitask_mlp_mc_dropout",
        "deterministic_mmoe",
        "ensemble_mmoe",
    }:
        from .models_torch import parameter_count

        members = model if isinstance(model, list) else [model]
        counts = [parameter_count(member.model) for member in members]
        return {
            "member_count": len(members),
            "parameter_count_per_member": counts,
            "parameter_count_total": int(sum(counts)),
            "best_epoch_per_member": [member.best_epoch for member in members],
            "epochs_completed_per_member": [
                member.epochs_completed for member in members
            ],
            "checkpoint_selection_metric_per_member": [
                member.selection_metric for member in members
            ],
            "best_validation_rmse_per_member": [
                member.best_validation_rmse for member in members
            ],
            "validation_nll_at_selected_checkpoint_per_member": [
                member.best_validation_nll for member in members
            ],
        }
    if model_name == "pca_gpr":
        return {"retained_pca_components": model.retained_components}
    if model_name == "xgboost":
        return {
            "best_iteration_per_output": [
                int(item.best_iteration) if item.best_iteration is not None else None
                for item in model.models
            ]
        }
    return {}


def _save_fitted_model(
    model_name: str,
    model: Any,
    selected_config: dict[str, Any],
    prepared,
    variance_scale: np.ndarray | None,
    residual_floor: np.ndarray | None,
    result_dir: Path,
) -> None:
    scalers = {
        "x_mean": prepared.x_scaler.mean,
        "x_scale": prepared.x_scaler.scale,
        "y_mean": prepared.y_scaler.mean,
        "y_scale": prepared.y_scaler.scale,
    }
    if model_name in {"gpr", "pca_gpr", "xgboost"}:
        import joblib

        joblib.dump(
            {
                "model_name": model_name,
                "model": model,
                "selected_config": selected_config,
                "variance_scale": variance_scale,
                "residual_floor": residual_floor,
                **scalers,
            },
            result_dir / "fitted_model.joblib",
        )
        return

    import torch

    members = model if isinstance(model, list) else [model]
    torch.save(
        {
            "model_name": model_name,
            "state_dicts": [member.model.state_dict() for member in members],
            "selected_config": selected_config,
            "variance_scale": variance_scale,
            "residual_floor": residual_floor,
            **scalers,
        },
        result_dir / "fitted_model.pt",
    )


def _write_result(
    model_name: str,
    seed: int,
    selected: CandidateResult,
    candidates: list[CandidateResult],
    prepared,
    dataset,
    final_model: Any,
    final_fit_seconds: float,
    search_seconds: float,
    device: str,
    threads: int,
    ensemble_workers: int,
    ensemble_member_fit_seconds_sum: float | None,
    output_root: Path,
) -> None:
    result_dir = output_root / f"seed_{seed}" / model_name
    result_dir.mkdir(parents=True, exist_ok=True)
    test_prediction = _predict(model_name, final_model, prepared.x_test, device)
    test_mean_physical = prepared.y_scaler.inverse_transform(test_prediction.mean)
    per_output = per_output_regression_metrics(
        dataset.test_y, test_mean_physical, dataset.output_names
    )
    per_output.to_csv(result_dir / "per_output_metrics.csv", index=False)
    group_regression_metrics(per_output).to_csv(
        result_dir / "group_metrics.csv", index=False
    )
    control = control_stress_metrics(
        dataset.test_y, test_mean_physical, dataset.output_names
    )

    selection_rows = []
    for candidate in candidates:
        selection_rows.append(
            {
                "config_index": candidate.config_index,
                "config": json.dumps(_jsonable(candidate.config), ensure_ascii=False),
                "validation_rmse_standardized_macro": (
                    candidate.validation_rmse_standardized_macro
                ),
                "validation_nll_macro": candidate.validation_nll_macro,
                "fit_seconds": candidate.fit_seconds,
                "selected": candidate.config_index == selected.config_index,
            }
        )
    pd.DataFrame(selection_rows).to_csv(result_dir / "model_selection.csv", index=False)

    arrays: dict[str, np.ndarray] = {
        "y_true_standardized": prepared.y_test,
        "mean_standardized": test_prediction.mean,
        "y_true_physical": dataset.test_y,
        "mean_physical": test_mean_physical,
    }
    variance_scale = None
    residual_floor = None
    if test_prediction.variance is not None:
        calibration_prediction = _predict(
            model_name, final_model, prepared.x_calibration, device
        )
        if calibration_prediction.variance is None:
            raise RuntimeError("A probabilistic model returned no calibration variance.")
        if model_name == "ensemble_mmoe":
            residual_floor = fit_residual_variance_floor(
                prepared.y_calibration,
                calibration_prediction.mean,
                calibration_prediction.variance,
            )
            calibrated_variance = (
                test_prediction.variance + residual_floor[None, :]
            )
            # Retained only for compatibility with legacy result readers; the
            # final DE-MMoE uses additive calibration, not a scale multiplier.
            variance_scale = np.ones_like(residual_floor)
        else:
            variance_scale = fit_variance_scale(
                prepared.y_calibration,
                calibration_prediction.mean,
                calibration_prediction.variance,
            )
            calibrated_variance = test_prediction.variance * variance_scale[None, :]
        raw_nll = gaussian_nll(
            prepared.y_test, test_prediction.mean, test_prediction.variance
        )
        calibrated_nll = gaussian_nll(
            prepared.y_test, test_prediction.mean, calibrated_variance
        )
        probability_summary = _group_probability_metrics(
            dataset.output_names, raw_nll, calibrated_nll
        )
        probability_summary.to_csv(result_dir / "probability_metrics.csv", index=False)
        raw_coverage = coverage_table(
            prepared.y_test, test_prediction.mean, test_prediction.variance
        ).assign(variance="raw")
        calibrated_coverage = coverage_table(
            prepared.y_test, test_prediction.mean, calibrated_variance
        ).assign(variance="calibrated")
        pd.concat((raw_coverage, calibrated_coverage), ignore_index=True).to_csv(
            result_dir / "coverage_metrics.csv", index=False
        )
        arrays.update(
            variance_standardized=test_prediction.variance,
            calibrated_variance_standardized=calibrated_variance,
            variance_scale=variance_scale,
        )
        if test_prediction.model_variance is not None:
            arrays["model_variance_standardized"] = test_prediction.model_variance
        if test_prediction.residual_variance is not None:
            arrays["residual_variance_standardized"] = test_prediction.residual_variance
        if residual_floor is not None:
            arrays["residual_floor_standardized"] = residual_floor
            arrays["residual_variance_standardized"] = np.broadcast_to(
                residual_floor[None, :],
                test_prediction.variance.shape,
            ).copy()
    np.savez_compressed(result_dir / "test_predictions.npz", **arrays)
    _save_fitted_model(
        model_name,
        final_model,
        selected.config,
        prepared,
        variance_scale,
        residual_floor,
        result_dir,
    )

    inference_seconds = _median_inference_seconds(
        lambda: _predict(model_name, final_model, prepared.x_test, device)
    )
    metadata = {
        "model": model_name,
        "seed": seed,
        "device": device,
        "thread_limit": threads,
        "selected_config_index": selected.config_index,
        "selected_config": _jsonable(selected.config),
        "validation_rmse_standardized_macro": (
            selected.validation_rmse_standardized_macro
        ),
        "validation_nll_macro": selected.validation_nll_macro,
        "search_seconds": search_seconds,
        "final_model_fit_seconds": final_fit_seconds,
        "ensemble_member_fit_seconds_sum": ensemble_member_fit_seconds_sum,
        "ensemble_workers": (
            min(ensemble_workers, len(final_model))
            if model_name == "ensemble_mmoe"
            else None
        ),
        "ensemble_execution": (
            "parallel" if model_name == "ensemble_mmoe" and ensemble_workers > 1
            else "serial" if model_name == "ensemble_mmoe"
            else None
        ),
        "final_model_fit_time_definition": "wall_clock_seconds",
        "test_inference_seconds_median": inference_seconds,
        "control_stress_metrics": control,
        "probabilistic": test_prediction.variance is not None,
        "training_objective": (
            "group_balanced_standardized_mse"
            if model_name == "ensemble_mmoe"
            else "mse"
            if model_name in {
                "single_mlp_mc_dropout",
                "multitask_mlp_mc_dropout",
            }
            else None
        ),
        "checkpoint_selection_metric": (
            "validation_rmse_standardized_macro"
            if model_name == "ensemble_mmoe"
            else "validation_mse_deterministic_mean"
            if model_name in {
                "single_mlp_mc_dropout",
                "multitask_mlp_mc_dropout",
            }
            else None
        ),
        "uncertainty_method": (
            "mc_dropout_model_variance"
            if model_name in {
                "single_mlp_mc_dropout",
                "multitask_mlp_mc_dropout",
            }
            else "five_member_disagreement_plus_calibrated_residual_floor"
            if model_name == "ensemble_mmoe"
            else None
        ),
        "mc_samples": (
            MC_DROPOUT_SAMPLES
            if model_name in {
                "single_mlp_mc_dropout",
                "multitask_mlp_mc_dropout",
            }
            else None
        ),
        "model_diagnostics": _model_diagnostics(model_name, final_model),
        "architecture_selection_objective": (
            "equal_output_standardized_mse"
            if model_name == "ensemble_mmoe"
            else None
        ),
        "selected_search_member_reused": (
            False if model_name == "ensemble_mmoe" else None
        ),
        "variance_calibration": (
            "taskwise_additive_residual_floor"
            if model_name == "ensemble_mmoe"
            else "taskwise_multiplicative_scale"
            if test_prediction.variance is not None
            else None
        ),
        "input_names": list(dataset.input_names),
        "output_names": list(dataset.output_names),
        "split": {
            "train": len(prepared.indices.train),
            "validation": len(prepared.indices.validation),
            "calibration": len(prepared.indices.calibration),
            "independent_test": len(prepared.y_test),
        },
    }
    with (result_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def _refresh_summary_files(output_root: Path) -> None:
    accuracy_rows: list[pd.DataFrame] = []
    probability_rows: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, Any]] = []
    for metadata_path in sorted(output_root.glob("seed_*/*/run_metadata.json")):
        result_dir = metadata_path.parent
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        seed = int(metadata["seed"])
        model_name = str(metadata["model"])
        accuracy_path = result_dir / "group_metrics.csv"
        if accuracy_path.exists():
            accuracy_rows.append(
                pd.read_csv(accuracy_path).assign(seed=seed, model=model_name)
            )
        probability_path = result_dir / "probability_metrics.csv"
        if probability_path.exists():
            probability_rows.append(
                pd.read_csv(probability_path).assign(seed=seed, model=model_name)
            )
        runtime_rows.append(
            {
                "seed": seed,
                "model": model_name,
                "search_seconds": metadata["search_seconds"],
                "final_model_fit_seconds": metadata["final_model_fit_seconds"],
                "ensemble_member_fit_seconds_sum": metadata.get(
                    "ensemble_member_fit_seconds_sum"
                ),
                "ensemble_workers": metadata.get("ensemble_workers"),
                "test_inference_seconds_median": metadata[
                    "test_inference_seconds_median"
                ],
            }
        )
    if accuracy_rows:
        pd.concat(accuracy_rows, ignore_index=True).to_csv(
            output_root / "summary_accuracy.csv", index=False
        )
    if probability_rows:
        pd.concat(probability_rows, ignore_index=True).to_csv(
            output_root / "summary_probability.csv", index=False
        )
    if runtime_rows:
        pd.DataFrame(runtime_rows).to_csv(
            output_root / "summary_runtime.csv", index=False
        )


def run_one(
    model_name: str,
    seed: int,
    *,
    device: str,
    threads: int,
    ensemble_workers: int,
    smoke: bool,
    output_root: Path,
) -> None:
    dataset = load_dataset()
    prepared = prepare_split(dataset, make_split(seed))
    candidates: list[CandidateResult] = []
    selected: CandidateResult | None = None
    search_start = time.perf_counter()
    for config_index, config in enumerate(_grid_for(model_name, smoke)):
        model, prediction, fit_seconds = _fit_candidate(
            model_name,
            config,
            prepared,
            seed=seed,
            device=device,
            smoke=smoke,
        )
        validation_nll = None
        if prediction.variance is not None:
            validation_nll = float(
                np.mean(
                    gaussian_nll(
                        prepared.y_validation, prediction.mean, prediction.variance
                    )
                )
            )
        candidate = CandidateResult(
            config_index=config_index,
            config=config,
            validation_rmse_standardized_macro=_validation_standardized_rmse(
                prepared.y_validation, prediction.mean
            ),
            validation_nll_macro=validation_nll,
            fit_seconds=fit_seconds,
            model=model,
        )
        if selected is None or _candidate_key(candidate) < _candidate_key(selected):
            if selected is not None:
                selected.model = None
            selected = candidate
        else:
            candidate.model = None
        candidates.append(candidate)
    search_seconds = time.perf_counter() - search_start
    if selected is None or selected.model is None:
        raise RuntimeError("No valid candidate model was selected.")
    final_model: Any = selected.model
    final_fit_seconds = selected.fit_seconds
    ensemble_member_fit_seconds_sum = None
    if model_name == "ensemble_mmoe":
        (
            final_model,
            final_fit_seconds,
            ensemble_member_fit_seconds_sum,
        ) = _complete_ensemble(
            selected,
            prepared,
            seed=seed,
            device=device,
            smoke=smoke,
            workers=ensemble_workers,
            threads=threads,
        )
    _write_result(
        model_name,
        seed,
        selected,
        candidates,
        prepared,
        dataset,
        final_model,
        final_fit_seconds,
        search_seconds,
        device,
        threads,
        ensemble_workers,
        ensemble_member_fit_seconds_sum,
        output_root,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=(*MODEL_NAMES, "all"),
        default=["all"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--ensemble-workers",
        type=int,
        default=1,
        help=(
            "Independent CPU processes used for DE-MMoE members. "
            "Use 1 for the original serial behavior or up to 5 for parallel fitting."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one configuration and a tiny neural/XGBoost budget for code checks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun model/seed combinations that already have complete metadata.",
    )
    parser.add_argument(
        "--defer-summary",
        action="store_true",
        help="Skip shared summary refresh so different seeds can run concurrently.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown_seeds = sorted(set(args.seeds) - set(SEEDS))
    if unknown_seeds:
        raise ValueError(f"Seeds must use the frozen protocol: {SEEDS}; got {unknown_seeds}")
    if args.threads < 1:
        raise ValueError("--threads must be a positive integer.")
    if args.ensemble_workers < 1:
        raise ValueError("--ensemble-workers must be a positive integer.")
    if args.ensemble_workers > 1 and args.device != "cpu":
        raise ValueError("--ensemble-workers > 1 currently requires --device cpu.")
    models = MODEL_NAMES if "all" in args.models else tuple(args.models)
    neural_models = {
        "single_mlp",
        "single_mlp_mc_dropout",
        "multitask_mlp",
        "multitask_mlp_mc_dropout",
        "deterministic_mmoe",
        "ensemble_mmoe",
    }
    if neural_models.intersection(models):
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is required for the selected neural models.") from error
        torch.set_num_threads(args.threads)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        thread_context = nullcontext()
    else:
        thread_context = threadpool_limits(limits=args.threads)
    output_root = (
        args.output
        if args.output is not None
        else RESULTS_DIR.parent / "smoke_results"
        if args.smoke
        else RESULTS_DIR
    )
    output_root.mkdir(parents=True, exist_ok=True)
    split_manifest().to_csv(output_root / "split_manifest.csv", index=False)
    with thread_context:
        for seed in args.seeds:
            for model_name in models:
                metadata_path = (
                    output_root / f"seed_{seed}" / model_name / "run_metadata.json"
                )
                if metadata_path.exists() and not args.overwrite:
                    print(f"Skipping completed {model_name}, seed={seed}", flush=True)
                    continue
                print(f"Running {model_name}, seed={seed}, smoke={args.smoke}", flush=True)
                run_one(
                    model_name,
                    seed,
                    device=args.device,
                    threads=args.threads,
                    ensemble_workers=args.ensemble_workers,
                    smoke=args.smoke,
                    output_root=output_root,
                )
                if not args.defer_summary:
                    _refresh_summary_files(output_root)


if __name__ == "__main__":
    main()
