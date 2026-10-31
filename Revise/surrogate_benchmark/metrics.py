"""Shared deterministic and probabilistic metrics for all benchmark models."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


NOMINAL_LEVELS = (0.50, 0.80, 0.90, 0.95)


def _as_matching_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.ndim != 2 or pred.shape != true.shape:
        raise ValueError(f"Expected matching 2D arrays, got {true.shape} and {pred.shape}.")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Metric inputs contain NaN or infinite values.")
    return true, pred


def per_output_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_names: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    true, pred = _as_matching_matrices(y_true, y_pred)
    names = list(output_names)
    if len(names) != true.shape[1]:
        raise ValueError("Output-name count does not match the response dimension.")
    residual = pred - true
    mse = np.mean(residual**2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residual), axis=0)
    centered = true - np.mean(true, axis=0, keepdims=True)
    denominator = np.sum(centered**2, axis=0)
    r2 = 1.0 - np.sum(residual**2, axis=0) / np.maximum(denominator, 1e-12)
    test_std = np.std(true, axis=0, ddof=1)
    nrmse = rmse / np.maximum(test_std, 1e-12)
    return pd.DataFrame(
        {
            "output_index": np.arange(true.shape[1], dtype=int),
            "output_name": names,
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "nrmse_std": nrmse,
        }
    )


def group_regression_metrics(per_output: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = {
        "Volu": per_output.iloc[[0]],
        "St": per_output.iloc[[1]],
        "stress_39": per_output.iloc[2:],
        "all_41": per_output,
    }
    for group_name, frame in groups.items():
        rows.append(
            {
                "group": group_name,
                "output_count": len(frame),
                "r2_macro": float(frame["r2"].mean()),
                "r2_minimum": float(frame["r2"].min()),
                "rmse_macro_physical": float(frame["rmse"].mean()),
                "nrmse_macro": float(frame["nrmse_std"].mean()),
            }
        )
    return pd.DataFrame(rows)


def control_stress_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_names: tuple[str, ...] | list[str],
) -> dict[str, float]:
    true, pred = _as_matching_matrices(y_true, y_pred)
    names = list(output_names)
    s1_indices = [index for index, name in enumerate(names) if "_S1_" in name]
    s3_indices = [index for index, name in enumerate(names) if "_S3_" in name]
    if not s1_indices or not s3_indices:
        raise ValueError("Both S1 and S3 stress-response columns are required.")
    true_max_s1 = np.max(true[:, s1_indices], axis=1)
    pred_max_s1 = np.max(pred[:, s1_indices], axis=1)
    true_min_s3 = np.min(true[:, s3_indices], axis=1)
    pred_min_s3 = np.min(pred[:, s3_indices], axis=1)

    def errors(reference: np.ndarray, estimate: np.ndarray) -> tuple[float, float]:
        difference = estimate - reference
        return float(np.mean(np.abs(difference))), float(np.sqrt(np.mean(difference**2)))

    s1_mae, s1_rmse = errors(true_max_s1, pred_max_s1)
    s3_mae, s3_rmse = errors(true_min_s3, pred_min_s3)
    return {
        "maximum_s1_mae": s1_mae,
        "maximum_s1_rmse": s1_rmse,
        "minimum_s3_mae": s3_mae,
        "minimum_s3_rmse": s3_rmse,
    }


def gaussian_nll(
    y_true: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
) -> np.ndarray:
    true, prediction = _as_matching_matrices(y_true, mean)
    var = np.asarray(variance, dtype=np.float64)
    if var.shape != true.shape:
        raise ValueError("Variance shape differs from the response shape.")
    var = np.maximum(var, 1e-12)
    values = 0.5 * (np.log(2.0 * np.pi * var) + (true - prediction) ** 2 / var)
    return np.mean(values, axis=0)


def fit_variance_scale(
    y_calibration: np.ndarray,
    mean: np.ndarray,
    total_variance: np.ndarray,
    minimum: float = 0.05,
    maximum: float = 20.0,
) -> np.ndarray:
    true, prediction = _as_matching_matrices(y_calibration, mean)
    variance = np.maximum(np.asarray(total_variance, dtype=np.float64), 1e-12)
    if variance.shape != true.shape:
        raise ValueError("Variance shape differs from the calibration responses.")
    scale = np.mean((true - prediction) ** 2 / variance, axis=0)
    return np.clip(scale, minimum, maximum)


def fit_residual_variance_floor(
    y_calibration: np.ndarray,
    mean: np.ndarray,
    model_variance: np.ndarray,
) -> np.ndarray:
    """Fit one additive non-negative residual variance per output by NLL."""
    true, prediction = _as_matching_matrices(y_calibration, mean)
    variance = np.maximum(np.asarray(model_variance, dtype=np.float64), 1e-12)
    if variance.shape != true.shape:
        raise ValueError("Variance shape differs from the calibration responses.")
    residual_squared = (true - prediction) ** 2
    floors = np.zeros(variance.shape[1], dtype=np.float64)
    for task in range(variance.shape[1]):
        task_variance = variance[:, task]
        task_residual = residual_squared[:, task]
        upper = max(
            1.0,
            100.0 * float(np.mean(task_residual)),
            100.0 * float(np.mean(task_variance)),
        )

        def objective(floor: float) -> float:
            total = np.maximum(task_variance + floor, 1e-12)
            return float(np.mean(np.log(total) + task_residual / total))

        result = minimize_scalar(
            objective,
            bounds=(0.0, upper),
            method="bounded",
            options={"xatol": 1e-10},
        )
        candidate = float(result.x)
        floors[task] = candidate if objective(candidate) < objective(0.0) else 0.0
    return floors


def coverage_table(
    y_true: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
) -> pd.DataFrame:
    true, prediction = _as_matching_matrices(y_true, mean)
    var = np.maximum(np.asarray(variance, dtype=np.float64), 1e-12)
    if var.shape != true.shape:
        raise ValueError("Variance shape differs from the response shape.")
    standard_deviation = np.sqrt(var)
    rows = []
    for nominal in NOMINAL_LEVELS:
        z_value = NormalDist().inv_cdf(0.5 + nominal / 2.0)
        inside = np.abs(true - prediction) <= z_value * standard_deviation
        coverage_by_output = np.mean(inside, axis=0)
        width_by_output = np.mean(2.0 * z_value * standard_deviation, axis=0)
        rows.append(
            {
                "nominal": nominal,
                "coverage_macro": float(np.mean(coverage_by_output)),
                "coverage_objectives": float(np.mean(coverage_by_output[:2])),
                "coverage_stress": float(np.mean(coverage_by_output[2:])),
                "width_macro_standardized": float(np.mean(width_by_output)),
            }
        )
    return pd.DataFrame(rows)


def weighted_geometric_mean(
    values: np.ndarray,
    weights: np.ndarray | None = None,
    epsilon: float = 1e-12,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("Values must have shape (samples, tasks).")
    if weights is None:
        weight_array = np.full(array.shape[1], 1.0 / array.shape[1])
    else:
        weight_array = np.asarray(weights, dtype=np.float64)
        if weight_array.shape != (array.shape[1],):
            raise ValueError("Weight count differs from the task count.")
        if np.any(weight_array < 0.0) or weight_array.sum() <= 0.0:
            raise ValueError("Weights must be non-negative with a positive sum.")
        weight_array = weight_array / weight_array.sum()
    return np.exp(
        np.sum(np.log(np.maximum(array, epsilon)) * weight_array[None, :], axis=1)
    )
