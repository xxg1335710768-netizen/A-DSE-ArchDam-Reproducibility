"""Append-only records for A-DSE runs."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ._common import INPUT_DIM, OBJECTIVE_COUNT, finite_matrix


OBJECTIVE_NAMES = ("V_dam", "S_t")


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"


class ADSEExperimentLogger:
    """Write every A-DSE event immediately to a dedicated run directory."""

    def __init__(
        self,
        run_directory: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
        resume: bool = False,
    ) -> None:
        self.root = Path(run_directory)
        if self.root.exists() and any(self.root.iterdir()) and not resume:
            raise FileExistsError(
                f"Run directory is not empty: {self.root}. Use resume=True to append."
            )
        self.root.mkdir(parents=True, exist_ok=True)
        self.front_directory = self.root / "pareto_fronts"
        self.front_directory.mkdir(exist_ok=True)
        metadata_path = self.root / "metadata.json"
        if not metadata_path.exists():
            payload = {
                "schema_version": 3,
                "created_at": _timestamp(),
                "objective_names": OBJECTIVE_NAMES,
                "input_dimension": INPUT_DIM,
                "uncertainty_score": (
                    "weighted geometric mean of standardized model standard "
                    "deviations; weights use the current three-point mean errors"
                ),
                "accuracy_gate": (
                    "maximum of the two objective-wise RMS relative errors "
                    "over the three current Pareto representatives"
                ),
                "fe_feasible_usage": (
                    "diagnostic only; held-out gate representatives remain "
                    "excluded from surrogate training"
                ),
                **dict(metadata or {}),
            }
            metadata_path.write_text(
                json.dumps(_json_value(payload), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _append(self, filename: str, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return
        path = self.root / filename
        frame = pd.DataFrame(rows)
        if path.exists():
            existing_columns = pd.read_csv(path, nrows=0).columns.tolist()
            if frame.columns.tolist() != existing_columns:
                raise ValueError(f"Log schema differs from existing file: {path}")
        frame.to_csv(path, mode="a", header=not path.exists(), index=False)

    def log_gate_check(
        self,
        *,
        outer_iter: int,
        inner_iter: int,
        model_id: str,
        pf_revision: int,
        model_updated_since_pf: bool,
        objective_weights: np.ndarray,
        labels: Sequence[str],
        x: np.ndarray,
        prediction: np.ndarray,
        fe_values: np.ndarray,
        relative_error: np.ndarray,
        fe_feasible: np.ndarray,
        distances: np.ndarray,
        statuses: Sequence[str],
        gate_passed: bool,
        new_fe_count: int,
        cumulative_fe_count: int,
    ) -> None:
        points = finite_matrix(x, INPUT_DIM, "gate x")
        predicted = np.asarray(prediction, dtype=np.float64)
        fe = np.asarray(fe_values, dtype=np.float64)[:, :OBJECTIVE_COUNT]
        errors = np.asarray(relative_error, dtype=np.float64)
        feasible = np.asarray(fe_feasible, dtype=bool).reshape(-1)
        distances = np.asarray(distances, dtype=np.float64).reshape(-1)
        weights = np.asarray(objective_weights, dtype=np.float64).reshape(-1)
        count = len(points)
        if any(len(values) != count for values in (predicted, fe, errors, distances)):
            raise ValueError("Gate log row counts differ.")
        if predicted.shape[1] < OBJECTIVE_COUNT or errors.shape[1] != OBJECTIVE_COUNT:
            raise ValueError("Gate predictions and errors require two objectives.")
        if len(labels) != count or len(statuses) != count:
            raise ValueError("Gate labels or statuses do not match the points.")
        if len(feasible) != count:
            raise ValueError("Gate FE feasibility does not match the points.")
        if len(weights) != OBJECTIVE_COUNT or not np.isclose(weights.sum(), 1.0):
            raise ValueError("Gate objective weights must contain two values summing to one.")

        worst_flat = int(np.argmax(errors))
        worst_point, worst_objective = np.unravel_index(worst_flat, errors.shape)
        objective_rms_error = np.sqrt(np.mean(errors**2, axis=0))
        gate_error = float(np.max(objective_rms_error))
        shared = {
            "timestamp": _timestamp(),
            "outer_iter": outer_iter,
            "inner_iter": inner_iter,
            "model_id": model_id,
            "pf_revision": int(pf_revision),
            "model_updated_since_pf": bool(model_updated_since_pf),
            "gate_passed": bool(gate_passed),
            "all_fe_feasible": bool(np.all(feasible)),
            "gate_error": gate_error,
            "rms_error_f1": float(objective_rms_error[0]),
            "rms_error_f2": float(objective_rms_error[1]),
            "worst_error": float(errors[worst_point, worst_objective]),
            "worst_error_point": labels[worst_point],
            "worst_error_objective": OBJECTIVE_NAMES[worst_objective],
            "new_fe_count": int(new_fe_count),
            "cumulative_fe_count": int(cumulative_fe_count),
            "mean_error_f1": float(errors[:, 0].mean()),
            "mean_error_f2": float(errors[:, 1].mean()),
            "uncertainty_weight_f1": float(weights[0]),
            "uncertainty_weight_f2": float(weights[1]),
        }
        rows = []
        for index, label in enumerate(labels):
            row = {
                **shared,
                "representative": label,
                "status": statuses[index],
                "distance_rms": float(distances[index]),
                "pred_f1": float(predicted[index, 0]),
                "pred_f2": float(predicted[index, 1]),
                "fe_f1": float(fe[index, 0]),
                "fe_f2": float(fe[index, 1]),
                "err_f1": float(errors[index, 0]),
                "err_f2": float(errors[index, 1]),
                "point_worst_error": float(errors[index].max()),
                "fe_feasible": bool(feasible[index]),
            }
            row.update({f"x{j + 1}": float(value) for j, value in enumerate(points[index])})
            rows.append(row)
        self._append("gate_checks.csv", rows)

    def log_uncertainty_samples(
        self,
        *,
        outer_iter: int,
        inner_iter: int,
        model_id: str,
        pf_revision: int,
        objective_weights: np.ndarray,
        scores: np.ndarray,
        diversity_fallback: np.ndarray,
        x: np.ndarray,
        objective_model_std: np.ndarray,
        prediction: np.ndarray,
        archive_distance: np.ndarray,
        batch_distance: np.ndarray,
        fe_values: np.ndarray,
        cumulative_fe_count: int,
    ) -> None:
        points = finite_matrix(x, INPUT_DIM, "uncertainty x")
        std = np.asarray(objective_model_std, dtype=np.float64)[:, :OBJECTIVE_COUNT]
        predicted = np.asarray(prediction, dtype=np.float64)[:, :OBJECTIVE_COUNT]
        distances = np.asarray(archive_distance, dtype=np.float64).reshape(-1)
        batch_distances = np.asarray(batch_distance, dtype=np.float64).reshape(-1)
        weights = np.asarray(objective_weights, dtype=np.float64).reshape(-1)
        weighted_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        fallback = np.asarray(diversity_fallback, dtype=bool).reshape(-1)
        fe = np.asarray(fe_values, dtype=np.float64)
        if any(
            len(values) != len(points)
            for values in (
                std,
                predicted,
                distances,
                batch_distances,
                weighted_scores,
                fallback,
                fe,
            )
        ):
            raise ValueError("Uncertainty log row counts differ.")
        if len(weights) != OBJECTIVE_COUNT or not np.isclose(weights.sum(), 1.0):
            raise ValueError(
                "Uncertainty objective weights must contain two values summing to one."
            )
        relative_error = np.abs(predicted - fe[:, :OBJECTIVE_COUNT]) / np.maximum(
            np.abs(fe[:, :OBJECTIVE_COUNT]), 1e-12
        )
        rows = []
        for index, point in enumerate(points):
            row = {
                "timestamp": _timestamp(),
                "outer_iter": outer_iter,
                "inner_iter": inner_iter,
                "model_id": model_id,
                "pf_revision": int(pf_revision),
                "selection_rank": index + 1,
                "model_std_f1": float(std[index, 0]),
                "model_std_f2": float(std[index, 1]),
                "uncertainty_weight_f1": float(weights[0]),
                "uncertainty_weight_f2": float(weights[1]),
                "joint_uncertainty": float(weighted_scores[index]),
                "diversity_fallback": bool(fallback[index]),
                "distance_to_archive": float(distances[index]),
                "distance_to_batch": float(batch_distances[index]),
                "pred_f1": float(predicted[index, 0]),
                "pred_f2": float(predicted[index, 1]),
                "fe_f1": float(fe[index, 0]),
                "fe_f2": float(fe[index, 1]),
                "err_f1": float(relative_error[index, 0]),
                "err_f2": float(relative_error[index, 1]),
                "point_worst_error": float(relative_error[index].max()),
                "cumulative_fe_count": int(cumulative_fe_count),
            }
            row.update({f"x{j + 1}": float(value) for j, value in enumerate(point)})
            row.update(
                {f"fe_output_{j + 1:02d}": float(value) for j, value in enumerate(fe[index])}
            )
            rows.append(row)
        self._append("uncertainty_samples.csv", rows)

    def log_fe_failure(
        self,
        *,
        outer_iter: int,
        inner_iter: int,
        model_id: str,
        stage: str,
        role: str,
        x: np.ndarray,
        message: str,
        cumulative_fe_count: int,
    ) -> None:
        point = finite_matrix(x, INPUT_DIM, "failed FE x")[0]
        row = {
            "timestamp": _timestamp(),
            "outer_iter": outer_iter,
            "inner_iter": inner_iter,
            "model_id": model_id,
            "stage": stage,
            "role": role,
            "message": str(message),
            "cumulative_successful_fe_count": int(cumulative_fe_count),
        }
        row.update({f"x{i + 1}": float(value) for i, value in enumerate(point)})
        self._append("fe_failures.csv", [row])

    def log_gradient_shap(
        self, *, outer_iter: int, model_id: str, result: Any
    ) -> None:
        ranking = np.argsort(-np.asarray(result.combined_importance))
        rank = np.empty(len(ranking), dtype=int)
        rank[ranking] = np.arange(1, len(ranking) + 1)
        rows = []
        for feature_index, feature in enumerate(result.feature_names):
            rows.append(
                {
                    "timestamp": _timestamp(),
                    "outer_iter": outer_iter,
                    "model_id": model_id,
                    "feature": feature,
                    "mean_abs_f1": float(result.mean_absolute_by_task[0, feature_index]),
                    "mean_abs_f2": float(result.mean_absolute_by_task[1, feature_index]),
                    "normalized_f1": float(result.normalized_by_task[0, feature_index]),
                    "normalized_f2": float(result.normalized_by_task[1, feature_index]),
                    "combined_importance": float(result.combined_importance[feature_index]),
                    "importance_rank": int(rank[feature_index]),
                }
            )
        self._append("variable_importance.csv", rows)

    def log_bounds_update(
        self,
        *,
        outer_iter: int,
        model_id: str,
        previous_lower: np.ndarray,
        previous_upper: np.ndarray,
        update: Any,
    ) -> None:
        lower = np.asarray(previous_lower, dtype=np.float64).reshape(-1)
        upper = np.asarray(previous_upper, dtype=np.float64).reshape(-1)
        rows = []
        for index in range(INPUT_DIM):
            rows.append(
                {
                    "timestamp": _timestamp(),
                    "outer_iter": outer_iter,
                    "model_id": model_id,
                    "feature": f"x{index + 1}",
                    "previous_lower": float(lower[index]),
                    "previous_upper": float(upper[index]),
                    "raw_pf_min": float(update.raw_pareto_min[index]),
                    "raw_pf_max": float(update.raw_pareto_max[index]),
                    "ema_pf_min": float(update.ema_pareto_min[index]),
                    "ema_pf_max": float(update.ema_pareto_max[index]),
                    "pareto_span": float(update.pareto_span[index]),
                    "effective_span": float(update.effective_span[index]),
                    "importance": float(update.importance[index]),
                    "eta": float(update.eta),
                    "expansion": float(update.expansion[index]),
                    "new_lower": float(update.lower[index]),
                    "new_upper": float(update.upper[index]),
                }
            )
        self._append("design_space_bounds.csv", rows)

    def log_pareto_front(
        self,
        *,
        outer_iter: int,
        inner_iter: int,
        model_id: str,
        pf_revision: int,
        x: np.ndarray,
        objectives: np.ndarray,
        cumulative_fe_count: int,
        training_size: int,
    ) -> Path:
        points = finite_matrix(x, INPUT_DIM, "pareto x")
        values = np.asarray(objectives, dtype=np.float64)
        if values.ndim != 2 or values.shape != (len(points), OBJECTIVE_COUNT):
            raise ValueError("Pareto objectives must have shape (N, 2).")
        filename = (
            f"outer_{outer_iter:03d}_inner_{inner_iter:03d}_"
            f"{_safe_name(model_id)}.npz"
        )
        path = self.front_directory / filename
        np.savez_compressed(path, x=points, objectives=values)
        self._append(
            "pareto_summary.csv",
            [
                {
                    "timestamp": _timestamp(),
                    "outer_iter": outer_iter,
                    "inner_iter": inner_iter,
                    "model_id": model_id,
                    "pf_revision": int(pf_revision),
                    "cumulative_fe_count": int(cumulative_fe_count),
                    "training_size": int(training_size),
                    "point_count": len(points),
                    "f1_min": float(values[:, 0].min()),
                    "f1_max": float(values[:, 0].max()),
                    "f2_min": float(values[:, 1].min()),
                    "f2_max": float(values[:, 1].max()),
                    "front_file": str(path.relative_to(self.root)),
                }
            ],
        )
        return path

    def log_iteration_summary(
        self,
        *,
        outer_iter: int,
        model_id: str,
        refinement_cycles: int,
        pf_optimizations: int,
        model_updates: int,
        added_fe_count: int,
        cumulative_fe_count: int,
        training_size: int,
        gate_error: float,
        objective_rms_error: np.ndarray,
        worst_error: float,
        objective_weights: np.ndarray,
        failed_fe_count: int = 0,
    ) -> None:
        weights = np.asarray(objective_weights, dtype=np.float64).reshape(-1)
        rms_error = np.asarray(objective_rms_error, dtype=np.float64).reshape(-1)
        if len(weights) != OBJECTIVE_COUNT or not np.isclose(weights.sum(), 1.0):
            raise ValueError("Summary objective weights must sum to one.")
        if len(rms_error) != OBJECTIVE_COUNT or np.any(rms_error < 0.0):
            raise ValueError("Summary objective RMS errors must contain two values.")
        self._append(
            "iteration_summary.csv",
            [
                {
                    "timestamp": _timestamp(),
                    "outer_iter": outer_iter,
                    "model_id": model_id,
                    "refinement_cycles": int(refinement_cycles),
                    "pf_optimizations": int(pf_optimizations),
                    "model_updates": int(model_updates),
                    "added_fe_count": int(added_fe_count),
                    "failed_fe_count": int(failed_fe_count),
                    "cumulative_fe_count": int(cumulative_fe_count),
                    "training_size": int(training_size),
                    "final_gate_error": float(gate_error),
                    "final_rms_error_f1": float(rms_error[0]),
                    "final_rms_error_f2": float(rms_error[1]),
                    "final_worst_error": float(worst_error),
                    "final_uncertainty_weight_f1": float(weights[0]),
                    "final_uncertainty_weight_f2": float(weights[1]),
                }
            ],
        )
