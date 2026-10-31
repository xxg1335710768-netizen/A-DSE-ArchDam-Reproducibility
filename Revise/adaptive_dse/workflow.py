"""High-level inner loop for the three-point A-DSE accuracy gate.

The caller supplies optimization, FE and retraining callbacks.  Importing this
module never starts Rhino or ANSYS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

from ._common import (
    FEEvaluationFailure,
    INPUT_DIM,
    OBJECTIVE_COUNT,
    finite_matrix,
    finite_vector,
)
from .design_space_exploration import (
    OnlineValidationSet,
    ParetoRepresentatives,
    ValidationPlan,
    complete_accuracy_check,
    normalized_rms_distance,
    objective_error_weights,
    plan_accuracy_check,
    rank_pareto_candidates_by_role,
    select_uncertainty_points,
)
from .experiment_log import ADSEExperimentLogger

if TYPE_CHECKING:
    from .ensemble_surrogate import EnsembleMMoESurrogate


class FEBudgetExceeded(RuntimeError):
    """Raised when the optional cumulative FE budget would be exceeded."""


class CandidatePoolExhausted(RuntimeError):
    """Raised when the current PF has no unevaluated candidate for refinement."""


@dataclass(frozen=True)
class GateRefinementResult:
    surrogate: EnsembleMMoESurrogate
    training_x: np.ndarray
    training_y: np.ndarray
    validation: OnlineValidationSet
    pareto_x: np.ndarray
    pareto_objectives: np.ndarray
    fe_evaluations: int
    failed_fe_evaluations: int
    refinement_cycles: int
    pf_optimizations: int
    model_updates: int
    heldout_validation_x: np.ndarray
    heldout_validation_y: np.ndarray
    points_for_next_training_x: np.ndarray
    points_for_next_training_y: np.ndarray
    history: pd.DataFrame


def _next_role_candidate(
    pareto_x: np.ndarray,
    pareto_objectives: np.ndarray,
    role: str,
    excluded_x: np.ndarray,
    reference_width: np.ndarray,
    minimum_distance: float,
) -> int:
    ranking = rank_pareto_candidates_by_role(
        pareto_x, pareto_objectives, role
    )
    if len(excluded_x) == 0:
        return int(ranking[0])
    for source_index in ranking:
        point = pareto_x[[int(source_index)]]
        distance = normalized_rms_distance(
            point, excluded_x, reference_width
        ).min()
        if distance >= minimum_distance:
            return int(source_index)
    raise ValueError(
        f"No remaining Pareto candidate can replace failed role '{role}'."
    )


def _append_unique(
    base_x: np.ndarray,
    base_y: np.ndarray,
    added_x: np.ndarray,
    added_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(added_x) == 0:
        return base_x, base_y
    known = {tuple(np.round(row, 12)) for row in base_x}
    keep = []
    for index, row in enumerate(added_x):
        key = tuple(np.round(row, 12))
        if key not in known:
            keep.append(index)
            known.add(key)
    if not keep:
        return base_x, base_y
    return (
        np.vstack((base_x, added_x[keep])),
        np.vstack((base_y, added_y[keep])),
    )


def _exact_row_index(rows: np.ndarray, point: np.ndarray) -> int | None:
    """Return the exact-design row index used for FE result reuse."""

    if len(rows) == 0:
        return None
    matches = np.flatnonzero(
        np.all(
            np.isclose(rows, point, rtol=0.0, atol=1e-12),
            axis=1,
        )
    )
    return None if len(matches) == 0 else int(matches[0])


def refine_pareto_until_accurate(
    surrogate: EnsembleMMoESurrogate,
    training_x: np.ndarray,
    training_y: np.ndarray,
    *,
    optimize: Callable[
        [EnsembleMMoESurrogate], tuple[np.ndarray, np.ndarray]
    ],
    evaluate_fe: Callable[[np.ndarray], np.ndarray],
    retrain: Callable[
        [np.ndarray, np.ndarray], EnsembleMMoESurrogate
    ],
    reference_width: np.ndarray,
    representativeness_threshold: float,
    previous_validation: OnlineValidationSet | None = None,
    heldout_validation_x: np.ndarray | None = None,
    heldout_validation_y: np.ndarray | None = None,
    relative_error_limit: float = 0.01,
    uncertainty_batch_size: int = 3,
    strict_duplicate_distance: float = 1e-8,
    uncertainty_diversity_distance: float = 0.025,
    uncertainty_weight_floor: float = 0.20,
    acquisition_mode: str = "uncertainty",
    acquisition_seed: int = 2025,
    accuracy_trigger_enabled: bool = True,
    candidate_feasible: Callable[
        [EnsembleMMoESurrogate, np.ndarray], np.ndarray
    ] | None = None,
    fe_output_feasible: Callable[[np.ndarray], np.ndarray] | None = None,
    fe_budget: int | None = None,
    experiment_logger: ADSEExperimentLogger | None = None,
    outer_iteration: int = 0,
    model_id_prefix: str = "de_mmoe",
    cumulative_fe_start: int = 0,
    failed_fe_x: np.ndarray | None = None,
) -> GateRefinementResult:
    """Repeat optimize, validate, acquire and retrain until the gate passes.

    Every surrogate update is followed by a new NSGA-II optimization and a
    newly planned three-point check.  Gate points remain permanently held out
    from training.  FE feasibility is logged for diagnosis but does not
    determine the objective-accuracy gate.
    """

    train_x = finite_matrix(training_x, INPUT_DIM, "training_x").copy()
    train_y = np.asarray(training_y, dtype=np.float64).copy()
    if (
        train_y.ndim != 2
        or len(train_y) != len(train_x)
        or train_y.shape[1] < OBJECTIVE_COUNT
        or not np.isfinite(train_y).all()
    ):
        raise ValueError("training_y must have shape (N, at least 2).")
    width = finite_vector(reference_width, INPUT_DIM, "reference_width")
    if np.any(width <= 0.0):
        raise ValueError("reference_width must contain 16 positive values.")
    if uncertainty_batch_size <= 0:
        raise ValueError("uncertainty_batch_size must be positive.")
    if strict_duplicate_distance < 0.0:
        raise ValueError("strict_duplicate_distance must be nonnegative.")
    if uncertainty_diversity_distance < strict_duplicate_distance:
        raise ValueError(
            "uncertainty_diversity_distance cannot be below the strict tolerance."
        )
    if not 0.0 <= uncertainty_weight_floor < 0.5:
        raise ValueError("uncertainty_weight_floor must lie in [0, 0.5).")
    if acquisition_mode not in {"uncertainty", "random"}:
        raise ValueError(
            "acquisition_mode must be either 'uncertainty' or 'random'."
        )
    if int(acquisition_seed) < 0:
        raise ValueError("acquisition_seed must be nonnegative.")
    if fe_budget is not None and fe_budget <= 0:
        raise ValueError("fe_budget must be positive when provided.")
    if outer_iteration < 0 or cumulative_fe_start < 0:
        raise ValueError("Iteration and cumulative FE counts must be nonnegative.")

    output_count = train_y.shape[1]
    validation = previous_validation
    if (heldout_validation_x is None) != (heldout_validation_y is None):
        raise ValueError(
            "heldout_validation_x and heldout_validation_y must be provided together."
        )
    if heldout_validation_x is None:
        validation_archive_x = np.empty((0, INPUT_DIM), dtype=np.float64)
        validation_archive_y = np.empty((0, output_count), dtype=np.float64)
    else:
        archive_x = np.asarray(heldout_validation_x, dtype=np.float64)
        archive_y = np.asarray(heldout_validation_y, dtype=np.float64)
        if archive_x.size == 0:
            if archive_x.shape != (0, INPUT_DIM) or archive_y.shape != (
                0, output_count
            ):
                raise ValueError(
                    "Empty held-out validation arrays must have shapes "
                    f"(0, {INPUT_DIM}) and (0, {output_count})."
                )
            validation_archive_x = archive_x.copy()
            validation_archive_y = archive_y.copy()
        else:
            validation_archive_x = finite_matrix(
                archive_x, INPUT_DIM, "heldout_validation_x"
            ).copy()
            validation_archive_y = archive_y.copy()
            if validation_archive_y.shape != (
                len(validation_archive_x), output_count
            ):
                raise ValueError(
                    "heldout_validation_y must match heldout_validation_x "
                    "and outputs."
                )
    if validation is not None:
        for point, output in zip(validation.x, validation.y_fe):
            if _exact_row_index(validation_archive_x, point) is None:
                validation_archive_x = np.vstack(
                    (validation_archive_x, point.reshape(1, -1))
                )
                validation_archive_y = np.vstack(
                    (validation_archive_y, output.reshape(1, -1))
                )
    for point in validation_archive_x:
        if _exact_row_index(train_x, point) is not None:
            raise ValueError(
                "A held-out validation point is already present in training_x."
            )
    fe_count = 0
    failed_fe_count = 0
    failed_archive = np.empty((0, INPUT_DIM), dtype=np.float64)
    if failed_fe_x is not None and np.asarray(failed_fe_x).size:
        failed_archive = finite_matrix(
            failed_fe_x, INPUT_DIM, "failed_fe_x"
        ).copy()
    cycle = 0
    history_rows: list[dict[str, float | int | bool | str]] = []

    def run_fe(points: np.ndarray) -> np.ndarray:
        nonlocal fe_count
        requested = len(points)
        if fe_budget is not None and fe_count + requested > fe_budget:
            raise FEBudgetExceeded(
                f"The accuracy gate requested {requested} additional FE runs "
                f"after {fe_count}/{fe_budget} had been used."
            )
        if requested == 0:
            return np.empty((0, output_count), dtype=np.float64)
        values = np.asarray(evaluate_fe(points), dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        expected_shape = (requested, output_count)
        if values.shape != expected_shape:
            raise ValueError(
                "evaluate_fe returned an unexpected shape: "
                f"{values.shape}, expected {expected_shape}."
            )
        if not np.isfinite(values).all():
            raise ValueError("evaluate_fe returned NaN or infinite values.")
        fe_count += requested
        return values

    def cached_validation(point: np.ndarray) -> np.ndarray | None:
        # Once a point enters the fitting archive it can no longer serve as an
        # independent online validation result for a later model.
        if _exact_row_index(train_x, point) is not None:
            return None
        index = _exact_row_index(validation_archive_x, point)
        if index is None:
            return None
        return validation_archive_y[index].copy()

    def remember_validation(point: np.ndarray, output: np.ndarray) -> None:
        nonlocal validation_archive_x, validation_archive_y
        if _exact_row_index(validation_archive_x, point) is not None:
            return
        validation_archive_x = np.vstack(
            (validation_archive_x, point.reshape(1, -1))
        )
        validation_archive_y = np.vstack(
            (validation_archive_y, output.reshape(1, -1))
        )

    def register_failure(
        error: FEEvaluationFailure,
        *,
        stage: str,
        role: str,
        inner_iteration: int,
        model_id: str,
    ) -> None:
        nonlocal failed_archive, failed_fe_count
        point = np.asarray(error.point, dtype=np.float64).reshape(1, -1)
        if point.shape[1] != INPUT_DIM:
            raise ValueError("A failed FE point does not contain 16 variables.")
        if len(failed_archive) == 0 or not np.any(
            np.all(
                np.isclose(failed_archive, point[0], rtol=0.0, atol=1e-12),
                axis=1,
            )
        ):
            failed_archive = np.vstack((failed_archive, point))
        failed_fe_count += 1
        if experiment_logger is not None:
            experiment_logger.log_fe_failure(
                outer_iter=outer_iteration,
                inner_iter=inner_iteration,
                model_id=model_id,
                stage=stage,
                role=role,
                x=point,
                message=str(error),
                cumulative_fe_count=cumulative_fe_start + fe_count,
            )

    pareto_x: np.ndarray | None = None
    pareto_objectives: np.ndarray | None = None
    pf_revision = -1
    pf_optimizations = 0
    model_updates = 0

    while True:
        cycle += 1
        inner_iteration = cycle - 1
        model_id = (
            f"{model_id_prefix}_outer_{outer_iteration:03d}_"
            f"inner_{inner_iteration:03d}"
        )

        generated_x, generated_objectives = optimize(surrogate)
        pareto_x = finite_matrix(generated_x, INPUT_DIM, "pareto_x")
        pareto_objectives = finite_matrix(
            generated_objectives, OBJECTIVE_COUNT, "pareto_objectives"
        )
        if len(pareto_x) != len(pareto_objectives):
            raise ValueError("Pareto design and objective row counts differ.")
        pf_revision += 1
        pf_optimizations += 1
        if experiment_logger is not None:
            experiment_logger.log_pareto_front(
                outer_iter=outer_iteration,
                inner_iter=inner_iteration,
                model_id=model_id,
                pf_revision=pf_revision,
                x=pareto_x,
                objectives=pareto_objectives,
                cumulative_fe_count=cumulative_fe_start + fe_count,
                training_size=len(train_x),
            )

        predict_objectives = lambda values: surrogate.predict(
            values
        ).mean_physical[:, :OBJECTIVE_COUNT]
        initial_plan = plan_accuracy_check(
            pareto_x,
            pareto_objectives,
            predict_objectives,
            width,
            previous_validation=validation,
            representativeness_threshold=representativeness_threshold,
        )
        check_x = initial_plan.check_x.copy()
        prediction_before_fe = initial_plan.prediction_before_fe.copy()
        representative_distance = initial_plan.representative_distance.copy()
        representative_source = initial_plan.representatives.source_indices.copy()
        representative_x = initial_plan.representatives.x.copy()
        representative_objectives = initial_plan.representatives.objectives.copy()
        report_reused = initial_plan.reused.copy()
        report_requires_fe = initial_plan.requires_fe.copy()
        statuses = np.where(report_reused, "regional_reused", "new").tolist()
        new_gate_fe_count = 0
        replacement_count = np.zeros(3, dtype=int)

        resolved_y: list[np.ndarray | None] = [None, None, None]
        if initial_plan.previous_validation is not None:
            for index in np.flatnonzero(initial_plan.reused):
                resolved_y[int(index)] = (
                    initial_plan.previous_validation.y_fe[int(index)].copy()
                )

        for representative_index in np.flatnonzero(initial_plan.requires_fe):
            role = initial_plan.representatives.labels[representative_index]
            while True:
                point = check_x[representative_index]
                already_failed = len(failed_archive) and np.any(
                    normalized_rms_distance(
                        point.reshape(1, -1), failed_archive, width
                    ) < strict_duplicate_distance
                )
                already_trained = _exact_row_index(train_x, point) is not None
                if already_failed or already_trained:
                    other_points = np.delete(check_x, representative_index, axis=0)
                    exclusion_parts = [train_x, other_points]
                    if len(failed_archive):
                        exclusion_parts.append(failed_archive)
                    source_index = _next_role_candidate(
                        pareto_x,
                        pareto_objectives,
                        role,
                        np.vstack(exclusion_parts),
                        width,
                        strict_duplicate_distance,
                    )
                    check_x[representative_index] = pareto_x[source_index]
                    representative_source[representative_index] = source_index
                    representative_x[representative_index] = pareto_x[source_index]
                    representative_objectives[representative_index] = (
                        pareto_objectives[source_index]
                    )
                    prediction_before_fe[representative_index] = (
                        predict_objectives(pareto_x[[source_index]])[0]
                    )
                    if validation is None:
                        representative_distance[representative_index] = np.inf
                    else:
                        representative_distance[representative_index] = (
                            normalized_rms_distance(
                                pareto_x[[source_index]],
                                validation.x[[representative_index]],
                                width,
                            )[0, 0]
                        )
                    replacement_count[representative_index] += 1
                    statuses[representative_index] = (
                        "failed_replaced" if already_failed else "trained_replaced"
                    )
                    continue

                archived = cached_validation(point)
                if archived is not None:
                    resolved_y[representative_index] = archived
                    statuses[representative_index] = (
                        "replacement_archive_reused"
                        if replacement_count[representative_index]
                        else "archive_reused"
                    )
                    break
                try:
                    result = run_fe(point.reshape(1, -1))[0]
                    resolved_y[representative_index] = result
                    remember_validation(point, result)
                    new_gate_fe_count += 1
                    break
                except FEEvaluationFailure as error:
                    register_failure(
                        error,
                        stage="gate",
                        role=role,
                        inner_iteration=inner_iteration,
                        model_id=model_id,
                    )

        representatives = ParetoRepresentatives(
            labels=initial_plan.representatives.labels,
            source_indices=representative_source,
            x=representative_x,
            objectives=representative_objectives,
        )
        if any(value is None for value in resolved_y):
            raise RuntimeError("A gate representative has no FE response.")
        resolved_validation = OnlineValidationSet(
            labels=representatives.labels,
            x=check_x.copy(),
            y_fe=np.vstack(resolved_y),
        )
        plan = ValidationPlan(
            representatives=representatives,
            check_x=check_x,
            prediction_before_fe=prediction_before_fe,
            representative_distance=representative_distance,
            reused=np.ones(3, dtype=bool),
            requires_fe=np.zeros(3, dtype=bool),
            previous_validation=resolved_validation,
        )
        feasibility_mask = (
            None
            if fe_output_feasible is None
            else np.asarray(
                fe_output_feasible(resolved_validation.y_fe), dtype=bool
            ).reshape(-1)
        )
        check = complete_accuracy_check(
            plan,
            np.empty((0, output_count)),
            relative_error_limit=relative_error_limit,
            feasibility_mask=feasibility_mask,
        )
        validation = check.validation_set

        weights = objective_error_weights(
            check.relative_error,
            minimum_weight=uncertainty_weight_floor,
        )
        if experiment_logger is not None:
            experiment_logger.log_gate_check(
                outer_iter=outer_iteration,
                inner_iter=inner_iteration,
                model_id=model_id,
                pf_revision=pf_revision,
                model_updated_since_pf=False,
                objective_weights=weights,
                labels=plan.representatives.labels,
                x=plan.check_x,
                prediction=plan.prediction_before_fe,
                fe_values=check.validation_set.y_fe,
                relative_error=check.relative_error,
                fe_feasible=check.fe_feasible,
                distances=plan.representative_distance,
                statuses=statuses,
                gate_passed=check.passed,
                new_fe_count=new_gate_fe_count,
                cumulative_fe_count=cumulative_fe_start + fe_count,
            )

        mean_error = check.relative_error.mean(axis=0)
        row: dict[str, float | int | bool | str] = {
            "outer_iter": outer_iteration,
            "inner_iter": inner_iteration,
            "model_id": model_id,
            "cycle": cycle,
            "pf_revision": pf_revision,
            "model_updated_since_pf": False,
            "fe_evaluations": fe_count,
            "failed_fe_evaluations": failed_fe_count,
            "reused_representatives": int(np.sum(report_reused)),
            "new_representatives": int(np.sum(report_requires_fe)),
            "mean_error_volume": float(mean_error[0]),
            "mean_error_stress": float(mean_error[1]),
            "rms_error_volume": float(check.objective_rms_relative_error[0]),
            "rms_error_stress": float(check.objective_rms_relative_error[1]),
            "gate_error": check.gate_error,
            "all_fe_feasible": check.all_feasible,
            "uncertainty_weight_volume": float(weights[0]),
            "uncertainty_weight_stress": float(weights[1]),
            "worst_relative_error": check.worst_relative_error,
            "passed": check.passed,
            "uncertainty_selected": 0,
        }
        for index, label in enumerate(plan.representatives.labels):
            row[f"distance_{label}"] = float(
                plan.representative_distance[index]
            )
            row[f"error_{label}_volume"] = float(
                check.relative_error[index, 0]
            )
            row[f"error_{label}_stress"] = float(
                check.relative_error[index, 1]
            )
        history_rows.append(row)

        if check.passed and accuracy_trigger_enabled:
            next_x = np.empty((0, INPUT_DIM), dtype=np.float64)
            next_y = np.empty((0, output_count), dtype=np.float64)
            if experiment_logger is not None:
                experiment_logger.log_iteration_summary(
                    outer_iter=outer_iteration,
                    model_id=model_id,
                    refinement_cycles=cycle,
                    pf_optimizations=pf_optimizations,
                    model_updates=model_updates,
                    added_fe_count=fe_count,
                    cumulative_fe_count=cumulative_fe_start + fe_count,
                    training_size=len(train_x),
                    gate_error=check.gate_error,
                    objective_rms_error=check.objective_rms_relative_error,
                    worst_error=check.worst_relative_error,
                    objective_weights=weights,
                    failed_fe_count=failed_fe_count,
                )
            return GateRefinementResult(
                surrogate=surrogate,
                training_x=train_x,
                training_y=train_y,
                validation=validation,
                pareto_x=pareto_x,
                pareto_objectives=pareto_objectives,
                fe_evaluations=fe_count,
                failed_fe_evaluations=failed_fe_count,
                refinement_cycles=cycle,
                pf_optimizations=pf_optimizations,
                model_updates=model_updates,
                heldout_validation_x=validation_archive_x.copy(),
                heldout_validation_y=validation_archive_y.copy(),
                points_for_next_training_x=next_x,
                points_for_next_training_y=next_y,
                history=pd.DataFrame(history_rows),
            )

        prediction = surrogate.predict(pareto_x)
        if candidate_feasible is None:
            feasible_mask = np.ones(len(pareto_x), dtype=bool)
        else:
            feasible_mask = np.asarray(
                candidate_feasible(surrogate, pareto_x), dtype=bool
            ).reshape(-1)
            if len(feasible_mask) != len(pareto_x):
                raise ValueError(
                    "candidate_feasible must return one value per PF candidate."
                )

        evaluated_archive = train_x
        target_count = uncertainty_batch_size
        if fe_budget is not None:
            target_count = min(target_count, fe_budget - fe_count)
        if target_count <= 0:
            raise FEBudgetExceeded(
                "The dynamic accuracy gate failed after the FE budget was used."
            )

        selected_indices: list[int] = []
        selected_x: list[np.ndarray] = []
        selected_std: list[np.ndarray] = []
        selected_scores: list[float] = []
        selected_fallback: list[bool] = []
        selected_distance: list[float] = []
        selected_batch_distance: list[float] = []
        selected_y: list[np.ndarray] = []
        selection_priorities = None
        if acquisition_mode == "random":
            # A deterministic seed per outer/inner cycle makes the random-AL
            # ablation reproducible without changing the candidate pool,
            # feasibility filter, archive exclusion or distance rules.
            random_generator = np.random.default_rng(
                int(acquisition_seed)
                + 1_000_003 * int(outer_iteration)
                + 1_009 * int(inner_iteration)
            )
            selection_priorities = random_generator.random(len(pareto_x))
        while len(selected_x) < target_count:
            exclusion_parts = [evaluated_archive, validation_archive_x]
            if len(failed_archive):
                exclusion_parts.append(failed_archive)
            strict_archive = np.vstack(exclusion_parts)
            batch_x = (
                np.vstack(selected_x)
                if selected_x
                else np.empty((0, INPUT_DIM), dtype=np.float64)
            )
            selection = select_uncertainty_points(
                pareto_x,
                prediction.model_std_standardized[:, :OBJECTIVE_COUNT],
                strict_archive,
                width,
                batch_selected_x=batch_x,
                eligible_mask=feasible_mask,
                objective_weights=weights,
                selection_priorities=selection_priorities,
                count=1,
                strict_duplicate_distance=strict_duplicate_distance,
                batch_diversity_distance=uncertainty_diversity_distance,
            )
            if len(selection.indices) == 0:
                break
            candidate_index = int(selection.indices[0])
            candidate_x = selection.x[[0]]
            try:
                candidate_y = run_fe(candidate_x)[0]
            except FEEvaluationFailure as error:
                register_failure(
                    error,
                    stage=(
                        "uncertainty"
                        if acquisition_mode == "uncertainty"
                        else "random_active_learning"
                    ),
                    role=(
                        "weighted_joint_uncertainty"
                        if acquisition_mode == "uncertainty"
                        else "random_pf_candidate"
                    ),
                    inner_iteration=inner_iteration,
                    model_id=model_id,
                )
                continue
            selected_indices.append(candidate_index)
            selected_x.append(candidate_x[0].copy())
            selected_std.append(selection.objective_model_std[0].copy())
            selected_scores.append(float(selection.scores[0]))
            selected_fallback.append(bool(selection.diversity_fallback[0]))
            selected_distance.append(float(selection.minimum_archive_distance[0]))
            selected_batch_distance.append(
                float(selection.minimum_batch_distance[0])
            )
            selected_y.append(candidate_y.copy())

        if not selected_x:
            raise CandidatePoolExhausted(
                "The current Pareto front contains no unevaluated, currently "
                "feasible point for uncertainty refinement."
            )

        row["uncertainty_selected"] = len(selected_x)
        uncertainty_indices = np.asarray(selected_indices, dtype=np.int64)
        uncertainty_x = np.vstack(selected_x)
        uncertainty_std = np.vstack(selected_std)
        uncertainty_scores = np.asarray(selected_scores, dtype=np.float64)
        uncertainty_fallback = np.asarray(selected_fallback, dtype=bool)
        uncertainty_distance = np.asarray(selected_distance, dtype=np.float64)
        uncertainty_batch_distance = np.asarray(
            selected_batch_distance, dtype=np.float64
        )
        added_y = np.vstack(selected_y)
        if experiment_logger is not None:
            experiment_logger.log_uncertainty_samples(
                outer_iter=outer_iteration,
                inner_iter=inner_iteration,
                model_id=model_id,
                pf_revision=pf_revision,
                objective_weights=weights,
                scores=uncertainty_scores,
                diversity_fallback=uncertainty_fallback,
                x=uncertainty_x,
                objective_model_std=uncertainty_std,
                prediction=prediction.mean_physical[
                    uncertainty_indices, :OBJECTIVE_COUNT
                ],
                archive_distance=uncertainty_distance,
                batch_distance=uncertainty_batch_distance,
                fe_values=added_y,
                cumulative_fe_count=cumulative_fe_start + fe_count,
            )
        train_x, train_y = _append_unique(
            train_x, train_y, uncertainty_x, added_y
        )
        surrogate = retrain(train_x, train_y)
        model_updates += 1

        if not accuracy_trigger_enabled:
            # M4 ablation: the three-point check is diagnostic only.  Every
            # outer iteration receives one AL batch even when the diagnostic
            # gate already passes; after one retraining, NSGA-II is rerun once
            # and that updated PF is used by the outer A-DSE update.
            post_model_id = (
                f"{model_id_prefix}_outer_{outer_iteration:03d}_post_update"
            )
            generated_x, generated_objectives = optimize(surrogate)
            pareto_x = finite_matrix(generated_x, INPUT_DIM, "pareto_x")
            pareto_objectives = finite_matrix(
                generated_objectives,
                OBJECTIVE_COUNT,
                "pareto_objectives",
            )
            if len(pareto_x) != len(pareto_objectives):
                raise ValueError("Pareto design and objective row counts differ.")
            pf_revision += 1
            pf_optimizations += 1
            if experiment_logger is not None:
                experiment_logger.log_pareto_front(
                    outer_iter=outer_iteration,
                    inner_iter=inner_iteration + 1,
                    model_id=post_model_id,
                    pf_revision=pf_revision,
                    x=pareto_x,
                    objectives=pareto_objectives,
                    cumulative_fe_count=cumulative_fe_start + fe_count,
                    training_size=len(train_x),
                )
                experiment_logger.log_iteration_summary(
                    outer_iter=outer_iteration,
                    model_id=post_model_id,
                    refinement_cycles=cycle,
                    pf_optimizations=pf_optimizations,
                    model_updates=model_updates,
                    added_fe_count=fe_count,
                    cumulative_fe_count=cumulative_fe_start + fe_count,
                    training_size=len(train_x),
                    gate_error=check.gate_error,
                    objective_rms_error=check.objective_rms_relative_error,
                    worst_error=check.worst_relative_error,
                    objective_weights=weights,
                    failed_fe_count=failed_fe_count,
                )
            return GateRefinementResult(
                surrogate=surrogate,
                training_x=train_x,
                training_y=train_y,
                validation=validation,
                pareto_x=pareto_x,
                pareto_objectives=pareto_objectives,
                fe_evaluations=fe_count,
                failed_fe_evaluations=failed_fe_count,
                refinement_cycles=cycle,
                pf_optimizations=pf_optimizations,
                model_updates=model_updates,
                heldout_validation_x=validation_archive_x.copy(),
                heldout_validation_y=validation_archive_y.copy(),
                points_for_next_training_x=np.empty(
                    (0, INPUT_DIM), dtype=np.float64
                ),
                points_for_next_training_y=np.empty(
                    (0, output_count), dtype=np.float64
                ),
                history=pd.DataFrame(history_rows),
            )
