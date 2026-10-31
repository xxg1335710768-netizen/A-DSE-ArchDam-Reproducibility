"""Core logic for reliability-gated adaptive design-space exploration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from ._common import (
    INPUT_DIM,
    OBJECTIVE_COUNT,
    finite_matrix as _matrix,
    finite_vector as _vector,
    maximum_normalize_rows,
)

REPRESENTATIVE_LABELS = (
    "volume_extreme",
    "balanced_middle",
    "stress_extreme",
)
REPRESENTATIVE_COUNT = len(REPRESENTATIVE_LABELS)


def load_reference_bounds(
    registry_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the initial sampling ranges used only as distance/width scales."""

    with Path(registry_path).open("r", encoding="utf-8") as stream:
        registry = json.load(stream)
    bounds = registry["design_bounds"]
    lower = _vector(bounds["lower"], INPUT_DIM, "lower bounds")
    upper = _vector(bounds["upper"], INPUT_DIM, "upper bounds")
    if np.any(upper <= lower):
        raise ValueError("Every initial upper bound must exceed its lower bound.")
    return lower, upper


def nondominated_mask(objectives: np.ndarray) -> np.ndarray:
    """Return the nondominated mask for a minimization problem."""

    values = _matrix(objectives, OBJECTIVE_COUNT, "objectives")
    mask = np.ones(len(values), dtype=bool)
    for index, row in enumerate(values):
        dominates = np.all(values <= row, axis=1) & np.any(values < row, axis=1)
        if np.any(dominates):
            mask[index] = False
    return mask


@dataclass(frozen=True)
class ParetoRepresentatives:
    labels: tuple[str, str, str]
    source_indices: np.ndarray
    x: np.ndarray
    objectives: np.ndarray


def rank_pareto_candidates_by_role(
    pareto_x: np.ndarray,
    pareto_objectives: np.ndarray,
    role: str,
) -> np.ndarray:
    """Rank original PF row indices for one representative role.

    The order is increasing volume for ``volume_extreme``, increasing stress
    objective for ``stress_extreme``, and increasing normalized distance from
    the ideal point for ``balanced_middle``. Stable sorting preserves the
    original PF order for exact ties.
    """

    if role not in REPRESENTATIVE_LABELS:
        raise ValueError(f"Unknown Pareto representative role: {role}")
    x = _matrix(pareto_x, INPUT_DIM, "pareto_x")
    objectives = _matrix(
        pareto_objectives, OBJECTIVE_COUNT, "pareto_objectives"
    )
    if len(x) != len(objectives):
        raise ValueError("Pareto design and objective row counts differ.")
    unique_index = np.sort(np.unique(x, axis=0, return_index=True)[1])
    unique_objectives = objectives[unique_index]
    nondominated = nondominated_mask(unique_objectives)
    source = unique_index[nondominated]
    front_y = unique_objectives[nondominated]
    if len(source) < 3:
        raise ValueError("At least three unique nondominated points are required.")
    if role == "volume_extreme":
        score = front_y[:, 0]
    elif role == "stress_extreme":
        score = front_y[:, 1]
    else:
        minimum = front_y.min(axis=0)
        span = front_y.max(axis=0) - minimum
        span = np.where(span > 1e-15, span, 1.0)
        score = np.linalg.norm((front_y - minimum) / span, axis=1)
    return source[np.argsort(score, kind="stable")]


def select_pareto_representatives(
    pareto_x: np.ndarray, pareto_objectives: np.ndarray
) -> ParetoRepresentatives:
    """Select both objective extremes and the ideal-point-nearest solution."""

    x = _matrix(pareto_x, INPUT_DIM, "pareto_x")
    objectives = _matrix(
        pareto_objectives, OBJECTIVE_COUNT, "pareto_objectives"
    )
    if len(x) != len(objectives):
        raise ValueError("Pareto design and objective row counts differ.")

    volume_ranking = rank_pareto_candidates_by_role(
        x, objectives, "volume_extreme"
    )
    stress_ranking = rank_pareto_candidates_by_role(
        x, objectives, "stress_extreme"
    )
    middle_ranking = rank_pareto_candidates_by_role(
        x, objectives, "balanced_middle"
    )
    volume_source = int(volume_ranking[0])
    stress_source = int(stress_ranking[0])
    if volume_source == stress_source:
        raise ValueError("The two objective extremes collapse to the same point.")
    middle_source = next(
        int(index)
        for index in middle_ranking
        if int(index) not in {volume_source, stress_source}
    )
    selected = np.asarray(
        [volume_source, middle_source, stress_source], dtype=np.int64
    )
    return ParetoRepresentatives(
        labels=REPRESENTATIVE_LABELS,
        source_indices=selected,
        x=x[selected],
        objectives=objectives[selected],
    )


def normalized_rms_distance(
    query: np.ndarray, reference: np.ndarray, reference_width: np.ndarray
) -> np.ndarray:
    """Pairwise normalized root-mean-square design-parameter distance."""

    query_array = _matrix(query, INPUT_DIM, "query")
    reference_array = _matrix(reference, INPUT_DIM, "reference")
    width = _vector(reference_width, INPUT_DIM, "reference_width")
    if np.any(width <= 0.0):
        raise ValueError("reference_width must be positive.")
    difference = (
        query_array[:, None, :] - reference_array[None, :, :]
    ) / width.reshape(1, 1, -1)
    return np.sqrt(np.mean(difference**2, axis=2))


@dataclass(frozen=True)
class OnlineValidationSet:
    labels: tuple[str, str, str]
    x: np.ndarray
    y_fe: np.ndarray

    def __post_init__(self) -> None:
        points = _matrix(self.x, INPUT_DIM, "validation x")
        outputs = np.asarray(self.y_fe, dtype=np.float64)
        if (
            outputs.ndim != 2
            or outputs.shape[0] != REPRESENTATIVE_COUNT
            or outputs.shape[1] < OBJECTIVE_COUNT
        ):
            raise ValueError(
                "validation y_fe must have shape (3, at least 2)."
            )
        if not np.isfinite(outputs).all():
            raise ValueError("validation y_fe contains NaN or infinite values.")
        if len(points) != REPRESENTATIVE_COUNT:
            raise ValueError("The online validation set must contain three points.")
        if tuple(self.labels) != REPRESENTATIVE_LABELS:
            raise ValueError("Unexpected validation-point labels.")


@dataclass(frozen=True)
class ValidationPlan:
    representatives: ParetoRepresentatives
    check_x: np.ndarray
    prediction_before_fe: np.ndarray
    representative_distance: np.ndarray
    reused: np.ndarray
    requires_fe: np.ndarray
    previous_validation: OnlineValidationSet | None


def plan_accuracy_check(
    pareto_x: np.ndarray,
    pareto_objectives: np.ndarray,
    predict_objectives: Callable[[np.ndarray], np.ndarray],
    reference_width: np.ndarray,
    *,
    previous_validation: OnlineValidationSet | None = None,
    representativeness_threshold: float,
) -> ValidationPlan:
    """Reuse only representative old FE points; request FE for changed regions."""

    if representativeness_threshold < 0.0:
        raise ValueError("representativeness_threshold must be nonnegative.")
    representatives = select_pareto_representatives(
        pareto_x, pareto_objectives
    )
    if previous_validation is None:
        distances = np.full(REPRESENTATIVE_COUNT, np.inf, dtype=np.float64)
        reused = np.zeros(REPRESENTATIVE_COUNT, dtype=bool)
        check_x = representatives.x.copy()
    else:
        pairwise = normalized_rms_distance(
            representatives.x, previous_validation.x, reference_width
        )
        distances = np.diag(pairwise)
        reused = distances <= representativeness_threshold
        check_x = np.where(
            reused[:, None], previous_validation.x, representatives.x
        )

    prediction = np.asarray(predict_objectives(check_x), dtype=np.float64)
    if prediction.ndim == 1:
        prediction = prediction.reshape(1, -1)
    if (
        prediction.shape[0] != REPRESENTATIVE_COUNT
        or prediction.shape[1] < OBJECTIVE_COUNT
    ):
        raise ValueError("predict_objectives must return at least two outputs.")
    prediction = prediction[:, :OBJECTIVE_COUNT]
    if not np.isfinite(prediction).all():
        raise ValueError("Objective predictions contain NaN or infinite values.")

    return ValidationPlan(
        representatives=representatives,
        check_x=check_x,
        prediction_before_fe=prediction,
        representative_distance=distances,
        reused=reused,
        requires_fe=~reused,
        previous_validation=previous_validation,
    )


@dataclass(frozen=True)
class AccuracyCheck:
    validation_set: OnlineValidationSet
    relative_error: np.ndarray
    objective_rms_relative_error: np.ndarray
    gate_error: float
    worst_relative_error: float
    fe_feasible: np.ndarray
    all_feasible: bool
    passed: bool


def complete_accuracy_check(
    plan: ValidationPlan,
    new_fe_outputs: np.ndarray,
    *,
    relative_error_limit: float = 0.01,
    denominator_floor: float = 1e-12,
    feasibility_mask: np.ndarray | None = None,
) -> AccuracyCheck:
    """Evaluate the three-point, two-objective relative-RMSE gate.

    The two objective-wise statistics are the root mean squares of the three
    representative-point relative errors.  The gate statistic is their
    maximum.  The largest of the six individual errors is retained only as a
    diagnostic and does not determine whether the gate passes.
    """

    if relative_error_limit <= 0.0:
        raise ValueError("relative_error_limit must be positive.")
    needed = int(np.sum(plan.requires_fe))
    if needed:
        new_values = np.asarray(new_fe_outputs, dtype=np.float64)
        if new_values.ndim == 1:
            new_values = new_values.reshape(1, -1)
        if new_values.ndim != 2 or new_values.shape[1] < OBJECTIVE_COUNT:
            raise ValueError("new_fe_outputs must have at least two columns.")
        if not np.isfinite(new_values).all():
            raise ValueError("new_fe_outputs contains NaN or infinite values.")
        if len(new_values) != needed:
            raise ValueError(f"Expected {needed} new FE results, got {len(new_values)}.")
    else:
        if plan.previous_validation is None:
            raise RuntimeError("No FE outputs are available for the accuracy check.")
        new_values = np.empty(
            (0, plan.previous_validation.y_fe.shape[1]), dtype=np.float64
        )

    previous = plan.previous_validation
    if previous is None:
        if np.any(plan.reused):
            raise RuntimeError("A reused point has no previous FE result.")
        y_fe = new_values.copy()
    else:
        if needed and new_values.shape[1] != previous.y_fe.shape[1]:
            raise ValueError("Old and new FE output counts differ.")
        y_fe = previous.y_fe.copy()
        y_fe[plan.requires_fe] = new_values

    objective_fe = y_fe[:, :OBJECTIVE_COUNT]
    denominator = np.maximum(np.abs(objective_fe), denominator_floor)
    relative_error = (
        np.abs(objective_fe - plan.prediction_before_fe) / denominator
    )
    objective_rms_error = np.sqrt(np.mean(relative_error**2, axis=0))
    gate_error = float(np.max(objective_rms_error))
    worst = float(np.max(relative_error))
    if feasibility_mask is None:
        feasible = np.ones(REPRESENTATIVE_COUNT, dtype=bool)
    else:
        feasible = np.asarray(feasibility_mask, dtype=bool).reshape(-1)
        if len(feasible) != REPRESENTATIVE_COUNT:
            raise ValueError("feasibility_mask must contain three values.")
    all_feasible = bool(np.all(feasible))
    validation_set = OnlineValidationSet(
        labels=REPRESENTATIVE_LABELS,
        x=plan.check_x.copy(),
        y_fe=y_fe,
    )
    return AccuracyCheck(
        validation_set=validation_set,
        relative_error=relative_error,
        objective_rms_relative_error=objective_rms_error,
        gate_error=gate_error,
        worst_relative_error=worst,
        fe_feasible=feasible.copy(),
        all_feasible=all_feasible,
        # FE feasibility is retained only as a diagnostic of the surrogate's
        # constraint classification.  A successful FE result remains valid
        # training data regardless of whether that design satisfies the
        # engineering constraints, and feasibility does not enter the
        # objective-accuracy gate.
        passed=bool(gate_error < relative_error_limit),
    )


@dataclass(frozen=True)
class UncertaintySelection:
    indices: np.ndarray
    x: np.ndarray
    objective_model_std: np.ndarray
    minimum_archive_distance: np.ndarray
    minimum_batch_distance: np.ndarray
    scores: np.ndarray
    objective_weights: np.ndarray
    diversity_fallback: np.ndarray


def objective_error_weights(
    relative_error: np.ndarray,
    *,
    minimum_weight: float = 0.20,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Convert the three-point mean objective errors to two bounded weights."""

    errors = np.asarray(relative_error, dtype=np.float64)
    if errors.ndim != 2 or errors.shape[1] != OBJECTIVE_COUNT:
        raise ValueError("relative_error must have shape (N, 2).")
    if len(errors) == 0 or np.any(errors < 0.0) or not np.isfinite(errors).all():
        raise ValueError("relative_error must be nonempty, finite and nonnegative.")
    if not 0.0 <= minimum_weight < 0.5:
        raise ValueError("minimum_weight must lie in [0, 0.5).")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")

    mean_error = errors.mean(axis=0)
    error_share = (mean_error + epsilon) / (
        mean_error.sum() + OBJECTIVE_COUNT * epsilon
    )
    weights = minimum_weight + (1.0 - 2.0 * minimum_weight) * error_share
    return weights / weights.sum()


def select_uncertainty_points(
    pareto_x: np.ndarray,
    objective_model_std: np.ndarray,
    training_x: np.ndarray,
    reference_width: np.ndarray,
    *,
    validation_x: np.ndarray | None = None,
    batch_selected_x: np.ndarray | None = None,
    eligible_mask: np.ndarray | None = None,
    objective_weights: np.ndarray | None = None,
    selection_priorities: np.ndarray | None = None,
    count: int = 2,
    strict_duplicate_distance: float = 1e-8,
    batch_diversity_distance: float = 0.025,
) -> UncertaintySelection:
    """Select weighted-disagreement points with strict and batch distances.

    ``strict_duplicate_distance`` applies only to the permanent FE/validation
    archive.  ``batch_diversity_distance`` separates points selected in the
    current batch and is relaxed directly to the strict tolerance only when
    the soft diversity condition leaves no eligible candidate.
    """

    x = _matrix(pareto_x, INPUT_DIM, "pareto_x")
    std = np.asarray(objective_model_std, dtype=np.float64)
    if (
        std.ndim != 2
        or std.shape[0] != len(x)
        or std.shape[1] < OBJECTIVE_COUNT
    ):
        raise ValueError("objective_model_std must have shape (N, at least 2).")
    if np.any(std < 0.0) or not np.isfinite(std).all():
        raise ValueError("Model standard deviations must be finite and nonnegative.")
    archive = _matrix(training_x, INPUT_DIM, "training_x")
    if validation_x is not None:
        validation = _matrix(validation_x, INPUT_DIM, "validation_x")
        archive = np.vstack((archive, validation))
    if batch_selected_x is None or np.asarray(batch_selected_x).size == 0:
        batch = np.empty((0, INPUT_DIM), dtype=np.float64)
    else:
        batch = _matrix(batch_selected_x, INPUT_DIM, "batch_selected_x")
    if count <= 0:
        raise ValueError("count must be positive.")
    if strict_duplicate_distance < 0.0 or batch_diversity_distance < 0.0:
        raise ValueError("Distance thresholds must be nonnegative.")
    if eligible_mask is None:
        allowed = np.ones(len(x), dtype=bool)
    else:
        allowed = np.asarray(eligible_mask, dtype=bool).reshape(-1)
        if len(allowed) != len(x):
            raise ValueError("eligible_mask must contain one value per candidate.")
    if objective_weights is None:
        weights = np.full(OBJECTIVE_COUNT, 1.0 / OBJECTIVE_COUNT)
    else:
        weights = _vector(
            objective_weights, OBJECTIVE_COUNT, "objective_weights"
        )
        if np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("objective_weights must be nonnegative and sum to one.")

    scores = np.prod(std[:, :OBJECTIVE_COUNT] ** weights, axis=1)
    if selection_priorities is None:
        priorities = scores
    else:
        priorities = np.asarray(selection_priorities, dtype=np.float64).reshape(-1)
        if len(priorities) != len(x):
            raise ValueError(
                "selection_priorities must contain one value per candidate."
            )
        if not np.isfinite(priorities).all():
            raise ValueError("selection_priorities must be finite.")
    archive_distances = normalized_rms_distance(
        x, archive, reference_width
    ).min(axis=1)
    remaining = np.ones(len(x), dtype=bool)
    selected = []
    fallback_flags = []
    selected_batch_distances = []
    for _ in range(count):
        strict_eligible = (
            remaining
            & allowed
            & (archive_distances >= strict_duplicate_distance)
        )
        if not np.any(strict_eligible):
            break
        current_batch = batch
        if selected:
            selected_points = x[np.asarray(selected, dtype=np.int64)]
            current_batch = (
                np.vstack((current_batch, selected_points))
                if len(current_batch)
                else selected_points
            )
        used_fallback = False
        eligible = strict_eligible
        batch_distance = np.full(len(x), np.inf, dtype=np.float64)
        if len(current_batch):
            batch_distance = normalized_rms_distance(
                x, current_batch, reference_width
            ).min(axis=1)
            strict_eligible = strict_eligible & (
                batch_distance >= strict_duplicate_distance
            )
            if not np.any(strict_eligible):
                break
            eligible = strict_eligible
            diverse = strict_eligible & (
                batch_distance >= batch_diversity_distance
            )
            if np.any(diverse):
                eligible = diverse
            else:
                used_fallback = True
        candidates = np.flatnonzero(eligible)
        # The full method ranks by weighted geometric uncertainty.  A supplied
        # priority vector supports the random-AL ablation while retaining the
        # identical archive, feasibility and batch-distance rules.  Returned
        # scores always remain the actual joint uncertainty for diagnostics.
        local_order = np.lexsort(
            (-archive_distances[candidates], -priorities[candidates])
        )
        chosen = int(candidates[local_order[0]])
        selected.append(chosen)
        fallback_flags.append(used_fallback)
        selected_batch_distances.append(float(batch_distance[chosen]))
        remaining[chosen] = False

    indices = np.asarray(selected, dtype=np.int64)
    archive_distance = archive_distances[indices]
    return UncertaintySelection(
        indices=indices,
        x=x[indices],
        objective_model_std=std[indices, :OBJECTIVE_COUNT],
        minimum_archive_distance=archive_distance,
        minimum_batch_distance=np.asarray(
            selected_batch_distances, dtype=np.float64
        ),
        scores=scores[indices],
        objective_weights=weights.copy(),
        diversity_fallback=np.asarray(fallback_flags, dtype=bool),
    )


def importance_from_table(
    table: pd.DataFrame,
    *,
    feature_order: Sequence[str],
    target_columns: Sequence[str],
) -> pd.DataFrame:
    """Align by feature name, maximum-normalize per target, then average."""

    if "feature" not in table.columns:
        raise ValueError("The importance table has no 'feature' column.")
    if table["feature"].duplicated().any():
        raise ValueError("The importance table contains duplicate feature names.")
    order = tuple(feature_order)
    if len(order) != len(set(order)):
        raise ValueError("feature_order contains duplicates.")
    missing_features = sorted(set(order).difference(table["feature"]))
    missing_columns = sorted(set(target_columns).difference(table.columns))
    if missing_features or missing_columns:
        raise ValueError(
            f"Missing features={missing_features}, columns={missing_columns}."
        )

    aligned = table.set_index("feature").loc[list(order)].copy()
    raw = aligned.loc[:, list(target_columns)].to_numpy(dtype=np.float64).T
    if np.any(raw < 0.0) or not np.isfinite(raw).all():
        raise ValueError("Mean absolute attributions must be finite and nonnegative.")
    normalized = maximum_normalize_rows(raw)

    result = pd.DataFrame({"feature": order})
    for target_index, column in enumerate(target_columns):
        result[column] = raw[target_index]
        result[f"{column}_normalized"] = normalized[target_index]
    result["combined_importance"] = normalized.mean(axis=0)
    return result


@dataclass(frozen=True)
class BoundsUpdate:
    lower: np.ndarray
    upper: np.ndarray
    ema_pareto_min: np.ndarray
    ema_pareto_max: np.ndarray
    raw_pareto_min: np.ndarray
    raw_pareto_max: np.ndarray
    pareto_span: np.ndarray
    effective_span: np.ndarray
    expansion: np.ndarray
    importance: np.ndarray
    eta: float


def update_design_space(
    pareto_x: np.ndarray,
    importance: np.ndarray,
    initial_lower: np.ndarray,
    initial_upper: np.ndarray,
    previous_ema_min: np.ndarray,
    previous_ema_max: np.ndarray,
    *,
    iteration_index: int,
    total_iterations: int,
    gamma: float = 0.4,
    eta_start: float = 0.15,
    eta_end: float = 0.03,
    reference_span_ratio: float = 0.10,
    importance_multiplier: float = 1.0,
) -> BoundsUpdate:
    """Update the NSGA-II search window after the accuracy gate passes."""

    x = _matrix(pareto_x, INPUT_DIM, "pareto_x")
    weights = _vector(importance, INPUT_DIM, "importance")
    if np.any(weights < -1e-12) or np.any(weights > 1.0 + 1e-12):
        raise ValueError("importance must lie in [0, 1].")
    weights = np.clip(weights, 0.0, 1.0)
    lower_initial = _vector(initial_lower, INPUT_DIM, "initial_lower")
    upper_initial = _vector(initial_upper, INPUT_DIM, "initial_upper")
    initial_width = upper_initial - lower_initial
    if np.any(initial_width <= 0.0):
        raise ValueError("Initial design-space widths must be positive.")
    old_min = _vector(previous_ema_min, INPUT_DIM, "previous_ema_min")
    old_max = _vector(previous_ema_max, INPUT_DIM, "previous_ema_max")
    if np.any(old_max < old_min):
        raise ValueError("previous_ema_max is below previous_ema_min.")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1].")
    if total_iterations <= 0 or not 0 <= iteration_index <= total_iterations:
        raise ValueError("iteration_index must lie in [0, total_iterations].")
    if eta_start < 0.0 or eta_end < 0.0:
        raise ValueError("eta values must be nonnegative.")
    if reference_span_ratio < 0.0:
        raise ValueError("reference_span_ratio must be nonnegative.")
    if not 0.0 <= importance_multiplier <= 1.0:
        raise ValueError("importance_multiplier must lie in [0, 1].")

    raw_min = x.min(axis=0)
    raw_max = x.max(axis=0)
    ema_min = gamma * old_min + (1.0 - gamma) * raw_min
    ema_max = gamma * old_max + (1.0 - gamma) * raw_max
    span = np.maximum(ema_max - ema_min, 0.0)
    effective_span = np.maximum(span, reference_span_ratio * initial_width)
    progress = iteration_index / float(total_iterations)
    eta = eta_start - (eta_start - eta_end) * progress
    expansion = eta * effective_span * (
        1.0 + importance_multiplier * weights
    )

    return BoundsUpdate(
        lower=ema_min - expansion,
        upper=ema_max + expansion,
        ema_pareto_min=ema_min,
        ema_pareto_max=ema_max,
        raw_pareto_min=raw_min,
        raw_pareto_max=raw_max,
        pareto_span=span,
        effective_span=effective_span,
        expansion=expansion,
        importance=weights,
        eta=float(eta),
    )
