"""Reliability-gated adaptive design-space exploration."""

from ._common import FEEvaluationFailure
from .design_space_exploration import (
    AccuracyCheck,
    BoundsUpdate,
    OnlineValidationSet,
    ParetoRepresentatives,
    UncertaintySelection,
    ValidationPlan,
    complete_accuracy_check,
    importance_from_table,
    load_reference_bounds,
    normalized_rms_distance,
    objective_error_weights,
    plan_accuracy_check,
    rank_pareto_candidates_by_role,
    select_pareto_representatives,
    select_uncertainty_points,
    update_design_space,
)
from .ensemble_surrogate import (
    EnsembleMMoESurrogate,
    EnsemblePrediction,
    GradientShapResult,
    gradient_shap_objectives,
    save_gradient_shap_report,
)
from .experiment_log import ADSEExperimentLogger
from .workflow import (
    CandidatePoolExhausted,
    FEBudgetExceeded,
    GateRefinementResult,
    refine_pareto_until_accurate,
)
from .training import (
    EnsembleTrainingResult,
    save_training_result,
    train_five_member_from_template,
)

__all__ = [
    "AccuracyCheck",
    "ADSEExperimentLogger",
    "BoundsUpdate",
    "CandidatePoolExhausted",
    "EnsembleMMoESurrogate",
    "EnsemblePrediction",
    "EnsembleTrainingResult",
    "GradientShapResult",
    "FEBudgetExceeded",
    "FEEvaluationFailure",
    "GateRefinementResult",
    "OnlineValidationSet",
    "ParetoRepresentatives",
    "UncertaintySelection",
    "ValidationPlan",
    "complete_accuracy_check",
    "gradient_shap_objectives",
    "importance_from_table",
    "load_reference_bounds",
    "normalized_rms_distance",
    "objective_error_weights",
    "plan_accuracy_check",
    "rank_pareto_candidates_by_role",
    "refine_pareto_until_accurate",
    "save_gradient_shap_report",
    "save_training_result",
    "select_pareto_representatives",
    "select_uncertainty_points",
    "update_design_space",
    "train_five_member_from_template",
]
