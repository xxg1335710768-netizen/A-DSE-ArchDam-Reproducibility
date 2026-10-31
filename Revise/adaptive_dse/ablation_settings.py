"""Public, FE-independent definitions of the Full and M1--M4 experiments."""

COMMON = {
    "initial_sample_count": 80,
    "outer_iterations": 30,
    "seed": 2025,
    "nsga2_population_size": 200,
    "nsga2_generations": 200,
    "uncertainty_batch_size": 3,
    "gate_relative_error_limit": 0.01,
    "representativeness_threshold": 0.01,
    "strict_duplicate_distance": 1e-8,
    "uncertainty_diversity_distance": 0.025,
    "uncertainty_weight_floor": 0.20,
}

FULL = {
    **COMMON,
    "method": "Full",
    "design_space_mode": "importance",
    "acquisition_mode": "uncertainty",
    "accuracy_trigger_enabled": True,
}

M1 = {
    **COMMON,
    "method": "M1",
    "design_space_mode": "fixed",
    "acquisition_mode": "uncertainty",
    "accuracy_trigger_enabled": True,
}

M2 = {
    **COMMON,
    "method": "M2",
    "design_space_mode": "uniform",
    "acquisition_mode": "uncertainty",
    "accuracy_trigger_enabled": True,
}

M3 = {
    **COMMON,
    "method": "M3",
    "design_space_mode": "importance",
    "acquisition_mode": "random",
    "accuracy_trigger_enabled": True,
    "acquisition_seed": 2025,
}

M4 = {
    **COMMON,
    "method": "M4",
    "design_space_mode": "importance",
    "acquisition_mode": "uncertainty",
    "accuracy_trigger_enabled": False,
}

ALL_METHODS = {item["method"]: item for item in (FULL, M1, M2, M3, M4)}
