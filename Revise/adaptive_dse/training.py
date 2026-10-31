"""Retrain the fixed five-member MMoE architecture during A-DSE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch

from Revise.surrogate_benchmark.config import NEURAL_TRAINING
from Revise.surrogate_benchmark.data_protocol import ArrayStandardizer
from Revise.surrogate_benchmark.models_torch import (
    DeterministicMMoE,
    TorchTrainingConfig,
    fit_with_early_stopping,
)

from ._common import ENSEMBLE_SIZE, INPUT_DIM, finite_matrix
from .ensemble_surrogate import EnsembleMMoESurrogate


@dataclass(frozen=True)
class EnsembleTrainingResult:
    surrogate: EnsembleMMoESurrogate
    selected_config: dict[str, int]
    task_weights: np.ndarray
    validation_task_weights: np.ndarray
    member_best_epochs: tuple[int, ...]
    member_epochs_completed: tuple[int, ...]
    fit_seconds: float


def _load_payload(path: str | Path, device: str) -> dict:
    template_path = Path(path)
    if not template_path.is_file():
        raise FileNotFoundError(template_path)
    try:
        return torch.load(template_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(template_path, map_location=device)


def train_five_member_from_template(
    training_x: np.ndarray,
    training_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    template_model_path: str | Path,
    seed: int = 2025,
    device: str = "cpu",
    training_config: TorchTrainingConfig | None = None,
) -> EnsembleTrainingResult:
    """Retrain five members without repeating architecture search."""

    x_train = finite_matrix(training_x, INPUT_DIM, "training_x")
    y_train = np.asarray(training_y, dtype=np.float64)
    x_validation = finite_matrix(validation_x, INPUT_DIM, "validation_x")
    y_validation = np.asarray(validation_y, dtype=np.float64)
    if y_train.ndim != 2 or len(y_train) != len(x_train):
        raise ValueError("training_y has an invalid shape.")
    if y_validation.ndim != 2 or len(y_validation) != len(x_validation):
        raise ValueError("validation_y has an invalid shape.")
    if y_train.shape[1] != y_validation.shape[1]:
        raise ValueError("Training and validation output counts differ.")
    if not np.isfinite(y_train).all() or not np.isfinite(y_validation).all():
        raise ValueError("Training or validation data contain non-finite values.")

    template = _load_payload(template_model_path, device)
    required = {"selected_config", "task_weights", "validation_task_weights"}
    missing = sorted(required.difference(template))
    if missing:
        raise ValueError(f"Template payload misses fields: {missing}")
    selected_config = dict(template["selected_config"])
    task_weights = np.asarray(template["task_weights"], dtype=np.float64)
    validation_weights = np.asarray(
        template["validation_task_weights"], dtype=np.float64
    )
    if task_weights.size != y_train.shape[1]:
        raise ValueError("Template task weights do not match the outputs.")

    x_scaler = ArrayStandardizer.fit(x_train)
    y_scaler = ArrayStandardizer.fit(y_train)
    x_train_std = x_scaler.transform(x_train)
    y_train_std = y_scaler.transform(y_train)
    x_validation_std = x_scaler.transform(x_validation)
    y_validation_std = y_scaler.transform(y_validation)
    config = training_config or TorchTrainingConfig(**NEURAL_TRAINING)

    def factory() -> DeterministicMMoE:
        return DeterministicMMoE(
            d_in=INPUT_DIM,
            num_tasks=y_train.shape[1],
            activation_name="swish",
            **selected_config,
        )

    fitted_members = []
    start = time.perf_counter()
    for member_index in range(ENSEMBLE_SIZE):
        member_seed = seed if member_index == 0 else seed + 1000 + member_index
        fitted_members.append(
            fit_with_early_stopping(
                factory,
                x_train_std,
                y_train_std,
                x_validation_std,
                y_validation_std,
                probabilistic=False,
                config=config,
                seed=member_seed,
                device=device,
                selection_metric="rmse",
                task_weights=task_weights,
                validation_task_weights=validation_weights,
            )
        )
    fit_seconds = time.perf_counter() - start
    surrogate = EnsembleMMoESurrogate(
        [fitted.model for fitted in fitted_members],
        x_scaler.mean,
        x_scaler.scale,
        y_scaler.mean,
        y_scaler.scale,
        device=device,
    )
    return EnsembleTrainingResult(
        surrogate=surrogate,
        selected_config=selected_config,
        task_weights=task_weights,
        validation_task_weights=validation_weights,
        member_best_epochs=tuple(
            int(fitted.best_epoch) for fitted in fitted_members
        ),
        member_epochs_completed=tuple(
            int(fitted.epochs_completed) for fitted in fitted_members
        ),
        fit_seconds=float(fit_seconds),
    )


def save_training_result(
    path: str | Path, result: EnsembleTrainingResult
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    surrogate = result.surrogate
    torch.save(
        {
            "state_dicts": [member.state_dict() for member in surrogate.members],
            "selected_config": result.selected_config,
            "task_weights": result.task_weights,
            "validation_task_weights": result.validation_task_weights,
            "x_mean": surrogate.x_mean,
            "x_scale": surrogate.x_scale,
            "y_mean": surrogate.y_mean,
            "y_scale": surrogate.y_scale,
            "member_best_epochs": result.member_best_epochs,
            "member_epochs_completed": result.member_epochs_completed,
            "fit_seconds": result.fit_seconds,
        },
        output_path,
    )
