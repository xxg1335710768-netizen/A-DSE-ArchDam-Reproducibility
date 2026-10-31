"""Five-member DE-MMoE inference and ensemble GradientSHAP.

The active-learning score uses member disagreement only.  Residual variance
floors are intentionally not exposed here because they are used for prediction
intervals, not for design-point acquisition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from Revise.surrogate_benchmark.models_torch import DeterministicMMoE

from ._common import (
    ENSEMBLE_SIZE,
    INPUT_DIM,
    finite_matrix,
    maximum_normalize_rows,
)

@dataclass(frozen=True)
class EnsemblePrediction:
    mean_standardized: np.ndarray
    model_variance_standardized: np.ndarray
    mean_physical: np.ndarray
    model_variance_physical: np.ndarray

    @property
    def model_std_standardized(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.model_variance_standardized, 0.0))

    @property
    def model_std_physical(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.model_variance_physical, 0.0))


class _EnsembleMean(nn.Module):
    """Differentiable standardized ensemble mean from physical inputs."""

    def __init__(
        self,
        members: Sequence[nn.Module],
        x_mean: np.ndarray,
        x_scale: np.ndarray,
        output_indices: Sequence[int],
    ) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)
        self.register_buffer(
            "x_mean", torch.as_tensor(x_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "x_scale", torch.as_tensor(x_scale, dtype=torch.float32)
        )
        self.register_buffer(
            "output_indices",
            torch.as_tensor(tuple(output_indices), dtype=torch.long),
        )

    def forward(self, x_physical: torch.Tensor) -> torch.Tensor:
        x_standardized = (x_physical - self.x_mean) / self.x_scale
        member_outputs = torch.stack(
            [member(x_standardized) for member in self.members], dim=0
        )
        mean = member_outputs.mean(dim=0)
        return mean.index_select(1, self.output_indices)


class EnsembleMMoESurrogate:
    """Loaded five-member mean-first DE-MMoE used by A-DSE."""

    def __init__(
        self,
        members: Sequence[DeterministicMMoE],
        x_mean: np.ndarray,
        x_scale: np.ndarray,
        y_mean: np.ndarray,
        y_scale: np.ndarray,
        *,
        device: str = "cpu",
    ) -> None:
        if len(members) != ENSEMBLE_SIZE:
            raise ValueError(
                f"Expected {ENSEMBLE_SIZE} ensemble members, got {len(members)}."
            )
        self.device = torch.device(device)
        self.members = tuple(member.to(self.device).eval() for member in members)
        self.x_mean = np.asarray(x_mean, dtype=np.float64).reshape(-1)
        self.x_scale = np.asarray(x_scale, dtype=np.float64).reshape(-1)
        self.y_mean = np.asarray(y_mean, dtype=np.float64).reshape(-1)
        self.y_scale = np.asarray(y_scale, dtype=np.float64).reshape(-1)
        if np.any(self.x_scale <= 0.0) or np.any(self.y_scale <= 0.0):
            raise ValueError("All standardization scales must be positive.")
        if self.x_mean.size != INPUT_DIM or self.x_scale.size != INPUT_DIM:
            raise ValueError(f"Input scalers must contain {INPUT_DIM} values.")
        if self.y_mean.size == 0 or self.y_mean.size != self.y_scale.size:
            raise ValueError("Output scalers must have equal, nonzero sizes.")

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> "EnsembleMMoESurrogate":
        model_path = Path(path)
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        try:
            payload = torch.load(
                model_path, map_location=device, weights_only=False
            )
        except TypeError:
            payload = torch.load(model_path, map_location=device)

        required = {
            "state_dicts",
            "selected_config",
            "x_mean",
            "x_scale",
            "y_mean",
            "y_scale",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(f"Model payload misses fields: {missing}")

        x_mean = np.asarray(payload["x_mean"], dtype=np.float64)
        y_mean = np.asarray(payload["y_mean"], dtype=np.float64)
        members = []
        for state_dict in payload["state_dicts"]:
            member = DeterministicMMoE(
                d_in=x_mean.size,
                num_tasks=y_mean.size,
                activation_name="swish",
                **payload["selected_config"],
            )
            member.load_state_dict(state_dict)
            members.append(member)
        return cls(
            members,
            x_mean,
            payload["x_scale"],
            y_mean,
            payload["y_scale"],
            device=device,
        )

    def predict(
        self, x_physical: np.ndarray, *, batch_size: int = 4096
    ) -> EnsemblePrediction:
        x = finite_matrix(x_physical, INPUT_DIM, "x_physical")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        chunks = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                values = (x[start : start + batch_size] - self.x_mean) / self.x_scale
                tensor = torch.as_tensor(
                    values, dtype=torch.float32, device=self.device
                )
                chunks.append(
                    torch.stack([member(tensor) for member in self.members], dim=0)
                    .cpu()
                    .numpy()
                )
        member_predictions = np.concatenate(chunks, axis=1).astype(np.float64)
        mean_standardized = member_predictions.mean(axis=0)
        model_variance_standardized = member_predictions.var(axis=0, ddof=1)
        mean_physical = mean_standardized * self.y_scale + self.y_mean
        model_variance_physical = (
            model_variance_standardized * self.y_scale.reshape(1, -1) ** 2
        )
        return EnsemblePrediction(
            mean_standardized=mean_standardized,
            model_variance_standardized=model_variance_standardized,
            mean_physical=mean_physical,
            model_variance_physical=model_variance_physical,
        )

    def differentiable_mean(
        self, output_indices: Sequence[int] = (0, 1)
    ) -> nn.Module:
        indices = tuple(int(index) for index in output_indices)
        if not indices or min(indices) < 0 or max(indices) >= self.y_mean.size:
            raise ValueError("Invalid output indices.")
        module = _EnsembleMean(
            self.members, self.x_mean, self.x_scale, indices
        ).to(self.device)
        module.eval()
        return module


@dataclass(frozen=True)
class GradientShapResult:
    output_indices: tuple[int, ...]
    feature_names: tuple[str, ...]
    signed_values: np.ndarray
    mean_absolute_by_task: np.ndarray
    normalized_by_task: np.ndarray
    combined_importance: np.ndarray
    baseline_indices: np.ndarray
    reconstruction_relative_mae: np.ndarray


def _antithetic_alphas(
    rng: np.random.Generator, sample_count: int, batch_size: int
) -> np.ndarray:
    if sample_count <= 0:
        raise ValueError("path_samples must be positive.")
    half = sample_count // 2
    first = rng.random((half, batch_size, 1), dtype=np.float32)
    values = [first, 1.0 - first]
    if sample_count % 2:
        values.append(rng.random((1, batch_size, 1), dtype=np.float32))
    return np.concatenate(values, axis=0)


def gradient_shap_objectives(
    surrogate: EnsembleMMoESurrogate,
    background_x: np.ndarray,
    explain_x: np.ndarray,
    *,
    output_indices: Sequence[int] = (0, 1),
    feature_names: Sequence[str] | None = None,
    baseline_count: int = 30,
    path_samples: int = 64,
    batch_size: int = 32,
    seed: int = 2025,
) -> GradientShapResult:
    """Expected-gradient form of GradientSHAP for the ensemble mean.

    Signed attributions are retained per sample.  Design-variable importance
    uses the mean absolute attribution over the complete predicted Pareto set.
    """

    background = finite_matrix(
        background_x, INPUT_DIM, "background_x", dtype=np.float32
    )
    explained = finite_matrix(
        explain_x, INPUT_DIM, "explain_x", dtype=np.float32
    )
    if baseline_count <= 0:
        raise ValueError("baseline_count must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    names = tuple(feature_names or [f"x{i + 1}" for i in range(INPUT_DIM)])
    if len(names) != INPUT_DIM or len(set(names)) != INPUT_DIM:
        raise ValueError("feature_names must contain 16 unique names.")
    outputs = tuple(int(index) for index in output_indices)
    module = surrogate.differentiable_mean(outputs)
    device = surrogate.device

    baseline_count = min(int(baseline_count), len(background))
    selection_rng = np.random.default_rng(seed)
    baseline_indices = np.sort(
        selection_rng.choice(len(background), baseline_count, replace=False)
    )
    selected_background = background[baseline_indices]
    background_tensor = torch.as_tensor(
        selected_background, dtype=torch.float32, device=device
    )

    task_count = len(outputs)
    sample_count, dimension = explained.shape
    signed_values = np.zeros(
        (task_count, sample_count, dimension), dtype=np.float64
    )

    for task_position in range(task_count):
        # Resetting the generator gives both objectives identical Monte Carlo paths.
        path_rng = np.random.default_rng(seed + 1)
        for start in range(0, sample_count, batch_size):
            stop = min(start + batch_size, sample_count)
            x_batch = torch.as_tensor(
                explained[start:stop], dtype=torch.float32, device=device
            )
            current_batch = stop - start
            accumulated = torch.zeros(
                (current_batch, dimension), dtype=torch.float32, device=device
            )

            for baseline_row in background_tensor:
                baseline = baseline_row.reshape(1, -1).expand(current_batch, -1)
                delta = x_batch - baseline
                alphas = torch.as_tensor(
                    _antithetic_alphas(path_rng, path_samples, current_batch),
                    dtype=torch.float32,
                    device=device,
                )
                interpolated = (
                    baseline.unsqueeze(0) + alphas * delta.unsqueeze(0)
                ).reshape(-1, dimension)
                interpolated = interpolated.detach().requires_grad_(True)
                output = module(interpolated)[:, task_position]
                gradients = torch.autograd.grad(
                    output.sum(), interpolated, create_graph=False
                )[0].reshape(path_samples, current_batch, dimension)
                accumulated += (gradients * delta.unsqueeze(0)).mean(dim=0)

            accumulated /= float(baseline_count)
            signed_values[task_position, start:stop] = (
                accumulated.detach().cpu().numpy()
            )

    mean_absolute = np.mean(np.abs(signed_values), axis=1)
    normalized = maximum_normalize_rows(mean_absolute)
    combined = normalized.mean(axis=0)

    with torch.no_grad():
        explain_tensor = torch.as_tensor(
            explained, dtype=torch.float32, device=device
        )
        prediction = module(explain_tensor).cpu().numpy()
        baseline_mean = module(background_tensor).mean(dim=0).cpu().numpy()

    reconstructed = baseline_mean[:, None] + signed_values.sum(axis=2)
    prediction_by_task = prediction.T
    scale = np.maximum(np.ptp(prediction, axis=0), 1e-12)
    reconstruction_error = (
        np.mean(np.abs(prediction_by_task - reconstructed), axis=1) / scale
    )

    return GradientShapResult(
        output_indices=outputs,
        feature_names=names,
        signed_values=signed_values,
        mean_absolute_by_task=mean_absolute,
        normalized_by_task=normalized,
        combined_importance=combined,
        baseline_indices=baseline_indices,
        reconstruction_relative_mae=reconstruction_error,
    )


def save_gradient_shap_report(
    path: str | Path, result: GradientShapResult
) -> None:
    """Save aligned importance first; save a separate sorted ranking sheet."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned = pd.DataFrame({"feature": result.feature_names})
    for task_position, output_index in enumerate(result.output_indices):
        aligned[f"mean_abs_output_{output_index}"] = (
            result.mean_absolute_by_task[task_position]
        )
        aligned[f"normalized_output_{output_index}"] = (
            result.normalized_by_task[task_position]
        )
    aligned["combined_importance"] = result.combined_importance
    ranking = aligned.sort_values(
        "combined_importance", ascending=False
    ).reset_index(drop=True)
    diagnostics = pd.DataFrame(
        {
            "output_index": result.output_indices,
            "reconstruction_relative_mae": result.reconstruction_relative_mae,
        }
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        aligned.to_excel(writer, sheet_name="importance_aligned", index=False)
        ranking.to_excel(writer, sheet_name="importance_ranking", index=False)
        diagnostics.to_excel(writer, sheet_name="diagnostics", index=False)
        for task_position, output_index in enumerate(result.output_indices):
            pd.DataFrame(
                result.signed_values[task_position],
                columns=result.feature_names,
            ).to_excel(
                writer,
                sheet_name=f"signed_output_{output_index}"[:31],
                index=False,
            )
