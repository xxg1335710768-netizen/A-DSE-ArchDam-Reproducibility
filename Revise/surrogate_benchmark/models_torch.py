"""Neural baselines, MMoE ablations and a shared early-stopping trainer."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..probabilistic_mmoe import (
    PredictiveMoments,
    ProbabilisticMMoE,
    ensemble_moments,
    gaussian_nll,
    set_seed,
)


def activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


def dense_stack(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    activation_name: str,
    dropout: float,
) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    previous = int(input_dim)
    for width in hidden_dims:
        layers.append(nn.Linear(previous, int(width)))
        layers.append(activation(activation_name))
        if dropout > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        previous = int(width)
    return nn.Sequential(*layers), previous


class IndependentTaskMLP(nn.Module):
    """A set of output-specific MLPs with no shared trainable parameters."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float = 0.0,
        activation_name: str = "swish",
    ) -> None:
        super().__init__()
        networks = []
        for _ in range(output_dim):
            body, last_dim = dense_stack(
                input_dim,
                hidden_dims,
                activation_name,
                dropout,
            )
            networks.append(nn.Sequential(body, nn.Linear(last_dim, 1)))
        self.networks = nn.ModuleList(networks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([network(x) for network in self.networks], dim=1)


class MultiTaskMLP(nn.Module):
    """Shared feature extractor followed by one mean head per output."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float = 0.0,
        activation_name: str = "swish",
        head_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.backbone, shared_dim = dense_stack(
            input_dim,
            hidden_dims,
            activation_name,
            dropout,
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(shared_dim, head_hidden),
                    activation(activation_name),
                    nn.Linear(head_hidden, 1),
                )
                for _ in range(output_dim)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.backbone(x)
        return torch.cat([head(shared) for head in self.heads], dim=1)


class ProbabilisticMultiTaskMLP(nn.Module):
    """Matched probabilistic MLP used as the deep-ensemble architecture control."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        activation_name: str = "swish",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone, shared_dim = dense_stack(
            input_dim,
            hidden_dims,
            activation_name,
            dropout,
        )
        self.heads = nn.ModuleList(
            [nn.Linear(shared_dim, 2) for _ in range(output_dim)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.backbone(x)
        outputs = torch.stack([head(shared) for head in self.heads], dim=1)
        mean = outputs[:, :, 0]
        log_variance = outputs[:, :, 1].clamp(-10.0, 5.0)
        return mean, log_variance


class DeterministicExpert(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation_name: str,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation(activation_name),
            nn.Linear(hidden_dim, hidden_dim),
            activation(activation_name),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DeterministicMMoE(nn.Module):
    """Mean-only MMoE used to isolate the contribution of the gate architecture."""

    def __init__(
        self,
        d_in: int,
        num_experts: int,
        expert_hidden: int,
        tower_hidden: int,
        num_tasks: int,
        activation_name: str = "swish",
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [
                DeterministicExpert(d_in, expert_hidden, activation_name)
                for _ in range(num_experts)
            ]
        )
        self.gates = nn.ModuleList(
            [nn.Linear(d_in, num_experts) for _ in range(num_tasks)]
        )
        self.towers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(expert_hidden, tower_hidden),
                    activation(activation_name),
                    nn.Linear(tower_hidden, 1),
                )
                for _ in range(num_tasks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        task_outputs = []
        for gate, tower in zip(self.gates, self.towers):
            weights = F.softmax(gate(x), dim=-1)
            fused = torch.bmm(weights.unsqueeze(1), expert_outputs).squeeze(1)
            task_outputs.append(tower(fused))
        return torch.cat(task_outputs, dim=1)


@dataclass(frozen=True)
class TorchTrainingConfig:
    max_epochs: int = 500
    patience: int = 30
    min_delta: float = 1e-6
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 5e-2
    gradient_clip: float = 5.0


@dataclass
class FittedTorchModel:
    model: nn.Module
    best_epoch: int
    best_validation_loss: float
    epochs_completed: int
    fit_seconds: float
    probabilistic: bool
    selection_metric: str
    best_validation_nll: float | None
    best_validation_rmse: float
    seed: int


def _tensor(values: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def fit_with_early_stopping(
    model_factory: Callable[[], nn.Module],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    probabilistic: bool,
    config: TorchTrainingConfig,
    seed: int,
    device: str = "cpu",
    selection_metric: str = "loss",
    task_weights: np.ndarray | None = None,
    validation_task_weights: np.ndarray | None = None,
) -> FittedTorchModel:
    if selection_metric not in {"loss", "rmse"}:
        raise ValueError("selection_metric must be 'loss' or 'rmse'.")
    if probabilistic and task_weights is not None:
        raise ValueError("Task-weighted loss is currently defined for mean-only models.")
    def normalize_weights(
        values: np.ndarray | None,
        label: str,
    ) -> np.ndarray | None:
        if values is None:
            return None
        normalized = np.asarray(values, dtype=np.float64)
        expected_shape = (np.asarray(y_train).shape[1],)
        if normalized.shape != expected_shape:
            raise ValueError(
                f"Expected {label} with shape {expected_shape}, "
                f"got {normalized.shape}."
            )
        if (
            not np.isfinite(normalized).all()
            or np.any(normalized < 0.0)
            or normalized.sum() <= 0.0
        ):
            raise ValueError(
                f"{label} must be finite, non-negative and sum positive."
            )
        return normalized / normalized.sum()

    normalized_task_weights = normalize_weights(task_weights, "task_weights")
    normalized_validation_weights = normalize_weights(
        validation_task_weights,
        "validation_task_weights",
    )
    if normalized_validation_weights is None:
        normalized_validation_weights = normalized_task_weights
    set_seed(seed)
    model = model_factory().to(device)
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(
        torch.as_tensor(x_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_x = _tensor(x_validation, device)
    validation_y = _tensor(y_validation, device)
    weight_tensor = (
        _tensor(normalized_task_weights, device)
        if normalized_task_weights is not None
        else None
    )
    validation_weight_tensor = (
        _tensor(normalized_validation_weights, device)
        if normalized_validation_weights is not None
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start = time.perf_counter()
    epochs_completed = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if probabilistic:
                mean, log_variance = model(batch_x)
                loss = gaussian_nll(mean, log_variance, batch_y)
            else:
                prediction = model(batch_x)
                per_task_mse = torch.mean((prediction - batch_y).square(), dim=0)
                loss = (
                    torch.sum(per_task_mse * weight_tensor)
                    if weight_tensor is not None
                    else torch.mean(per_task_mse)
                )
            loss.backward()
            if config.gradient_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            if probabilistic:
                mean, log_variance = model(validation_x)
                validation_nll = float(
                    gaussian_nll(mean, log_variance, validation_y).cpu()
                )
                validation_rmse = float(
                    torch.sqrt(
                        torch.mean((validation_y - mean).square(), dim=0)
                    ).mean().cpu()
                )
                validation_loss = (
                    validation_nll
                    if selection_metric == "loss"
                    else validation_rmse
                )
            else:
                prediction = model(validation_x)
                per_task_mse = torch.mean(
                    (validation_y - prediction).square(), dim=0
                )
                per_task_rmse = torch.sqrt(per_task_mse)
                validation_mse = float(
                    (
                        torch.sum(per_task_mse * validation_weight_tensor)
                        if validation_weight_tensor is not None
                        else torch.mean(per_task_mse)
                    ).cpu()
                )
                validation_rmse = float(
                    (
                        torch.sum(per_task_rmse * validation_weight_tensor)
                        if validation_weight_tensor is not None
                        else torch.mean(per_task_rmse)
                    ).cpu()
                )
                validation_nll = None
                validation_loss = (
                    validation_mse
                    if selection_metric == "loss"
                    else validation_rmse
                )
        epochs_completed = epoch
        if validation_loss < best_loss - config.min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("Early stopping did not record a valid model state.")
    model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.no_grad():
        if probabilistic:
            mean, log_variance = model(validation_x)
            best_validation_nll = float(
                gaussian_nll(mean, log_variance, validation_y).cpu()
            )
            best_validation_rmse = float(
                torch.sqrt(
                    torch.mean((validation_y - mean).square(), dim=0)
                ).mean().cpu()
            )
        else:
            prediction = model(validation_x)
            best_validation_nll = None
            per_task_rmse = torch.sqrt(
                torch.mean((validation_y - prediction).square(), dim=0)
            )
            best_validation_rmse = float(
                (
                    torch.sum(per_task_rmse * validation_weight_tensor)
                    if validation_weight_tensor is not None
                    else torch.mean(per_task_rmse)
                ).cpu()
            )
    return FittedTorchModel(
        model=model,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        epochs_completed=epochs_completed,
        fit_seconds=time.perf_counter() - start,
        probabilistic=probabilistic,
        selection_metric=selection_metric,
        best_validation_nll=best_validation_nll,
        best_validation_rmse=best_validation_rmse,
        seed=seed,
    )


@torch.no_grad()
def predict_deterministic(
    fitted: FittedTorchModel,
    x: np.ndarray,
    *,
    device: str = "cpu",
) -> np.ndarray:
    if fitted.probabilistic:
        raise ValueError("predict_deterministic received a probabilistic model.")
    fitted.model.to(device).eval()
    return fitted.model(_tensor(x, device)).cpu().numpy()


@torch.no_grad()
def predict_deterministic_ensemble(
    fitted_models: list[FittedTorchModel],
    x: np.ndarray,
    *,
    device: str = "cpu",
) -> PredictiveMoments:
    """Return mean and member-disagreement variance for mean-only members."""
    if not fitted_models or any(model.probabilistic for model in fitted_models):
        raise ValueError("A non-empty list of deterministic members is required.")
    predictions = np.stack(
        [predict_deterministic(fitted, x, device=device) for fitted in fitted_models],
        axis=0,
    )
    model_variance = np.maximum(predictions.var(axis=0, ddof=0), 1e-12)
    return PredictiveMoments(
        mean=predictions.mean(axis=0),
        epistemic_var=model_variance,
        aleatoric_var=np.zeros_like(model_variance),
    )


def _enable_dropout_only(model: nn.Module) -> None:
    """Enable stochastic dropout masks without changing other layer modes."""
    model.eval()
    positive_dropout_layers = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout) and module.p > 0.0:
            module.train()
            positive_dropout_layers += 1
    if positive_dropout_layers == 0:
        raise ValueError("MC Dropout requires at least one positive-rate dropout layer.")


@torch.no_grad()
def predict_mc_dropout(
    fitted: FittedTorchModel,
    x: np.ndarray,
    *,
    mc_samples: int,
    seed: int | None = None,
    device: str = "cpu",
) -> PredictiveMoments:
    """Estimate mean and model variance from stochastic dropout predictions.

    The network is still trained by MSE and has no conditional-variance head.
    Consequently, the returned epistemic component is the MC Dropout model
    variance and the residual component is zero.  A calibration-set scale may
    later convert this raw variance into an approximate predictive variance for
    interval and NLL evaluation, but it does not create a variance decomposition.
    """
    if fitted.probabilistic:
        raise ValueError("MC Dropout expects a mean-only MSE-trained model.")
    if mc_samples < 2:
        raise ValueError("mc_samples must be at least 2.")
    model = fitted.model.to(device)
    _enable_dropout_only(model)
    tensor = _tensor(np.asarray(x), device)
    inference_seed = fitted.seed + 50_000 if seed is None else int(seed)
    device_object = torch.device(device)
    fork_devices: list[int] = []
    if device_object.type == "cuda":
        fork_devices = [
            device_object.index
            if device_object.index is not None
            else torch.cuda.current_device()
        ]
    predictions: list[np.ndarray] = []
    try:
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(inference_seed)
            if device_object.type == "cuda":
                torch.cuda.manual_seed_all(inference_seed)
            for _ in range(mc_samples):
                predictions.append(model(tensor).cpu().numpy())
    finally:
        model.eval()
    draws = np.stack(predictions, axis=0)
    model_variance = np.maximum(draws.var(axis=0, ddof=1), 1e-12)
    return PredictiveMoments(
        mean=draws.mean(axis=0),
        epistemic_var=model_variance,
        aleatoric_var=np.zeros_like(model_variance),
    )


@torch.no_grad()
def predict_probabilistic(
    fitted: FittedTorchModel,
    x: np.ndarray,
    *,
    device: str = "cpu",
) -> PredictiveMoments:
    if not fitted.probabilistic:
        raise ValueError("predict_probabilistic received a deterministic model.")
    return ensemble_moments([fitted.model], np.asarray(x), device=device)


@torch.no_grad()
def predict_ensemble(
    fitted_models: list[FittedTorchModel],
    x: np.ndarray,
    *,
    device: str = "cpu",
) -> PredictiveMoments:
    if not fitted_models or not all(model.probabilistic for model in fitted_models):
        raise ValueError("A non-empty list of probabilistic members is required.")
    return ensemble_moments(
        [fitted.model for fitted in fitted_models],
        np.asarray(x),
        device=device,
    )


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def probabilistic_mmoe_factory(
    *,
    input_dim: int,
    output_dim: int,
    num_experts: int,
    expert_hidden: int,
    tower_hidden: int,
    activation_name: str = "swish",
) -> Callable[[], ProbabilisticMMoE]:
    return lambda: ProbabilisticMMoE(
        d_in=input_dim,
        num_experts=num_experts,
        expert_hidden=expert_hidden,
        tower_hidden=tower_hidden,
        num_tasks=output_dim,
        activation=activation_name,
        dropout=0.0,
    )
