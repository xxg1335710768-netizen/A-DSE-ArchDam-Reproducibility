"""MMoE models with explicit, testable uncertainty semantics.

The legacy implementation samples both a Gaussian expert representation and a
Gaussian task output, then reports the empirical standard deviation of those
draws. That quantity is a predictive standard deviation, but it does not by
itself identify epistemic uncertainty because no posterior over network weights
is learned.

This module retains the legacy model for controlled comparisons and provides
two defensible alternatives:

* MC-dropout MMoE: epistemic variance is the variance of predictive means over
  dropout masks; expected task variance is the conditional residual component.
* Deep-ensemble MMoE: epistemic variance is disagreement between independently
  trained members; expected task variance is the conditional residual component.

In both cases total predictive variance follows the law of total variance.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


LOGVAR_MIN = -10.0
LOGVAR_MAX = 5.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _activation(name: str) -> nn.Module:
    return {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "swish": nn.SiLU(),
        "silu": nn.SiLU(),
    }.get(name.lower(), nn.SiLU())


def gaussian_nll(
    pred_mu: torch.Tensor,
    pred_logvar: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Full Gaussian NLL with a squared residual and bounded log-variance."""
    logvar = pred_logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)
    return (
        0.5 * torch.exp(-logvar) * (target - pred_mu).square()
        + 0.5 * logvar
        + 0.5 * math.log(2.0 * math.pi)
    ).mean()


class LegacyGaussianExpert(nn.Module):
    """Original input-dependent Gaussian latent expert used as the baseline."""

    def __init__(self, input_dim: int, hidden_dim: int, activation: str = "swish") -> None:
        super().__init__()
        self.fc_mu = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _activation(activation),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _activation(activation),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        sample = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return sample, mu, logvar


class LegacyLatentMMoE(nn.Module):
    """Exact architecture of the submitted code, retained for fair comparison."""

    def __init__(
        self,
        d_in: int,
        num_experts: int,
        expert_hidden: int,
        tower_hidden: int,
        num_tasks: int,
        activation: str = "swish",
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [LegacyGaussianExpert(d_in, expert_hidden, activation) for _ in range(num_experts)]
        )
        self.gates = nn.ModuleList([nn.Linear(d_in, num_experts) for _ in range(num_tasks)])
        self.towers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(expert_hidden, tower_hidden),
                    _activation(activation),
                    nn.Linear(tower_hidden, 2),
                )
                for _ in range(num_tasks)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        samples = torch.stack([expert(x)[0] for expert in self.experts], dim=1)
        means: list[torch.Tensor] = []
        logvars: list[torch.Tensor] = []
        for gate, tower in zip(self.gates, self.towers):
            weights = F.softmax(gate(x), dim=-1)
            fused = torch.bmm(weights.unsqueeze(1), samples).squeeze(1)
            out = tower(fused)
            means.append(out[:, :1])
            logvars.append(out[:, 1:2])
        return torch.cat(means, dim=1), torch.cat(logvars, dim=1)


class DeterministicExpert(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            _activation(activation),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProbabilisticMMoE(nn.Module):
    """Deterministic MMoE mean/variance heads with optional MC dropout."""

    def __init__(
        self,
        d_in: int,
        num_experts: int,
        expert_hidden: int,
        tower_hidden: int,
        num_tasks: int,
        activation: str = "swish",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [
                DeterministicExpert(d_in, expert_hidden, activation, dropout)
                for _ in range(num_experts)
            ]
        )
        self.gates = nn.ModuleList([nn.Linear(d_in, num_experts) for _ in range(num_tasks)])
        self.towers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(expert_hidden, tower_hidden),
                    _activation(activation),
                    nn.Dropout(dropout),
                    nn.Linear(tower_hidden, 2),
                )
                for _ in range(num_tasks)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        means: list[torch.Tensor] = []
        logvars: list[torch.Tensor] = []
        for gate, tower in zip(self.gates, self.towers):
            weights = F.softmax(gate(x), dim=-1)
            fused = torch.bmm(weights.unsqueeze(1), expert_outputs).squeeze(1)
            out = tower(fused)
            means.append(out[:, :1])
            logvars.append(out[:, 1:2].clamp(LOGVAR_MIN, LOGVAR_MAX))
        return torch.cat(means, dim=1), torch.cat(logvars, dim=1)


@dataclass
class PredictiveMoments:
    mean: np.ndarray
    epistemic_var: np.ndarray
    aleatoric_var: np.ndarray
    calibrated_total_var: np.ndarray | None = None

    @property
    def total_var(self) -> np.ndarray:
        return np.maximum(self.epistemic_var + self.aleatoric_var, 1e-12)

    @property
    def interval_var(self) -> np.ndarray:
        """Variance used for NLL and intervals after optional calibration."""
        if self.calibrated_total_var is None:
            return self.total_var
        return np.maximum(self.calibrated_total_var, 1e-12)


def train_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    device: str = "cpu",
    gradient_clip: float = 5.0,
) -> nn.Module:
    """Train one member with a reproducible shuffle and fixed training budget."""
    set_seed(seed)
    model = model.to(device)
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(
        torch.as_tensor(x_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred_mu, pred_logvar = model(xb)
            loss = gaussian_nll(pred_mu, pred_logvar, yb)
            loss.backward()
            if gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
    model.eval()
    return model


def _as_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32, device=device)


@torch.no_grad()
def legacy_original_sampling(
    model: LegacyLatentMMoE,
    x: np.ndarray,
    *,
    mc_samples: int,
    device: str = "cpu",
) -> PredictiveMoments:
    """Reproduce the submitted inference: sample latent state and task output."""
    model.to(device).eval()
    xt = _as_tensor(x, device)
    samples: list[np.ndarray] = []
    for _ in range(mc_samples):
        mu, logvar = model(xt)
        draw = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar.clamp(LOGVAR_MIN, LOGVAR_MAX))
        samples.append(draw.cpu().numpy())
    draws = np.stack(samples, axis=0)
    total_var = np.maximum(draws.var(axis=0, ddof=0), 1e-12)
    return PredictiveMoments(
        mean=draws.mean(axis=0),
        epistemic_var=total_var,
        aleatoric_var=np.zeros_like(total_var),
    )


@torch.no_grad()
def legacy_moment_decomposition(
    model: LegacyLatentMMoE,
    x: np.ndarray,
    *,
    mc_samples: int,
    device: str = "cpu",
) -> PredictiveMoments:
    """Correct the legacy predictive variance by the law of total variance.

    ``epistemic_var`` here is retained as a compatibility field. Scientifically
    it is variance induced by the stochastic latent representation, not a
    posterior-over-weights estimate and should be described that way.
    """
    model.to(device).eval()
    xt = _as_tensor(x, device)
    means: list[np.ndarray] = []
    conditional_vars: list[np.ndarray] = []
    for _ in range(mc_samples):
        mu, logvar = model(xt)
        means.append(mu.cpu().numpy())
        conditional_vars.append(torch.exp(logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)).cpu().numpy())
    mean_stack = np.stack(means, axis=0)
    var_stack = np.stack(conditional_vars, axis=0)
    return PredictiveMoments(
        mean=mean_stack.mean(axis=0),
        epistemic_var=np.maximum(mean_stack.var(axis=0, ddof=0), 0.0),
        aleatoric_var=np.maximum(var_stack.mean(axis=0), 1e-12),
    )


def _enable_mc_dropout(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_moments(
    model: ProbabilisticMMoE,
    x: np.ndarray,
    *,
    mc_samples: int,
    device: str = "cpu",
) -> PredictiveMoments:
    """Separate dropout disagreement and expected conditional output variance."""
    model.to(device)
    _enable_mc_dropout(model)
    xt = _as_tensor(x, device)
    means: list[np.ndarray] = []
    conditional_vars: list[np.ndarray] = []
    for _ in range(mc_samples):
        mu, logvar = model(xt)
        means.append(mu.cpu().numpy())
        conditional_vars.append(torch.exp(logvar).cpu().numpy())
    model.eval()
    mean_stack = np.stack(means, axis=0)
    var_stack = np.stack(conditional_vars, axis=0)
    return PredictiveMoments(
        mean=mean_stack.mean(axis=0),
        epistemic_var=np.maximum(mean_stack.var(axis=0, ddof=0), 0.0),
        aleatoric_var=np.maximum(var_stack.mean(axis=0), 1e-12),
    )


@torch.no_grad()
def ensemble_moments(
    models: Sequence[ProbabilisticMMoE],
    x: np.ndarray,
    *,
    device: str = "cpu",
) -> PredictiveMoments:
    """Exact first two moments of an equally weighted Gaussian ensemble."""
    xt = _as_tensor(x, device)
    means: list[np.ndarray] = []
    conditional_vars: list[np.ndarray] = []
    for model in models:
        model.to(device).eval()
        mu, logvar = model(xt)
        means.append(mu.cpu().numpy())
        conditional_vars.append(torch.exp(logvar).cpu().numpy())
    mean_stack = np.stack(means, axis=0)
    var_stack = np.stack(conditional_vars, axis=0)
    return PredictiveMoments(
        mean=mean_stack.mean(axis=0),
        epistemic_var=np.maximum(mean_stack.var(axis=0, ddof=0), 0.0),
        aleatoric_var=np.maximum(var_stack.mean(axis=0), 1e-12),
    )


def fit_variance_temperature(
    y_true: np.ndarray,
    moments: PredictiveMoments,
    *,
    min_scale: float = 0.05,
    max_scale: float = 20.0,
) -> np.ndarray:
    """Fit one held-out NLL-optimal multiplicative variance scale per task."""
    base_var = np.maximum(moments.total_var, 1e-12)
    squared_standardized_residual = (np.asarray(y_true) - moments.mean) ** 2 / base_var
    scale = np.mean(squared_standardized_residual, axis=0)
    return np.clip(scale, min_scale, max_scale)


def apply_variance_temperature(
    moments: PredictiveMoments,
    scale: np.ndarray,
) -> PredictiveMoments:
    """Attach a calibrated total variance while preserving raw components.

    The calibration scale is learned from total predictive residuals.  It does
    not establish that the epistemic and conditional-residual components have
    each been calibrated separately.  With task-wise constant scales and a
    weighted geometric mean, scaling model standard deviations would multiply
    every candidate score by the same constant and would not change its rank.
    """
    scale = np.asarray(scale, dtype=float).reshape(1, -1)
    return PredictiveMoments(
        mean=moments.mean.copy(),
        epistemic_var=moments.epistemic_var.copy(),
        aleatoric_var=moments.aleatoric_var.copy(),
        calibrated_total_var=moments.total_var * scale,
    )


def weighted_geometric_mean(
    values: np.ndarray,
    weights: Iterable[float] | None = None,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Weighted geometric mean used by Eq. (13), with explicit normalization."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must have shape (n_samples, n_tasks)")
    task_count = values.shape[1]
    if weights is None:
        weight_array = np.full(task_count, 1.0 / task_count)
    else:
        weight_array = np.asarray(list(weights), dtype=float)
        if weight_array.shape != (task_count,):
            raise ValueError(f"expected {task_count} weights, got {weight_array.shape}")
        if np.any(weight_array < 0) or not np.isfinite(weight_array).all():
            raise ValueError("weights must be finite and non-negative")
        total = weight_array.sum()
        if total <= 0:
            raise ValueError("at least one weight must be positive")
        weight_array = weight_array / total
    return np.exp(np.sum(np.log(np.maximum(values, eps)) * weight_array[None, :], axis=1))
