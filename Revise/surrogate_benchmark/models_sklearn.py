"""Scikit-learn and XGBoost baselines with a common prediction interface."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF


@dataclass(frozen=True)
class GaussianPrediction:
    mean: np.ndarray
    variance: np.ndarray


def build_kernel(name: str, input_dim: int):
    amplitude = ConstantKernel(1.0, (1e-2, 1e2))
    length_scale = np.ones(input_dim, dtype=np.float64)
    bounds = (1e-2, 1e3)
    normalized_name = name.lower()
    if normalized_name == "rbf":
        correlation = RBF(length_scale, bounds)
    elif normalized_name == "matern32":
        correlation = Matern(length_scale, bounds, nu=1.5)
    elif normalized_name == "matern52":
        correlation = Matern(length_scale, bounds, nu=2.5)
    else:
        raise ValueError(f"Unsupported GPR kernel: {name}")
    return amplitude * correlation


class IndependentGPR:
    """One exact GPR per standardized output."""

    def __init__(
        self,
        *,
        kernel: str,
        alpha: float,
        n_restarts_optimizer: int,
        random_state: int,
    ) -> None:
        self.kernel_name = kernel
        self.alpha = float(alpha)
        self.n_restarts_optimizer = int(n_restarts_optimizer)
        self.random_state = int(random_state)
        self.models: list[GaussianProcessRegressor] = []

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "IndependentGPR":
        x = np.asarray(x_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
            raise ValueError("GPR training arrays must be matching 2D matrices.")
        template = build_kernel(self.kernel_name, x.shape[1])
        self.models = []
        for output_index in range(y.shape[1]):
            model = GaussianProcessRegressor(
                kernel=deepcopy(template),
                alpha=self.alpha,
                normalize_y=False,
                n_restarts_optimizer=self.n_restarts_optimizer,
                random_state=self.random_state + output_index,
            )
            model.fit(x, y[:, output_index])
            self.models.append(model)
        return self

    def predict_distribution(self, x: np.ndarray) -> GaussianPrediction:
        if not self.models:
            raise RuntimeError("IndependentGPR has not been fitted.")
        features = np.asarray(x, dtype=np.float64)
        means = []
        variances = []
        for model in self.models:
            mean, standard_deviation = model.predict(features, return_std=True)
            means.append(mean)
            variances.append(np.maximum(standard_deviation**2, 1e-12))
        return GaussianPrediction(
            mean=np.column_stack(means),
            variance=np.column_stack(variances),
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_distribution(x).mean


class PCALatentGPR:
    """PCA output compression followed by independent latent-space GPRs."""

    def __init__(
        self,
        *,
        kernel: str,
        explained_variance: float,
        alpha: float,
        n_restarts_optimizer: int,
        random_state: int,
    ) -> None:
        if not 0.0 < explained_variance <= 1.0:
            raise ValueError("explained_variance must be in (0, 1].")
        self.kernel_name = kernel
        self.explained_variance = float(explained_variance)
        self.alpha = float(alpha)
        self.n_restarts_optimizer = int(n_restarts_optimizer)
        self.random_state = int(random_state)
        self.pca: PCA | None = None
        self.models: list[GaussianProcessRegressor] = []
        self.reconstruction_variance: np.ndarray | None = None

    @property
    def retained_components(self) -> int:
        if self.pca is None:
            raise RuntimeError("PCALatentGPR has not been fitted.")
        return int(self.pca.n_components_)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "PCALatentGPR":
        x = np.asarray(x_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
            raise ValueError("PCA-GPR training arrays must be matching 2D matrices.")
        self.pca = PCA(n_components=self.explained_variance, svd_solver="full")
        latent = self.pca.fit_transform(y)
        reconstruction = self.pca.inverse_transform(latent)
        self.reconstruction_variance = np.maximum(
            np.mean((y - reconstruction) ** 2, axis=0),
            1e-12,
        )

        template = build_kernel(self.kernel_name, x.shape[1])
        self.models = []
        for latent_index in range(latent.shape[1]):
            model = GaussianProcessRegressor(
                kernel=deepcopy(template),
                alpha=self.alpha,
                normalize_y=True,
                n_restarts_optimizer=self.n_restarts_optimizer,
                random_state=self.random_state + latent_index,
            )
            model.fit(x, latent[:, latent_index])
            self.models.append(model)
        return self

    def predict_distribution(self, x: np.ndarray) -> GaussianPrediction:
        if self.pca is None or self.reconstruction_variance is None or not self.models:
            raise RuntimeError("PCALatentGPR has not been fitted.")
        features = np.asarray(x, dtype=np.float64)
        latent_means = []
        latent_variances = []
        for model in self.models:
            mean, standard_deviation = model.predict(features, return_std=True)
            latent_means.append(mean)
            latent_variances.append(np.maximum(standard_deviation**2, 1e-12))
        mean_matrix = np.column_stack(latent_means)
        variance_matrix = np.column_stack(latent_variances)
        output_mean = self.pca.inverse_transform(mean_matrix)
        output_variance = (
            variance_matrix @ (self.pca.components_**2)
            + self.reconstruction_variance[None, :]
        )
        return GaussianPrediction(
            mean=output_mean,
            variance=np.maximum(output_variance, 1e-12),
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_distribution(x).mean


class IndependentXGBoost:
    """One deterministic XGBoost regressor per standardized output."""

    def __init__(self, *, params: dict[str, Any], random_state: int) -> None:
        self.params = dict(params)
        self.random_state = int(random_state)
        self.models: list[Any] = []

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_validation: np.ndarray,
        y_validation: np.ndarray,
    ) -> "IndependentXGBoost":
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise RuntimeError("XGBoost is required for the XGBoost baseline.") from error

        x_fit = np.asarray(x_train, dtype=np.float64)
        y_fit = np.asarray(y_train, dtype=np.float64)
        x_val = np.asarray(x_validation, dtype=np.float64)
        y_val = np.asarray(y_validation, dtype=np.float64)
        self.models = []
        for output_index in range(y_fit.shape[1]):
            model = XGBRegressor(
                **self.params,
                objective="reg:squarederror",
                random_state=self.random_state + output_index,
                n_jobs=1,
                tree_method="hist",
                early_stopping_rounds=50,
            )
            model.fit(
                x_fit,
                y_fit[:, output_index],
                eval_set=[(x_val, y_val[:, output_index])],
                verbose=False,
            )
            self.models.append(model)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("IndependentXGBoost has not been fitted.")
        features = np.asarray(x, dtype=np.float64)
        return np.column_stack([model.predict(features) for model in self.models])
