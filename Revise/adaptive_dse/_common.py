"""Small shared validators and constants for the A-DSE package."""

from __future__ import annotations

import numpy as np


INPUT_DIM = 16
OBJECTIVE_COUNT = 2
ENSEMBLE_SIZE = 5


class FEEvaluationFailure(RuntimeError):
    """A selected design did not produce a usable finite-element result."""

    def __init__(self, point: np.ndarray, message: str) -> None:
        self.point = np.asarray(point, dtype=np.float64).reshape(-1).copy()
        super().__init__(message)


def finite_matrix(
    values: np.ndarray,
    columns: int,
    name: str,
    *,
    dtype=np.float64,
) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != columns or len(array) == 0:
        raise ValueError(f"{name} must have shape (N, {columns}), with N > 0.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def finite_vector(values: np.ndarray, length: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != length:
        raise ValueError(f"{name} must contain {length} values.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def maximum_normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    maxima = array.max(axis=1, keepdims=True)
    normalized = np.zeros_like(array)
    np.divide(array, maxima, out=normalized, where=maxima > 1e-15)
    return normalized
