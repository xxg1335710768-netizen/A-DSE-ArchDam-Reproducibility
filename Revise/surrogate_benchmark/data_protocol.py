"""Dataset validation, leakage-safe splits and standardization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CALIBRATION_COUNT,
    DATASET_PATH,
    DEVELOPMENT_COUNT,
    INPUT_COUNT,
    OUTPUT_COUNT,
    SEEDS,
    TEST_COUNT,
    TRAIN_COUNT,
    VALIDATION_COUNT,
)


@dataclass(frozen=True)
class DatasetBundle:
    development_x: np.ndarray
    development_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]


@dataclass(frozen=True)
class SplitIndices:
    seed: int
    train: np.ndarray
    validation: np.ndarray
    calibration: np.ndarray

    def validate(self) -> None:
        combined = np.concatenate((self.train, self.validation, self.calibration))
        if len(self.train) != TRAIN_COUNT:
            raise ValueError(f"Expected {TRAIN_COUNT} training rows.")
        if len(self.validation) != VALIDATION_COUNT:
            raise ValueError(f"Expected {VALIDATION_COUNT} validation rows.")
        if len(self.calibration) != CALIBRATION_COUNT:
            raise ValueError(f"Expected {CALIBRATION_COUNT} calibration rows.")
        if len(np.unique(combined)) != DEVELOPMENT_COUNT:
            raise ValueError("Train, validation and calibration indices overlap.")
        if set(combined.tolist()) != set(range(DEVELOPMENT_COUNT)):
            raise ValueError("Split indices do not cover the 400 development rows.")


@dataclass(frozen=True)
class ArrayStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "ArrayStandardizer":
        array = np.asarray(values, dtype=np.float64)
        mean = np.mean(array, axis=0)
        scale = np.std(array, axis=0, ddof=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.scale + self.mean


@dataclass(frozen=True)
class PreparedSplit:
    indices: SplitIndices
    x_scaler: ArrayStandardizer
    y_scaler: ArrayStandardizer
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_calibration: np.ndarray
    y_calibration: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_dataset(path: Path = DATASET_PATH) -> DatasetBundle:
    train_x = pd.read_excel(path, sheet_name="TrainX")
    train_y = pd.read_excel(path, sheet_name="TrainY")
    test_x = pd.read_excel(path, sheet_name="ValX")
    test_y = pd.read_excel(path, sheet_name="ValY")

    expected_shapes = {
        "TrainX": (DEVELOPMENT_COUNT, INPUT_COUNT),
        "TrainY": (DEVELOPMENT_COUNT, OUTPUT_COUNT),
        "ValX": (TEST_COUNT, INPUT_COUNT),
        "ValY": (TEST_COUNT, OUTPUT_COUNT),
    }
    actual_shapes = {
        "TrainX": train_x.shape,
        "TrainY": train_y.shape,
        "ValX": test_x.shape,
        "ValY": test_y.shape,
    }
    if actual_shapes != expected_shapes:
        raise ValueError(f"Unexpected dataset shapes: {actual_shapes}")
    if train_x.columns.tolist() != test_x.columns.tolist():
        raise ValueError("TrainX and ValX columns differ.")
    if train_y.columns.tolist() != test_y.columns.tolist():
        raise ValueError("TrainY and ValY columns differ.")
    if train_y.columns[:2].tolist() != ["Volu", "St"]:
        raise ValueError("The first two outputs must be Volu and St.")

    arrays = [
        train_x.to_numpy(dtype=np.float64),
        train_y.to_numpy(dtype=np.float64),
        test_x.to_numpy(dtype=np.float64),
        test_y.to_numpy(dtype=np.float64),
    ]
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("The dataset contains NaN or infinite values.")
    development_x, development_y, independent_x, independent_y = arrays
    if len(np.unique(development_x, axis=0)) != DEVELOPMENT_COUNT:
        raise ValueError("Duplicate design vectors occur in the development set.")
    if len(np.unique(independent_x, axis=0)) != TEST_COUNT:
        raise ValueError("Duplicate design vectors occur in the independent test set.")
    combined_x = np.concatenate((development_x, independent_x), axis=0)
    if len(np.unique(combined_x, axis=0)) != DEVELOPMENT_COUNT + TEST_COUNT:
        raise ValueError("Development and independent test designs overlap.")
    if not np.allclose(independent_x[0], 1.0):
        raise ValueError("Independent test sample 0 is not [1, ..., 1].")

    return DatasetBundle(
        development_x=development_x,
        development_y=development_y,
        test_x=independent_x,
        test_y=independent_y,
        input_names=tuple(str(name) for name in train_x.columns),
        output_names=tuple(str(name) for name in train_y.columns),
    )


def make_split(seed: int) -> SplitIndices:
    permutation = np.random.default_rng(seed).permutation(DEVELOPMENT_COUNT)
    train_end = TRAIN_COUNT
    validation_end = train_end + VALIDATION_COUNT
    split = SplitIndices(
        seed=int(seed),
        train=np.sort(permutation[:train_end]),
        validation=np.sort(permutation[train_end:validation_end]),
        calibration=np.sort(permutation[validation_end:]),
    )
    split.validate()
    return split


def prepare_split(dataset: DatasetBundle, split: SplitIndices) -> PreparedSplit:
    split.validate()
    x_scaler = ArrayStandardizer.fit(dataset.development_x[split.train])
    y_scaler = ArrayStandardizer.fit(dataset.development_y[split.train])
    return PreparedSplit(
        indices=split,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        x_train=x_scaler.transform(dataset.development_x[split.train]),
        y_train=y_scaler.transform(dataset.development_y[split.train]),
        x_validation=x_scaler.transform(dataset.development_x[split.validation]),
        y_validation=y_scaler.transform(dataset.development_y[split.validation]),
        x_calibration=x_scaler.transform(dataset.development_x[split.calibration]),
        y_calibration=y_scaler.transform(dataset.development_y[split.calibration]),
        x_test=x_scaler.transform(dataset.test_x),
        y_test=y_scaler.transform(dataset.test_y),
    )


def split_manifest() -> pd.DataFrame:
    rows: list[dict[str, int | str]] = []
    for seed in SEEDS:
        split = make_split(seed)
        for role, indices in (
            ("train", split.train),
            ("validation", split.validation),
            ("calibration", split.calibration),
        ):
            rows.extend(
                {"seed": seed, "development_index": int(index), "role": role}
                for index in indices
            )
    return pd.DataFrame(rows)
