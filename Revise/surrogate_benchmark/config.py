"""Frozen configuration for the surrogate-model benchmark."""

from __future__ import annotations

from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = BENCHMARK_DIR.parent
DATASET_PATH = PACKAGE_DIR / "dataset" / "SurrogateDataset.xlsx"
RESULTS_DIR = BENCHMARK_DIR / "results"
SPLIT_DIR = BENCHMARK_DIR / "splits"

SEEDS = (2025, 2026, 2027, 2028, 2029)
TRAIN_COUNT = 256
VALIDATION_COUNT = 64
CALIBRATION_COUNT = 80
DEVELOPMENT_COUNT = TRAIN_COUNT + VALIDATION_COUNT + CALIBRATION_COUNT
TEST_COUNT = 80
INPUT_COUNT = 16
OUTPUT_COUNT = 41
OBJECTIVE_INDICES = (0, 1)
STRESS_START_INDEX = 2

NEURAL_TRAINING = {
    "max_epochs": 500,
    "patience": 30,
    "min_delta": 1e-6,
    "batch_size": 32,
    "learning_rate": 3e-4,
    "weight_decay": 5e-2,
    "gradient_clip": 5.0,
}

GPR_GRID = tuple(
    {"kernel": kernel, "alpha": alpha, "n_restarts_optimizer": 2}
    for kernel in ("rbf", "matern32", "matern52")
    for alpha in (1e-6, 1e-4, 1e-2)
)

PCA_GPR_GRID = tuple(
    {
        "kernel": kernel,
        "explained_variance": explained_variance,
        "alpha": 1e-6,
        "n_restarts_optimizer": 2,
    }
    for kernel in ("rbf", "matern32", "matern52")
    for explained_variance in (0.95, 0.99, 0.995)
)

XGBOOST_GRID = tuple(
    {
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "n_estimators": 1000,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    }
    for max_depth in (3, 5, 7)
    for learning_rate in (0.03, 0.05, 0.10)
)

MLP_GRID = tuple(
    {"hidden_dims": hidden_dims, "dropout": dropout}
    for hidden_dims in ((64, 64), (128, 64), (128, 128))
    for dropout in (0.0, 0.1, 0.2)
)

# MC Dropout is an uncertainty-estimation method rather than only a training
# regularizer.  Zero dropout would make its predictive variance identically
# zero, so the probability benchmark uses a separate positive-rate
# grid with the same nine-configuration budget as the other model families.
MC_DROPOUT_GRID = tuple(
    {"hidden_dims": hidden_dims, "dropout": dropout}
    for hidden_dims in ((64, 64), (128, 64), (128, 128))
    for dropout in (0.1, 0.2, 0.3)
)
MC_DROPOUT_SAMPLES = 100

MMOE_GRID = tuple(
    {
        "num_experts": num_experts,
        "expert_hidden": expert_hidden,
        "tower_hidden": tower_hidden,
    }
    for num_experts in (2, 4, 8)
    for expert_hidden, tower_hidden in ((64, 32), (128, 32), (128, 64))
)
