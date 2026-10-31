"""Check the released dataset, fixed model, key results, and excluded files."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Revise" / "dataset" / "SurrogateDataset.xlsx"
MODEL = (
    ROOT
    / "Revise"
    / "surrogate_benchmark"
    / "results"
    / "seed_2025"
    / "ensemble_mmoe"
    / "fitted_model.pt"
)
EXPECTED_DATASET_SHA256 = (
    "d03a5af15e3ac03dd018f6922578473a206eadfb09be40621104a944e4a35c41"
)
EXPECTED_MODEL_SHA256 = (
    "a4388e11d8f6ce6b87aa5e5852202d99a58e44e0066272b42674ee1bc09632af"
)
EXPECTED_SHEETS = {
    "TrainX": (400, 16),
    "TrainY": (400, 41),
    "ValX": (80, 16),
    "ValY": (80, 41),
}
FORBIDDEN_SUFFIXES = {".3dm", ".db", ".out", ".err", ".bat", ".cmd"}
FORBIDDEN_NAMES = {"fe_evaluator.py", "main_Dataset.py", "PATH.py"}
EXPECTED_TABLE5_MODELS = {
    "GPR",
    "XGB",
    "MLP",
    "PCA-GPR",
    "MT-MLP",
    "MMoE",
    "DE-MMoE",
}
EXPECTED_SEEDS = {2025, 2026, 2027, 2028, 2029}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if sha256(DATASET) != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Dataset checksum differs from the formal artifact.")
    if sha256(MODEL) != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Fixed DE-MMoE checksum differs from the formal artifact.")

    workbook = pd.ExcelFile(DATASET)
    if workbook.sheet_names != list(EXPECTED_SHEETS):
        raise RuntimeError(f"Unexpected workbook sheets: {workbook.sheet_names}")
    for sheet_name, expected_shape in EXPECTED_SHEETS.items():
        frame = pd.read_excel(DATASET, sheet_name=sheet_name)
        if frame.shape != expected_shape:
            raise RuntimeError(
                f"{sheet_name} shape {frame.shape} != {expected_shape}."
            )
        if frame.isna().any().any():
            raise RuntimeError(f"{sheet_name} contains missing values.")

    forbidden = [
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_file()
        and (path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name in FORBIDDEN_NAMES)
    ]
    if forbidden:
        raise RuntimeError(f"Excluded project-specific files found: {forbidden}")

    required_results = (
        ROOT / "Revise" / "surrogate_benchmark" / "public_results" / "table5" / "table5_summary.csv",
        ROOT / "Revise" / "adaptive_dse" / "public_results" / "optimizer_comparison" / "optimizer_metrics_summary.csv",
        ROOT / "Revise" / "adaptive_dse" / "public_results" / "pareto_fe_validation" / "validation_summary.csv",
    )
    missing = [path.relative_to(ROOT) for path in required_results if not path.is_file()]
    if missing:
        raise RuntimeError(f"Required public results are missing: {missing}")

    table5_path = (
        ROOT
        / "Revise"
        / "surrogate_benchmark"
        / "public_results"
        / "table5"
        / "table5_by_seed.csv"
    )
    table5 = pd.read_csv(table5_path)
    if set(table5["model"]) != EXPECTED_TABLE5_MODELS:
        raise RuntimeError("Table 5 model set is incomplete.")
    for model, group in table5.groupby("model"):
        if set(group["seed"]) != EXPECTED_SEEDS or len(group) != len(EXPECTED_SEEDS):
            raise RuntimeError(f"Incomplete repeated results for {model}.")

    required_model_code = (
        ROOT / "Revise" / "surrogate_benchmark" / "models_sklearn.py",
        ROOT / "Revise" / "surrogate_benchmark" / "models_torch.py",
        ROOT / "Revise" / "surrogate_benchmark" / "run_benchmark.py",
        ROOT / "Revise" / "probabilistic_mmoe.py",
    )
    missing_code = [path.relative_to(ROOT) for path in required_model_code if not path.is_file()]
    if missing_code:
        raise RuntimeError(f"Required model code is missing: {missing_code}")

    print("Public package verification passed.")
    print("Dataset sheets:", EXPECTED_SHEETS)
    print("Dataset SHA-256:", EXPECTED_DATASET_SHA256)
    print("Model SHA-256:", EXPECTED_MODEL_SHA256)
    print("Table 5 repetitions: 7 models x 5 seeds")


if __name__ == "__main__":
    main()
