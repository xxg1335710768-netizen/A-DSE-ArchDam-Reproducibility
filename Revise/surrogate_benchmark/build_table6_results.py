"""Build manuscript Tables 6 and 7 from completed benchmark runs.

Predictive metrics are reconstructed from the per-seed result directories.
Fitting times are joined from the separately measured fixed-configuration
runtime benchmark; hyperparameter-search time is never substituted for fitting
time.  No model is retrained and no prior result is overwritten.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .config import SEEDS
from .data_protocol import load_dataset, make_split, prepare_split
from .metrics import (
    coverage_table,
    fit_residual_variance_floor,
    gaussian_nll,
    group_regression_metrics,
    per_output_regression_metrics,
)
from .models_torch import DeterministicMMoE


BASE_DIR = Path(__file__).resolve().parent
BASE_RESULTS = BASE_DIR / "results"
# The public benchmark writes every formal model to one result tree.  The
# separate directories used during method development are intentionally not
# part of the release.
MC_RESULTS = BASE_RESULTS
DE_RESULTS = BASE_RESULTS
OUTPUT_DIR = BASE_DIR / "table6_results"
PUBLISHED_RUNTIME_REFERENCE = (
    BASE_DIR / "public_results" / "table6" / "fitting_time_by_seed.csv"
)
RERUN_RUNTIME_REFERENCE = (
    BASE_DIR / "fixed_runtime_all_models" / "all_models_runtime_by_seed.csv"
)

SOURCES = {
    "GPR": (BASE_RESULTS, "gpr", "total predictive variance"),
    "XGB": (BASE_RESULTS, "xgboost", None),
    "MLP": (MC_RESULTS, "single_mlp_mc_dropout", "MC Dropout model variance"),
    "PCA-GPR": (BASE_RESULTS, "pca_gpr", "reconstructed total variance"),
    "MT-MLP": (
        MC_RESULTS,
        "multitask_mlp_mc_dropout",
        "MC Dropout model variance",
    ),
    "DE-MMoE": (DE_RESULTS, "ensemble_mmoe", "ensemble model variance"),
}


def _score(variance: np.ndarray) -> np.ndarray:
    objective_std = np.sqrt(np.maximum(variance[:, :2], 1e-12))
    return np.sqrt(objective_std[:, 0] * objective_std[:, 1])


def _utility(
    y_true: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
) -> dict[str, float]:
    error = np.mean(np.abs(y_true[:, :2] - mean[:, :2]), axis=1)
    score = _score(variance)
    high_error = error >= np.quantile(error, 0.8)
    top_count = max(1, int(np.ceil(0.2 * len(score))))
    top_indices = np.argsort(score)[-top_count:]
    return {
        "spearman_error": float(spearmanr(score, error).statistic),
        "auroc_high_error": float(roc_auc_score(high_error, score)),
        "top20_error_enrichment": float(error[top_indices].mean() / error.mean()),
    }


def _workflow_minutes(
    root: Path,
    stored_name: str,
    seed: int,
    *,
    include_extra_members: bool,
) -> float:
    del root
    labels = {
        "gpr": "GPR",
        "xgboost": "XGB",
        "single_mlp_mc_dropout": "MLP",
        "pca_gpr": "PCA-GPR",
        "multitask_mlp_mc_dropout": "MT-MLP",
        "ensemble_mmoe": "DE-MMoE" if include_extra_members else "MMoE",
    }
    if stored_name not in labels:
        raise ValueError(f"No runtime label for {stored_name!r}.")
    runtime_path = (
        RERUN_RUNTIME_REFERENCE
        if RERUN_RUNTIME_REFERENCE.is_file()
        else PUBLISHED_RUNTIME_REFERENCE
    )
    if not runtime_path.is_file():
        raise FileNotFoundError(
            "The fixed-configuration runtime reference is missing: "
            f"{runtime_path}"
        )
    runtime = pd.read_csv(runtime_path)
    row = runtime.loc[
        (runtime["seed"] == seed) & (runtime["method"] == labels[stored_name])
    ]
    if len(row) != 1:
        raise RuntimeError(
            f"Expected one fitting-time row for seed={seed}, "
            f"model={labels[stored_name]}."
        )
    return float(row.iloc[0]["wall_minutes"])


def _existing_rows() -> tuple[list[dict], list[dict], list[dict]]:
    table5 = []
    probability = []
    utility = []
    dataset = load_dataset()
    for label, (root, stored_name, uncertainty_type) in SOURCES.items():
        for seed in SEEDS:
            result_dir = root / f"seed_{seed}" / stored_name
            group = pd.read_csv(result_dir / "group_metrics.csv")
            all41 = group.loc[group["group"] == "all_41"].iloc[0]
            table5_row = {
                "seed": seed,
                "model": label,
                "r2": float(all41["r2_macro"]),
                "rmse": float(all41["rmse_macro_physical"]),
                "nll": np.nan,
                "time_min": _workflow_minutes(
                    root,
                    stored_name,
                    seed,
                    include_extra_members=label == "DE-MMoE",
                ),
            }
            probability_path = result_dir / "probability_metrics.csv"
            if probability_path.is_file():
                probability_frame = pd.read_csv(probability_path)
                probability_all41 = probability_frame.loc[
                    probability_frame["group"] == "all_41"
                ].iloc[0]
                table5_row["nll"] = float(
                    probability_all41["nll_calibrated_macro"]
                )

                coverage = pd.read_csv(result_dir / "coverage_metrics.csv")
                calibrated = coverage.loc[coverage["variance"] == "calibrated"]
                row95 = calibrated.loc[np.isclose(calibrated["nominal"], 0.95)].iloc[0]
                probability.append(
                    {
                        "seed": seed,
                        "model": label,
                        "nll_raw": float(probability_all41["nll_raw_macro"]),
                        "nll_calibrated": table5_row["nll"],
                        "picp95": float(row95["coverage_macro"]),
                        "mpiw95": float(row95["width_macro_standardized"]),
                        "calibration_error": float(
                            np.mean(
                                np.abs(
                                    calibrated["coverage_macro"]
                                    - calibrated["nominal"]
                                )
                            )
                        ),
                    }
                )

                with np.load(result_dir / "test_predictions.npz") as arrays:
                    if label in {"MLP", "MT-MLP", "DE-MMoE"}:
                        variance = arrays["model_variance_standardized"]
                    else:
                        variance = arrays["variance_standardized"]
                    utility.append(
                        {
                            "seed": seed,
                            "model": label,
                            "uncertainty_type": uncertainty_type,
                            **_utility(
                                arrays["y_true_standardized"],
                                arrays["mean_standardized"],
                                variance,
                            ),
                        }
                    )
            table5.append(table5_row)
    return table5, probability, utility


def _single_mmoe_rows() -> tuple[list[dict], list[dict], list[dict]]:
    dataset = load_dataset()
    table5 = []
    probability = []
    utility = []
    for seed in SEEDS:
        prepared = prepare_split(dataset, make_split(seed))
        result_dir = DE_RESULTS / f"seed_{seed}" / "ensemble_mmoe"
        payload = torch.load(
            result_dir / "fitted_model.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = DeterministicMMoE(
            d_in=16,
            num_tasks=41,
            activation_name="swish",
            **payload["selected_config"],
        )
        model.load_state_dict(payload["state_dicts"][0])
        model.eval()
        with torch.no_grad():
            test_mean = model(
                torch.as_tensor(prepared.x_test, dtype=torch.float32)
            ).numpy()
            calibration_mean = model(
                torch.as_tensor(prepared.x_calibration, dtype=torch.float32)
            ).numpy()
        zero_test_variance = np.full_like(test_mean, 1e-12)
        zero_calibration_variance = np.full_like(calibration_mean, 1e-12)
        residual_floor = fit_residual_variance_floor(
            prepared.y_calibration,
            calibration_mean,
            zero_calibration_variance,
        )
        calibrated_variance = zero_test_variance + residual_floor[None, :]
        mean_physical = prepared.y_scaler.inverse_transform(test_mean)
        per_output = per_output_regression_metrics(
            dataset.test_y,
            mean_physical,
            dataset.output_names,
        )
        all41 = group_regression_metrics(per_output).loc[
            lambda frame: frame["group"] == "all_41"
        ].iloc[0]
        raw_nll = float("nan")
        nll = float(
            np.mean(gaussian_nll(prepared.y_test, test_mean, calibrated_variance))
        )
        coverage = coverage_table(prepared.y_test, test_mean, calibrated_variance)
        row95 = coverage.loc[np.isclose(coverage["nominal"], 0.95)].iloc[0]
        table5.append(
            {
                "seed": seed,
                "model": "MMoE",
                "r2": float(all41["r2_macro"]),
                "rmse": float(all41["rmse_macro_physical"]),
                "nll": nll,
                "time_min": _workflow_minutes(
                    DE_RESULTS,
                    "ensemble_mmoe",
                    seed,
                    include_extra_members=False,
                ),
            }
        )
        probability.append(
            {
                "seed": seed,
                "model": "MMoE",
                "nll_raw": raw_nll,
                "nll_calibrated": nll,
                "picp95": float(row95["coverage_macro"]),
                "mpiw95": float(row95["width_macro_standardized"]),
                "calibration_error": float(
                    np.mean(np.abs(coverage["coverage_macro"] - coverage["nominal"]))
                ),
            }
        )
        # A single mean-only MMoE has no sample-dependent model uncertainty.
    return table5, probability, utility


def _aggregate(frame: pd.DataFrame, values: list[str]) -> pd.DataFrame:
    grouped = frame.groupby("model", sort=False)[values].agg(["mean", "std", "var"])
    grouped.columns = [f"{value}_{stat}" for value, stat in grouped.columns]
    return grouped.reset_index()


def _format_pm(row: pd.Series, value: str, digits: int = 6) -> str:
    if not np.isfinite(row[f"{value}_mean"]):
        return "—"
    return f"{row[f'{value}_mean']:.{digits}f} ± {row[f'{value}_std']:.{digits}f}"


def _markdown(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    torch.set_num_threads(1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table5_rows, probability_rows, _ = _existing_rows()
    single_table5, single_probability, _ = _single_mmoe_rows()
    table5 = pd.DataFrame(table5_rows + single_table5)
    probability = pd.DataFrame(probability_rows + single_probability)
    order = ["GPR", "XGB", "MLP", "PCA-GPR", "MT-MLP", "MMoE", "DE-MMoE"]
    order_map = {name: index for index, name in enumerate(order)}
    for frame in (table5, probability):
        frame["model_order"] = frame["model"].map(order_map)
        frame.sort_values(["model_order", "seed"], inplace=True)
        frame.drop(columns="model_order", inplace=True)

    table5.rename(columns={"time_min": "fitting_time_min"}).to_csv(
        OUTPUT_DIR / "table6_by_seed.csv", index=False
    )
    probability[["seed", "model", "picp95", "mpiw95"]].to_csv(
        OUTPUT_DIR / "table7_by_seed.csv", index=False
    )
    table5_summary = _aggregate(table5, ["r2", "rmse", "nll", "time_min"])
    probability_summary = _aggregate(
        probability,
        ["nll_raw", "nll_calibrated", "picp95", "mpiw95", "calibration_error"],
    )
    table5_summary.to_csv(OUTPUT_DIR / "table6_summary.csv", index=False)
    probability_summary[
        ["model", "picp95_mean", "picp95_std", "mpiw95_mean", "mpiw95_std"]
    ].to_csv(OUTPUT_DIR / "table7_summary.csv", index=False)

    table_rows = []
    for model in order:
        row = table5_summary.loc[table5_summary["model"] == model].iloc[0]
        table_rows.append(
            [
                model,
                _format_pm(row, "r2"),
                _format_pm(row, "rmse"),
                _format_pm(row, "nll"),
                _format_pm(row, "time_min", digits=3),
            ]
        )
    probability_table_rows = []
    for _, row in probability_summary.iterrows():
        probability_table_rows.append(
            [
                str(row["model"]),
                _format_pm(row, "nll_raw"),
                _format_pm(row, "nll_calibrated"),
                _format_pm(row, "picp95"),
                _format_pm(row, "mpiw95"),
                _format_pm(row, "calibration_error"),
            ]
        )
    report = "\n".join(
        [
            "# Manuscript Tables 6 and 7",
            "",
            "## Table 6: surrogate performance",
            "",
            _markdown(
                ["Model", "R²", "RMSE", "Calibrated NLL", "Time (min)"],
                table_rows,
            ),
            "",
            "Values are mean ± sample standard deviation over five development splits; the fixed 80-point test set is shared. Fitting time is wall-clock time for the frozen final configuration and excludes hyperparameter search, data preparation, inference, uncertainty sampling, metrics, and serialization.",
            "",
            "## Table 7: predictive interval quality",
            "",
            _markdown(
                ["Model", "Raw NLL", "Calibrated NLL", "PICP95", "MPIW95", "Calibration error"],
                probability_table_rows,
            ),
        ]
    )
    (OUTPUT_DIR / "TABLE6_TABLE7.md").write_text(
        report + "\n",
        encoding="utf-8",
    )
    print("Tables 6 and 7 written to", OUTPUT_DIR, flush=True)


if __name__ == "__main__":
    main()
