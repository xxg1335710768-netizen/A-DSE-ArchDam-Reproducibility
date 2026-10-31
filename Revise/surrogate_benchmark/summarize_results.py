"""Aggregate the five-seed surrogate benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .config import RESULTS_DIR, SEEDS
from .run_benchmark import _refresh_summary_files


BOOTSTRAP_REPEATS = 2_000
BOOTSTRAP_SEED = 91_021
PROBABILISTIC_MODELS = ("gpr", "pca_gpr", "ensemble_mmoe")
CORE_MODELS = (
    "gpr",
    "xgboost",
    "single_mlp",
    "pca_gpr",
    "multitask_mlp",
    "deterministic_mmoe",
    "ensemble_mmoe",
)


def _require_complete_results() -> None:
    missing: list[str] = []
    invalid: list[str] = []
    for seed in SEEDS:
        for model in CORE_MODELS:
            result_dir = RESULTS_DIR / f"seed_{seed}" / model
            for filename in (
                "run_metadata.json",
                "group_metrics.csv",
                "per_output_metrics.csv",
                "test_predictions.npz",
            ):
                if not (result_dir / filename).is_file():
                    missing.append(f"seed_{seed}/{model}/{filename}")
            model_filename = (
                "fitted_model.joblib"
                if model in {"gpr", "pca_gpr", "xgboost"}
                else "fitted_model.pt"
            )
            if not (result_dir / model_filename).is_file():
                missing.append(f"seed_{seed}/{model}/{model_filename}")
            if model in PROBABILISTIC_MODELS:
                for filename in ("probability_metrics.csv", "coverage_metrics.csv"):
                    if not (result_dir / filename).is_file():
                        missing.append(f"seed_{seed}/{model}/{filename}")
            prediction_path = result_dir / "test_predictions.npz"
            if prediction_path.is_file():
                with np.load(prediction_path) as arrays:
                    required = {
                        "y_true_standardized",
                        "mean_standardized",
                        "y_true_physical",
                        "mean_physical",
                    }
                    if model in PROBABILISTIC_MODELS:
                        required.update(
                            {
                                "variance_standardized",
                                "calibrated_variance_standardized",
                                "variance_scale",
                            }
                        )
                    if model == "ensemble_mmoe":
                        required.update(
                            {
                                "model_variance_standardized",
                                "residual_variance_standardized",
                            }
                        )
                    absent = sorted(required - set(arrays.files))
                    if absent:
                        invalid.append(
                            f"seed_{seed}/{model}: missing arrays {absent}"
                        )
                    for key in required.intersection(arrays.files):
                        values = arrays[key]
                        expected_shape = (41,) if key == "variance_scale" else (80, 41)
                        if values.shape != expected_shape or not np.isfinite(values).all():
                            invalid.append(
                                f"seed_{seed}/{model}: invalid {key} shape/values"
                            )
                        if key == "residual_variance_standardized":
                            invalid_variance = np.any(values < 0.0)
                        else:
                            invalid_variance = "variance" in key and np.any(
                                values <= 0.0
                            )
                        if invalid_variance:
                            invalid.append(
                                f"seed_{seed}/{model}: non-positive {key}"
                            )
    if missing:
        preview = "\n".join(missing[:20])
        raise RuntimeError(
            f"Benchmark is incomplete ({len(missing)} missing files):\n{preview}"
        )
    if invalid:
        preview = "\n".join(invalid[:20])
        raise RuntimeError(
            f"Benchmark has invalid prediction artifacts ({len(invalid)}):\n{preview}"
        )


def _read_seed_tables(filename: str, models: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        for model in models:
            path = RESULTS_DIR / f"seed_{seed}" / model / filename
            frames.append(pd.read_csv(path).assign(seed=seed, model=model))
    return pd.concat(frames, ignore_index=True)


def _aggregate(
    table: pd.DataFrame,
    group_columns: list[str],
    value_columns: list[str],
) -> pd.DataFrame:
    grouped = table.groupby(group_columns, sort=False)[value_columns].agg(
        ["mean", "std"]
    )
    grouped.columns = [f"{value}_{stat}" for value, stat in grouped.columns]
    return grouped.reset_index()


def _bootstrap_mean_interval(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPEATS, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return float(lower), float(upper)


def _paired_accuracy(accuracy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    proposed = accuracy[accuracy["model"] == "ensemble_mmoe"]
    for comparator in CORE_MODELS:
        if comparator == "ensemble_mmoe":
            continue
        reference = accuracy[accuracy["model"] == comparator]
        for group in ("Volu", "St", "stress_39", "all_41"):
            left = proposed[proposed["group"] == group]
            right = reference[reference["group"] == group]
            paired = left.merge(right, on=["seed", "group"], suffixes=("_new", "_ref"))
            delta_r2 = paired["r2_macro_new"].to_numpy() - paired[
                "r2_macro_ref"
            ].to_numpy()
            delta_nrmse = paired["nrmse_macro_ref"].to_numpy() - paired[
                "nrmse_macro_new"
            ].to_numpy()
            r2_ci = _bootstrap_mean_interval(delta_r2)
            nrmse_ci = _bootstrap_mean_interval(delta_nrmse)
            rows.append(
                {
                    "comparator": comparator,
                    "group": group,
                    "delta_r2_mean": float(delta_r2.mean()),
                    "delta_r2_ci95_lower": r2_ci[0],
                    "delta_r2_ci95_upper": r2_ci[1],
                    "delta_nrmse_improvement_mean": float(delta_nrmse.mean()),
                    "delta_nrmse_ci95_lower": nrmse_ci[0],
                    "delta_nrmse_ci95_upper": nrmse_ci[1],
                    "ensemble_wins_r2": int(np.sum(delta_r2 > 0.0)),
                    "ensemble_wins_nrmse": int(np.sum(delta_nrmse > 0.0)),
                    "paired_seeds": len(paired),
                }
            )
    return pd.DataFrame(rows)


def _paired_probability(probability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    proposed = probability[probability["model"] == "ensemble_mmoe"]
    for comparator in ("gpr", "pca_gpr"):
        reference = probability[probability["model"] == comparator]
        for group in ("Volu", "St", "stress_39", "all_41"):
            paired = proposed[proposed["group"] == group].merge(
                reference[reference["group"] == group],
                on=["seed", "group"],
                suffixes=("_new", "_ref"),
            )
            delta = paired["nll_calibrated_macro_ref"].to_numpy() - paired[
                "nll_calibrated_macro_new"
            ].to_numpy()
            ci = _bootstrap_mean_interval(delta)
            rows.append(
                {
                    "comparator": comparator,
                    "group": group,
                    "calibrated_nll_improvement_mean": float(delta.mean()),
                    "ci95_lower": ci[0],
                    "ci95_upper": ci[1],
                    "ensemble_wins": int(np.sum(delta > 0.0)),
                    "paired_seeds": len(paired),
                }
            )
    return pd.DataFrame(rows)


def _uncertainty_score(variance: np.ndarray) -> np.ndarray:
    objective_std = np.sqrt(np.maximum(variance[:, :2], 1e-12))
    return np.sqrt(objective_std[:, 0] * objective_std[:, 1])


def _uncertainty_utility() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for model in PROBABILISTIC_MODELS:
            path = RESULTS_DIR / f"seed_{seed}" / model / "test_predictions.npz"
            with np.load(path) as arrays:
                true_error = np.mean(
                    np.abs(
                        arrays["y_true_standardized"][:, :2]
                        - arrays["mean_standardized"][:, :2]
                    ),
                    axis=1,
                )
                score_sources = {
                    f"{model}_total": arrays["variance_standardized"]
                }
                if model == "ensemble_mmoe":
                    score_sources["ensemble_mmoe_model"] = arrays[
                        "model_variance_standardized"
                    ]
                for score_name, variance in score_sources.items():
                    score = _uncertainty_score(variance)
                    high_error = true_error >= np.quantile(true_error, 0.8)
                    top_count = max(1, int(np.ceil(0.2 * len(score))))
                    top_indices = np.argsort(score)[-top_count:]
                    enrichment = true_error[top_indices].mean() / true_error.mean()
                    rows.append(
                        {
                            "seed": seed,
                            "score": score_name,
                            "spearman_error": float(spearmanr(score, true_error).statistic),
                            "auroc_high_error": float(roc_auc_score(high_error, score)),
                            "top20_error_enrichment": float(enrichment),
                        }
                    )
    return pd.DataFrame(rows)


def _aggregate_uncertainty_utility(utility: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = (
        "spearman_error",
        "auroc_high_error",
        "top20_error_enrichment",
    )
    for score, frame in utility.groupby("score", sort=False):
        row: dict[str, float | str] = {"score": score}
        for metric in metrics:
            values = frame[metric].to_numpy(dtype=float)
            lower, upper = _bootstrap_mean_interval(values)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_ci95_lower"] = lower
            row[f"{metric}_ci95_upper"] = upper
        rows.append(row)
    return pd.DataFrame(rows)


def _selected_configurations() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for model in CORE_MODELS:
            path = RESULTS_DIR / f"seed_{seed}" / model / "run_metadata.json"
            with path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)
            diagnostics = metadata.get("model_diagnostics", {})
            rows.append(
                {
                    "seed": seed,
                    "model": model,
                    "selected_config_index": metadata["selected_config_index"],
                    "selected_config": json.dumps(
                        metadata["selected_config"], ensure_ascii=False, sort_keys=True
                    ),
                    "parameter_count_total": diagnostics.get("parameter_count_total"),
                    "retained_pca_components": diagnostics.get(
                        "retained_pca_components"
                    ),
                    "best_epoch_per_member": json.dumps(
                        diagnostics.get("best_epoch_per_member"), ensure_ascii=False
                    ),
                }
            )
    return pd.DataFrame(rows)


def _control_stress_table() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for model in CORE_MODELS:
            path = RESULTS_DIR / f"seed_{seed}" / model / "run_metadata.json"
            with path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)
            rows.append(
                {
                    "seed": seed,
                    "model": model,
                    **metadata["control_stress_metrics"],
                }
            )
    return pd.DataFrame(rows)


def _runtime_table() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for model in CORE_MODELS:
            result_dir = RESULTS_DIR / f"seed_{seed}" / model
            with (result_dir / "run_metadata.json").open(
                "r", encoding="utf-8"
            ) as file:
                metadata = json.load(file)
            selection = pd.read_csv(result_dir / "model_selection.csv")
            selected_mask = selection["selected"]
            if selected_mask.dtype != bool:
                selected_mask = selected_mask.astype(str).str.lower().eq("true")
            selected_rows = selection[selected_mask]
            if len(selected_rows) != 1:
                raise RuntimeError(f"Expected one selected configuration for {seed}/{model}.")
            selected_fit = float(selected_rows.iloc[0]["fit_seconds"])
            final_fit = float(metadata["final_model_fit_seconds"])
            selected_member_reused = bool(
                metadata.get("selected_search_member_reused", True)
            )
            if model == "ensemble_mmoe":
                additional_members = (
                    max(final_fit - selected_fit, 0.0)
                    if selected_member_reused
                    else final_fit
                )
            else:
                additional_members = 0.0
            search_seconds = float(metadata["search_seconds"])
            rows.append(
                {
                    "seed": seed,
                    "model": model,
                    "search_seconds": search_seconds,
                    "selected_candidate_fit_seconds": selected_fit,
                    "additional_ensemble_members_seconds": additional_members,
                    "complete_workflow_seconds": search_seconds + additional_members,
                    "test_inference_seconds_median": float(
                        metadata["test_inference_seconds_median"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _calibration_summary(coverage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, model, variance), frame in coverage.groupby(
        ["seed", "model", "variance"], sort=False
    ):
        ordered = frame.sort_values("nominal")
        row_95 = ordered[np.isclose(ordered["nominal"], 0.95)]
        if len(row_95) != 1:
            raise RuntimeError(f"Missing unique 95% coverage row for {seed}/{model}.")
        selected = row_95.iloc[0]
        rows.append(
            {
                "seed": int(seed),
                "model": model,
                "variance": variance,
                "mean_absolute_calibration_error": float(
                    np.mean(np.abs(ordered["coverage_macro"] - ordered["nominal"]))
                ),
                "picp95_macro": float(selected["coverage_macro"]),
                "picp95_objectives": float(selected["coverage_objectives"]),
                "picp95_stress": float(selected["coverage_stress"]),
                "mpiw95_standardized": float(selected["width_macro_standardized"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    _require_complete_results()
    _refresh_summary_files(RESULTS_DIR)

    accuracy = _read_seed_tables("group_metrics.csv", CORE_MODELS)
    probability = _read_seed_tables("probability_metrics.csv", PROBABILISTIC_MODELS)
    coverage = _read_seed_tables("coverage_metrics.csv", PROBABILISTIC_MODELS)
    runtime = _runtime_table()
    runtime.to_csv(RESULTS_DIR / "runtime_by_seed.csv", index=False)

    _aggregate(
        accuracy,
        ["model", "group"],
        ["r2_macro", "r2_minimum", "rmse_macro_physical", "nrmse_macro"],
    ).to_csv(RESULTS_DIR / "aggregate_accuracy.csv", index=False)
    _aggregate(
        probability,
        ["model", "group"],
        ["nll_raw_macro", "nll_calibrated_macro"],
    ).to_csv(RESULTS_DIR / "aggregate_probability.csv", index=False)
    _aggregate(
        coverage,
        ["model", "variance", "nominal"],
        [
            "coverage_macro",
            "coverage_objectives",
            "coverage_stress",
            "width_macro_standardized",
        ],
    ).to_csv(RESULTS_DIR / "aggregate_coverage.csv", index=False)
    calibration = _calibration_summary(coverage)
    calibration.to_csv(RESULTS_DIR / "calibration_summary_by_seed.csv", index=False)
    _aggregate(
        calibration,
        ["model", "variance"],
        [
            "mean_absolute_calibration_error",
            "picp95_macro",
            "picp95_objectives",
            "picp95_stress",
            "mpiw95_standardized",
        ],
    ).to_csv(RESULTS_DIR / "aggregate_calibration_summary.csv", index=False)
    _aggregate(
        runtime,
        ["model"],
        [
            "search_seconds",
            "selected_candidate_fit_seconds",
            "additional_ensemble_members_seconds",
            "complete_workflow_seconds",
            "test_inference_seconds_median",
        ],
    ).to_csv(RESULTS_DIR / "aggregate_runtime.csv", index=False)

    _paired_accuracy(accuracy).to_csv(
        RESULTS_DIR / "paired_accuracy_vs_ensemble_mmoe.csv", index=False
    )
    _paired_probability(probability).to_csv(
        RESULTS_DIR / "paired_probability_vs_ensemble_mmoe.csv", index=False
    )
    utility = _uncertainty_utility()
    utility.to_csv(RESULTS_DIR / "uncertainty_utility_by_seed.csv", index=False)
    _aggregate_uncertainty_utility(utility).to_csv(
        RESULTS_DIR / "aggregate_uncertainty_utility.csv", index=False
    )
    _selected_configurations().to_csv(
        RESULTS_DIR / "selected_configurations.csv", index=False
    )
    control_stress = _control_stress_table()
    control_stress.to_csv(RESULTS_DIR / "control_stress_by_seed.csv", index=False)
    _aggregate(
        control_stress,
        ["model"],
        [
            "maximum_s1_mae",
            "maximum_s1_rmse",
            "minimum_s3_mae",
            "minimum_s3_rmse",
        ],
    ).to_csv(RESULTS_DIR / "aggregate_control_stress.csv", index=False)
    print("Benchmark summaries written to", RESULTS_DIR, flush=True)


if __name__ == "__main__":
    main()
