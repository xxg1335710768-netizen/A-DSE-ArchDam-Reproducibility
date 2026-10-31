"""Verify the published dataset, result tables, and experiment settings."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Revise" / "dataset" / "SurrogateDataset.xlsx"
MODEL = ROOT / "Revise" / "surrogate_benchmark" / "results" / "seed_2025" / "ensemble_mmoe" / "fitted_model.pt"
TABLE_DIR = ROOT / "Revise" / "surrogate_benchmark" / "public_results" / "table6"
ABLATION_DIR = ROOT / "Revise" / "adaptive_dse" / "public_results" / "ablation"
OPTIMIZER_DIR = ROOT / "Revise" / "adaptive_dse" / "public_results" / "optimizer_comparison"

EXPECTED_DATASET_SHA256 = "1f9aa8db86de10d3024856d906ef5324724cf1661039ab6b2bfad9bca58ff353"
EXPECTED_MODEL_SHA256 = "a4388e11d8f6ce6b87aa5e5852202d99a58e44e0066272b42674ee1bc09632af"
EXPECTED_SHEETS = {"TrainX": (400, 16), "TrainY": (400, 41), "ValX": (80, 16), "ValY": (80, 41)}
EXPECTED_SEEDS = {2025, 2026, 2027, 2028, 2029}
EXPECTED_MODELS = {"GPR", "XGB", "MLP", "PCA-GPR", "MT-MLP", "MMoE", "DE-MMoE"}
EXPECTED_INTERVAL_MODELS = EXPECTED_MODELS - {"XGB"}
EXPECTED_BUDGETS = {2400, 5000, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 80000, 100000, 120000}

EXPECTED_TABLE6 = {
    "GPR": (0.9875153717345938, 0.0004010777225135241, 0.03588171310962018, 0.0003055393034421, -1.0388817968498063, 0.004671958068989354, 3.9508495859999675, 0.2949298087747503),
    "XGB": (0.9347813023710888, 0.002795204469623933, 0.13049705441000795, 0.001966933877610731, np.nan, np.nan, 1.0420599296665751, 0.1342008885687296),
    "MLP": (0.9839401313892822, 0.0003851334322175297, 0.04618984598184304, 0.0016162589668754094, -0.8623404757246389, 0.014465912609347549, 2.3856654176667025, 0.22535308815113164),
    "PCA-GPR": (0.9861768164330373, 0.00031610761074507717, 0.0495669402004457, 0.0015172588652907824, -0.9042089438092591, 0.014394799611036659, 1.7208331650000765, 0.06784389232745733),
    "MT-MLP": (0.9845549894892824, 0.0003313481930864395, 0.054642898946893136, 0.0016226750844381638, -0.8228056536748625, 0.027078012653873322, 1.1977023543334022, 0.13861778268621486),
    "MMoE": (0.9843052157500946, 0.0003707398990239705, 0.04511773855902399, 0.000661750232294514, -0.8505100021860776, 0.007267438356597517, 0.6319989656666681, 0.1239022951892355),
    "DE-MMoE": (0.986493292115124, 0.0003128173122984424, 0.038863759224832484, 0.0006329837695914091, -0.9823278325115806, 0.015459403122314605, 0.7426582723334043, 0.10532793695309667),
}
EXPECTED_TABLE7 = {
    "GPR": (0.9396341463414635, 0.005752427519546775, 0.37714685030828, 0.01491561311250848),
    "MLP": (0.9409756097560976, 0.005528305832509009, 0.4481992112525444, 0.011224274878722108),
    "PCA-GPR": (0.9488414634146342, 0.004275907840958033, 0.4076673192746497, 0.009987645964901022),
    "MT-MLP": (0.936829268292683, 0.007327862979594363, 0.4428493896156163, 0.014584326201478696),
    "MMoE": (0.9457317073170731, 0.004496305245046607, 0.4339581343415869, 0.009476400873405517),
    "DE-MMoE": (0.9476829268292685, 0.002463671201321564, 0.3966870077434711, 0.004719915085173363),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_close(actual: float, expected: float, label: str) -> None:
    if np.isnan(expected):
        if not np.isnan(actual):
            raise RuntimeError(f"{label}: expected a missing value, got {actual}.")
    elif not np.isclose(actual, expected, rtol=1e-10, atol=1e-12):
        raise RuntimeError(f"{label}: {actual} != {expected}.")


def verify_table6() -> None:
    repeated = pd.read_csv(TABLE_DIR / "table6_by_seed.csv")
    summary = pd.read_csv(TABLE_DIR / "table6_summary.csv")
    timing = pd.read_csv(TABLE_DIR / "fitting_time_by_seed.csv")
    if set(repeated["model"]) != EXPECTED_MODELS:
        raise RuntimeError("Table 6 model set is incomplete.")
    for model, group in repeated.groupby("model"):
        if len(group) != 5 or set(group["seed"]) != EXPECTED_SEEDS:
            raise RuntimeError(f"Incomplete Table 6 repetitions for {model}.")
    merged = repeated.merge(
        timing[["seed", "method", "wall_minutes"]],
        left_on=["seed", "model"],
        right_on=["seed", "method"],
        validate="one_to_one",
    )
    if not np.allclose(merged["fitting_time_min"], merged["wall_minutes"], rtol=0, atol=1e-12):
        raise RuntimeError("Table 6 time does not match the fixed-fitting timing record.")
    columns = ["r2_mean", "r2_std", "rmse_mean", "rmse_std", "nll_mean", "nll_std", "fitting_time_min_mean", "fitting_time_min_std"]
    for model, expected in EXPECTED_TABLE6.items():
        row = summary.loc[summary["model"] == model]
        if len(row) != 1:
            raise RuntimeError(f"Missing Table 6 summary row for {model}.")
        for column, value in zip(columns, expected):
            assert_close(float(row.iloc[0][column]), value, f"Table 6 {model} {column}")


def verify_table7() -> None:
    repeated = pd.read_csv(TABLE_DIR / "table7_by_seed.csv")
    summary = pd.read_csv(TABLE_DIR / "table7_summary.csv")
    if set(repeated["model"]) != EXPECTED_INTERVAL_MODELS:
        raise RuntimeError("Table 7 model set is incomplete.")
    for model, group in repeated.groupby("model"):
        if len(group) != 5 or set(group["seed"]) != EXPECTED_SEEDS:
            raise RuntimeError(f"Incomplete Table 7 repetitions for {model}.")
    columns = ["picp95_mean", "picp95_std", "mpiw95_mean", "mpiw95_std"]
    for model, expected in EXPECTED_TABLE7.items():
        row = summary.loc[summary["model"] == model]
        if len(row) != 1:
            raise RuntimeError(f"Missing Table 7 summary row for {model}.")
        for column, value in zip(columns, expected):
            assert_close(float(row.iloc[0][column]), value, f"Table 7 {model} {column}")


def verify_repeated_experiments() -> None:
    sensitivity = pd.read_csv(ROOT / "Revise" / "surrogate_benchmark" / "public_results" / "sample_size" / "metrics_by_seed_budget.csv")
    if set(sensitivity["sample_size"]) != set(range(25, 401, 25)):
        raise RuntimeError("Unexpected sample-size sensitivity budgets.")
    for size, group in sensitivity.groupby("sample_size"):
        if len(group) != 5 or set(group["seed"]) != EXPECTED_SEEDS:
            raise RuntimeError(f"Incomplete sensitivity repetitions at n={size}.")

    optimizer = pd.read_csv(OPTIMIZER_DIR / "optimizer_metrics_runs.csv")
    if set(optimizer["algorithm"]) != {"NSGA-II", "MOEA/D", "MOPSO-CD"}:
        raise RuntimeError("Optimizer set is incomplete.")
    for algorithm, group in optimizer.groupby("algorithm"):
        if set(group["budget"]) != EXPECTED_BUDGETS:
            raise RuntimeError(f"Unexpected budgets for {algorithm}.")
        for budget, runs in group.groupby("budget"):
            if len(runs) != 10 or set(runs["seed"]) != set(range(2025, 2035)):
                raise RuntimeError(f"Incomplete optimizer runs for {algorithm} at {budget}.")

    with (OPTIMIZER_DIR / "run_manifest.json").open(encoding="utf-8-sig") as stream:
        manifest = json.load(stream)
    metadata_path = ROOT / manifest["model_metadata"]
    if manifest["dataset_sha256"] != sha256(DATASET):
        raise RuntimeError("Optimizer manifest dataset checksum is stale.")
    if manifest["model_checkpoint_sha256"] != sha256(MODEL):
        raise RuntimeError("Optimizer manifest model checksum is stale.")
    if manifest["model_metadata_sha256"] != sha256(metadata_path):
        raise RuntimeError("Optimizer manifest metadata checksum is stale.")


def verify_ablation_status() -> None:
    expected = {
        "Full": (200, 200),
        "M1": (100, 100),
        "M2": (200, 200),
        "M3": (200, 200),
        "M4": (100, 100),
    }
    for method, (population, generations) in expected.items():
        with (ABLATION_DIR / method / "metadata.json").open(encoding="utf-8-sig") as stream:
            metadata = json.load(stream)
        nsga2 = metadata["nsga2"]
        if nsga2["population_size"] != population or nsga2["generations"] != generations:
            raise RuntimeError(f"Unexpected NSGA-II setting for {method}.")


def verify_hygiene() -> None:
    forbidden_suffixes = {".3dm", ".db", ".out", ".err", ".bat", ".cmd"}
    forbidden_names = {
        "fe_evaluator.py",
        "main_Dataset.py",
        "PATH.py",
        "REVIEWER_RESPONSE_DRAFT.md",
        "AUDIT_NOTES.md",
        "LHS_sampling_registry.json",
        "validation_results.csv",
    }
    forbidden = [
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts
        and (path.suffix.lower() in forbidden_suffixes or path.name in forbidden_names)
    ]
    if forbidden:
        raise RuntimeError(f"Excluded project-specific files found: {forbidden}")

    absolute_pattern = re.compile(r"[A-Za-z]:[\\/]")
    text_suffixes = {".py", ".md", ".json", ".csv", ".txt"}
    leaked = []
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or any(part in {".git", ".pytest_cache", "__pycache__"} for part in path.parts)
            or path.suffix.lower() not in text_suffixes
        ):
            continue
        if absolute_pattern.search(path.read_text(encoding="utf-8-sig", errors="replace")):
            leaked.append(path.relative_to(ROOT))
    if leaked:
        raise RuntimeError(f"Machine-specific absolute paths found: {leaked}")

    workbook_markers = re.compile(
        r"(?:^|[\"'>])[A-Za-z]:[\\/](?!/)|Projects_Manage|CodexSandbox|PC-LAB|Administrator|"
        r"revised_adse|full_run_cache|new_fe",
        re.IGNORECASE,
    )
    workbook_leaks = []
    for path in ROOT.rglob("*.xlsx"):
        with zipfile.ZipFile(path) as archive:
            if any(
                workbook_markers.search(
                    archive.read(name).decode("utf-8", errors="ignore")
                )
                for name in archive.namelist()
                if name.endswith(".xml")
            ):
                workbook_leaks.append(path.relative_to(ROOT))
    if workbook_leaks:
        raise RuntimeError(f"Internal workbook metadata found: {workbook_leaks}")

    obsolete = [
        ROOT / "Revise" / "surrogate_benchmark" / "public_results" / "table5",
        ROOT / "Revise" / "surrogate_benchmark" / "build_table5_results.py",
        OPTIMIZER_DIR / "optimizer_comparison_results_final_front.xlsx",
    ]
    existing = [path.relative_to(ROOT) for path in obsolete if path.exists()]
    if existing:
        raise RuntimeError(f"Obsolete or duplicate artifacts remain: {existing}")


def main() -> None:
    if sha256(DATASET) != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Dataset checksum differs from the published artifact.")
    if sha256(MODEL) != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Seed-2025 DE-MMoE checksum differs from the published artifact.")

    workbook = pd.ExcelFile(DATASET)
    if workbook.sheet_names != list(EXPECTED_SHEETS):
        raise RuntimeError(f"Unexpected workbook sheets: {workbook.sheet_names}")
    for sheet, shape in EXPECTED_SHEETS.items():
        frame = pd.read_excel(DATASET, sheet_name=sheet)
        if frame.shape != shape or frame.isna().any().any():
            raise RuntimeError(f"Invalid dataset sheet {sheet}: {frame.shape}.")

    required = [
        TABLE_DIR / "table6_by_seed.csv",
        TABLE_DIR / "table6_summary.csv",
        TABLE_DIR / "table7_by_seed.csv",
        TABLE_DIR / "table7_summary.csv",
        OPTIMIZER_DIR / "optimizer_comparison_results.xlsx",
        ROOT / "Revise" / "adaptive_dse" / "public_results" / "pareto_fe_validation" / "validation_summary.csv",
    ]
    missing = [path.relative_to(ROOT) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Required public artifacts are missing: {missing}")

    verify_table6()
    verify_table7()
    verify_repeated_experiments()
    verify_ablation_status()
    verify_hygiene()

    print("Public package verification passed.")
    print("Dataset:", EXPECTED_SHEETS)
    print("Table 6: 7 models x 5 seeds; manuscript values and fitting times verified")
    print("Table 7: 6 probabilistic models x 5 seeds verified")
    print("Optimizer comparison: 3 algorithms x 12 budgets x 10 seeds verified")
    print("Ablations: settings and result files verified")


if __name__ == "__main__":
    main()
