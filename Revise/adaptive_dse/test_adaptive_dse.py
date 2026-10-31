from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from ._common import FEEvaluationFailure
from .design_space_exploration import (
    OnlineValidationSet,
    complete_accuracy_check,
    importance_from_table,
    objective_error_weights,
    plan_accuracy_check,
    rank_pareto_candidates_by_role,
    select_pareto_representatives,
    select_uncertainty_points,
    update_design_space,
)
from .ensemble_surrogate import (
    EnsembleMMoESurrogate,
    gradient_shap_objectives,
)
from .experiment_log import ADSEExperimentLogger
from .ablation_settings import M1, M2, M3, M4
from .workflow import refine_pareto_until_accurate


class AdaptiveDesignSpaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lower = np.zeros(16, dtype=np.float64)
        self.upper = np.ones(16, dtype=np.float64)
        self.width = self.upper - self.lower

    def test_importance_is_realigned_before_use(self) -> None:
        table = pd.DataFrame(
            {
                "feature": ["x3", "x1", "x2"],
                "mean_abs_t1": [9.0, 3.0, 6.0],
                "mean_abs_t2": [3.0, 6.0, 0.0],
            }
        )
        result = importance_from_table(
            table,
            feature_order=("x1", "x2", "x3"),
            target_columns=("mean_abs_t1", "mean_abs_t2"),
        )
        self.assertEqual(result["feature"].tolist(), ["x1", "x2", "x3"])
        np.testing.assert_allclose(
            result["mean_abs_t1_normalized"], [1.0 / 3.0, 2.0 / 3.0, 1.0]
        )
        np.testing.assert_allclose(
            result["mean_abs_t2_normalized"], [1.0, 0.0, 0.5]
        )

    def test_three_representatives_keep_both_extremes(self) -> None:
        x = np.zeros((4, 16), dtype=np.float64)
        x[:, 0] = np.linspace(0.0, 1.0, 4)
        objectives = np.array(
            [[0.0, 1.0], [0.2, 0.6], [0.5, 0.5], [1.0, 0.0]],
            dtype=np.float64,
        )
        result = select_pareto_representatives(x, objectives)
        self.assertEqual(result.source_indices.tolist(), [0, 1, 3])
        self.assertEqual(result.labels[0], "volume_extreme")
        self.assertEqual(result.labels[2], "stress_extreme")

    def test_role_candidate_rankings_support_same_role_replacement(self) -> None:
        x = np.zeros((5, 16), dtype=np.float64)
        x[:, 0] = [0.1, 0.3, 0.5, 0.7, 0.9]
        objectives = np.column_stack((x[:, 0], 1.0 - x[:, 0]))
        self.assertEqual(
            rank_pareto_candidates_by_role(
                x, objectives, "volume_extreme"
            )[:2].tolist(),
            [0, 1],
        )
        self.assertEqual(
            rank_pareto_candidates_by_role(
                x, objectives, "balanced_middle"
            )[0],
            2,
        )
        self.assertEqual(
            rank_pareto_candidates_by_role(
                x, objectives, "stress_extreme"
            )[:2].tolist(),
            [4, 3],
        )

    def test_fe_feasibility_is_diagnostic_not_an_accuracy_gate(self) -> None:
        x = np.zeros((5, 16), dtype=np.float64)
        x[:, 0] = np.linspace(0.0, 1.0, 5)
        objectives = np.column_stack(
            (np.linspace(0.0, 1.0, 5), np.linspace(1.0, 0.0, 5))
        )
        representatives = select_pareto_representatives(x, objectives)

        def predict(values: np.ndarray) -> np.ndarray:
            return np.column_stack((values[:, 0] + 10.0, 20.0 - values[:, 0]))

        prediction = predict(representatives.x)
        previous = OnlineValidationSet(
            representatives.labels,
            representatives.x.copy(),
            prediction.copy(),
        )
        plan = plan_accuracy_check(
            x,
            objectives,
            predict,
            self.width,
            previous_validation=previous,
            representativeness_threshold=1e-8,
        )
        self.assertTrue(np.all(plan.reused))
        result = complete_accuracy_check(plan, np.empty((0, 2)))
        self.assertTrue(result.passed)

        infeasible = complete_accuracy_check(
            plan,
            np.empty((0, 2)),
            relative_error_limit=0.01,
            feasibility_mask=np.array([True, False, True]),
        )
        self.assertTrue(infeasible.passed)
        self.assertFalse(infeasible.all_feasible)
        self.assertEqual(result.worst_relative_error, 0.0)

    def test_gate_uses_objective_wise_three_point_rms_error(self) -> None:
        x = np.zeros((5, 16), dtype=np.float64)
        x[:, 0] = np.linspace(0.0, 1.0, 5)
        objectives = np.column_stack((x[:, 0], 1.0 - x[:, 0]))
        representatives = select_pareto_representatives(x, objectives)
        y_fe = np.full((3, 2), 100.0, dtype=np.float64)

        def predict(values: np.ndarray) -> np.ndarray:
            prediction = np.full((len(values), 2), 100.0, dtype=np.float64)
            prediction[np.isclose(values[:, 0], 0.5), 1] = 101.5
            return prediction

        previous = OnlineValidationSet(
            representatives.labels, representatives.x.copy(), y_fe
        )
        plan = plan_accuracy_check(
            x,
            objectives,
            predict,
            self.width,
            previous_validation=previous,
            representativeness_threshold=1e-8,
        )
        result = complete_accuracy_check(
            plan, np.empty((0, 2)), relative_error_limit=0.01
        )
        self.assertAlmostEqual(result.worst_relative_error, 0.015)
        self.assertAlmostEqual(
            result.objective_rms_relative_error[1], 0.015 / np.sqrt(3.0)
        )
        self.assertTrue(result.passed)

    def test_uncertainty_selection_uses_joint_score(self) -> None:
        x = np.zeros((4, 16), dtype=np.float64)
        x[:, 0] = [0.1, 0.3, 0.6, 0.9]
        std = np.array(
            [[10.0, 0.001], [5.0, 5.0], [3.0, 3.0], [2.0, 2.0]],
            dtype=np.float64,
        )
        training = np.zeros((1, 16), dtype=np.float64)
        result = select_uncertainty_points(
            x, std, training, self.width, count=2
        )
        self.assertEqual(result.indices.tolist(), [1, 2])

    def test_random_priorities_change_only_the_selection_order(self) -> None:
        x = np.zeros((3, 16), dtype=np.float64)
        x[:, 0] = [0.2, 0.5, 0.8]
        std = np.array(
            [[10.0, 10.0], [5.0, 5.0], [2.0, 2.0]], dtype=np.float64
        )
        result = select_uncertainty_points(
            x,
            std,
            np.zeros((1, 16), dtype=np.float64),
            self.width,
            selection_priorities=np.array([0.1, 0.2, 0.9]),
            count=1,
        )
        self.assertEqual(result.indices.tolist(), [2])
        self.assertAlmostEqual(result.scores[0], 2.0)

    def test_uncertainty_selection_skips_evaluated_point(self) -> None:
        x = np.zeros((3, 16), dtype=np.float64)
        x[:, 0] = [0.0, 0.3, 0.6]
        std = np.array(
            [[10.0, 10.0], [5.0, 5.0], [3.0, 3.0]], dtype=np.float64
        )
        training = np.zeros((1, 16), dtype=np.float64)
        result = select_uncertainty_points(
            x, std, training, self.width, count=1
        )
        self.assertEqual(result.indices.tolist(), [1])

    def test_error_weights_keep_floor_and_sum_to_one(self) -> None:
        errors = np.array(
            [[0.001, 0.010], [0.001, 0.012], [0.00085, 0.0125]],
            dtype=np.float64,
        )
        weights = objective_error_weights(errors, minimum_weight=0.20)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertGreater(weights[1], weights[0])
        self.assertGreaterEqual(weights.min(), 0.20)
        self.assertLessEqual(weights.max(), 0.80)

    def test_batch_diversity_is_separate_from_archive_deduplication(self) -> None:
        x = np.zeros((4, 16), dtype=np.float64)
        x[:, 0] = [0.020, 0.400, 0.401, 0.800]
        std = np.array(
            [[4.0, 4.0], [10.0, 10.0], [9.0, 9.0], [8.0, 8.0]],
            dtype=np.float64,
        )
        archive = np.zeros((1, 16), dtype=np.float64)
        result = select_uncertainty_points(
            x,
            std,
            archive,
            self.width,
            count=2,
            strict_duplicate_distance=1e-8,
            batch_diversity_distance=0.025,
        )
        # x=0.020 remains eligible despite being close to the FE archive;
        # x=0.401 is skipped because it is too close to the first batch point.
        self.assertEqual(result.indices.tolist(), [1, 3])

    def test_uncertainty_selection_refilters_fixed_pool_feasibility(self) -> None:
        x = np.zeros((3, 16), dtype=np.float64)
        x[:, 0] = [0.2, 0.5, 0.8]
        std = np.array(
            [[10.0, 10.0], [5.0, 5.0], [4.0, 4.0]], dtype=np.float64
        )
        result = select_uncertainty_points(
            x,
            std,
            np.zeros((1, 16), dtype=np.float64),
            self.width,
            eligible_mask=np.array([False, True, True]),
            count=1,
        )
        self.assertEqual(result.indices.tolist(), [1])

    def test_batch_diversity_relaxes_directly_to_strict_tolerance(self) -> None:
        x = np.zeros((2, 16), dtype=np.float64)
        x[:, 0] = [0.400, 0.401]
        std = np.array([[10.0, 10.0], [9.0, 9.0]], dtype=np.float64)
        result = select_uncertainty_points(
            x,
            std,
            np.zeros((1, 16), dtype=np.float64),
            self.width,
            count=2,
            strict_duplicate_distance=1e-8,
            batch_diversity_distance=0.025,
        )
        self.assertEqual(result.indices.tolist(), [0, 1])
        self.assertEqual(result.diversity_fallback.tolist(), [False, True])

    def test_reference_span_prevents_zero_expansion(self) -> None:
        pareto = np.full((3, 16), 0.5, dtype=np.float64)
        importance = np.zeros(16, dtype=np.float64)
        result = update_design_space(
            pareto,
            importance,
            self.lower,
            self.upper,
            np.full(16, 0.5),
            np.full(16, 0.5),
            iteration_index=0,
            total_iterations=10,
            gamma=0.0,
            eta_start=0.15,
            eta_end=0.03,
            reference_span_ratio=0.10,
        )
        np.testing.assert_allclose(result.effective_span, 0.10)
        np.testing.assert_allclose(result.expansion, 0.015)
        np.testing.assert_allclose(result.upper - result.lower, 0.03)

    def test_uniform_adse_removes_only_the_importance_gain(self) -> None:
        pareto = np.vstack(
            (np.full(16, 0.4), np.full(16, 0.6))
        ).astype(np.float64)
        importance = np.linspace(0.0, 1.0, 16)
        result = update_design_space(
            pareto,
            importance,
            self.lower,
            self.upper,
            np.full(16, 0.4),
            np.full(16, 0.6),
            iteration_index=0,
            total_iterations=10,
            gamma=0.0,
            eta_start=0.15,
            eta_end=0.03,
            reference_span_ratio=0.10,
            importance_multiplier=0.0,
        )
        np.testing.assert_allclose(
            result.expansion, 0.15 * result.effective_span
        )
        np.testing.assert_allclose(result.importance, importance)

    def test_four_ablation_entry_configs_change_only_named_mechanisms(self) -> None:
        m1 = M1
        m2 = M2
        m3 = M3
        m4 = M4
        self.assertEqual(m1["design_space_mode"], "fixed")
        self.assertEqual(m1["acquisition_mode"], "uncertainty")
        self.assertTrue(m1["accuracy_trigger_enabled"])
        self.assertEqual(m2["design_space_mode"], "uniform")
        self.assertEqual(m2["acquisition_mode"], "uncertainty")
        self.assertTrue(m2["accuracy_trigger_enabled"])
        self.assertEqual(m3["design_space_mode"], "importance")
        self.assertEqual(m3["acquisition_mode"], "random")
        self.assertTrue(m3["accuracy_trigger_enabled"])
        self.assertEqual(m4["design_space_mode"], "importance")
        self.assertEqual(m4["acquisition_mode"], "uncertainty")
        self.assertFalse(m4["accuracy_trigger_enabled"])
        for config in (m1, m2, m3, m4):
            self.assertEqual(config["initial_sample_count"], 80)
            self.assertEqual(config["outer_iterations"], 30)
            self.assertEqual(config["uncertainty_batch_size"], 3)

    def test_no_accuracy_trigger_always_adds_one_uq_batch(self) -> None:
        pareto_x = np.zeros((7, 16), dtype=np.float64)
        pareto_x[:, 0] = np.linspace(0.0, 1.0, 7)
        pareto_objectives = np.column_stack(
            (10.0 + pareto_x[:, 0], 20.0 - pareto_x[:, 0])
        )

        class AccurateSurrogate:
            def predict(self, values: np.ndarray) -> SimpleNamespace:
                mean = np.column_stack(
                    (10.0 + values[:, 0], 20.0 - values[:, 0])
                )
                std = np.column_stack(
                    (0.1 + values[:, 0], 1.1 - values[:, 0])
                )
                return SimpleNamespace(
                    mean_physical=mean,
                    model_std_standardized=std,
                )

        optimize_calls = []

        def optimize(_surrogate):
            optimize_calls.append(True)
            return pareto_x, pareto_objectives

        def evaluate_fe(values: np.ndarray) -> np.ndarray:
            return np.column_stack(
                (10.0 + values[:, 0], 20.0 - values[:, 0])
            )

        retrain_calls = []

        def retrain(x_values, y_values):
            retrain_calls.append((x_values.copy(), y_values.copy()))
            return AccurateSurrogate()

        initial_x = np.full((1, 16), 0.25, dtype=np.float64)
        result = refine_pareto_until_accurate(
            AccurateSurrogate(),
            initial_x,
            evaluate_fe(initial_x),
            optimize=optimize,
            evaluate_fe=evaluate_fe,
            retrain=retrain,
            reference_width=self.width,
            representativeness_threshold=1e-12,
            accuracy_trigger_enabled=False,
            uncertainty_batch_size=3,
        )
        self.assertEqual(result.refinement_cycles, 1)
        self.assertEqual(result.pf_optimizations, 2)
        self.assertEqual(result.model_updates, 1)
        self.assertEqual(result.fe_evaluations, 6)
        self.assertEqual(len(optimize_calls), 2)
        self.assertEqual(len(retrain_calls), 1)
        self.assertEqual(len(result.training_x), 4)
        for validation_point in result.validation.x:
            self.assertFalse(
                np.any(
                    np.all(
                        np.isclose(
                            result.training_x,
                            validation_point,
                            rtol=0.0,
                            atol=1e-12,
                        ),
                        axis=1,
                    )
                )
            )

    def test_failed_gate_adds_three_points_then_reoptimizes(self) -> None:
        pareto_x = np.zeros((7, 16), dtype=np.float64)
        pareto_x[:, 0] = np.linspace(0.0, 1.0, 7)
        true_objectives = np.column_stack(
            (10.0 + pareto_x[:, 0], 20.0 - pareto_x[:, 0])
        )

        class FakeSurrogate:
            def __init__(self, accurate: bool) -> None:
                self.accurate = accurate

            def predict(self, values: np.ndarray) -> SimpleNamespace:
                true = np.column_stack(
                    (10.0 + values[:, 0], 20.0 - values[:, 0])
                )
                mean = true if self.accurate else true * 1.02
                std = np.column_stack(
                    (0.1 + values[:, 0], 1.1 - values[:, 0])
                )
                return SimpleNamespace(
                    mean_physical=mean,
                    model_std_standardized=std,
                )

        def optimize(_surrogate):
            return pareto_x, true_objectives

        def evaluate_fe(values: np.ndarray) -> np.ndarray:
            return np.column_stack(
                (10.0 + values[:, 0], 20.0 - values[:, 0])
            )

        retrain_calls = []

        def retrain(x_values, y_values):
            retrain_calls.append((x_values.copy(), y_values.copy()))
            return FakeSurrogate(accurate=True)

        initial_x = np.full((1, 16), 0.25, dtype=np.float64)
        initial_y = evaluate_fe(initial_x)
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        log_directory = Path(temporary.name) / "run"
        logger = ADSEExperimentLogger(log_directory, metadata={"seed": 2025})
        result = refine_pareto_until_accurate(
            FakeSurrogate(accurate=False),
            initial_x,
            initial_y,
            optimize=optimize,
            evaluate_fe=evaluate_fe,
            retrain=retrain,
            reference_width=self.width,
            representativeness_threshold=1e-12,
            relative_error_limit=0.01,
            heldout_validation_x=np.empty((0, 16), dtype=np.float64),
            heldout_validation_y=np.empty((0, 2), dtype=np.float64),
            experiment_logger=logger,
            outer_iteration=2,
            cumulative_fe_start=1,
        )
        self.assertEqual(result.refinement_cycles, 2)
        self.assertEqual(result.pf_optimizations, 2)
        self.assertEqual(result.model_updates, 1)
        self.assertEqual(result.fe_evaluations, 6)
        self.assertEqual(len(retrain_calls), 1)
        self.assertEqual(len(result.training_x), 4)
        self.assertEqual(len(result.points_for_next_training_x), 0)
        gate_log = pd.read_csv(log_directory / "gate_checks.csv")
        uncertainty_log = pd.read_csv(
            log_directory / "uncertainty_samples.csv"
        )
        pareto_log = pd.read_csv(log_directory / "pareto_summary.csv")
        summary_log = pd.read_csv(log_directory / "iteration_summary.csv")
        self.assertEqual(len(gate_log), 6)
        self.assertEqual(len(uncertainty_log), 3)
        self.assertEqual(len(pareto_log), 2)
        self.assertEqual(len(summary_log), 1)
        self.assertEqual(summary_log.loc[0, "cumulative_fe_count"], 7)
        self.assertIn("gate_error", gate_log.columns)
        self.assertIn("rms_error_f2", gate_log.columns)
        first_gate = gate_log.loc[gate_log["inner_iter"].eq(0), "x1"].to_numpy()
        new_gate = gate_log.loc[gate_log["inner_iter"].eq(1), "x1"].to_numpy()
        np.testing.assert_allclose(first_gate, new_gate)
        np.testing.assert_allclose(
            gate_log.loc[gate_log["inner_iter"].eq(0), "fe_f1"],
            gate_log.loc[gate_log["inner_iter"].eq(1), "fe_f1"],
        )

    def test_failed_gate_fe_uses_next_candidate_of_same_role(self) -> None:
        pareto_x = np.zeros((5, 16), dtype=np.float64)
        pareto_x[:, 0] = [0.1, 0.3, 0.5, 0.7, 0.9]
        pareto_objectives = np.column_stack(
            (pareto_x[:, 0], 1.0 - pareto_x[:, 0])
        )

        class AccurateSurrogate:
            def predict(self, values: np.ndarray) -> SimpleNamespace:
                mean = np.column_stack(
                    (10.0 + values[:, 0], 20.0 - values[:, 0])
                )
                return SimpleNamespace(
                    mean_physical=mean,
                    model_std_standardized=np.ones((len(values), 2)),
                )

        def evaluate_fe(values: np.ndarray) -> np.ndarray:
            if np.isclose(values[0, 0], 0.1):
                raise FEEvaluationFailure(values[0], "synthetic gate failure")
            return np.column_stack(
                (10.0 + values[:, 0], 20.0 - values[:, 0])
            )

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        log_directory = Path(temporary.name) / "run"
        result = refine_pareto_until_accurate(
            AccurateSurrogate(),
            np.zeros((1, 16), dtype=np.float64),
            np.array([[10.0, 20.0]], dtype=np.float64),
            optimize=lambda _surrogate: (pareto_x, pareto_objectives),
            evaluate_fe=evaluate_fe,
            retrain=lambda x_values, y_values: AccurateSurrogate(),
            reference_width=self.width,
            representativeness_threshold=1e-12,
            experiment_logger=ADSEExperimentLogger(log_directory),
        )
        self.assertEqual(result.fe_evaluations, 3)
        self.assertEqual(result.failed_fe_evaluations, 1)
        self.assertEqual(result.pf_optimizations, 1)
        self.assertEqual(result.model_updates, 0)
        self.assertAlmostEqual(result.validation.x[0, 0], 0.3)
        failures = pd.read_csv(log_directory / "fe_failures.csv")
        self.assertEqual(failures.loc[0, "stage"], "gate")
        self.assertEqual(failures.loc[0, "role"], "volume_extreme")
        self.assertAlmostEqual(failures.loc[0, "x1"], 0.1)

    def test_failed_representative_replacement_is_archived_and_reused(self) -> None:
        pareto_x = np.zeros((7, 16), dtype=np.float64)
        pareto_x[:, 0] = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
        pareto_objectives = np.column_stack(
            (pareto_x[:, 0], 1.0 - pareto_x[:, 0])
        )

        class FakeSurrogate:
            def __init__(self, accurate: bool) -> None:
                self.accurate = accurate

            def predict(self, values: np.ndarray) -> SimpleNamespace:
                true = np.column_stack(
                    (10.0 + values[:, 0], 20.0 - values[:, 0])
                )
                mean = true if self.accurate else true * 1.02
                priority = np.ones(len(values), dtype=np.float64)
                priority[np.isclose(values[:, 0], 0.3)] = 10.0
                priority[np.isclose(values[:, 0], 0.7)] = 9.0
                return SimpleNamespace(
                    mean_physical=mean,
                    model_std_standardized=np.column_stack(
                        (priority, priority)
                    ),
                )

        attempted: list[float] = []

        def evaluate_fe(values: np.ndarray) -> np.ndarray:
            attempted.append(float(values[0, 0]))
            if np.isclose(values[0, 0], 0.1):
                raise FEEvaluationFailure(
                    values[0], "synthetic failed representative"
                )
            return np.column_stack(
                (10.0 + values[:, 0], 20.0 - values[:, 0])
            )

        result = refine_pareto_until_accurate(
            FakeSurrogate(accurate=False),
            np.zeros((1, 16), dtype=np.float64),
            np.array([[10.0, 20.0]], dtype=np.float64),
            optimize=lambda _surrogate: (pareto_x, pareto_objectives),
            evaluate_fe=evaluate_fe,
            retrain=lambda x_values, y_values: FakeSurrogate(accurate=True),
            reference_width=self.width,
            representativeness_threshold=1e-12,
        )

        self.assertEqual(result.refinement_cycles, 2)
        self.assertEqual(result.pf_optimizations, 2)
        self.assertEqual(result.model_updates, 1)
        self.assertEqual(result.fe_evaluations, 6)
        self.assertEqual(result.failed_fe_evaluations, 1)
        self.assertEqual(sum(np.isclose(value, 0.2) for value in attempted), 1)
        self.assertEqual(attempted, [0.1, 0.2, 0.5, 0.9, 0.3, 0.7, 0.8])
        self.assertAlmostEqual(result.validation.x[0, 0], 0.2)

    def test_failed_uncertainty_fe_skips_to_next_joint_score(self) -> None:
        pareto_x = np.zeros((7, 16), dtype=np.float64)
        pareto_x[:, 0] = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
        pareto_objectives = np.column_stack(
            (pareto_x[:, 0], 1.0 - pareto_x[:, 0])
        )

        class FakeSurrogate:
            def __init__(self, accurate: bool) -> None:
                self.accurate = accurate

            def predict(self, values: np.ndarray) -> SimpleNamespace:
                true = np.column_stack(
                    (10.0 + values[:, 0], 20.0 - values[:, 0])
                )
                mean = true if self.accurate else true * 1.02
                priority = np.ones(len(values), dtype=np.float64)
                priority[np.isclose(values[:, 0], 0.2)] = 10.0
                priority[np.isclose(values[:, 0], 0.3)] = 9.0
                priority[np.isclose(values[:, 0], 0.7)] = 8.0
                priority[np.isclose(values[:, 0], 0.8)] = 7.0
                return SimpleNamespace(
                    mean_physical=mean,
                    model_std_standardized=np.column_stack(
                        (priority, priority)
                    ),
                )

        attempted = []

        def evaluate_fe(values: np.ndarray) -> np.ndarray:
            attempted.append(float(values[0, 0]))
            if np.isclose(values[0, 0], 0.2):
                raise FEEvaluationFailure(
                    values[0], "synthetic uncertainty failure"
                )
            return np.column_stack(
                (10.0 + values[:, 0], 20.0 - values[:, 0])
            )

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        log_directory = Path(temporary.name) / "run"
        result = refine_pareto_until_accurate(
            FakeSurrogate(accurate=False),
            np.zeros((1, 16), dtype=np.float64),
            np.array([[10.0, 20.0]], dtype=np.float64),
            optimize=lambda _surrogate: (pareto_x, pareto_objectives),
            evaluate_fe=evaluate_fe,
            retrain=lambda x_values, y_values: FakeSurrogate(accurate=True),
            reference_width=self.width,
            representativeness_threshold=1e-12,
            experiment_logger=ADSEExperimentLogger(log_directory),
        )
        self.assertEqual(result.fe_evaluations, 6)
        self.assertEqual(result.failed_fe_evaluations, 1)
        self.assertEqual(
            attempted[:7], [0.1, 0.5, 0.9, 0.2, 0.3, 0.7, 0.8]
        )
        uncertainty = pd.read_csv(log_directory / "uncertainty_samples.csv")
        np.testing.assert_allclose(uncertainty["x1"], [0.3, 0.7, 0.8])
        failures = pd.read_csv(log_directory / "fe_failures.csv")
        self.assertEqual(failures.loc[0, "stage"], "uncertainty")
        self.assertEqual(
            failures.loc[0, "role"], "weighted_joint_uncertainty"
        )
        self.assertAlmostEqual(failures.loc[0, "x1"], 0.2)

    def test_logger_writes_importance_and_bounds(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        log_directory = Path(temporary.name) / "run"
        logger = ADSEExperimentLogger(log_directory)
        importance = np.linspace(0.0, 1.0, 16)
        explanation = SimpleNamespace(
            feature_names=tuple(f"x{i + 1}" for i in range(16)),
            mean_absolute_by_task=np.vstack((importance, importance[::-1])),
            normalized_by_task=np.vstack((importance, importance[::-1])),
            combined_importance=np.full(16, 0.5),
        )
        logger.log_gradient_shap(
            outer_iter=1, model_id="model_1", result=explanation
        )
        update = update_design_space(
            np.full((3, 16), 0.5),
            importance,
            self.lower,
            self.upper,
            np.full(16, 0.5),
            np.full(16, 0.5),
            iteration_index=1,
            total_iterations=10,
        )
        logger.log_bounds_update(
            outer_iter=1,
            model_id="model_1",
            previous_lower=self.lower,
            previous_upper=self.upper,
            update=update,
        )
        self.assertEqual(
            len(pd.read_csv(log_directory / "variable_importance.csv")), 16
        )
        self.assertEqual(
            len(pd.read_csv(log_directory / "design_space_bounds.csv")), 16
        )


class EnsembleSmokeTest(unittest.TestCase):
    def test_model_and_gradient_shap(self) -> None:
        model_path = (
            Path(__file__).resolve().parents[1]
            / "surrogate_benchmark"
            / "results"
            / "seed_2025"
            / "ensemble_mmoe"
            / "fitted_model.pt"
        )
        if not model_path.is_file():
            self.skipTest("The published five-member model is unavailable.")
        surrogate = EnsembleMMoESurrogate.load(model_path)
        background = np.vstack(
            (
                np.ones(16),
                np.full(16, 0.98),
                np.full(16, 1.02),
                np.linspace(0.97, 1.03, 16),
            )
        )
        explained = background[:2]
        prediction = surrogate.predict(explained)
        self.assertEqual(prediction.mean_physical.shape, (2, 41))
        self.assertEqual(prediction.model_variance_standardized.shape, (2, 41))
        result = gradient_shap_objectives(
            surrogate,
            background,
            explained,
            baseline_count=2,
            path_samples=2,
            batch_size=2,
            seed=2025,
        )
        self.assertEqual(result.mean_absolute_by_task.shape, (2, 16))
        self.assertEqual(result.combined_importance.shape, (16,))
        self.assertTrue(np.isfinite(result.combined_importance).all())
        self.assertTrue(np.all(result.combined_importance >= 0.0))
        self.assertTrue(np.all(result.combined_importance <= 1.0))


if __name__ == "__main__":
    unittest.main()
