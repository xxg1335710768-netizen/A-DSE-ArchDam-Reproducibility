"""Focused tests for the constrained optimizer-comparison protocol.

Run from PyCharm or with::

    python -m unittest Revise.adaptive_dse.test_optimizer_comparison

These tests never call Rhino, ANSYS, or the FE workflow.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from ._common import INPUT_DIM
from .optimizer_comparison import (
    BinaryFeasibilityRankAndCrowding,
    CONFIG_PATH,
    CumulativeFeasibleArchive,
    FixedSampling,
    ConstrainedMOPSOCD,
    build_algorithm,
    generate_initial_lhs,
    load_comparison_config,
    nondominated_objectives_2d,
    normalized_hv,
)
from .design_space_exploration import load_reference_bounds
from .optimizer_constraints import (
    EngineeringConstraintEvaluator,
    SIGNED_STRESS_LOWER_MPA,
    SIGNED_STRESS_UPPER_MPA,
    feasibility_compare,
)
from .optimizer_comparison import BOUNDS_PATH


class _DeterministicToyProblem(Problem):
    def __init__(self) -> None:
        super().__init__(
            n_var=INPUT_DIM,
            n_obj=2,
            n_ieq_constr=1,
            xl=np.zeros(INPUT_DIM),
            xu=np.ones(INPUT_DIM),
        )

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        values = np.asarray(x, dtype=np.float64)
        out["F"] = np.column_stack(
            (np.mean(values**2, axis=1), np.mean((values - 0.7) ** 2, axis=1))
        )
        out["G"] = (0.42 - np.mean(values, axis=1))[:, None]


class ProtocolDefinitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = load_comparison_config(CONFIG_PATH)
        dataset = (CONFIG_PATH.parent / config["dataset"]).resolve()
        cls.output_names = tuple(
            pd.read_excel(dataset, sheet_name="TrainY", nrows=1).columns.astype(str)
        )
        cls.constraints = EngineeringConstraintEvaluator(cls.output_names)

    def test_budgets_seeds_and_population(self) -> None:
        config = load_comparison_config(CONFIG_PATH)
        self.assertEqual(config["population_size"], 200)
        self.assertEqual(
            config["budgets"],
            [
                2_400,
                5_000,
                10_000,
                15_000,
                20_000,
                30_000,
                40_000,
                50_000,
                60_000,
                80_000,
                100_000,
                120_000,
            ],
        )
        self.assertEqual(config["seeds"], list(range(2025, 2035)))

    def test_binary_feasibility_priority_does_not_rank_violation_degree(self) -> None:
        self.assertLess(feasibility_compare(True, False), 0)
        self.assertGreater(feasibility_compare(False, True), 0)
        self.assertEqual(feasibility_compare(False, False), 0)
        self.assertEqual(feasibility_compare(True, True), 0)

    def test_constraint_evaluator_returns_only_binary_cv(self) -> None:
        x = np.ones((2, INPUT_DIM), dtype=np.float64)
        y = np.zeros((2, 41), dtype=np.float64)
        y[1, 2:] = 100.0
        result = self.constraints.evaluate(x, y)
        self.assertTrue(np.isin(result.cv, (0.0, 1.0)).all())
        self.assertEqual(result.cv[0], 0.0)
        self.assertEqual(result.cv[1], 1.0)

    def test_all_39_signed_stress_responses_map_to_one_physical_limit(self) -> None:
        self.assertEqual(len(self.constraints.stress_names), 39)
        tension = [name for name in self.constraints.names if name.startswith("tension_lower::")]
        compression = [name for name in self.constraints.names if name.startswith("compression_upper::")]
        self.assertEqual(len(tension), 23)
        self.assertEqual(len(compression), 16)
        self.assertGreater(len(self.constraints.geometry_names), 0)
        self.assertEqual(
            self.constraints.count,
            len(self.constraints.geometry_names) + 39,
        )

    def test_stress_sign_and_limits(self) -> None:
        geometry_count = len(self.constraints.geometry_names)
        x = np.ones((1, INPUT_DIM), dtype=np.float64)
        y = np.zeros((1, 41), dtype=np.float64)
        tension_index = int(np.flatnonzero(self.constraints.tension_mask)[0])
        compression_index = int(np.flatnonzero(self.constraints.compression_mask)[0])

        y[0, 2 + tension_index] = SIGNED_STRESS_LOWER_MPA
        result = self.constraints.evaluate(x, y)
        self.assertLessEqual(result.g[0, geometry_count + tension_index], 1e-12)

        y[0, 2 + tension_index] = SIGNED_STRESS_LOWER_MPA - 0.01
        result = self.constraints.evaluate(x, y)
        self.assertGreater(result.g[0, geometry_count + tension_index], 0.0)

        y[0, 2 + tension_index] = 0.0
        y[0, 2 + compression_index] = SIGNED_STRESS_UPPER_MPA
        result = self.constraints.evaluate(x, y)
        self.assertLessEqual(
            result.g[0, geometry_count + compression_index], 1e-12
        )

        y[0, 2 + compression_index] = SIGNED_STRESS_UPPER_MPA + 0.01
        result = self.constraints.evaluate(x, y)
        self.assertGreater(result.g[0, geometry_count + compression_index], 0.0)

    def test_geometry_feasibility_matches_source_checker_on_lhs(self) -> None:
        lower, upper = load_reference_bounds(BOUNDS_PATH)
        points = generate_initial_lhs(314159, lower, upper, 200)
        responses = np.zeros((len(points), 41), dtype=np.float64)
        evaluated = self.constraints.evaluate(points, responses)
        geometry_feasible = np.all(
            evaluated.g[:, : len(self.constraints.geometry_names)] <= 1e-12,
            axis=1,
        )
        source_feasible = []
        for point in points:
            model = self.constraints.parameterization.DamParameter(T=1)
            model.InitialParameter()
            model.ParameterUpdate(point.tolist())
            source_feasible.append(bool(model.IsConstraintAvailable()))
        source_feasible = np.asarray(source_feasible, dtype=bool)
        mismatches = np.flatnonzero(geometry_feasible != source_feasible)
        detail = []
        for row in mismatches:
            violated = np.flatnonzero(
                evaluated.g[row, : len(self.constraints.geometry_names)] > 1e-12
            )
            detail.append(
                {
                    "row": int(row),
                    "continuous": bool(geometry_feasible[row]),
                    "source": bool(source_feasible[row]),
                    "violated": [self.constraints.geometry_names[i] for i in violated],
                }
            )
        self.assertEqual(detail, [], msg=str(detail))

    def test_paired_lhs_is_reproducible_and_fixed(self) -> None:
        lower = np.zeros(INPUT_DIM)
        upper = np.ones(INPUT_DIM)
        first = generate_initial_lhs(2025, lower, upper, 200)
        second = generate_initial_lhs(2025, lower, upper, 200)
        other = generate_initial_lhs(2026, lower, upper, 200)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other))
        sampling = FixedSampling(first)
        np.testing.assert_array_equal(sampling._do(None, 200), first)


class IndicatorAndArchiveTests(unittest.TestCase):
    def test_nsga_survival_ranks_infeasible_solutions_by_objectives(self) -> None:
        population = Population.new(
            "X", np.zeros((4, INPUT_DIM)),
            "F", np.asarray([[10.0, 10.0], [0.0, 5.0], [5.0, 0.0], [6.0, 6.0]]),
            "G", np.asarray([[0.0], [1.0], [1.0], [1.0]]),
            "CV", np.asarray([[0.0], [1.0], [1.0], [1.0]]),
        )
        survivors = BinaryFeasibilityRankAndCrowding().do(
            _DeterministicToyProblem(),
            population,
            n_survive=3,
            random_state=np.random.default_rng(7),
        )
        np.testing.assert_allclose(
            np.sort(survivors.get("F"), axis=0),
            np.sort(np.asarray([[10.0, 10.0], [0.0, 5.0], [5.0, 0.0]]), axis=0),
        )

    def test_objective_front_does_not_collapse_distinct_points(self) -> None:
        front = nondominated_objectives_2d(
            np.asarray([[2.0, 8.0], [8.0, 2.0], [5.0, 9.0]])
        )
        np.testing.assert_allclose(front, [[2.0, 8.0], [8.0, 2.0]])

    def test_fixed_reference_hv(self) -> None:
        self.assertEqual(normalized_hv(np.empty((0, 2)), np.zeros(2), np.ones(2)), 0.0)
        value = normalized_hv(
            np.asarray([[2.0, 8.0], [8.0, 2.0]]),
            np.zeros(2),
            np.full(2, 10.0),
        )
        self.assertAlmostEqual(value, 0.28, places=12)

    def test_external_archive_is_unbounded_feasible_and_nondominated(self) -> None:
        count = 250
        x = np.zeros((count, INPUT_DIM))
        x[:, 0] = np.arange(count)
        f = np.column_stack((np.arange(count), np.arange(count)[::-1]))
        archive = CumulativeFeasibleArchive()
        archive.update(x, f, np.ones(count, dtype=bool))
        self.assertEqual(len(archive.x), count)
        np.testing.assert_allclose(archive.f, nondominated_objectives_2d(f))

    def test_cumulative_archive_hv_cannot_decrease(self) -> None:
        archive = CumulativeFeasibleArchive()
        first_x = np.zeros((1, INPUT_DIM))
        archive.update(first_x, np.asarray([[8.0, 8.0]]), np.asarray([True]))
        first_hv = normalized_hv(archive.f, np.zeros(2), np.full(2, 10.0))
        second_x = np.ones((1, INPUT_DIM))
        archive.update(second_x, np.asarray([[2.0, 2.0]]), np.asarray([True]))
        second_hv = normalized_hv(archive.f, np.zeros(2), np.full(2, 10.0))
        self.assertGreaterEqual(second_hv, first_hv)

    def test_mopso_internal_archive_limit_is_not_external_limit(self) -> None:
        count = 250
        x = np.zeros((count, INPUT_DIM))
        x[:, 0] = np.arange(count)
        f = np.column_stack((np.arange(count), np.arange(count)[::-1]))
        population = Population.new(
            "X", x,
            "F", f,
            "G", np.full((count, 1), -1.0),
            "CV", np.zeros((count, 1)),
        )
        algorithm = ConstrainedMOPSOCD(pop_size=200, archive_size=200)
        algorithm.leader_archive = Population.empty()
        internal = algorithm._update_archive(population)
        self.assertEqual(len(internal), 200)

        external = CumulativeFeasibleArchive()
        external.update(x, f, np.ones(count, dtype=bool))
        self.assertEqual(len(external.x), 250)

    def test_mopso_infeasible_archive_uses_objective_fitness(self) -> None:
        design = np.zeros((3, INPUT_DIM))
        design[:, 0] = np.arange(3)
        population = Population.new(
            "X", design,
            "F", np.asarray([[0.0, 5.0], [5.0, 0.0], [6.0, 6.0]]),
            "G", np.ones((3, 1)),
            "CV", np.ones((3, 1)),
        )
        algorithm = ConstrainedMOPSOCD(pop_size=200, archive_size=200)
        algorithm.leader_archive = Population.empty()
        selected = algorithm._update_archive(population)
        np.testing.assert_allclose(
            nondominated_objectives_2d(selected.get("F")),
            np.asarray([[0.0, 5.0], [5.0, 0.0]]),
        )

    def test_algorithms_use_exact_budget_and_are_reproducible(self) -> None:
        config = load_comparison_config(CONFIG_PATH)
        initial = generate_initial_lhs(
            999, np.zeros(INPUT_DIM), np.ones(INPUT_DIM), 200
        )
        for name in ("NSGA-II", "MOEA/D", "MOPSO-CD"):
            populations = []
            for _ in range(2):
                result = minimize(
                    _DeterministicToyProblem(),
                    build_algorithm(name, config, initial, seed=999),
                    termination=("n_eval", 400),
                    seed=999,
                    verbose=False,
                    save_history=False,
                    copy_algorithm=False,
                )
                self.assertEqual(result.algorithm.evaluator.n_eval, 400)
                populations.append(
                    (
                        result.pop.get("X").copy(),
                        result.pop.get("F").copy(),
                        result.pop.get("CV").copy(),
                    )
                )
            for first, second in zip(populations[0], populations[1]):
                np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main(verbosity=2)
