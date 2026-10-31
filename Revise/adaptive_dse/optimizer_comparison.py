"""Fair constrained comparison of NSGA-II, batch MOEA/D and MOPSO-CD.

One fixed five-member DE-MMoE and one engineering constraint evaluator are
shared by all algorithms.  Every seed is a single continuous 120,000-evaluation
trajectory with twelve cumulative checkpoints.  No FE, AL, or A-DSE operation is
imported or executed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import friedmanchisquare, qmc, wilcoxon
import torch

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.moo.moead import ParallelMOEAD
from pymoo.algorithms.moo.mopso_cd import MOPSO_CD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.algorithm import Algorithm
from pymoo.core.callback import Callback
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection, compare
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.survival.rank_and_crowding.metrics import (
    get_crowding_function,
)
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside

from ._common import INPUT_DIM, OBJECTIVE_COUNT
from .design_space_exploration import load_reference_bounds
from .ensemble_surrogate import EnsembleMMoESurrogate
from .optimizer_constraints import (
    COMPRESSIVE_MAGNITUDE_LIMIT_MPA,
    EngineeringConstraintEvaluator,
    SIGNED_STRESS_LOWER_MPA,
    SIGNED_STRESS_UPPER_MPA,
    TENSILE_MAGNITUDE_LIMIT_MPA,
    feasibility_compare,
)
HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "optimizer_comparison_config.json"
REGISTRY_PATH = HERE.parent / "LHS_sampling_registry.json"
ALGORITHMS = ("NSGA-II", "MOEA/D", "MOPSO-CD")
HV_REFERENCE_NORMALIZED = np.ones(OBJECTIVE_COUNT, dtype=np.float64)
FLOAT_TOLERANCE = 1e-12


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _safe_name(algorithm: str) -> str:
    return algorithm.lower().replace("-", "").replace("/", "").replace(" ", "")


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (HERE / path).resolve()


def load_comparison_config(path: str | Path = CONFIG_PATH) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    required = {
        "experiment_name",
        "model_checkpoint",
        "model_metadata",
        "dataset",
        "model_split_seed",
        "population_size",
        "budgets",
        "seeds",
        "objective_normalization_min",
        "objective_normalization_max",
        "hv_ideal_physical",
        "hv_reference_physical",
        "signed_stress_lower_mpa",
        "signed_stress_upper_mpa",
        "nsga2",
        "moead",
        "mopso_cd",
        "timing",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Comparison config misses fields: {missing}")
    population_size = int(config["population_size"])
    if population_size != 200:
        raise ValueError("The formal comparison requires population/swarm size 200.")
    budgets = tuple(map(int, config["budgets"]))
    if budgets != (
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
    ):
        raise ValueError("The twelve formal evaluation budgets were changed.")
    seeds = tuple(map(int, config["seeds"]))
    if seeds != tuple(range(2025, 2035)):
        raise ValueError("The ten paired formal seeds must be 2025 through 2034.")
    if any(value % population_size for value in budgets):
        raise ValueError("Every checkpoint must align with a 200-candidate batch.")
    if int(config["moead"]["reference_directions"]) != population_size:
        raise ValueError("MOEA/D reference directions must equal population size.")
    if int(config["moead"]["offspring_per_batch"]) != population_size:
        raise ValueError("MOEA/D offspring batch must equal population size.")
    if int(config["mopso_cd"]["swarm_size"]) != population_size:
        raise ValueError("MOPSO-CD swarm size must equal population size.")
    if float(config["signed_stress_lower_mpa"]) != SIGNED_STRESS_LOWER_MPA:
        raise ValueError("Signed tensile-side stress limit must be -1.50 MPa.")
    if float(config["signed_stress_upper_mpa"]) != SIGNED_STRESS_UPPER_MPA:
        raise ValueError("Signed compressive-side stress limit must be 9.78 MPa.")
    return config


class FixedSampling(Sampling):
    """Return the same pre-generated LHS matrix to every optimizer."""

    def __init__(self, design: np.ndarray) -> None:
        super().__init__()
        values = np.asarray(design, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != INPUT_DIM:
            raise ValueError("Fixed LHS design must have shape (N, 16).")
        self.design = values.copy()

    def _do(self, problem, n_samples, **kwargs):
        if int(n_samples) != len(self.design):
            raise ValueError("Requested initial population differs from fixed LHS size.")
        return self.design.copy()


def generate_initial_lhs(
    seed: int, lower: np.ndarray, upper: np.ndarray, size: int
) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=INPUT_DIM, scramble=True, seed=int(seed))
    return qmc.scale(sampler.random(int(size)), lower, upper).astype(np.float64)


def nondominated_2d(
    design: np.ndarray, objectives: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Unique minimization front for two objectives, sorted by objective 1."""

    x = np.asarray(design, dtype=np.float64)
    f = np.asarray(objectives, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != INPUT_DIM:
        raise ValueError("design must have shape (N, 16).")
    if f.shape != (len(x), OBJECTIVE_COUNT):
        raise ValueError("objectives must have shape (N, 2).")
    if not len(x):
        return x.copy(), f.copy()
    finite = np.isfinite(x).all(axis=1) & np.isfinite(f).all(axis=1)
    x, f = x[finite], f[finite]
    if not len(x):
        return x, f
    _, unique_x = np.unique(np.round(x, 12), axis=0, return_index=True)
    unique_x = np.sort(unique_x)
    x, f = x[unique_x], f[unique_x]
    order = np.lexsort((f[:, 1], f[:, 0]))
    selected: list[int] = []
    best_second = np.inf
    for index in order:
        second = float(f[index, 1])
        if second < best_second - FLOAT_TOLERANCE:
            selected.append(int(index))
            best_second = second
    keep = np.asarray(selected, dtype=np.int64)
    return x[keep], f[keep]


def nondominated_objectives_2d(objectives: np.ndarray) -> np.ndarray:
    """Unique two-objective minimization front without design-space deduplication."""

    f = np.asarray(objectives, dtype=np.float64)
    if f.ndim != 2 or f.shape[1] != OBJECTIVE_COUNT:
        raise ValueError("objectives must have shape (N, 2).")
    if not len(f):
        return f.copy()
    f = f[np.isfinite(f).all(axis=1)]
    if not len(f):
        return f
    f = np.unique(np.round(f, 14), axis=0)
    order = np.lexsort((f[:, 1], f[:, 0]))
    selected: list[int] = []
    best_second = np.inf
    for index in order:
        second = float(f[index, 1])
        if second < best_second - FLOAT_TOLERANCE:
            selected.append(int(index))
            best_second = second
    return f[np.asarray(selected, dtype=np.int64)]


def normalized_hv(
    physical_front: np.ndarray,
    ideal_physical: np.ndarray,
    reference_physical: np.ndarray,
) -> float:
    """Two-dimensional normalized HV with fixed reference (1, 1)."""

    front = np.asarray(physical_front, dtype=np.float64)
    if not len(front):
        return 0.0
    scale = np.asarray(reference_physical) - np.asarray(ideal_physical)
    normalized = (front - ideal_physical) / scale
    contributes = np.all(normalized < HV_REFERENCE_NORMALIZED, axis=1)
    normalized = normalized[contributes]
    if not len(normalized):
        return 0.0
    values = nondominated_objectives_2d(normalized)
    total = 0.0
    previous_second = 1.0
    for first, second in values:
        if first < 1.0 and second < previous_second:
            total += (1.0 - float(first)) * (previous_second - float(second))
            previous_second = float(second)
    return float(total)


class CumulativeFeasibleArchive:
    """Unbounded external evaluation archive used only for fair indicators."""

    def __init__(self) -> None:
        self.x = np.empty((0, INPUT_DIM), dtype=np.float64)
        self.f = np.empty((0, OBJECTIVE_COUNT), dtype=np.float64)

    def update(
        self, design: np.ndarray, physical_objectives: np.ndarray, feasible: np.ndarray
    ) -> None:
        mask = np.asarray(feasible, dtype=bool)
        if not np.any(mask):
            return
        new_x = np.asarray(design, dtype=np.float64)[mask]
        new_f = np.asarray(physical_objectives, dtype=np.float64)[mask]
        if len(self.x):
            new_x = np.vstack((self.x, new_x))
            new_f = np.vstack((self.f, new_f))
        self.x, self.f = nondominated_2d(new_x, new_f)


class CommonConstrainedSurrogateProblem(Problem):
    """One fixed surrogate, normalization, and constraint map for all methods."""

    def __init__(
        self,
        surrogate: EnsembleMMoESurrogate,
        constraints: EngineeringConstraintEvaluator,
        lower: np.ndarray,
        upper: np.ndarray,
        objective_minimum: np.ndarray,
        objective_maximum: np.ndarray,
    ) -> None:
        super().__init__(
            n_var=INPUT_DIM,
            n_obj=OBJECTIVE_COUNT,
            # All engineering checks are aggregated into one binary hard
            # constraint for identical treatment by every optimizer.
            n_ieq_constr=1,
            xl=np.asarray(lower, dtype=np.float64),
            xu=np.asarray(upper, dtype=np.float64),
        )
        self.surrogate = surrogate
        self.constraints = constraints
        self.objective_minimum = np.asarray(objective_minimum, dtype=np.float64)
        self.objective_scale = (
            np.asarray(objective_maximum, dtype=np.float64) - self.objective_minimum
        )
        if np.any(self.objective_scale <= 0.0):
            raise ValueError("Fixed objective normalization range is invalid.")
        self.archive = CumulativeFeasibleArchive()
        self.evaluation_count = 0
        self.feasible_evaluation_count = 0
        self.surrogate_seconds = 0.0
        self.constraint_seconds = 0.0

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        design = np.asarray(x, dtype=np.float64)
        _sync_cuda()
        started = time.perf_counter()
        prediction = self.surrogate.predict(design).mean_physical
        _sync_cuda()
        self.surrogate_seconds += time.perf_counter() - started

        started = time.perf_counter()
        constraint_batch = self.constraints.evaluate(design, prediction)
        self.constraint_seconds += time.perf_counter() - started

        physical_objectives = prediction[:, :OBJECTIVE_COUNT]
        normalized_objectives = (
            physical_objectives - self.objective_minimum
        ) / self.objective_scale
        out["F"] = normalized_objectives
        out["G"] = constraint_batch.cv[:, None]

        self.evaluation_count += len(design)
        self.feasible_evaluation_count += int(constraint_batch.feasible.sum())
        self.archive.update(design, physical_objectives, constraint_batch.feasible)


def _population_cv(population: Population) -> np.ndarray:
    if len(population) == 0:
        return np.empty(0, dtype=np.float64)
    return np.asarray(population.get("CV"), dtype=np.float64).reshape(-1)


def _population_feasible(population: Population) -> np.ndarray:
    return _population_cv(population) <= FLOAT_TOLERANCE


class BinaryFeasibilityRankAndCrowding(RankAndCrowding):
    """NSGA-II survival: feasibility first, then objective rank/crowding."""

    def __init__(self) -> None:
        super().__init__()
        self.filter_infeasible = False

    def _do(self, problem, pop, *args, n_survive=None, random_state=None, **kwargs):
        limit = len(pop) if n_survive is None else min(int(n_survive), len(pop))
        feasible = _population_feasible(pop)
        survivors = Population.empty()
        for group in (pop[feasible], pop[~feasible]):
            remaining = limit - len(survivors)
            if remaining <= 0 or len(group) == 0:
                continue
            ranked = super()._do(
                problem,
                group,
                *args,
                n_survive=min(remaining, len(group)),
                random_state=random_state,
                **kwargs,
            )
            survivors = Population.merge(survivors, ranked)
        return survivors


def binary_feasibility_tournament(pop, pairs, algorithm, **kwargs):
    """NSGA-II tournament using objectives when feasibility states tie."""

    winners = np.empty(len(pairs), dtype=np.int64)
    for row, (first, second) in enumerate(pairs):
        first = int(first)
        second = int(second)
        first_feasible = bool(pop[first].CV[0] <= FLOAT_TOLERANCE)
        second_feasible = bool(pop[second].CV[0] <= FLOAT_TOLERANCE)
        state = feasibility_compare(first_feasible, second_feasible)
        if state < 0:
            winners[row] = first
            continue
        if state > 0:
            winners[row] = second
            continue

        rank_winner = compare(
            first,
            pop[first].get("rank"),
            second,
            pop[second].get("rank"),
            method="smaller_is_better",
        )
        if rank_winner is None:
            rank_winner = compare(
                first,
                pop[first].get("crowding"),
                second,
                pop[second].get("crowding"),
                method="larger_is_better",
                return_random_if_equal=True,
                random_state=algorithm.random_state,
            )
        winners[row] = int(rank_winner)
    return winners[:, None]


class ConstrainedParallelMOEAD(ParallelMOEAD):
    """Synchronous MOEA/D with binary-feasibility neighborhood replacement."""

    update_mode = "synchronous_batch_binary_feasibility"

    def _setup(self, problem, **kwargs) -> None:
        Algorithm._setup(self, problem, **kwargs)
        self.pop_size = len(self.ref_dirs)
        self.neighbors = np.argsort(
            cdist(self.ref_dirs, self.ref_dirs), axis=1, kind="quicksort"
        )[:, : self.n_neighbors]
        if self.decomposition is None:
            self.decomposition = Tchebicheff()
        self.ideal = np.full(problem.n_obj, np.inf, dtype=np.float64)

    def _initialize_advance(self, infills=None, **kwargs) -> None:
        GeneticAlgorithm._initialize_advance(self, infills, **kwargs)
        self._refresh_ideal(self.pop)

    def _refresh_ideal(self, candidates: Population) -> None:
        self.ideal = np.minimum(self.ideal, candidates.get("F").min(axis=0))

    def _candidate_better(self, candidate, incumbent, subproblem: int) -> bool:
        candidate_feasible = bool(candidate.CV[0] <= FLOAT_TOLERANCE)
        incumbent_feasible = bool(incumbent.CV[0] <= FLOAT_TOLERANCE)
        comparison = feasibility_compare(candidate_feasible, incumbent_feasible)
        if comparison != 0:
            return comparison < 0
        weight = self.ref_dirs[subproblem : subproblem + 1]
        candidate_value = float(
            np.asarray(
                self.decomposition.do(
                    candidate.F[None, :], weights=weight, ideal_point=self.ideal
                )
            ).reshape(-1)[0]
        )
        incumbent_value = float(
            np.asarray(
                self.decomposition.do(
                    incumbent.F[None, :], weights=weight, ideal_point=self.ideal
                )
            ).reshape(-1)[0]
        )
        return candidate_value < incumbent_value - FLOAT_TOLERANCE

    def _advance(self, infills=None, **kwargs) -> None:
        if infills is None:
            return
        if len(self.indices) != len(infills):
            raise RuntimeError("MOEA/D batch indices and offspring counts differ.")
        combined = Population.merge(self.pop, infills)
        self._refresh_ideal(combined)

        original = self.pop
        updated = original.copy()
        contenders: list[list[int]] = [[] for _ in range(len(original))]
        for offspring_index, subproblem in enumerate(self.indices):
            for neighbor in self.neighbors[int(subproblem)]:
                contenders[int(neighbor)].append(int(offspring_index))

        for slot, candidate_indices in enumerate(contenders):
            best = original[slot]
            for candidate_index in candidate_indices:
                candidate = infills[candidate_index]
                if self._candidate_better(candidate, best, slot):
                    best = candidate
            updated[slot] = best.copy()
        self.pop = updated


class ConstrainedMOPSOCD(MOPSO_CD):
    """MOPSO-CD with binary-feasibility pbest and archive logic."""

    def _setup(self, problem, **kwargs) -> None:
        Algorithm._setup(self, problem, **kwargs)
        # Algorithm.advance() reserves ``archive`` for pymoo Archive objects
        # exposing ``add``.  Keep the leader archive separate so no hidden
        # unconstrained archive update is performed by the base class.
        self.archive = None
        self.leader_archive = Population.empty()
        lower, upper = problem.bounds()
        self.v_max = self.max_velocity_rate * (upper - lower)
        self.velocities = None
        self.pbest = None
        self.pbest_f = None

    @staticmethod
    def _deduplicate(population: Population) -> Population:
        if len(population) <= 1:
            return population
        _, indices = np.unique(
            np.round(population.get("X"), 12), axis=0, return_index=True
        )
        return population[np.sort(indices)]

    def _update_archive(self, population: Population) -> Population:
        combined = (
            Population.merge(self.leader_archive, population)
            if len(self.leader_archive)
            else population
        )
        combined = self._deduplicate(combined)
        if not len(combined):
            return Population.empty()
        feasible = _population_feasible(combined)
        candidates = combined[feasible] if np.any(feasible) else combined
        front = NonDominatedSorting().do(
            candidates.get("F"), only_non_dominated_front=True
        )
        selected = candidates[front]
        if len(selected) > self.archive_size:
            crowding = get_crowding_function("cd").do(selected.get("F"))
            selected = selected[
                np.argsort(-crowding, kind="stable")[: self.archive_size]
            ]
        return selected

    def _initialize_advance(self, infills=None, **kwargs) -> None:
        self.pop = infills
        self.leader_archive = self._update_archive(infills)
        self.pbest = infills.copy()
        self.pbest_f = infills.get("F").copy()

    def _select_diverse_leaders(self):
        if not len(self.leader_archive):
            return [
                self.pop[self.random_state.integers(0, len(self.pop))]
                for _ in range(self.pop_size)
            ]
        crowding = get_crowding_function("cd").do(
            self.leader_archive.get("F")
        )
        leaders = []
        for _ in range(self.pop_size):
            first = int(self.random_state.integers(0, len(self.leader_archive)))
            second = int(self.random_state.integers(0, len(self.leader_archive)))
            if first == second:
                leaders.append(self.leader_archive[first])
                continue
            selected = first if crowding[first] >= crowding[second] else second
            leaders.append(self.leader_archive[selected])
        return leaders

    @staticmethod
    def _dominates(first: np.ndarray, second: np.ndarray) -> bool:
        return bool(np.all(first <= second) and np.any(first < second))

    def _update_pbest(self, new_population: Population) -> None:
        for index, candidate in enumerate(new_population):
            incumbent = self.pbest[index]
            candidate_feasible = bool(candidate.CV[0] <= FLOAT_TOLERANCE)
            incumbent_feasible = bool(incumbent.CV[0] <= FLOAT_TOLERANCE)
            comparison = feasibility_compare(candidate_feasible, incumbent_feasible)
            replace = comparison < 0
            if comparison == 0:
                if self._dominates(candidate.F, incumbent.F):
                    replace = True
                elif not self._dominates(incumbent.F, candidate.F):
                    replace = bool(self.random_state.random() < 0.5)
            if replace:
                self.pbest[index] = candidate.copy()
                self.pbest_f[index] = candidate.F.copy()

    def _advance(self, infills=None, **kwargs) -> None:
        if infills is None:
            return
        self.leader_archive = self._update_archive(
            Population.merge(self.pop, infills)
        )
        self._update_pbest(infills)
        self.pop = infills

    def _set_optimum(self, **kwargs) -> None:
        self.opt = (
            self.leader_archive.copy()
            if len(self.leader_archive)
            else Population.empty()
        )


def build_algorithm(
    name: str, config: dict, initial_design: np.ndarray, seed: int
) -> Algorithm:
    sampling = FixedSampling(initial_design)
    population_size = int(config["population_size"])
    if name == "NSGA-II":
        settings = config["nsga2"]
        return NSGA2(
            pop_size=population_size,
            sampling=sampling,
            selection=TournamentSelection(func_comp=binary_feasibility_tournament),
            survival=BinaryFeasibilityRankAndCrowding(),
            crossover=SBX(
                prob=float(settings["crossover_probability"]),
                prob_var=float(settings["crossover_variable_probability"]),
                eta=float(settings["crossover_eta"]),
            ),
            mutation=PM(
                prob=float(settings["mutation_probability"]),
                prob_var=float(settings["mutation_variable_probability"]),
                eta=float(settings["mutation_eta"]),
            ),
            eliminate_duplicates=bool(settings["eliminate_duplicates"]),
        )
    if name == "MOEA/D":
        settings = config["moead"]
        first = np.linspace(0.0, 1.0, population_size)
        reference_directions = np.column_stack((first, 1.0 - first))
        return ConstrainedParallelMOEAD(
            reference_directions,
            n_neighbors=int(settings["neighbors"]),
            decomposition=Tchebicheff(),
            prob_neighbor_mating=float(settings["neighbor_mating_probability"]),
            sampling=sampling,
            crossover=SBX(
                prob=float(settings["crossover_probability"]),
                prob_var=float(settings["crossover_variable_probability"]),
                eta=float(settings["crossover_eta"]),
            ),
            mutation=PM(
                prob=float(settings["mutation_probability"]),
                prob_var=float(settings["mutation_variable_probability"]),
                eta=float(settings["mutation_eta"]),
            ),
            n_offsprings=int(settings["offspring_per_batch"]),
        )
    if name == "MOPSO-CD":
        settings = config["mopso_cd"]
        return ConstrainedMOPSOCD(
            pop_size=int(settings["swarm_size"]),
            w=float(settings["inertia"]),
            c1=float(settings["cognitive"]),
            c2=float(settings["social"]),
            max_velocity_rate=float(settings["maximum_velocity_rate"]),
            archive_size=int(settings["archive_size"]),
            sampling=sampling,
            seed=int(seed),
        )
    raise ValueError(f"Unknown algorithm: {name}")


@dataclass(frozen=True)
class RunSettings:
    algorithm: str
    seed: int
    budgets: tuple[int, ...]
    output_directory: Path
    hv_ideal: np.ndarray
    hv_reference: np.ndarray


def _front_path(
    output_directory: Path, algorithm: str, seed: int, budget: int
) -> Path:
    return (
        output_directory
        / "fronts"
        / f"{_safe_name(algorithm)}_seed{seed}_budget{budget}.npz"
    )


def _marker_path(output_directory: Path, algorithm: str, seed: int) -> Path:
    return (
        output_directory
        / "completed_runs"
        / f"{_safe_name(algorithm)}_seed{seed}.json"
    )


class BudgetCheckpoint(Callback):
    def __init__(
        self, problem: CommonConstrainedSurrogateProblem, settings: RunSettings
    ) -> None:
        super().__init__()
        self.problem = problem
        self.settings = settings
        self.pending = list(settings.budgets)
        self.started = time.perf_counter()
        self.bookkeeping_seconds = 0.0

    def notify(self, algorithm: Algorithm) -> None:
        callback_started = time.perf_counter()
        evaluated = int(algorithm.evaluator.n_eval)
        while self.pending and evaluated >= self.pending[0]:
            budget = int(self.pending.pop(0))
            if evaluated != budget or self.problem.evaluation_count != budget:
                raise RuntimeError(
                    f"{self.settings.algorithm} crossed checkpoint {budget}: "
                    f"evaluator={evaluated}, problem={self.problem.evaluation_count}."
                )
            checkpoint_time = time.perf_counter()
            total_seconds = (
                checkpoint_time
                - self.started
                - self.bookkeeping_seconds
                - (checkpoint_time - callback_started)
            )
            population = algorithm.pop
            output_population = (
                algorithm.leader_archive
                if self.settings.algorithm == "MOPSO-CD"
                and len(algorithm.leader_archive)
                else population
            )
            output_feasible = _population_feasible(output_population)
            output_x, output_f = nondominated_2d(
                output_population.get("X")[output_feasible],
                (
                    output_population.get("F")[output_feasible]
                    * self.problem.objective_scale
                    + self.problem.objective_minimum
                ),
            )
            hv = normalized_hv(
                output_f,
                self.settings.hv_ideal,
                self.settings.hv_reference,
            )
            internal = getattr(
                algorithm,
                "leader_archive",
                getattr(algorithm, "archive", Population.empty()),
            )
            if not isinstance(internal, Population):
                internal = Population.empty()
            path = _front_path(
                self.settings.output_directory,
                self.settings.algorithm,
                self.settings.seed,
                budget,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                x=output_x,
                objectives=output_f,
                objectives_optimizer_normalized=(
                    output_f - self.problem.objective_minimum
                ) / self.problem.objective_scale,
                cumulative_archive_x=self.problem.archive.x,
                cumulative_archive_objectives=self.problem.archive.f,
                algorithm=np.asarray(self.settings.algorithm),
                seed=np.asarray(self.settings.seed),
                budget=np.asarray(budget),
                actual_evaluations=np.asarray(evaluated),
                feasible_evaluations=np.asarray(
                    self.problem.feasible_evaluation_count
                ),
                normalized_hv=np.asarray(hv),
                elapsed_seconds=np.asarray(total_seconds),
                surrogate_seconds=np.asarray(self.problem.surrogate_seconds),
                constraint_seconds=np.asarray(self.problem.constraint_seconds),
                optimizer_seconds=np.asarray(
                    total_seconds
                    - self.problem.surrogate_seconds
                    - self.problem.constraint_seconds
                ),
                population_x=population.get("X"),
                population_f_normalized=population.get("F"),
                population_f_physical=(
                    population.get("F") * self.problem.objective_scale
                    + self.problem.objective_minimum
                ),
                population_cv=_population_cv(population),
                internal_archive_x=(
                    internal.get("X")
                    if len(internal)
                    else np.empty((0, INPUT_DIM))
                ),
                internal_archive_f_normalized=(
                    internal.get("F")
                    if len(internal)
                    else np.empty((0, OBJECTIVE_COUNT))
                ),
                internal_archive_f_physical=(
                    internal.get("F") * self.problem.objective_scale
                    + self.problem.objective_minimum
                    if len(internal)
                    else np.empty((0, OBJECTIVE_COUNT))
                ),
                internal_archive_cv=_population_cv(internal),
            )
        self.bookkeeping_seconds += time.perf_counter() - callback_started


def _load_front(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: data[name].copy() for name in data.files}


def _snapshot_output_front(
    data: dict[str, np.ndarray], algorithm: str
) -> tuple[np.ndarray, np.ndarray]:
    """Recover the algorithm's output front, not the cumulative evaluation archive."""

    if algorithm == "MOPSO-CD" and len(data["internal_archive_x"]):
        design = np.asarray(data["internal_archive_x"], dtype=np.float64)
        objectives = np.asarray(
            data["internal_archive_f_physical"], dtype=np.float64
        )
        cv = np.asarray(data["internal_archive_cv"], dtype=np.float64).reshape(-1)
    else:
        design = np.asarray(data["population_x"], dtype=np.float64)
        objectives = np.asarray(data["population_f_physical"], dtype=np.float64)
        cv = np.asarray(data["population_cv"], dtype=np.float64).reshape(-1)
    feasible = cv <= FLOAT_TOLERANCE
    return nondominated_2d(design[feasible], objectives[feasible])


def _all_checkpoints_exist(
    directory: Path, algorithm: str, seed: int, budgets: Sequence[int]
) -> bool:
    return _marker_path(directory, algorithm, seed).is_file() and all(
        _front_path(directory, algorithm, seed, budget).is_file()
        for budget in budgets
    )


def _run_one(
    surrogate: EnsembleMMoESurrogate,
    constraints: EngineeringConstraintEvaluator,
    lower: np.ndarray,
    upper: np.ndarray,
    objective_minimum: np.ndarray,
    objective_maximum: np.ndarray,
    config: dict,
    initial_design: np.ndarray,
    settings: RunSettings,
) -> None:
    problem = CommonConstrainedSurrogateProblem(
        surrogate,
        constraints,
        lower,
        upper,
        objective_minimum,
        objective_maximum,
    )
    algorithm = build_algorithm(
        settings.algorithm, config, initial_design, settings.seed
    )
    callback = BudgetCheckpoint(problem, settings)
    minimize(
        problem,
        algorithm,
        termination=("n_eval", max(settings.budgets)),
        seed=settings.seed,
        callback=callback,
        verbose=False,
        save_history=False,
        copy_algorithm=False,
    )
    if callback.pending:
        raise RuntimeError(f"Missing checkpoints: {callback.pending}")
    if problem.evaluation_count != max(settings.budgets):
        raise RuntimeError("Final actual evaluation count is not the exact budget.")
    marker = _marker_path(
        settings.output_directory, settings.algorithm, settings.seed
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "completed_at": _timestamp(),
                "algorithm": settings.algorithm,
                "seed": settings.seed,
                "actual_evaluations": problem.evaluation_count,
                "feasible_evaluations": problem.feasible_evaluation_count,
                "all_checkpoints_saved": True,
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * float(values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def _plot_hv(
    summary: pd.DataFrame,
    initial_baseline: pd.DataFrame,
    population_size: int,
    directory: Path,
) -> None:
    style = {
        "NSGA-II": "#D55E00",
        "MOEA/D": "#0072B2",
        "MOPSO-CD": "#009E73",
    }
    initial_mean = float(initial_baseline["HV"].mean())
    initial_std = float(initial_baseline["HV"].std(ddof=1))
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    for algorithm in ALGORITHMS:
        frame = summary.loc[summary["algorithm"] == algorithm]
        x = np.concatenate(
            ([0], frame["budget"].to_numpy(dtype=np.int64) - population_size)
        )
        mean = np.concatenate(([initial_mean], frame["HV_mean"].to_numpy()))
        deviation = np.concatenate(([initial_std], frame["HV_std"].to_numpy()))
        axis.plot(x, mean, marker="o", linewidth=1.6, label=algorithm, color=style[algorithm])
        axis.fill_between(x, mean - deviation, mean + deviation, alpha=0.18, color=style[algorithm])
    axis.set_xscale("symlog", linthresh=population_size)
    axis.set_xlabel("Additional surrogate evaluations after common initialization")
    axis.set_ylabel("Normalized hypervolume")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(directory / "HV_convergence.png", dpi=300)
    plt.close(figure)


def _representative_fronts(
    metrics: pd.DataFrame, directory: Path, final_budget: int
) -> pd.DataFrame:
    rows = []
    for algorithm in ALGORITHMS:
        subset = metrics.loc[
            (metrics["algorithm"] == algorithm)
            & (metrics["budget"] == final_budget)
        ].copy()
        median = float(subset["HV"].median())
        selected = subset.iloc[(subset["HV"] - median).abs().argmin()]
        data = _load_front(
            _front_path(directory, algorithm, int(selected["seed"]), final_budget)
        )
        front_x, front_f = _snapshot_output_front(data, algorithm)
        for point_index, (design, objective) in enumerate(
            zip(front_x, front_f)
        ):
            rows.append(
                {
                    "algorithm": algorithm,
                    "representative_seed": int(selected["seed"]),
                    "final_HV": float(selected["HV"]),
                    "point_index": point_index,
                    "Volu": float(objective[0]),
                    "St": float(objective[1]),
                    **{f"x{i + 1}": float(value) for i, value in enumerate(design)},
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(
        directory / "representative_pareto_fronts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    figure, axis = plt.subplots(figsize=(6.4, 4.6))
    markers = {"NSGA-II": "o", "MOEA/D": "s", "MOPSO-CD": "^"}
    for algorithm in ALGORITHMS:
        group = frame.loc[frame["algorithm"] == algorithm]
        if len(group):
            axis.scatter(
                group["Volu"], group["St"], s=24, alpha=0.8,
                marker=markers[algorithm], label=algorithm,
            )
    axis.set_xlabel(r"$V_{dam}$ ($10^4$ m$^3$)")
    axis.set_ylabel(r"$S_t$")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(directory / "representative_pareto_fronts.png", dpi=300)
    plt.close(figure)
    return frame


def postprocess(
    directory: Path,
    budgets: Sequence[int],
    seeds: Sequence[int],
    hv_ideal: np.ndarray,
    hv_reference: np.ndarray,
    objective_minimum: np.ndarray,
    objective_maximum: np.ndarray,
    initial_designs: dict[int, np.ndarray],
    surrogate: EnsembleMMoESurrogate,
    constraints: EngineeringConstraintEvaluator,
    population_size: int,
) -> None:
    rows = []
    coverage_rows = []
    final_front_rows = []
    final_budget = max(budgets)
    objective_scale = objective_maximum - objective_minimum
    initial_rows = []
    for seed in seeds:
        design = initial_designs[int(seed)]
        prediction = surrogate.predict(design).mean_physical
        constraint_batch = constraints.evaluate(design, prediction)
        _, initial_front = nondominated_2d(
            design[constraint_batch.feasible],
            prediction[constraint_batch.feasible, :OBJECTIVE_COUNT],
        )
        initial_rows.append(
            {
                "seed": int(seed),
                "initial_evaluations": int(population_size),
                "additional_evaluations": 0,
                "feasible_evaluations": int(constraint_batch.feasible.sum()),
                "feasible_rate": float(constraint_batch.feasible.mean()),
                "nondominated_count": len(initial_front),
                "min_Volu": (
                    float(initial_front[:, 0].min()) if len(initial_front) else np.nan
                ),
                "min_St": (
                    float(initial_front[:, 1].min()) if len(initial_front) else np.nan
                ),
                "HV": normalized_hv(initial_front, hv_ideal, hv_reference),
            }
        )
    initial_baseline = pd.DataFrame(initial_rows)
    initial_baseline.to_csv(
        directory / "initial_hv_baseline.csv", index=False, encoding="utf-8-sig"
    )
    for algorithm in ALGORITHMS:
        for seed in seeds:
            for budget in budgets:
                data = _load_front(_front_path(directory, algorithm, seed, budget))
                front_x, front = _snapshot_output_front(data, algorithm)
                hv = normalized_hv(front, hv_ideal, hv_reference)
                actual = int(data["actual_evaluations"].item())
                feasible_count = int(data["feasible_evaluations"].item())
                rows.append(
                    {
                        "algorithm": algorithm,
                        "seed": int(seed),
                        "budget": int(budget),
                        "actual_evaluations": actual,
                        "feasible_evaluations": feasible_count,
                        "feasible_rate": feasible_count / actual,
                        "nondominated_count": len(front),
                        "min_Volu": float(front[:, 0].min()) if len(front) else np.nan,
                        "min_St": float(front[:, 1].min()) if len(front) else np.nan,
                        "HV": hv,
                        "elapsed_seconds": float(data["elapsed_seconds"].item()),
                        "surrogate_seconds": float(data["surrogate_seconds"].item()),
                        "constraint_seconds": float(data["constraint_seconds"].item()),
                        "optimizer_seconds": float(data["optimizer_seconds"].item()),
                    }
                )
                coverage_rows.append(
                    {
                        "algorithm": algorithm,
                        "seed": int(seed),
                        "budget": int(budget),
                        "points_better_than_ideal": int(
                            np.any(front < hv_ideal - FLOAT_TOLERANCE, axis=1).sum()
                        ) if len(front) else 0,
                        "points_not_dominated_by_reference": int(
                            np.any(front > hv_reference + FLOAT_TOLERANCE, axis=1).sum()
                        ) if len(front) else 0,
                    }
                )
                if budget == final_budget:
                    normalized = (front - objective_minimum) / objective_scale
                    for point_index, (design, objective, scaled) in enumerate(
                        zip(front_x, front, normalized)
                    ):
                        final_front_rows.append(
                            {
                                "algorithm": algorithm,
                                "seed": int(seed),
                                "budget": int(budget),
                                "point_index": point_index,
                                "Volu": float(objective[0]),
                                "St": float(objective[1]),
                                "Volu_optimizer_normalized": float(scaled[0]),
                                "St_optimizer_normalized": float(scaled[1]),
                                **{
                                    f"x{i + 1}": float(value)
                                    for i, value in enumerate(design)
                                },
                            }
                        )
    metrics = pd.DataFrame(rows)
    coverage = pd.DataFrame(coverage_rows)
    final_fronts = pd.DataFrame(final_front_rows)
    metrics.to_csv(directory / "optimizer_metrics_runs.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(directory / "hv_reference_diagnostics.csv", index=False, encoding="utf-8-sig")
    final_fronts.to_csv(
        directory / "final_pareto_fronts_all_runs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if int(coverage["points_not_dominated_by_reference"].max()) > 0:
        raise RuntimeError(
            "Fixed physical HV reference does not dominate every feasible front; "
            "see hv_reference_diagnostics.csv before changing one common setting."
        )

    summary = (
        metrics.groupby(["algorithm", "budget"], sort=False)
        .agg(
            HV_mean=("HV", "mean"), HV_std=("HV", "std"),
            feasible_rate_mean=("feasible_rate", "mean"),
            feasible_rate_std=("feasible_rate", "std"),
            nondominated_count_mean=("nondominated_count", "mean"),
            nondominated_count_std=("nondominated_count", "std"),
            min_Volu_mean=("min_Volu", "mean"), min_Volu_std=("min_Volu", "std"),
            min_St_mean=("min_St", "mean"), min_St_std=("min_St", "std"),
            elapsed_seconds_mean=("elapsed_seconds", "mean"),
            elapsed_seconds_std=("elapsed_seconds", "std"),
            surrogate_seconds_mean=("surrogate_seconds", "mean"),
            surrogate_seconds_std=("surrogate_seconds", "std"),
            constraint_seconds_mean=("constraint_seconds", "mean"),
            constraint_seconds_std=("constraint_seconds", "std"),
            optimizer_seconds_mean=("optimizer_seconds", "mean"),
            optimizer_seconds_std=("optimizer_seconds", "std"),
        )
        .reset_index()
    )
    summary.to_csv(directory / "optimizer_metrics_summary.csv", index=False, encoding="utf-8-sig")

    final = metrics.loc[metrics["budget"] == final_budget]
    pivot = final.pivot(index="seed", columns="algorithm", values="HV")
    if len(pivot) >= 3:
        friedman_p = float(
            friedmanchisquare(*(pivot[name] for name in ALGORITHMS)).pvalue
        )
    else:
        friedman_p = np.nan
    raw_p = []
    competitors = ("MOEA/D", "MOPSO-CD")
    for competitor in competitors:
        try:
            raw_p.append(
                float(
                    wilcoxon(
                        pivot["NSGA-II"], pivot[competitor], alternative="two-sided"
                    ).pvalue
                )
            )
        except ValueError:
            raw_p.append(1.0)
    adjusted = _holm_adjust(raw_p)
    statistics = pd.DataFrame(
        [
            {
                "budget": final_budget,
                "comparison": f"NSGA-II vs {competitor}",
                "friedman_overall_p": friedman_p,
                "wilcoxon_two_sided_raw_p": raw_p[index],
                "wilcoxon_two_sided_Holm_p": adjusted[index],
                "alpha": 0.05,
            }
            for index, competitor in enumerate(competitors)
        ]
    )
    statistics.to_csv(directory / "optimizer_statistical_tests.csv", index=False, encoding="utf-8-sig")
    representatives = _representative_fronts(metrics, directory, final_budget)
    _plot_hv(summary, initial_baseline, population_size, directory)
    workbook_path = directory / "optimizer_comparison_results.xlsx"
    try:
        writer = pd.ExcelWriter(workbook_path, engine="openpyxl")
    except PermissionError:
        workbook_path = directory / "optimizer_comparison_results_final_front.xlsx"
        writer = pd.ExcelWriter(workbook_path, engine="openpyxl")
    with writer:
        metrics.to_excel(writer, sheet_name="RunMetrics", index=False)
        summary.to_excel(writer, sheet_name="BudgetSummary", index=False)
        final.to_excel(writer, sheet_name="FinalRuns", index=False)
        statistics.to_excel(writer, sheet_name="Statistics", index=False)
        coverage.to_excel(writer, sheet_name="HVDiagnostics", index=False)
        representatives.to_excel(writer, sheet_name="RepresentativePF", index=False)
        final_fronts.to_excel(writer, sheet_name="FinalPFAllRuns", index=False)
        initial_baseline.to_excel(writer, sheet_name="InitialBaseline", index=False)


def _validate_definition(
    config: dict, model_path: Path, metadata_path: Path, dataset_path: Path
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if not model_path.is_file() or not metadata_path.is_file() or not dataset_path.is_file():
        raise FileNotFoundError("Model, metadata, or dataset path is missing.")
    targets = pd.read_excel(dataset_path, sheet_name="TrainY")
    independent = pd.read_excel(dataset_path, sheet_name="ValY")
    if targets.shape != (400, 41) or independent.shape != (80, 41):
        raise ValueError("Expected 400 development and 80 independent-test responses.")
    output_names = tuple(map(str, targets.columns))
    if tuple(independent.columns) != output_names:
        raise ValueError("Development and independent-test output names differ.")
    actual_min = targets.iloc[:, :2].min().to_numpy(dtype=np.float64)
    actual_max = targets.iloc[:, :2].max().to_numpy(dtype=np.float64)
    configured_min = np.asarray(config["objective_normalization_min"], dtype=np.float64)
    configured_max = np.asarray(config["objective_normalization_max"], dtype=np.float64)
    if not np.allclose(actual_min, configured_min, rtol=0.0, atol=1e-12):
        raise ValueError("Fixed objective-normalization minima do not match the manifest.")
    if not np.allclose(actual_max, configured_max, rtol=0.0, atol=1e-12):
        raise ValueError("Fixed objective-normalization maxima do not match the manifest.")
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    split = metadata.get("split", {})
    if [split.get(name) for name in ("train", "validation", "calibration", "independent_test")] != [256, 64, 80, 80]:
        raise ValueError("The fixed model data split is not 256/64/80 + 80.")
    if int(metadata.get("model_diagnostics", {}).get("member_count", -1)) != 5:
        raise ValueError("The common checkpoint is not a five-member ensemble.")
    if int(metadata.get("seed", -1)) != int(config["model_split_seed"]):
        raise ValueError("Checkpoint seed and configured model split seed differ.")
    if tuple(metadata.get("output_names", ())) != output_names:
        raise ValueError("Checkpoint metadata and dataset outputs differ.")
    return configured_min, configured_max, output_names


def _write_manifest(
    directory: Path,
    config: dict,
    model_path: Path,
    metadata_path: Path,
    dataset_path: Path,
    device: str,
    lower: np.ndarray,
    upper: np.ndarray,
    constraints: EngineeringConstraintEvaluator,
    budgets: Sequence[int],
    seeds: Sequence[int],
) -> None:
    manifest = {
        "created_at": _timestamp(),
        "configuration": str(CONFIG_PATH),
        "model_checkpoint": str(model_path),
        "model_checkpoint_sha256": _sha256(model_path),
        "model_metadata": str(metadata_path),
        "model_metadata_sha256": _sha256(metadata_path),
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "model_split_seed": int(config["model_split_seed"]),
        "development_samples": 400,
        "independent_test_usage": "excluded",
        "finite_element_evaluations": 0,
        "device": device,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "algorithms": list(ALGORITHMS),
        "budgets": list(map(int, budgets)),
        "seeds": list(map(int, seeds)),
        "initial_population_counted_in_budget": True,
        "one_continuous_run_per_algorithm_seed": True,
        "design_lower": lower.tolist(),
        "design_upper": upper.tolist(),
        "constraint_count": constraints.count,
        "constraint_handling": (
            "binary hard feasibility: CV=0 if all checks pass, otherwise CV=1; "
            "ties use each optimizer's objective-fitness rule"
        ),
        "optimizer_constraint_count": 1,
        "engineering_check_count": constraints.count,
        "constraint_sign_in_code": "pymoo G <= 0; manuscript margins g >= 0 are negated",
        "parameterization_invariants": {
            "half_chord_symmetry": (
                "X_l and X_r are fixed by the project valley geometry and are "
                "not changed by the 16 design multipliers"
            )
        },
        "objective_normalization": {
            "minimum_physical": config["objective_normalization_min"],
            "maximum_physical": config["objective_normalization_max"],
            "fixed_across_algorithms_and_runs": True,
        },
        "stress_sign_convention": {
            "negative": "tension",
            "positive": "compression",
            "tensile_magnitude_limit_MPa": TENSILE_MAGNITUDE_LIMIT_MPA,
            "compressive_magnitude_limit_MPa": COMPRESSIVE_MAGNITUDE_LIMIT_MPA,
            "signed_interval_MPa": [SIGNED_STRESS_LOWER_MPA, SIGNED_STRESS_UPPER_MPA],
            "responses_checked": list(constraints.stress_names),
        },
        "hv": {
            "ideal_physical": config["hv_ideal_physical"],
            "reference_physical": config["hv_reference_physical"],
            "normalized_reference": [1.0, 1.0],
            "indicator_front": (
                "algorithm output front: feasible nondominated current population "
                "for NSGA-II and MOEA/D; feasible nondominated leader archive "
                "for MOPSO-CD"
            ),
            "mopso_archive_capacity": int(config["mopso_cd"]["archive_size"]),
        },
        "algorithm_settings": {
            "NSGA-II": config["nsga2"],
            "MOEA/D": config["moead"],
            "MOPSO-CD": config["mopso_cd"],
        },
    }
    with (directory / "run_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    pd.DataFrame(
        {
            "constraint_index": np.arange(constraints.count),
            "constraint_name": constraints.names,
        }
    ).to_csv(directory / "constraint_mapping.csv", index=False, encoding="utf-8-sig")


def run_benchmark(
    *,
    preflight: bool = False,
    postprocess_only: bool = False,
) -> Path:
    config = load_comparison_config()
    budgets = tuple(map(int, config["budgets"]))
    seeds = tuple(map(int, config["seeds"]))
    experiment_name = str(config["experiment_name"])
    directory = HERE / "runs" / experiment_name
    directory.mkdir(parents=True, exist_ok=True)

    model_path = _resolve_path(config["model_checkpoint"])
    metadata_path = _resolve_path(config["model_metadata"])
    dataset_path = _resolve_path(config["dataset"])
    objective_minimum, objective_maximum, output_names = _validate_definition(
        config, model_path, metadata_path, dataset_path
    )
    lower, upper = load_reference_bounds(REGISTRY_PATH)
    constraints = EngineeringConstraintEvaluator(output_names)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(int(config["timing"]["torch_threads"]))
    surrogate = EnsembleMMoESurrogate.load(model_path, device=device)
    surrogate.predict(np.ones((2, INPUT_DIM), dtype=np.float64))
    hv_ideal = np.asarray(config["hv_ideal_physical"], dtype=np.float64)
    hv_reference = np.asarray(config["hv_reference_physical"], dtype=np.float64)
    if np.any(hv_reference <= hv_ideal):
        raise ValueError("Every physical HV reference coordinate must exceed ideal.")

    initial_directory = directory / "initial_designs"
    initial_directory.mkdir(exist_ok=True)
    initial_designs = {}
    population_size = int(config["population_size"])
    for seed in seeds:
        design = generate_initial_lhs(seed, lower, upper, population_size)
        initial_designs[int(seed)] = design
        np.savez_compressed(
            initial_directory / f"lhs_seed{seed}.npz",
            x=design,
            seed=np.asarray(seed),
            lower=lower,
            upper=upper,
        )
        for algorithm in ALGORITHMS:
            built = build_algorithm(algorithm, config, design, seed)
            if getattr(built, "pop_size", population_size) != population_size:
                raise RuntimeError(
                    f"{algorithm} does not use {population_size} initial candidates."
                )

    _write_manifest(
        directory,
        config,
        model_path,
        metadata_path,
        dataset_path,
        device,
        lower,
        upper,
        constraints,
        budgets,
        seeds,
    )
    print(
        f"Common five-member DE-MMoE: {model_path}\n"
        f"Constraints: {constraints.count} engineering checks aggregated to binary CV; "
        f"signed stress interval=[{SIGNED_STRESS_LOWER_MPA}, {SIGNED_STRESS_UPPER_MPA}] MPa\n"
        f"Algorithms: {', '.join(ALGORITHMS)}\n"
        f"Continuous checkpoints: {list(budgets)}; paired seeds: {list(seeds)}\n"
        f"Device: {device}; FE evaluations: 0\n"
        f"Output: {directory}",
        flush=True,
    )
    if preflight:
        print("PREFLIGHT PASSED: no optimization was executed.", flush=True)
        return directory

    if not postprocess_only:
        for seed_index, seed in enumerate(seeds):
            order = list(ALGORITHMS)
            shift = seed_index % len(order)
            order = order[shift:] + order[:shift]
            for algorithm in order:
                if _all_checkpoints_exist(directory, algorithm, seed, budgets):
                    print(f"Skip completed: {algorithm}, seed={seed}", flush=True)
                    continue
                started = time.perf_counter()
                _run_one(
                    surrogate,
                    constraints,
                    lower,
                    upper,
                    objective_minimum,
                    objective_maximum,
                    config,
                    initial_designs[int(seed)],
                    RunSettings(
                        algorithm=algorithm,
                        seed=int(seed),
                        budgets=tuple(budgets),
                        output_directory=directory,
                        hv_ideal=hv_ideal,
                        hv_reference=hv_reference,
                    ),
                )
                print(
                    f"Completed {algorithm}, seed={seed}, budget={max(budgets)}, "
                    f"time={(time.perf_counter() - started) / 60.0:.2f} min",
                    flush=True,
                )
    postprocess(
        directory,
        budgets,
        seeds,
        hv_ideal,
        hv_reference,
        objective_minimum,
        objective_maximum,
        initial_designs,
        surrogate,
        constraints,
        population_size,
    )
    print(f"Benchmark completed: {directory}", flush=True)
    return directory
