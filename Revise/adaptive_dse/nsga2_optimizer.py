"""Side-effect-free NSGA-II adapter for A-DSE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize

from ._common import INPUT_DIM, OBJECTIVE_COUNT, finite_matrix, finite_vector
from .design_space_exploration import nondominated_mask
from .ensemble_surrogate import EnsembleMMoESurrogate


@dataclass(frozen=True)
class NSGA2Settings:
    population_size: int = 100
    generations: int = 100
    crossover_probability: float = 0.9
    crossover_eta: float = 15.0
    mutation_eta: float = 20.0
    seed: int = 2025


class _SurrogateProblem(Problem):
    def __init__(
        self,
        surrogate: EnsembleMMoESurrogate,
        lower: np.ndarray,
        upper: np.ndarray,
        geometry_feasible: Callable[[np.ndarray], bool],
        tensile_limit_mpa: float,
        compressive_limit_mpa: float,
    ) -> None:
        super().__init__(
            n_var=INPUT_DIM,
            n_obj=OBJECTIVE_COUNT,
            n_ieq_constr=3,
            xl=lower,
            xu=upper,
        )
        self.surrogate = surrogate
        self.geometry_feasible = geometry_feasible
        self.tensile_limit_mpa = float(tensile_limit_mpa)
        self.compressive_limit_mpa = float(compressive_limit_mpa)

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        prediction = self.surrogate.predict(x).mean_physical
        objectives = prediction[:, :OBJECTIVE_COUNT]
        stress = prediction[:, OBJECTIVE_COUNT:]
        if stress.shape[1] == 0:
            raise ValueError("The surrogate has no structural stress outputs.")
        geometry_violation = np.fromiter(
            (0.0 if self.geometry_feasible(row) else 1.0 for row in x),
            dtype=np.float64,
            count=len(x),
        )
        out["F"] = objectives
        out["G"] = np.column_stack(
            (
                geometry_violation,
                stress.max(axis=1) - self.tensile_limit_mpa,
                self.compressive_limit_mpa - stress.min(axis=1),
            )
        )


def optimize_surrogate_nsga2(
    surrogate: EnsembleMMoESurrogate,
    lower: np.ndarray,
    upper: np.ndarray,
    geometry_feasible: Callable[[np.ndarray], bool],
    *,
    settings: NSGA2Settings | None = None,
    tensile_limit_mpa: float = 9.78,
    compressive_limit_mpa: float = -1.50,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the unique feasible nondominated front in physical units."""

    lower = finite_vector(lower, INPUT_DIM, "lower")
    upper = finite_vector(upper, INPUT_DIM, "upper")
    if np.any(upper <= lower):
        raise ValueError("Every NSGA-II upper bound must exceed its lower bound.")
    config = settings or NSGA2Settings()
    if config.population_size < 4 or config.generations <= 0:
        raise ValueError("NSGA-II population and generation settings are invalid.")

    problem = _SurrogateProblem(
        surrogate,
        lower,
        upper,
        geometry_feasible,
        tensile_limit_mpa,
        compressive_limit_mpa,
    )
    algorithm = NSGA2(
        pop_size=config.population_size,
        sampling=LHS(),
        crossover=SBX(
            prob=config.crossover_probability,
            eta=config.crossover_eta,
        ),
        mutation=PM(eta=config.mutation_eta),
        eliminate_duplicates=True,
    )
    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", config.generations),
        seed=config.seed,
        verbose=False,
        save_history=False,
    )
    if result.X is None or result.F is None:
        raise RuntimeError("NSGA-II found no feasible solution.")
    x = finite_matrix(np.atleast_2d(result.X), INPUT_DIM, "NSGA-II X")
    objectives = finite_matrix(
        np.atleast_2d(result.F), OBJECTIVE_COUNT, "NSGA-II objectives"
    )
    unique = np.sort(np.unique(np.round(x, 12), axis=0, return_index=True)[1])
    x = x[unique]
    objectives = objectives[unique]
    nondominated = nondominated_mask(objectives)
    x = x[nondominated]
    objectives = objectives[nondominated]
    if len(x) < 3:
        raise RuntimeError(
            f"NSGA-II returned only {len(x)} unique feasible nondominated points."
        )
    order = np.argsort(objectives[:, 0], kind="stable")
    return x[order], objectives[order]
