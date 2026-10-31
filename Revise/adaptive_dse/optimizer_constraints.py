"""Shared engineering constraints for the optimizer comparison experiment.

The FE response convention used by this project is signed: positive interface
stress is compression and negative interface stress is tension.  Consequently,
every one of the 39 control responses must lie in [-1.50, 9.78] MPa.  The
geometry margins below reproduce the Type-0 checks in
``ArcDamParameter4_GX.DamParameter.IsConstraintAvailable``.  The margins are
retained for diagnostics, while optimization uses only binary feasibility:
CV=0 when every constraint is satisfied and CV=1 otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

from ._common import INPUT_DIM, OBJECTIVE_COUNT


TENSILE_MAGNITUDE_LIMIT_MPA = 1.50
COMPRESSIVE_MAGNITUDE_LIMIT_MPA = 9.78
SIGNED_STRESS_LOWER_MPA = -TENSILE_MAGNITUDE_LIMIT_MPA
SIGNED_STRESS_UPPER_MPA = COMPRESSIVE_MAGNITUDE_LIMIT_MPA


@dataclass(frozen=True)
class ConstraintBatch:
    """Diagnostic margins plus binary feasibility and CV."""

    names: tuple[str, ...]
    g: np.ndarray
    cv: np.ndarray
    feasible: np.ndarray


def feasibility_compare(feasible_a: bool, feasible_b: bool) -> int:
    """Apply feasibility priority without ranking violation magnitudes.

    Returns -1 when A is preferred, +1 when B is preferred, and 0 when the
    two solutions share the same feasibility state.  A tie is deliberately
    left to the optimizer-specific objective-fitness rule.
    """

    if bool(feasible_a) != bool(feasible_b):
        return -1 if feasible_a else 1
    return 0


class EngineeringConstraintEvaluator:
    """Evaluate all geometry and 39-response stress constraints identically."""

    def __init__(self, output_names: Sequence[str]) -> None:
        names = tuple(str(name) for name in output_names)
        if len(names) != 41 or names[:2] != ("Volu", "St"):
            raise ValueError("Expected outputs Volu, St and 39 stress responses.")
        self.output_names = names
        self.stress_names = names[OBJECTIVE_COUNT:]
        if len(self.stress_names) != 39:
            raise ValueError("Exactly 39 stress responses are required.")
        invalid = [
            name
            for name in self.stress_names
            if not (
                (name.startswith("UP_") or name.startswith("DOWN_"))
                and ("_S1_" in name or "_S3_" in name)
            )
        ]
        if invalid:
            raise ValueError(f"Unrecognized stress response mapping: {invalid}")
        self.tension_mask = np.asarray(
            ["_S3_" in name for name in self.stress_names], dtype=bool
        )
        self.compression_mask = np.asarray(
            ["_S1_" in name for name in self.stress_names], dtype=bool
        )
        if np.any(self.tension_mask == self.compression_mask):
            raise ValueError("Every stress output must map to exactly one limit.")

        package_directory = Path(__file__).resolve().parents[1]
        package_text = str(package_directory)
        if package_text not in sys.path:
            sys.path.insert(0, package_text)
        self.parameterization = importlib.import_module("ArcDamParameter4_GX")

        nominal_names, _, nominal_scales = self._geometry_margins(
            np.ones(INPUT_DIM, dtype=np.float64)
        )
        self.geometry_names = nominal_names
        self.geometry_scales = nominal_scales
        self.names = (
            *self.geometry_names,
            *tuple(
                (
                    f"tension_lower::{name}"
                    if is_tension
                    else f"compression_upper::{name}"
                )
                for name, is_tension in zip(
                    self.stress_names, self.tension_mask
                )
            ),
        )

    @property
    def count(self) -> int:
        return len(self.names)

    @staticmethod
    def _arc_length(x_value: float, twice_radius: float) -> float:
        return x_value * (1.0 + 0.1 * (2.0 * x_value / twice_radius) ** 2)

    def _geometry_margins(
        self, point: np.ndarray
    ) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
        model = self.parameterization.DamParameter(T=1)
        model.InitialParameter()
        model.ParameterUpdate(np.asarray(point, dtype=float).tolist())
        if model.Type != 0:
            raise ValueError("The comparison is defined for the Type-0 parabolic dam.")

        names: list[str] = []
        margins: list[float] = []
        scales: list[float] = []

        def add(name: str, margin: float, scale: float) -> None:
            names.append(name)
            margins.append(float(margin))
            scales.append(float(scale))

        height = float(model.Z_top - model.Z_bot)
        normalized_elevations = np.asarray(
            [(elevation - model.Z_bot) / height for elevation in model.Gong_h],
            dtype=np.float64,
        )

        # Abutment central-angle admissibility at every modeled elevation.
        for layer, (left_angle, right_angle) in enumerate(
            zip(model.Angle_l, model.Angle_r)
        ):
            add(f"angle_left_lower_L{layer:02d}", left_angle, 90.0)
            add(f"angle_left_upper_L{layer:02d}", 90.0 - left_angle, 90.0)
            add(f"angle_right_lower_L{layer:02d}", right_angle, 90.0)
            add(f"angle_right_upper_L{layer:02d}", 90.0 - right_angle, 90.0)

        for layer in range(model.Layer_num):
            x_mid_l = np.asarray(
                [position / 10.0 * model.X_l[layer] for position in range(11)]
            )
            x_mid_r = np.asarray(
                [-position / 10.0 * model.X_r[layer] for position in range(11)]
            )
            angle_l = np.arctan(x_mid_l / model.R_l[layer])
            angle_r = np.arctan(-x_mid_r / model.R_r[layer])

            arc_total_l = self._arc_length(
                model.X_l[layer], 2.0 * model.R_l[layer]
            )
            arc_l = np.asarray(
                [
                    self._arc_length(value, 2.0 * model.R_l[layer])
                    for value in x_mid_l
                ]
            )
            thickness_l = model.T_m[layer] + (
                model.T_l[layer] - model.T_m[layer]
            ) * (arc_l / arc_total_l) ** 2
            x_up_l = x_mid_l + thickness_l / 2.0 * np.sin(angle_l)
            x_down_l = x_mid_l - thickness_l / 2.0 * np.sin(angle_l)

            arc_total_r = self._arc_length(
                model.X_r[layer], 2.0 * model.R_r[layer]
            )
            arc_r = np.asarray(
                [
                    self._arc_length(value, 2.0 * model.R_r[layer])
                    for value in x_mid_r
                ]
            )
            thickness_r = model.T_m[layer] + (
                model.T_r[layer] - model.T_m[layer]
            ) * (arc_r / arc_total_r) ** 2
            x_up_r = x_mid_r - thickness_r / 2.0 * np.sin(angle_r)
            x_down_r = x_mid_r + thickness_r / 2.0 * np.sin(angle_r)

            y_mid_l = (
                model.Y_up[layer]
                - model.T_m[layer] / 2.0
                + x_mid_l**2 / (-2.0 * model.R_l[layer])
            )
            y_mid_r = (
                model.Y_up[layer]
                - model.T_m[layer] / 2.0
                + x_mid_r**2 / (-2.0 * model.R_r[layer])
            )
            y_up_l = y_mid_l + thickness_l / 2.0 * np.cos(angle_l)
            y_down_l = y_mid_l - thickness_l / 2.0 * np.cos(angle_l)
            y_up_r = y_mid_r + thickness_r / 2.0 * np.cos(angle_r)
            y_down_r = y_mid_r - thickness_r / 2.0 * np.cos(angle_r)

            surface_radii = {
                "radius_left_up": -x_up_l[-1] ** 2
                / (2.0 * (y_up_l[-1] - model.Y_up[layer])),
                "radius_left_down": -x_down_l[-1] ** 2
                / (
                    2.0
                    * (
                        y_down_l[-1]
                        - (model.Y_up[layer] - model.T_m[layer])
                    )
                ),
                "radius_right_up": -x_up_r[-1] ** 2
                / (2.0 * (y_up_r[-1] - model.Y_up[layer])),
                "radius_right_down": -x_down_r[-1] ** 2
                / (
                    2.0
                    * (
                        y_down_r[-1]
                        - (model.Y_up[layer] - model.T_m[layer])
                    )
                ),
            }
            for label, radius in surface_radii.items():
                add(f"{label}_L{layer:02d}", radius, 100.0)

            monotonic = {
                "x_left_down": np.diff(x_down_l),
                "x_left_up": np.diff(x_up_l),
                "x_right_down": -np.diff(x_down_r),
                "x_right_up": -np.diff(x_up_r),
            }
            for label, values in monotonic.items():
                for position, margin in enumerate(values):
                    add(
                        f"{label}_L{layer:02d}_P{position:02d}",
                        margin,
                        10.0,
                    )

            separation = x_down_l - x_down_r
            for position, margin in enumerate(separation):
                add(
                    f"downstream_separation_L{layer:02d}_P{position:02d}",
                    margin,
                    10.0,
                )
            add(
                f"abutment_separation_L{layer:02d}",
                separation[-1] - 10.0,
                10.0,
            )

            # Match IsConstraintAvailable exactly: its final-layer branch checks
            # overhang and smoothness, while convexity is checked by the later
            # ``elif`` branches only for the preceding layers.
            if layer < model.Layer_num - 1:
                elevation = normalized_elevations[layer]
                add(
                    f"crown_upstream_convexity_L{layer:02d}",
                    -(6.0 * model.A1 * elevation + 2.0 * model.B1),
                    10.0,
                )
                add(
                    f"crown_downstream_convexity_L{layer:02d}",
                    -(
                        6.0 * (model.A1 - model.A2) * elevation
                        + 2.0 * (model.B1 - model.B2)
                    ),
                    10.0,
                )

        overhang_up = abs(model.C1 / height)
        overhang_down = abs((model.C1 - model.C2) / height)
        add("overhang_upstream", 0.35 - overhang_up, 0.35)
        add("overhang_downstream", 0.35 - overhang_down, 0.35)

        add(
            "base_angle_left_lower",
            model.Angle_l[-1] - (model.Angle_l_ini - 6.0),
            15.0,
        )
        add(
            "base_angle_left_upper",
            model.Angle_l_ini + 9.0 - model.Angle_l[-1],
            15.0,
        )
        add(
            "base_angle_right_lower",
            model.Angle_r[-1] - (model.Angle_r_ini - 5.0),
            13.0,
        )
        add(
            "base_angle_right_upper",
            model.Angle_r_ini + 8.0 - model.Angle_r[-1],
            13.0,
        )

        for label, values, allowance in (
            ("crown_thickness_smooth", model.T_m, 4.0),
            ("left_thickness_smooth", model.T_l, 5.0),
            ("right_thickness_smooth", model.T_r, 5.0),
        ):
            for position in range(len(values) - 1):
                permitted_drop = allowance * ((position + 1) / 10.0) ** 0.4
                add(
                    f"{label}_P{position:02d}",
                    values[position + 1] - values[position] + permitted_drop,
                    allowance,
                )

        for label, value, lower, upper in (
            ("crest_crown_thickness", model.T_m[0], 9.95, 18.0),
            ("crest_left_thickness", model.T_l[0], 10.0, 18.0),
            ("crest_right_thickness", model.T_r[0], 10.0, 18.0),
        ):
            span = upper - lower
            add(f"{label}_lower", value - lower, span)
            add(f"{label}_upper", upper - value, span)

        add("downstream_stationary_height_lower", model.m0__, height)
        add("downstream_stationary_height_upper", height - model.m0__, height)

        result = np.asarray(margins, dtype=np.float64)
        scale = np.asarray(scales, dtype=np.float64)
        if not np.isfinite(result).all() or np.any(scale <= 0.0):
            raise ValueError("Non-finite geometry constraint margin or scale.")
        return tuple(names), result, scale

    def evaluate(self, design: np.ndarray, prediction: np.ndarray) -> ConstraintBatch:
        points = np.asarray(design, dtype=np.float64)
        responses = np.asarray(prediction, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != INPUT_DIM:
            raise ValueError("design must have shape (N, 16).")
        if responses.shape != (len(points), 41):
            raise ValueError("prediction must have shape (N, 41).")

        geometry_g = np.empty((len(points), len(self.geometry_names)))
        for row_index, point in enumerate(points):
            try:
                names, margins, scales = self._geometry_margins(point)
                if names != self.geometry_names or not np.array_equal(
                    scales, self.geometry_scales
                ):
                    raise RuntimeError("Geometry constraint schema changed.")
                geometry_g[row_index] = -margins / scales
            except Exception:
                # A numerical failure is itself a severe geometry violation.  A
                # fixed value preserves schema and never marks the point feasible.
                geometry_g[row_index] = 1.0e6

        stress = responses[:, OBJECTIVE_COUNT:]
        stress_g = np.empty_like(stress)
        stress_g[:, self.tension_mask] = (
            SIGNED_STRESS_LOWER_MPA - stress[:, self.tension_mask]
        ) / TENSILE_MAGNITUDE_LIMIT_MPA
        stress_g[:, self.compression_mask] = (
            stress[:, self.compression_mask] - SIGNED_STRESS_UPPER_MPA
        ) / COMPRESSIVE_MAGNITUDE_LIMIT_MPA
        g = np.column_stack((geometry_g, stress_g))
        feasible = np.all(g <= 1e-12, axis=1)
        cv = (~feasible).astype(np.float64)
        return ConstraintBatch(self.names, g, cv, feasible)
