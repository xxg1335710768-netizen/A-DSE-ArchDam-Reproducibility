"""Dependency-light tests for the data and metric protocol."""

from __future__ import annotations

import math
import unittest

import numpy as np

from .config import SEEDS
from .data_protocol import load_dataset, make_split, prepare_split
from .metrics import control_stress_metrics, gaussian_nll, weighted_geometric_mean


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset()

    def test_dataset_contract(self) -> None:
        self.assertEqual(self.dataset.development_x.shape, (400, 16))
        self.assertEqual(self.dataset.development_y.shape, (400, 41))
        self.assertEqual(self.dataset.test_x.shape, (80, 16))
        self.assertEqual(self.dataset.test_y.shape, (80, 41))
        self.assertEqual(self.dataset.output_names[:2], ("Volu", "St"))
        self.assertEqual(len(self.dataset.output_names[2:]), 39)

    def test_five_splits_are_disjoint_and_reproducible(self) -> None:
        for seed in SEEDS:
            first = make_split(seed)
            second = make_split(seed)
            first.validate()
            np.testing.assert_array_equal(first.train, second.train)
            np.testing.assert_array_equal(first.validation, second.validation)
            np.testing.assert_array_equal(first.calibration, second.calibration)

    def test_scalers_fit_training_rows_only(self) -> None:
        prepared = prepare_split(self.dataset, make_split(SEEDS[0]))
        np.testing.assert_allclose(prepared.x_train.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(prepared.y_train.mean(axis=0), 0.0, atol=1e-12)
        self.assertGreater(
            float(np.max(np.abs(prepared.x_validation.mean(axis=0)))),
            1e-4,
        )

    def test_gaussian_nll_uses_squared_residual(self) -> None:
        true = np.array([[2.0]])
        mean = np.array([[0.0]])
        variance = np.array([[1.0]])
        expected = 0.5 * math.log(2.0 * math.pi) + 2.0
        self.assertAlmostEqual(float(gaussian_nll(true, mean, variance)[0]), expected)

    def test_control_stress_columns_match_dataset(self) -> None:
        metrics = control_stress_metrics(
            self.dataset.test_y,
            self.dataset.test_y,
            self.dataset.output_names,
        )
        np.testing.assert_allclose(list(metrics.values()), 0.0, atol=1e-12)

    def test_taskwise_constant_scale_preserves_geometric_mean_ranking(self) -> None:
        model_standard_deviation = np.array(
            [[0.4, 0.9], [0.8, 0.7], [1.1, 1.3], [0.6, 0.5]]
        )
        scales = np.sqrt(np.array([[2.0, 0.25]]))
        raw_score = weighted_geometric_mean(model_standard_deviation)
        calibrated_score = weighted_geometric_mean(
            model_standard_deviation * scales
        )
        np.testing.assert_array_equal(np.argsort(raw_score), np.argsort(calibrated_score))
        ratio = calibrated_score / raw_score
        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
