"""Small regression tests for the GPR baseline implementations."""

from __future__ import annotations

import unittest

import numpy as np

try:
    from .models_sklearn import IndependentGPR, PCALatentGPR, build_kernel
except ImportError as error:  # pragma: no cover - depends on the selected environment
    IndependentGPR = None
    PCALatentGPR = None
    build_kernel = None
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None


@unittest.skipIf(IMPORT_ERROR is not None, f"scikit-learn unavailable: {IMPORT_ERROR}")
class SklearnModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(2025)
        cls.x = rng.uniform(-1.0, 1.0, size=(24, 3))
        cls.y = np.column_stack(
            (
                cls.x[:, 0] + 0.2 * cls.x[:, 1],
                cls.x[:, 1] ** 2 - cls.x[:, 2],
                np.sin(cls.x[:, 0]) + cls.x[:, 2],
            )
        )

    def test_kernel_choice_is_not_ignored(self) -> None:
        self.assertIn("RBF", repr(build_kernel("rbf", 3)))
        self.assertIn("Matern", repr(build_kernel("matern32", 3)))
        self.assertIn("nu=1.5", repr(build_kernel("matern32", 3)))
        self.assertIn("nu=2.5", repr(build_kernel("matern52", 3)))

    def test_independent_gpr_returns_positive_outputwise_variance(self) -> None:
        model = IndependentGPR(
            kernel="matern32",
            alpha=1e-4,
            n_restarts_optimizer=0,
            random_state=2025,
        ).fit(self.x[:18], self.y[:18])
        prediction = model.predict_distribution(self.x[18:])
        self.assertEqual(prediction.mean.shape, (6, 3))
        self.assertEqual(prediction.variance.shape, (6, 3))
        self.assertTrue(np.isfinite(prediction.mean).all())
        self.assertTrue((prediction.variance > 0.0).all())

    def test_pca_gpr_reconstructs_mean_and_variance_to_output_space(self) -> None:
        model = PCALatentGPR(
            kernel="rbf",
            explained_variance=0.95,
            alpha=1e-4,
            n_restarts_optimizer=0,
            random_state=2025,
        ).fit(self.x[:18], self.y[:18])
        prediction = model.predict_distribution(self.x[18:])
        self.assertEqual(prediction.mean.shape, (6, 3))
        self.assertEqual(prediction.variance.shape, (6, 3))
        self.assertTrue((prediction.variance > 0.0).all())
        self.assertGreaterEqual(model.retained_components, 1)
        self.assertLessEqual(model.retained_components, 3)


if __name__ == "__main__":
    unittest.main()
