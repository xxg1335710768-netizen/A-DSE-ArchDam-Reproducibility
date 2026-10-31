"""Focused tests for neural uncertainty helpers."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from .config import MC_DROPOUT_GRID
from .models_torch import (
    DeterministicMMoE,
    FittedTorchModel,
    MultiTaskMLP,
    predict_deterministic_ensemble,
    predict_mc_dropout,
)


class MCDropoutTests(unittest.TestCase):
    def test_probability_grid_has_nine_positive_dropout_configurations(self) -> None:
        self.assertEqual(len(MC_DROPOUT_GRID), 9)
        self.assertTrue(all(config["dropout"] > 0.0 for config in MC_DROPOUT_GRID))

    def test_prediction_is_reproducible_and_has_positive_model_variance(self) -> None:
        torch.manual_seed(11)
        model = MultiTaskMLP(
            input_dim=3,
            output_dim=2,
            hidden_dims=(8, 8),
            dropout=0.2,
        )
        fitted = FittedTorchModel(
            model=model,
            best_epoch=1,
            best_validation_loss=0.0,
            epochs_completed=1,
            fit_seconds=0.0,
            probabilistic=False,
            selection_metric="loss",
            best_validation_nll=None,
            best_validation_rmse=0.0,
            seed=2025,
        )
        x = np.arange(18, dtype=np.float32).reshape(6, 3) / 10.0
        first = predict_mc_dropout(fitted, x, mc_samples=40, seed=91)
        second = predict_mc_dropout(fitted, x, mc_samples=40, seed=91)
        np.testing.assert_allclose(first.mean, second.mean, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            first.epistemic_var,
            second.epistemic_var,
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(np.all(first.epistemic_var > 0.0))
        np.testing.assert_allclose(first.aleatoric_var, 0.0)
        self.assertFalse(model.training)

    def test_zero_dropout_is_rejected(self) -> None:
        model = MultiTaskMLP(
            input_dim=2,
            output_dim=1,
            hidden_dims=(4,),
            dropout=0.0,
        )
        fitted = FittedTorchModel(
            model=model,
            best_epoch=1,
            best_validation_loss=0.0,
            epochs_completed=1,
            fit_seconds=0.0,
            probabilistic=False,
            selection_metric="loss",
            best_validation_nll=None,
            best_validation_rmse=0.0,
            seed=1,
        )
        with self.assertRaises(ValueError):
            predict_mc_dropout(
                fitted,
                np.ones((2, 2), dtype=np.float32),
                mc_samples=2,
            )


class DeterministicEnsembleTests(unittest.TestCase):
    def test_member_disagreement_is_returned_as_model_variance(self) -> None:
        first_model = DeterministicMMoE(2, 2, 4, 3, 2)
        second_model = DeterministicMMoE(2, 2, 4, 3, 2)
        with torch.no_grad():
            for parameter in first_model.parameters():
                parameter.zero_()
            for parameter in second_model.parameters():
                parameter.zero_()
            second_model.towers[0][-1].bias.fill_(2.0)
            second_model.towers[1][-1].bias.fill_(4.0)

        def fitted(model: torch.nn.Module, seed: int) -> FittedTorchModel:
            return FittedTorchModel(
                model=model,
                best_epoch=1,
                best_validation_loss=0.0,
                epochs_completed=1,
                fit_seconds=0.0,
                probabilistic=False,
                selection_metric="rmse",
                best_validation_nll=None,
                best_validation_rmse=0.0,
                seed=seed,
            )

        moments = predict_deterministic_ensemble(
            [fitted(first_model, 1), fitted(second_model, 2)],
            np.ones((3, 2), dtype=np.float32),
        )
        np.testing.assert_allclose(moments.mean, [[1.0, 2.0]] * 3)
        np.testing.assert_allclose(moments.epistemic_var, [[1.0, 4.0]] * 3)
        np.testing.assert_allclose(moments.aleatoric_var, 0.0)


if __name__ == "__main__":
    unittest.main()
