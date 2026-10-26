import numpy as np

from src.nn.losses import MSE
from src.utils.metrics import mse


def test_mse_averages_over_examples_with_four_outputs():
    y_pred = np.array([[1.0, 2.0, 3.0, 4.0],
                       [2.0, 0.0, 0.0, 0.0]])
    y_true = np.zeros_like(y_pred)

    # Squared errors sum to 30 and 4 for the two examples: (30 + 4) / 2
    assert MSE().value(y_pred, y_true) == 17.0
    assert mse(y_pred, y_true) == 17.0
    np.testing.assert_array_equal(MSE().gradient(y_pred, y_true), y_pred)
