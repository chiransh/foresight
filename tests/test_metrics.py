import numpy as np

from foresight.metrics import mape, smape, wape


def test_mape_perfect_prediction_is_zero():
    y = np.array([100.0, 200.0, 300.0])
    assert mape(y, y) == 0.0


def test_mape_known_value():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    # errors: 10/100=10%, 20/200=10% -> mean 10%
    assert mape(y_true, y_pred) == 10.0


def test_wape_weights_by_volume_not_by_series_count():
    y_true = np.array([1000.0, 10.0])
    y_pred = np.array([900.0, 20.0])
    # total abs error = 100 + 10 = 110; total true = 1010
    assert round(wape(y_true, y_pred), 2) == round(110 / 1010 * 100, 2)


def test_smape_symmetric_between_over_and_under_prediction():
    over = smape(np.array([100.0]), np.array([120.0]))
    under = smape(np.array([120.0]), np.array([100.0]))
    assert round(over, 6) == round(under, 6)


def test_metrics_ignore_zero_true_rows_for_mape():
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([5.0, 110.0])
    # the zero-actual row would blow up a naive MAPE; it should be excluded
    assert mape(y_true, y_pred) == 10.0
