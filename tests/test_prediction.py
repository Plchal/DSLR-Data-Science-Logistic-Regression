"""Our own test for the prediction functions."""

import numpy as np
from pandas import DataFrame
from pytest import CaptureFixture

from model import (
    LogisticRegression,
    MultinomialLogisticRegression,
    PredictionSet,
    clean_data_frame,
    create_multinomial_prediction_set,
)
from utils.variable import HOUSES


def _make_weights() -> DataFrame:
    return DataFrame(
        {
            "Ravenclaw": [0.0, 10.0, 0.0],
            "Slytherin": [0.0, 0.0, 0.0],
            "Gryffindor": [0.0, 0.0, 0.0],
            "Hufflepuff": [0.0, 0.0, 10.0],
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
            "feature": ["beta0", "f1", "f2"],
        }
    )


def test_predict_softmax_expected_house() -> None:
    """The house with the highest logit must be predicted."""
    prediction_set = PredictionSet(
        design_matrix=np.array([[1.0, 100.0, 0.0], [1.0, 0.0, 100.0]]),
        weights=_make_weights()[HOUSES].to_numpy(dtype=float),
    )
    logistic_regression = LogisticRegression(MultinomialLogisticRegression())
    probabilities = logistic_regression.create_prediction(prediction_set)
    indices = np.argmax(probabilities, axis=1)
    result = [HOUSES[index] for index in indices]
    assert result == ["Ravenclaw", "Hufflepuff"]


def test_create_multiclass_prediction_set() -> None:
    """A valid weights/data pair must build a usable prediction set."""
    df = DataFrame({"f1": [100.0, 0.0], "f2": [0.0, 100.0]})
    prediction_set = create_multinomial_prediction_set(_make_weights(), df)
    assert prediction_set is not None
    assert prediction_set.design_matrix.shape == (2, 3)
    assert prediction_set.weights.shape == (3, len(HOUSES))
    logistic_regression = LogisticRegression(MultinomialLogisticRegression())
    probabilities = logistic_regression.create_prediction(prediction_set)
    indices = np.argmax(probabilities, axis=1)
    result = [HOUSES[index] for index in indices]
    assert result == ["Ravenclaw", "Hufflepuff"]


def test_create_multiclass_prediction_set_nan_imputed() -> None:
    """Missing values must be imputed with the mean (standardized zero)."""
    df = DataFrame({"f1": [np.nan], "f2": [100.0]})
    prediction_set = create_multinomial_prediction_set(_make_weights(), df)
    assert prediction_set is not None
    logistic_regression = LogisticRegression(MultinomialLogisticRegression())
    probabilities = logistic_regression.create_prediction(prediction_set)
    indices = np.argmax(probabilities, axis=1)
    result = [HOUSES[index] for index in indices]
    assert result == ["Hufflepuff"]


def test_create_multiclass_prediction_set_missing_feature(capsys: CaptureFixture[str]) -> None:
    """A feature absent from the data must fail with an explicit message."""
    df = DataFrame({"f2": [100.0]})
    result = clean_data_frame(_make_weights(), df, "weights_softmax.csv", "dataset.csv")
    captured = capsys.readouterr()
    assert result is None
    assert "Feature f1 not defined in dataset.csv" in captured.out


def test_create_multiclass_prediction_set_bad_columns(capsys: CaptureFixture[str]) -> None:
    """Weights with a missing house column must fail."""
    weights = _make_weights().drop(columns=["Hufflepuff"])
    result = clean_data_frame(weights, DataFrame({"f1": [1.0], "f2": [1.0]}), "weights.csv", "dataset.csv")
    captured = capsys.readouterr()
    assert result is None
    assert "Bad number of column in weights.csv" in captured.out
