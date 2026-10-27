"""Model Prediction."""

import argparse
import sys

import numpy as np
from pandas import DataFrame

from model import clean_data_frame, create_binary_prediction_set, create_multinomial_prediction_set
from model.binary_logistic_regression import BinaryLogisticRegression
from model.logistic_regression import LogisticRegression
from model.multinomial_logistic_regression import MultinomialLogisticRegression
from utils import load_csv, write_csv
from utils.variable import HOUSES, PREDICTION_FILE, WEIGHTS_FILE, WEIGHTS_SOFTMAX_FILE


def _multinomial_prediction(weights: DataFrame, test_df: DataFrame) -> list[str]:
    multiclass_set = create_multinomial_prediction_set(weights, test_df)
    logistic_regression = LogisticRegression(MultinomialLogisticRegression())
    probabilities = logistic_regression.create_prediction(multiclass_set)
    indices = np.argmax(probabilities, axis=1)
    return [HOUSES[index] for index in indices]


def _binary_prediction(weights: DataFrame, test_df: DataFrame) -> list[str]:
    prediction_sets = create_binary_prediction_set(weights, test_df)
    logistic_regression = LogisticRegression(BinaryLogisticRegression())
    predictions_by_house = {house: logistic_regression.create_prediction(prediction_sets[house]) for house in HOUSES}
    size = len(predictions_by_house[HOUSES[0]])
    predictions = []
    for i in range(size):
        scores = {house: predictions_by_house[house][i] for house in HOUSES}
        house_of_max_score = max(scores, key=lambda house: scores[house])
        predictions.append(house_of_max_score)
    return predictions


def _pars_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A script that predicts the house of each observation.")
    parser.add_argument("file", help="Path of the file")
    parser.add_argument("--softmax", action="store_true", help="Use the softmax weights instead of one-vs-all")
    parser.add_argument("--output", type=str, default=PREDICTION_FILE, help="Output file")
    return parser.parse_args()


def main() -> int:
    """Generate a file containing predictions using the training data."""
    args = _pars_args()

    data_frame = load_csv(args.file)
    if data_frame is None:
        return 1

    weights_name = WEIGHTS_SOFTMAX_FILE if args.softmax else WEIGHTS_FILE
    weights = load_csv(weights_name)
    if weights is None:
        return 1

    test_df = clean_data_frame(weights, data_frame, weights_name, args.file)
    if test_df is None:
        return 1

    predictions = _multinomial_prediction(weights, test_df) if args.softmax else _binary_prediction(weights, test_df)
    size = len(predictions)
    result = write_csv(args.output, DataFrame({"Index": range(size), "Hogwarts House": predictions}))
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
