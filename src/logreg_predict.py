"""Model Prediction."""

import argparse
import sys

from pandas import DataFrame

from model import create_multiclass_prediction_set, create_prediction, create_prediction_set, predict_softmax
from utils import load_csv, write_csv
from utils.variable import (
    HOUSES,
    PREDICTION_FILE,
    WEIGHTS_FILE,
    WEIGHTS_SOFTMAX_FILE,
)


def main() -> int:
    """Generate a file containing predictions using the training data."""
    parser = argparse.ArgumentParser(description="A script that predicts the house of each observation.")
    parser.add_argument("file", help="Path of the file")
    parser.add_argument("--softmax", action="store_true", help="Use the softmax weights instead of one-vs-all")
    parser.add_argument("--output", type=str, default=PREDICTION_FILE, help="Output file")
    args = parser.parse_args()

    data_frame = load_csv(args.file)
    if data_frame is None:
        return 1

    weights_name = WEIGHTS_SOFTMAX_FILE if args.softmax else WEIGHTS_FILE
    weights = load_csv(weights_name)
    if weights is None:
        return 1

    if args.softmax:
        multiclass_set = create_multiclass_prediction_set(weights, data_frame, weights_name, args.file)
        if multiclass_set is None:
            return 1
        predictions = predict_softmax(multiclass_set)
    else:
        prediction_sets = create_prediction_set(weights, data_frame)
        if prediction_sets is None:
            return 1
        predictions_by_house = {house: create_prediction(prediction_sets[house]) for house in HOUSES}
        size = len(predictions_by_house[HOUSES[0]])
        predictions = []
        for i in range(size):
            scores = {house: predictions_by_house[house][i] for house in HOUSES}
            house_of_max_score = max(scores, key=lambda house: scores[house])
            predictions.append(house_of_max_score)

    size = len(predictions)
    result = write_csv(args.output, DataFrame({"Index": range(size), "Hogwarts House": predictions}))
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
