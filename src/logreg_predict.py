"""Model Prediction."""

import argparse
import sys

from pandas import DataFrame

from model import create_prediction, create_prediction_set
from utils import load_csv, write_csv
from utils.variable import HOUSES, PREDICTION_FILE, WEIGHTS_FILE


def main() -> int:
    """Generate a file containing predictions using the training data."""
    data_frame = _load_dataset()
    if data_frame is None:
        return 1
    weights = load_csv(WEIGHTS_FILE)
    if weights is None:
        return 1
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
    result = write_csv(PREDICTION_FILE, DataFrame({"Index": range(size), "Hogwarts House": predictions}))
    return 0 if result else 1


def _load_dataset() -> DataFrame | None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path of the file")
    args = parser.parse_args()
    return load_csv(args.file)


if __name__ == "__main__":
    sys.exit(main())
