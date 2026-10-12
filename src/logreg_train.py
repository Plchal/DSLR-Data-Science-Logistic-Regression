"""Model Training."""

import argparse
import sys

from pandas import DataFrame

from create_training_set import create_training_set_dict
from logistic_regression import logistic_regression
from utils import load_csv, write_csv
from variable import HOUSES, WEIGHTS_FILE


def main() -> int:
    """The program generates a file containing the weights that will be used for the prediction."""
    data_frame = _load_dataset()
    if data_frame is None:
        return 1
    training_set_dict = create_training_set_dict(data_frame)
    if training_set_dict is None:
        return 1
    weights = DataFrame({house: logistic_regression(training_set_dict[house]) for house in HOUSES})
    stats = training_set_dict[HOUSES[0]].stats
    weights["mean"] = [0, *stats.loc["mean"].tolist()]
    weights["std"] = [1, *stats.loc["std"].tolist()]
    weights["feature"] = training_set_dict[HOUSES[0]].features_name
    result = write_csv(WEIGHTS_FILE, weights)
    return 0 if result else 1


def _load_dataset() -> DataFrame | None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path of the file")
    args = parser.parse_args()
    return load_csv(args.file)


if __name__ == "__main__":
    sys.exit(main())
