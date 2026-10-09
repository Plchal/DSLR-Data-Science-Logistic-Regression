"""Model Training."""

import argparse
import sys

from pandas import DataFrame

from src.logistic_regression import TrainingSet, logistic_regression
from src.stats import get_stats
from src.utils import load_csv, write_csv


def main() -> int:
    """The program generates a file containing the weights that will be used for the prediction."""
    data_frame = _load_dataset()
    if data_frame is None:
        return 0
    stats = get_stats(data_frame.iloc[:, 2:])
    training_set = _create_training_set(data_frame)
    weights = logistic_regression(training_set, stats)
    filename = "weights"
    result = write_csv(filename + ".csv", DataFrame(weights))
    return 0 if result else 1


def _load_dataset() -> DataFrame | None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path of the file")
    args = parser.parse_args()
    return load_csv(args.file)


def _create_training_set(data_frame: DataFrame) -> TrainingSet:
    house_col = "Hogwarts House"

    data_frame = data_frame.dropna()
    data_frame[house_col] = (data_frame[house_col] == "Ravenclaw").astype(int)  # TODO : use other House.
    x_raw = data_frame.iloc[:, 2:].to_numpy()
    return TrainingSet(x_raw, data_frame[house_col].to_numpy())


if __name__ == "__main__":
    sys.exit(main())
