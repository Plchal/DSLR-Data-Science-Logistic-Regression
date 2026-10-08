"""Model Training."""

import argparse
import sys

from pandas import DataFrame, Series

from logistic_regression import TrainingSet, logistic_regression
from stats import get_stats
from utils import load_csv, write_csv
from variable import HOUSES, Array1D, Array2D


def main() -> int:
    """The program generates a file containing the weights that will be used for the prediction."""
    data_frame = _load_dataset()
    if data_frame is None:
        return 1
    stats = get_stats(data_frame.iloc[:, 2:])  # TODO: not start to index 2 + check where dropna.
    training_set_dict = _create_training_set_dict(data_frame)
    if training_set_dict is None:
        return 1
    weight_dict = {}
    for house in HOUSES:
        weight_dict[house] = logistic_regression(training_set_dict[house], stats)
    filename = "weights"
    result = write_csv(filename + ".csv", DataFrame(weight_dict))
    return 0 if result else 1


def _load_dataset() -> DataFrame | None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path of the file")
    args = parser.parse_args()
    return load_csv(args.file)


def _create_training_set_dict(data_frame: DataFrame) -> dict[str, TrainingSet] | None:
    house_col = "Hogwarts House"

    if house_col not in data_frame.columns:
        print(house_col, "column not find")
        return None
    if len(data_frame.index) == 0:
        print("No data for the training")
        return None
    _clean_data_set(data_frame)
    target_dict = _get_tagets(data_frame[house_col])
    data_frame.drop(columns=[house_col], inplace=True)
    x_raw = data_frame.select_dtypes(include=["number"]).to_numpy()
    return _get_training_set(target_dict, x_raw)


def _clean_data_set(data_frame: DataFrame) -> None:
    index_col = "Index"

    if index_col in data_frame.columns:
        data_frame.drop(columns=[index_col], inplace=True)
    data_frame.dropna(inplace=True)


def _get_tagets(house_col: Series) -> dict[str, Array1D]:
    target_dict = {}
    for house in HOUSES:
        target_dict[house] = (house_col == house).astype(int).to_numpy()
    return target_dict


def _get_training_set(target_dict: dict[str, Array1D], x_raw: Array2D) -> dict[str, TrainingSet]:
    training_set_dict = {}
    for house in HOUSES:
        training_set_dict[house] = TrainingSet(x_raw, target_dict[house])
    return training_set_dict


if __name__ == "__main__":
    sys.exit(main())
