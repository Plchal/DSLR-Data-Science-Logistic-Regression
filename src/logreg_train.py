"""Model Training."""

import argparse
import sys

from pandas import DataFrame

from create_training_set import create_training_set_dict
from logistic_regression import HyperParameter, logistic_regression
from utils import load_csv, write_csv
from variable import DATASET, HOUSES, WEIGHTS_FILE


def main() -> int:
    """The program generates a file containing the weights that will be used for the prediction."""
    parser = argparse.ArgumentParser(
        description="A script that creates the .csv file containing the parameters of our logistic regression."
    )
    parser.add_argument("-f", "--file", type=str, help="Path of the file")
    parser.add_argument("-mb", "--mini_batch", action="store_true", help="Use mini_batch optimisation")
    parser.add_argument("-sgd", action="store_true", help="Use stochastic_gradient_descent optimisation")
    args = parser.parse_args()

    data_frame = load_csv(args.file) if args.file else load_csv(DATASET)
    if data_frame is None:
        return 1

    training_set_dict = create_training_set_dict(data_frame)
    if training_set_dict is None:
        return 1

    hyperparam = HyperParameter(mini_batch=args.mini_batch, sgd=args.sgd)

    weights = DataFrame({house: logistic_regression(training_set_dict[house], hyperparam) for house in HOUSES})
    stats = training_set_dict[HOUSES[0]].stats
    weights["mean"] = [0, *stats.loc["mean"].tolist()]
    weights["std"] = [1, *stats.loc["std"].tolist()]
    weights["feature"] = training_set_dict[HOUSES[0]].features_name
    result = write_csv(WEIGHTS_FILE, weights)
    print(weights)
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
