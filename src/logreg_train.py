"""Model Training."""

import argparse
import sys
from typing import get_args

from pandas import DataFrame

from create_training_set import create_multiclass_training_set, create_training_set_dict
from logistic_regression import HyperParameter, logistic_regression, softmax_regression
from utils import load_csv, write_csv
from variable import (
    DATASET,
    DEFAULT_BATCH_SIZE,
    HOUSES,
    WEIGHTS_FILE,
    WEIGHTS_SOFTMAX_FILE,
    Optimizer,
)


def main() -> int:
    """The program generates a file containing the weights that will be used for the prediction."""
    parser = argparse.ArgumentParser(
        description="A script that creates the .csv file containing the parameters of our logistic regression."
    )
    parser.add_argument("-f", "--file", type=str, help="Path of the file")
    parser.add_argument("--softmax", action="store_true", help="Train one softmax classifier instead of one-vs-all")
    parser.add_argument("--optimizer", choices=list(get_args(Optimizer)), default="batch")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--iteration", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None, help="Output file (default depends on the model)")
    args = parser.parse_args()

    data_frame = load_csv(args.file) if args.file else load_csv(DATASET)
    if data_frame is None:
        return 1

    optimizer: Optimizer = args.optimizer
    hyperparam = HyperParameter(
        alpha=args.alpha,
        iteration=args.iteration,
        optimizer=optimizer,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    if args.softmax:
        multiclass_set = create_multiclass_training_set(data_frame)
        if multiclass_set is None:
            return 1
        result = softmax_regression(multiclass_set, hyperparam)
        weights = DataFrame(result.weights, columns=HOUSES)
        stats = multiclass_set.stats
        features_name = multiclass_set.features_name
        final_loss = result.loss_history[-1]
        default_output = WEIGHTS_SOFTMAX_FILE
    else:
        training_set_dict = create_training_set_dict(data_frame)
        if training_set_dict is None:
            return 1
        results = {house: logistic_regression(training_set_dict[house], hyperparam) for house in HOUSES}
        weights = DataFrame({house: results[house].weights for house in HOUSES})
        stats = training_set_dict[HOUSES[0]].stats
        features_name = training_set_dict[HOUSES[0]].features_name
        final_loss = sum(result.loss_history[-1] for result in results.values()) / len(HOUSES)
        default_output = WEIGHTS_FILE

    weights["mean"] = [0, *stats.loc["mean"].tolist()]
    weights["std"] = [1, *stats.loc["std"].tolist()]
    weights["feature"] = features_name

    output = args.output or default_output
    written = write_csv(output, weights)
    mode = "softmax" if args.softmax else "one-vs-all"
    print(f"[{mode}] optimizer={optimizer} alpha={args.alpha} iter={args.iteration} loss={final_loss:.4f} -> {output}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
