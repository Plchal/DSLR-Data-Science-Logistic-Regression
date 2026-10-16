"""Model Training."""

import argparse
import sys
from typing import TypedDict, get_args

from pandas import DataFrame

from model import (
    HyperParameter,
    create_multiclass_training_set,
    create_training_set_dict,
    logistic_regression,
    softmax_regression,
)
from utils import load_csv, write_csv
from utils.loss_plot import plot_loss
from utils.variable import (
    BEST_FEATURES_FOR_TRAIN,
    COLS_TO_DROP,
    DATASET,
    DEFAULT_BATCH_SIZE,
    HOUSES,
    WEIGHTS_FILE,
    WEIGHTS_SOFTMAX_FILE,
    Optimizer,
)


class TrainResult(TypedDict):
    """TODO."""

    weights: DataFrame
    stats: DataFrame
    features_name: list[str]
    final_loss: float
    default_output: str
    histories: dict[str, list[float]]


def _pars_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A script that creates the .csv file containing the parameters of our logistic regression."
    )
    parser.add_argument("-f", "--file", type=str, help="Path of the file")
    parser.add_argument("--softmax", action="store_true", help="Train one softmax classifier instead of one-vs-all")
    parser.add_argument("--features", type=str, default=None, help="Comma-separated list of features to train on")
    parser.add_argument("--optimizer", choices=list(get_args(Optimizer)), default="batch")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--iteration", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None, help="Output file (default depends on the model)")
    parser.add_argument("-d", "--display", action="store_true", help="Display loss evolution")
    return parser.parse_args()


def _resolve_features(data_frame: DataFrame, features_arg: str | None) -> list[str] | None:
    valid = [col for col in data_frame.columns if col not in COLS_TO_DROP]
    if features_arg:
        features = [f.strip() for f in features_arg.split(",")]
        missing = [f for f in features if f not in valid]
        if missing:
            print(f"Error : The subject '{missing[0]}' does not exist in the dataset.")
            print(f"Subjects existing : {' , '.join(valid)}")
            return None
        return features
    return BEST_FEATURES_FOR_TRAIN


def _train_softmax(df: DataFrame, features: list[str], hp: HyperParameter) -> TrainResult | None:
    multiclass_set = create_multiclass_training_set(df, features)
    if multiclass_set is None:
        return None
    res = softmax_regression(multiclass_set, hp)
    return TrainResult(
        weights=DataFrame(res.weights, columns=HOUSES),
        stats=multiclass_set.stats,
        features_name=multiclass_set.features_name,
        final_loss=res.loss_history[-1],
        default_output=WEIGHTS_SOFTMAX_FILE,
        histories={"softmax": res.loss_history},
    )


def _train_ovr(df: DataFrame, features: list[str], hp: HyperParameter) -> TrainResult | None:
    tdict = create_training_set_dict(df, features)
    if tdict is None:
        return None
    results = {h: logistic_regression(tdict[h], hp) for h in HOUSES}
    final_loss = sum(r.loss_history[-1] for r in results.values()) / len(HOUSES)
    return TrainResult(
        weights=DataFrame({h: results[h].weights for h in HOUSES}),
        stats=tdict[HOUSES[0]].stats,
        features_name=tdict[HOUSES[0]].features_name,
        final_loss=final_loss,
        default_output=WEIGHTS_FILE,
        histories={h: results[h].loss_history for h in HOUSES},
    )


def main() -> int:
    """The program generates a file containing the weights that will be used for the prediction."""
    args = _pars_args()

    data_frame = load_csv(args.file) if args.file else load_csv(DATASET)
    if data_frame is None:
        return 1

    features = _resolve_features(data_frame, args.features)
    if features is None:
        return 1

    optimizer: Optimizer = args.optimizer
    hyperparam = HyperParameter(
        alpha=args.alpha,
        iteration=args.iteration,
        optimizer=optimizer,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    out = (
        _train_softmax(data_frame, features, hyperparam)
        if args.softmax
        else _train_ovr(data_frame, features, hyperparam)
    )
    if out is None:
        return 1

    weights = out["weights"]
    weights["mean"] = [0, *out["stats"].loc["mean"].tolist()]
    weights["std"] = [1, *out["stats"].loc["std"].tolist()]
    weights["feature"] = out["features_name"]

    output = args.output or out["default_output"]
    written = write_csv(output, weights)
    mode = "softmax" if args.softmax else "one-vs-all"

    print(
        f"[{mode}] optimizer={hyperparam.optimizer} alpha={args.alpha} iter={args.iteration} loss={out['final_loss']:.4f} -> {output}"
    )
    if args.display:
        plot_loss(out["histories"], optimizer)
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
