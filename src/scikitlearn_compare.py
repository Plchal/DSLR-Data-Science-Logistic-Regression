"""Script for the comparative evaluation of the logistic regression model."""

import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.multiclass import OneVsRestClassifier

sys.path.insert(0, "src")
import model
from model import BinaryLogisticRegression, PredictionSet, TrainingSet, standard_score
from utils import load_csv
from utils.variable import BEST_FEATURES_FOR_TRAIN, BETA0_NAME, DATASET, EPSILON, HOUSES, Matrix

FEATURES = BEST_FEATURES_FOR_TRAIN


def _preprocess(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    med = train[features].median()
    tr, te = train[features].fillna(med), test[features].fillna(med)
    stats = pd.DataFrame([tr.mean().values, tr.std(ddof=1).values], index=["mean", "std"], columns=features)
    return tr, te, stats


def _build_design_matrix(feature: Matrix, stats: pd.DataFrame) -> Matrix:
    mean = stats.loc["mean"].values
    std = stats.loc["std"].values
    standardized = standard_score(feature, mean, std)
    one_value_col = np.ones(feature.shape[0])
    return np.column_stack([one_value_col, standardized])


def _predict_ours_ovr(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, stats: pd.DataFrame) -> Matrix:
    proba = np.zeros((len(x_test), len(HOUSES)))
    for k, house in enumerate(HOUSES):
        target = (y_train == house).astype(float).to_numpy()
        ts = TrainingSet(list(x_train.columns), x_train.to_numpy(), target, stats)
        logistic_regression = model.LogisticRegression(BinaryLogisticRegression())
        weights = logistic_regression.train(ts).weights
        prediction_set = PredictionSet(_build_design_matrix(x_test.to_numpy(), stats), weights)
        proba[:, k] = logistic_regression.create_prediction(prediction_set)
    return proba


def _predict_ours_softmax(
    x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, stats: pd.DataFrame
) -> Matrix:
    target = np.column_stack([(y_train == house).astype(float).to_numpy() for house in HOUSES])
    ts = TrainingSet([BETA0_NAME, *x_train.columns], x_train.to_numpy(), target, stats)
    logistic_regression = model.LogisticRegression(model.MultinomialLogisticRegression())
    weights = logistic_regression.train(ts).weights
    prediction_set = PredictionSet(_build_design_matrix(x_test.to_numpy(), stats), weights)
    return logistic_regression.create_prediction(prediction_set)


def _make_sk_ovr() -> OneVsRestClassifier:
    return OneVsRestClassifier(LogisticRegression(C=np.inf, solver="lbfgs", max_iter=5000))  # type: ignore[no-untyped-call]


def _make_sk_multinomial() -> LogisticRegression:
    return LogisticRegression(C=np.inf, solver="lbfgs", max_iter=5000)  # type: ignore[no-untyped-call]


def _align(proba: Matrix, classes: np.ndarray) -> Matrix:
    return proba[:, [list(classes).index(house) for house in HOUSES]]


def _normalize(proba: Matrix) -> Matrix:
    clipped = np.clip(proba, EPSILON, 1.0)
    return clipped / clipped.sum(axis=1, keepdims=True)


def _evaluate(proba: Matrix, y_true: pd.Series) -> tuple[float, float]:
    labels = np.array(HOUSES)[proba.argmax(1)]
    order = [HOUSES.index(house) for house in sorted(HOUSES)]
    return accuracy_score(y_true, labels), log_loss(y_true, _normalize(proba)[:, order])


def _record(
    results: dict[str, list[tuple[float, float, float]]], name: str, proba: Matrix, y_true: pd.Series, elapsed: float
) -> None:
    results[name].append((*_evaluate(proba, y_true), elapsed))


def run_cv(df: pd.DataFrame, features: list[str], model_name: str) -> None:
    """Performs cross-validation and displays comparative performance."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=0)  # type: ignore[no-untyped-call]
    results: dict[str, list[tuple[float, float, float]]] = {
        "ours OvR": [],
        "sklearn OvR": [],
        "ours softmax": [],
        "sklearn softmax": [],
    }
    for tr_idx, te_idx in cv.split(df, df["Hogwarts House"]):  # type: ignore[no-untyped-call]
        train, test = df.iloc[tr_idx], df.iloc[te_idx]
        tr, te, stats = _preprocess(train, test, features)
        y_train = train["Hogwarts House"]
        y_test = test["Hogwarts House"]
        xtr = _build_design_matrix(tr.to_numpy(), stats)
        xte = _build_design_matrix(te.to_numpy(), stats)

        for name, predict in {
            "ours OvR": _predict_ours_ovr,
            "ours softmax": _predict_ours_softmax,
        }.items():
            start = time.perf_counter()
            proba = predict(tr, te, y_train, stats)
            _record(results, name, proba, y_test, time.perf_counter() - start)

        start = time.perf_counter()
        sk_ovr = _make_sk_ovr().fit(xtr, y_train)
        elapsed_ovr = time.perf_counter() - start
        _record(results, "sklearn OvR", _align(sk_ovr.predict_proba(xte), sk_ovr.classes_), y_test, elapsed_ovr)

        start = time.perf_counter()
        sk_softmax = _make_sk_multinomial().fit(xtr, y_train)
        elapsed_softmax = time.perf_counter() - start
        _record(
            results,
            "sklearn softmax",
            _align(sk_softmax.predict_proba(xte), sk_softmax.classes_),
            y_test,
            elapsed_softmax,
        )

    print(f"\n=== {model_name} ({len(features)} features) ===")
    print(f"{'model':<18}{'accuracy':>18}{'log-loss':>12}{'time (s)':>12}")
    for name, values in results.items():
        acc = np.array([v[0] for v in values])
        loss = np.array([v[1] for v in values])
        timings = np.array([v[2] for v in values])
        print(
            f"{name:<18}{np.mean(acc) * 100:8.2f}% ± {np.std(acc) * 100:5.2f}"
            f"{np.mean(loss):12.4f}{np.mean(timings):12.3f}"
        )


def main() -> int:
    """Load the training dataset and run the comparison."""
    df = load_csv(DATASET)
    if df is None:
        return 1
    run_cv(df, FEATURES, "F4 selection")
    return 0


if __name__ == "__main__":
    sys.exit(main())
