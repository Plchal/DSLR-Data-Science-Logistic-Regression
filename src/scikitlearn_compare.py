"""Script for the comparative evaluation of the logistic regression model."""

import sys

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.multiclass import OneVsRestClassifier

sys.path.insert(0, "src")
from model import TrainingSet, _build_design_matrix, _sigmoid, logistic_regression
from utils.variable import HOUSES

FEATURES = ["Astronomy", "Herbology", "Defense Against the Dark Arts", "Ancient Runes"]  # ou les 13
df = pd.read_csv("dataset_train.csv")


def _preprocess(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    med = train[features].median()
    tr, te = train[features].fillna(med), test[features].fillna(med)
    stats = pd.DataFrame([tr.mean().values, tr.std(ddof=1).values], index=["mean", "std"], columns=features)
    return tr, te, stats


def _predict_ours(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, stats: pd.DataFrame) -> np.ndarray:
    proba = np.zeros((len(x_test), len(HOUSES)))
    for k, house in enumerate(HOUSES):
        target = (y_train == house).astype(float).to_numpy()
        ts = TrainingSet(list(x_train.columns), x_train.to_numpy(), target, stats)
        weights = logistic_regression(ts).weights
        proba[:, k] = _sigmoid(_build_design_matrix(x_test.to_numpy(), stats) @ weights)
    return np.array(HOUSES)[proba.argmax(1)]


def _make_sk() -> OneVsRestClassifier:
    return OneVsRestClassifier(LogisticRegression(C=1e6, max_iter=5000))  # type: ignore[no-untyped-call]


def run_cv(features: list[str], model_name: str) -> None:
    """Performs cross-validation and displays comparative performance."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=0)  # type: ignore[no-untyped-call]
    ours, theirs, dummy = [], [], []
    for tr_idx, te_idx in cv.split(df, df["Hogwarts House"]):  # type: ignore[no-untyped-call]
        train, test = df.iloc[tr_idx], df.iloc[te_idx]
        tr, te, stats = _preprocess(train, test, features)
        y_train = train["Hogwarts House"]
        y_test = test["Hogwarts House"]

        ours.append(accuracy_score(y_test, _predict_ours(tr, te, y_train, stats)))

        xtr = _build_design_matrix(tr.to_numpy(), stats)
        xte = _build_design_matrix(te.to_numpy(), stats)
        clf = _make_sk().fit(xtr, y_train)
        theirs.append(accuracy_score(y_test, clf.predict(xte)))

        dm = DummyClassifier(strategy="most_frequent").fit(xtr, y_train)  # type: ignore[no-untyped-call]
        dummy.append(accuracy_score(y_test, dm.predict(xte)))

    print(f"\n=== {model_name} ({len(features)} features) ===")
    print(f"  nôtre LR (GD)     : {np.mean(ours) * 100:6.2f}% ± {np.std(ours) * 100:.2f}")
    print(f"  sklearn LR        : {np.mean(theirs) * 100:6.2f}% ± {np.std(theirs) * 100:.2f}")


if __name__ == "__main__":
    run_cv(FEATURES, "sélection F4")
