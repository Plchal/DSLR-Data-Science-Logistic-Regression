"""Function to create the prediction set dictionary."""

import pandas as pd
from pandas import DataFrame

from logistic_regression import PredictionSet
from variable import BETA0_NAME, HOUSES, PREDICTION_FILE, WEIGHTS_FILE


def create_prediction_set(weights: DataFrame, df: DataFrame) -> dict[str, PredictionSet] | None:
    """Create the prediction set dictionary.

    Args:
        weights (DataFrame): Data frame representing the weights in the CSV file created by the training.
        df (DataFrame): Data frame containing the data from the CSV file to be analyzed.

    Returns:
        dict[str, TrainingSet] | None: Dictionary of training set of each house or None on failed.
    """
    if not _parse_weights(weights):
        return None
    test_df = _clean_test_df(weights, df)
    if test_df is None:
        return None
    if not _parse_test_df(test_df):
        return None
    return _create_prediction_data(weights, test_df)


def _parse_weights(weights: DataFrame) -> bool:
    if weights.isnull().values.any():
        print("Null value in", WEIGHTS_FILE)
        return False
    min_row = 2
    if weights.shape[0] < min_row:
        print("Bad number of rows in", WEIGHTS_FILE)
        return False
    return _parse_weights_col(weights)


def _parse_weights_col(weights: DataFrame) -> bool:
    nb_col = 7

    if weights.shape[1] != nb_col:
        print("Bad number of column in", WEIGHTS_FILE)
        return False
    if "feature" not in weights.columns:
        print("Column feature not defined in", WEIGHTS_FILE)
        return False
    for col_name in [*HOUSES, "mean", "std"]:
        if col_name not in weights.columns:
            print("Column", col_name, "not defined in", WEIGHTS_FILE)
            return False
        try:
            pd.to_numeric(weights[col_name])
        except ValueError:
            print("Not numeric value in", WEIGHTS_FILE)
            return False
    return True


def _clean_test_df(weights: DataFrame, df: DataFrame) -> DataFrame | None:
    test_df = DataFrame()
    for feature in weights["feature"].values:
        if feature == BETA0_NAME:
            continue
        if feature not in df.columns:
            print("Feature", feature, "not defined in", PREDICTION_FILE)
            return None
        test_df[feature] = df[feature]
    return test_df


def _parse_test_df(test_df: DataFrame) -> bool:
    min_row = 1
    if test_df.shape[0] < min_row:
        print("No row in", PREDICTION_FILE)
        return False
    for col_name in test_df:
        try:
            pd.to_numeric(test_df[col_name])
        except ValueError:
            print("Not numeric value in", PREDICTION_FILE)
            return False
    return True


def _create_prediction_data(weights_df: DataFrame, test_df: DataFrame) -> dict[str, PredictionSet] | None:
    prediction_set_dict = {}
    for house in HOUSES:
        result = _create_prediction_house_data(weights_df, test_df, house)
        if result is None:
            return None
        prediction_set_dict[house] = result
    return prediction_set_dict


def _create_prediction_house_data(weights_df: DataFrame, test_df: DataFrame, house: str) -> PredictionSet | None:
    is_intercept = weights_df["feature"].eq(BETA0_NAME)  # Determine which row has the feature BETA0_NAME.
    intercept_weights = weights_df.loc[is_intercept, house].tolist()  # Store this row.
    feature_weights = weights_df.loc[~is_intercept]  # Store the other rows.

    features = feature_weights["feature"].tolist()
    means = feature_weights["mean"].tolist()
    stds = feature_weights["std"].tolist()
    weights = feature_weights[house].tolist()

    scores_df = test_df[features]

    list_features = []
    mean_list = []
    std_list = []
    list_weights = []

    for row in scores_df.itertuples(index=False, name=None):  # It's a tuple with all score of a row.
        valid = [i for (i, score) in enumerate(row) if not pd.isna(score)]  # Store all index where a score is defined.
        if not valid:  # If there are no valid values in the row.
            print("No feature available for one prediction")
            return None

        list_features.append([row[i] for i in valid])
        mean_list.append([means[i] for i in valid])
        std_list.append([stds[i] for i in valid])
        list_weights.append(intercept_weights + [weights[i] for i in valid])

    return PredictionSet(list_weights, list_features, mean_list, std_list)
