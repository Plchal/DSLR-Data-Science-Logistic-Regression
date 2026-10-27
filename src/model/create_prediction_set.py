"""Function to create the prediction set dictionary."""

import numpy as np
import pandas as pd
from pandas import DataFrame

from utils.variable import BETA0_NAME, HOUSES

from .dataclass import PredictionSet
from .logistic_regression import standard_score


def clean_data_frame(weights: DataFrame, df: DataFrame, weights_name: str, data_name: str) -> DataFrame | None:
    """Validate and clean the weights and test data frames.

    Args:
        weights (DataFrame): Data frame containing the trained weights.
        df (DataFrame): Data frame containing the test data.
        weights_name (str): Weights file name used in error messages.
        data_name (str): Test data file name used in error messages.

    Returns:
        DataFrame | None: The cleaned test data frame, or None if validation fails.
    """
    if not _parse_weights(weights, weights_name):
        return None
    test_df = _clean_test_df(weights, df, data_name)
    if test_df is None:
        return None
    if not _parse_test_df(test_df, data_name):
        return None
    return test_df


def _parse_weights(weights: DataFrame, file_name: str) -> bool:
    if weights.isnull().values.any():
        print("Null value in", file_name)
        return False
    min_row = 2
    if weights.shape[0] < min_row:
        print("Bad number of rows in", file_name)
        return False
    return _parse_weights_col(weights, file_name)


def _parse_weights_col(weights: DataFrame, file_name: str) -> bool:
    nb_col = 7

    if weights.shape[1] != nb_col:
        print("Bad number of column in", file_name)
        return False
    if "feature" not in weights.columns:
        print("Column feature not defined in", file_name)
        return False
    for col_name in [*HOUSES, "mean", "std"]:
        if col_name not in weights.columns:
            print("Column", col_name, "not defined in", file_name)
            return False
        try:
            pd.to_numeric(weights[col_name])
        except ValueError:
            print("Not numeric value in", file_name)
            return False
    return True


def _clean_test_df(weights: DataFrame, df: DataFrame, data_name: str) -> DataFrame | None:
    test_df = DataFrame()
    for feature in weights["feature"].values:
        if feature == BETA0_NAME:
            continue
        if feature not in df.columns:
            print("Feature", feature, "not defined in", data_name)
            return None
        test_df[feature] = df[feature]
    return test_df


def _parse_test_df(test_df: DataFrame, data_name: str) -> bool:
    min_row = 1
    if test_df.shape[0] < min_row:
        print("No row in", data_name)
        return False
    for col_name in test_df:
        try:
            pd.to_numeric(test_df[col_name])
        except ValueError:
            print("Not numeric value in", data_name)
            return False
    return True


def create_binary_prediction_set(weights_df: DataFrame, test_df: DataFrame) -> dict[str, PredictionSet]:
    """Create a prediction set for each house using binary classification."""
    return {house: _create_prediction_set(weights_df, test_df, house) for house in HOUSES}


def create_multinomial_prediction_set(weights_df: DataFrame, test_df: DataFrame) -> PredictionSet:
    """Create a prediction set for multi-class classification across all houses."""
    return _create_prediction_set(weights_df, test_df, HOUSES)


def _create_prediction_set(weights_df: DataFrame, test_df: DataFrame, house: str | list[str]) -> PredictionSet:
    weight_matrix = weights_df[house].to_numpy(dtype=float)

    i = weights_df[(weights_df.feature == BETA0_NAME)].index
    weights = weights_df.drop(i)
    means = weights["mean"].to_numpy(dtype=float)
    stds = weights["std"].to_numpy(dtype=float)

    raw = test_df.to_numpy(dtype=float)
    standardized = standard_score(raw, means, stds)
    standardized = np.nan_to_num(standardized, nan=0.0)

    intercept_col = np.ones(raw.shape[0])
    design_matrix = np.column_stack([intercept_col, standardized])
    return PredictionSet(design_matrix, weight_matrix)
