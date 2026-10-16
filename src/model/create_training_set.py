"""Function to create the training set dictionary."""

from typing import cast

import numpy as np
from pandas import DataFrame, Series

from utils.variable import BETA0_NAME, HOUSES, Array1D, Array2D

from .logistic_regression import MulticlassTrainingSet, TrainingSet
from .stats import get_stats


def create_training_set_dict(df: DataFrame, features: list[str] | None = None) -> dict[str, TrainingSet] | None:
    """Create the training set dictionary.

    Args:
        df (DataFrame): Data frame representing the data in the CSV file.
        features (list[str] | None): Features to train on. Use every numeric column when None.

    Returns:
        dict[str, TrainingSet] | None: Dictionary of training set of each house or None on failed.
    """
    prepared = _prepare_dataset(df, features)
    if prepared is None:
        return None
    features_name, features_values, target_dict, stats = prepared
    return _create_training_data(features_name, features_values, target_dict, stats)


def create_multiclass_training_set(df: DataFrame, features: list[str] | None = None) -> MulticlassTrainingSet | None:
    """Create the multiclass (softmax) training set.

    Args:
        df (DataFrame): Data frame representing the data in the CSV file.
        features (list[str] | None): Features to train on. Use every numeric column when None.

    Returns:
        MulticlassTrainingSet | None: Training set with the one-hot target or None on failed.
    """
    prepared = _prepare_dataset(df, features)
    if prepared is None:
        return None
    features_name, features_values, target_dict, stats = prepared
    one_hot = np.column_stack([target_dict[house] for house in HOUSES])
    return MulticlassTrainingSet(features_name, features_values, one_hot, stats)


def _prepare_dataset(
    df: DataFrame, features: list[str] | None = None
) -> tuple[list[str], Array2D, dict[str, Array1D], DataFrame] | None:
    house_col = "Hogwarts House"
    if not _is_valid_house_col(df, house_col):
        return None
    df_clean = _clean_data(df, house_col)
    if not _has_data(df_clean):
        return None
    if not _is_valid_features(df_clean, features):
        return None
    stats = _create_stats(df_clean, features)
    features_name, features_values, target_dict = _get_features_and_target(df_clean, house_col, features)
    return features_name, features_values, target_dict, stats


def _is_valid_house_col(df: DataFrame, house_col: str) -> bool:
    if house_col not in df.columns:
        print(house_col, "column not find")
        return False
    mask = df[house_col].isin(HOUSES) | df[house_col].isna()
    if not mask.all():
        print(house_col, "with bad value(s)")
        return False
    return True


def _clean_data(df: DataFrame, house_col: str) -> DataFrame:
    index_col = "Index"

    df_clean = df.copy()
    if index_col in df_clean.columns:
        df_clean.drop(columns=[index_col], inplace=True)
    df_clean.drop_duplicates(inplace=True)
    df_clean = _remove_duplicate_same_house(df_clean, house_col)
    return df_clean


def _remove_duplicate_same_house(df_clean: DataFrame, house_col: str) -> DataFrame:
    cols_without_house = df_clean.drop(columns=[house_col]).columns
    df_filtered = df_clean[~df_clean.duplicated(subset=cols_without_house, keep=False)]
    return cast(DataFrame, df_filtered)


def _has_data(df_clean: DataFrame) -> bool:
    if len(df_clean.index) == 0:
        print("No data for the training")
        return False
    return True


def _is_valid_features(df_clean: DataFrame, features: list[str] | None) -> bool:
    if features is None:
        return True
    numeric = set(df_clean.select_dtypes(include=["number"]).columns)
    for feature in features:
        if feature not in df_clean.columns:
            print("Feature", feature, "not defined in the dataset")
            return False
        if feature not in numeric:
            print("Feature", feature, "is not numeric")
            return False
    return True


def _selected_columns(df_clean: DataFrame, features: list[str] | None) -> list[str]:
    if features is None:
        return df_clean.select_dtypes(include=["number"]).columns.to_list()
    return list(features)


def _create_stats(df_clean: DataFrame, features: list[str] | None = None) -> DataFrame:
    df_clean_stat = df_clean[_selected_columns(df_clean, features)]
    return get_stats(df_clean_stat)


def _get_features_and_target(
    df_clean: DataFrame, house_col: str, features: list[str] | None = None
) -> tuple[list[str], Array2D, dict[str, Array1D]]:
    selected = _selected_columns(df_clean, features)
    if features is None:
        df_clean.dropna(inplace=True)
    else:
        df_clean.dropna(subset=selected, inplace=True)
    target_dict = _get_tagets(df_clean[house_col])
    features_clean = df_clean[selected]
    features_name = [BETA0_NAME, *features_clean.columns.to_list()]
    features_arr = features_clean.to_numpy()
    return features_name, features_arr, target_dict


def _get_tagets(house_col: Series) -> dict[str, Array1D]:
    target_dict = {}
    for house in HOUSES:
        target_dict[house] = (house_col == house).astype(int).to_numpy()
    return target_dict


def _create_training_data(
    features_name: list[str], features: Array2D, target_dict: dict[str, Array1D], stats: DataFrame
) -> dict[str, TrainingSet]:
    training_set_dict = {}
    for house in HOUSES:
        training_set_dict[house] = TrainingSet(features_name, features, target_dict[house], stats)
    return training_set_dict
