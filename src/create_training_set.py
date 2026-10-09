"""Function to create the training set dictionary."""

from pandas import DataFrame, Series

from logistic_regression import TrainingSet
from stats import get_stats
from variable import HOUSES, Array1D, Array2D


def create_training_set_dict(df: DataFrame) -> dict[str, TrainingSet] | None:
    """Create the training set dictionary.

    Args:
        df (DataFrame): Data frame representing the data in the CSV file.

    Returns:
        dict[str, TrainingSet] | None: Dictionary of training set of each house or None on failed.
    """
    house_col = "Hogwarts House"

    if not _is_valid_house_col(df, house_col):
        return None
    df_clean = _clean_data(df, house_col)
    if not _has_data(df_clean):
        return None
    stats = _create_stats(df_clean)
    features, target_dict = _get_features_and_target(df_clean, house_col)
    return _create_training_data(features, target_dict, stats)


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

    df_clean = df
    if index_col in df_clean.columns:
        df_clean.drop(columns=[index_col], inplace=True)
    df_clean.drop_duplicates(inplace=True)
    _remove_duplicate_same_house(df_clean, house_col)
    return df_clean


def _remove_duplicate_same_house(df_clean: DataFrame, house_col: str) -> None:
    df_without_house = df_clean.drop(columns=[house_col])
    df_clean = df_clean[~df_clean.duplicated(subset=df_without_house, keep=False)]


def _has_data(df_clean: DataFrame) -> bool:
    if len(df_clean.index) == 0:
        print("No data for the training")
        return False
    return True


def _create_stats(df_clean: DataFrame) -> DataFrame:
    df_clean_stat = df_clean.select_dtypes(include=["number"])
    return get_stats(df_clean_stat)


def _get_features_and_target(df_clean: DataFrame, house_col: str) -> tuple[Array2D, dict[str, Array1D]]:
    df_clean.dropna(inplace=True)
    target_dict = _get_tagets(df_clean[house_col])
    df_clean.drop(columns=[house_col], inplace=True)
    features = df_clean.select_dtypes(include=["number"]).to_numpy()
    return features, target_dict


def _get_tagets(house_col: Series) -> dict[str, Array1D]:
    target_dict = {}
    for house in HOUSES:
        target_dict[house] = (house_col == house).astype(int).to_numpy()
    return target_dict


def _create_training_data(
    features: Array2D, target_dict: dict[str, Array1D], stats: DataFrame
) -> dict[str, TrainingSet]:
    training_set_dict = {}
    for house in HOUSES:
        training_set_dict[house] = TrainingSet(features, target_dict[house], stats)
    return training_set_dict
