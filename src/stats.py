"""Functions for obtaining statistics similar to the describe() function in pandas."""

from dataclasses import dataclass

import pandas as pd

from src.mathematics import (
    calculate_first_quartile,
    calculate_mean,
    calculate_median,
    calculate_std,
    calculate_third_quartile,
)


@dataclass
class Stats:
    """dataclass for storing statistics.

    Statistics store : count, mean, standard deviation, minimum, first quartile, median, third quartile, maximum,
    entendue and number of nan.
    """

    count: int = 0
    mean: float = 0
    std: float = 0
    min: float = 0
    first_quartile: float = 0
    median: float = 0
    third_quartile: float = 0
    max: float = 0
    etendue: float = 0
    nb_nan: int = 0


def get_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Get a DataFrame containing statistics.

    Args:
        df (pd.DataFrame): DataFrame which will be analyzed.

    Returns:
        pd.DataFrame: DataFrame containing statistics.
    """
    numeric_df = df.select_dtypes(include=["number"])
    data = {}
    for series_name, series in numeric_df.items():
        stats = _calculate_one_feature(series)
        data[series_name] = [
            stats.count,
            stats.mean,
            stats.std,
            stats.min,
            stats.first_quartile,
            stats.median,
            stats.third_quartile,
            stats.max,
            stats.etendue,
            stats.nb_nan,
        ]
    return pd.DataFrame(
        data,
        index=["count", "mean", "std", "min", "25%", "50%", "75%", "max", "etendue", "nan"],
    )


def _calculate_one_feature(feature: pd.Series) -> Stats:
    stats = Stats()
    feature_clean = feature.dropna()
    stats.nb_nan = feature.size - feature_clean.size
    if feature_clean.size == 0:
        return stats
    stats.count = feature_clean.size
    stats.mean = calculate_mean(feature_clean)
    stats.std = calculate_std(feature_clean, stats.mean)
    feature_sorted = feature_clean.sort_values().reset_index(drop=True)
    stats.min = feature_sorted.iloc[0]
    stats.max = feature_sorted.iloc[-1]
    stats.etendue = stats.max - stats.min
    stats.first_quartile = calculate_first_quartile(feature_sorted)
    stats.median = calculate_median(feature_sorted)
    stats.third_quartile = calculate_third_quartile(feature_sorted)
    return stats
