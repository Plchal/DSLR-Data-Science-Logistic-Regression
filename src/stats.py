"""Functions for obtaining statistics similar to the describe() function in pandas."""

import math
from dataclasses import dataclass

import pandas as pd


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
    stats.mean = sum(feature_clean) / stats.count
    stats.std = _calculate_std(feature_clean, stats.mean)
    feature_sorted = feature_clean.sort_values().reset_index(drop=True)
    stats.min = feature_sorted.iloc[0]
    stats.max = feature_sorted.iloc[-1]
    stats.etendue = stats.max - stats.min
    stats.first_quartile = _calculate_first_quartile(feature_sorted)
    stats.median = _calculate_median(feature_sorted)
    stats.third_quartile = _calculate_third_quartile(feature_sorted)
    return stats


def _calculate_first_quartile(feature: pd.Series) -> float:
    rank = (feature.size + 3) / 4 - 1
    return _calculate_quartile(feature, rank)


def _calculate_third_quartile(feature: pd.Series) -> float:
    rank = (3 * feature.size + 1) / 4 - 1
    return _calculate_quartile(feature, rank)


def _calculate_quartile(feature: pd.Series, rank: float) -> float:
    index = int(rank)
    floating_value = rank - index
    if floating_value == 0:
        return float(feature.iloc[index])
    if floating_value <= 1 / 3:
        return float((feature.iloc[index] * 3 + feature.iloc[index + 1]) / 4)
    if floating_value > 2 / 3:
        return float((feature.iloc[index] + feature.iloc[index + 1] * 3) / 4)
    return float((feature.iloc[index] + feature.iloc[index + 1]) / 2)


def _calculate_median(feature: pd.Series) -> float:
    size = feature.size
    if size % 2 != 0:
        return float(feature.iloc[(size - 1) // 2])
    before = feature.iloc[size // 2 - 1]
    after = feature.iloc[size // 2]
    return float((before + after) / 2)


def _calculate_std(feature: pd.Series, mean: float) -> float:
    min_size: int = 2

    if feature.size < min_size:
        return float("nan")
    variance = 0.0
    for value in feature:
        diff = value - mean
        variance += diff * diff
    variance = variance / (feature.size - 1)  # size-1 for Bessel's correction.
    variance = max(variance, 0.0)  # Avoid negative value (float imprecision).
    return math.sqrt(variance)
