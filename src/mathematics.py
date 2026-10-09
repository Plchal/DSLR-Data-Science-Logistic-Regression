"""Our own math functions."""

import math

import pandas as pd


def calculate_mean(feature: pd.Series) -> float:
    """Mean Calculation."""
    feature = feature.dropna()
    if feature.size == 0:
        return float("nan")

    total = 0.0
    for value in feature:
        total += value
    return total / feature.size


def calculate_variance_ddof0(values: list[float], mean: float) -> float:
    """Variance Calculation ddof=0 (population)."""
    if len(values) == 0:
        return float("nan")

    variance = 0.0
    for value in values:
        diff = value - mean
        variance += diff * diff
    variance = variance / len(values)  # Divisé par N (ddof=0)
    return max(variance, 0.0)


def calculate_first_quartile(feature: pd.Series) -> float:
    """First Quartile Calculation."""
    rank = (feature.size + 3) / 4 - 1
    return calculate_quartile(feature, rank)


def calculate_third_quartile(feature: pd.Series) -> float:
    """Third Quartile Calculation."""
    rank = (3 * feature.size + 1) / 4 - 1
    return calculate_quartile(feature, rank)


def calculate_quartile(feature: pd.Series, rank: float) -> float:
    """Second Quartile Calculation."""
    index = int(rank)
    floating_value = rank - index
    if floating_value == 0:
        return float(feature.iloc[index])
    if floating_value <= 1 / 3:
        return float((feature.iloc[index] * 3 + feature.iloc[index + 1]) / 4)
    if floating_value > 2 / 3:
        return float((feature.iloc[index] + feature.iloc[index + 1] * 3) / 4)
    return float((feature.iloc[index] + feature.iloc[index + 1]) / 2)


def calculate_median(feature: pd.Series) -> float:
    """Median Calculation."""
    size = feature.size
    if size % 2 != 0:
        return float(feature.iloc[(size - 1) // 2])
    before = feature.iloc[size // 2 - 1]
    after = feature.iloc[size // 2]
    return float((before + after) / 2)


def calculate_std(feature: pd.Series, mean: float) -> float:
    """Standart Deviation Calculation."""
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


def calculate_pearson_correlation_coefficient(x_feature: pd.Series, y_feature: pd.Series) -> float:
    """."""
    df = pd.concat([x_feature, y_feature], axis=1).dropna()
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]

    mean_x = calculate_mean(x_feature)
    mean_y = calculate_mean(y_feature)

    num = ((x - mean_x) * (y - mean_y)).sum()

    denom = math.sqrt(((x - mean_x) ** 2).sum() * ((y - mean_y) ** 2).sum())

    if denom == 0:
        return 0.0

    return float(num / denom)
