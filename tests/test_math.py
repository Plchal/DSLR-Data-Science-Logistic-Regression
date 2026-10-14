"""Our own test for mathematics.py."""

import math

import numpy as np
import pandas as pd

from utils import (
    calculate_first_quartile,
    calculate_mean,
    calculate_median,
    calculate_pearson_correlation_coefficient,
    calculate_std,
    calculate_third_quartile,
)


def test_mean() -> None:
    """Test mean."""
    data = pd.Series([3, 2, 7, 4, 9])
    result = calculate_mean(data)
    assert result == 5


def test_median_odd() -> None:
    """Test median odd."""
    data = pd.Series([1, 3, 3, 6, 7, 8, 9])
    result = calculate_median(data)
    assert result == 6


def test_median_even() -> None:
    """Test median odd."""
    data = pd.Series([1, 2, 3, 4, 5, 6, 8, 9])
    result = calculate_median(data)
    assert result == 4.5


def test_first_quartile() -> None:
    """Test first quartile."""
    data = pd.Series([1, 11, 15, 19, 20, 24, 28, 34, 37, 47, 50, 61, 68])
    result = calculate_first_quartile(data)
    assert result == 19


def test_first_quartile_low() -> None:
    """Test first quartile low."""
    data = pd.Series([1, 11, 15, 19, 20, 24, 28, 34, 37, 47])
    result = calculate_first_quartile(data)
    assert result == 16


def test_first_quartile_high() -> None:
    """Test first quartile high."""
    data = pd.Series([1, 11, 15, 19, 20, 24, 28, 34, 37, 47, 50, 61])
    result = calculate_first_quartile(data)
    assert result == 18


def test_third_quartile() -> None:
    """Test third quartile."""
    data = pd.Series([1, 11, 15, 19, 20, 24, 28, 34, 37, 47, 50, 61, 68])
    result = calculate_third_quartile(data)
    assert result == 47


def test_third_quartile_low() -> None:
    """Test third quartile low."""
    data = pd.Series([1, 11, 15, 19, 20, 24, 28, 34, 37, 47, 50, 61])
    result = calculate_third_quartile(data)
    assert result == 39.5


def test_third_quartile_high() -> None:
    """Test third quartile high."""
    data = pd.Series([1, 11, 15, 19, 20, 24, 28, 34, 37, 47])
    result = calculate_third_quartile(data)
    assert result == 32.5


def test_std() -> None:
    """Test std."""
    data = pd.Series([0, 0, 0, 1, 2, 9])
    mean = 2
    result = calculate_std(data, mean)
    assert result == math.sqrt(12.4)


def test_pearson_correlation_coefficient() -> None:
    """Test pearson correlation coefficient."""
    data_x = pd.Series([4, 6, 8, 10, 12, 14])
    data_y = pd.Series([65, 70, 75, 82, 88, 93])
    result = calculate_pearson_correlation_coefficient(data_x, data_y)
    expected = 201.00 / math.sqrt(40518.8)
    assert np.isclose(result, expected)


def test_pearson_correlation_coefficient_exact() -> None:
    """Test pearson correlation coefficient exact."""
    data_x = pd.Series([1, 2, 3, 4, 5])
    data_y = pd.Series([2, 4, 6, 8, 10])
    result = calculate_pearson_correlation_coefficient(data_x, data_y)
    assert np.isclose(result, 1)
