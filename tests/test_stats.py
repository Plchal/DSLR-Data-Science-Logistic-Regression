"""Our own test for stats.py."""

import pandas as pd
import pytest

from src import stats


@pytest.fixture
def dataframe() -> pd.DataFrame:
    """Reference DataFrame."""
    return pd.DataFrame({"values": [8, float("nan"), 2, 5, 1, 9, float("nan"), 3, 7, 4]})


def test_count(dataframe: pd.DataFrame) -> None:
    """Test count."""
    expected = dataframe["values"].count()
    result = stats.get_stats(dataframe).loc["count", "values"]
    assert result == expected


def test_mean(dataframe: pd.DataFrame) -> None:
    """Test mean."""
    expected = dataframe["values"].mean()
    result = stats.get_stats(dataframe).loc["mean", "values"]
    assert result == expected


def test_standard_deviation(dataframe: pd.DataFrame) -> None:
    """Test std."""
    expected = dataframe["values"].std()
    result = stats.get_stats(dataframe).loc["std", "values"]
    assert result == expected


def test_minimum(dataframe: pd.DataFrame) -> None:
    """Test min."""
    expected = dataframe["values"].min()
    result = stats.get_stats(dataframe).loc["min", "values"]
    assert result == expected


def test_first_quartile(dataframe: pd.DataFrame) -> None:
    """Test first quartile."""
    expected = dataframe["values"].quantile(0.25)
    result = stats.get_stats(dataframe).loc["25%", "values"]
    assert result == expected


def test_median(dataframe: pd.DataFrame) -> None:
    """Test median."""
    expected = dataframe["values"].median()
    result = stats.get_stats(dataframe).loc["50%", "values"]
    assert result == expected


def test_third_quartile(dataframe: pd.DataFrame) -> None:
    """Test third quartile."""
    expected = dataframe["values"].quantile(0.75)
    result = stats.get_stats(dataframe).loc["75%", "values"]
    assert result == expected


def test_maximum(dataframe: pd.DataFrame) -> None:
    """Test maximum."""
    expected = dataframe["values"].max()
    result = stats.get_stats(dataframe).loc["max", "values"]
    assert result == expected


def test_etendue(dataframe: pd.DataFrame) -> None:
    """Test minimum."""
    result = stats.get_stats(dataframe).loc["etendue", "values"]
    assert result == 8


def test_nb_nan(dataframe: pd.DataFrame) -> None:
    """Test nuber of nan."""
    result = stats.get_stats(dataframe).loc["nan", "values"]
    assert result == 2


@pytest.fixture
def multi_column_dataframe() -> None:
    """Reference DataFrame."""
    return pd.DataFrame(
        {
            "a": [8, 2, 5, 1, 9, 3, 7, 4],
            "b": [10, 20, 30, 40, 50, 60, 70, 80],
            "c": [-5, 0, 10, -2, 3, 8, -10, 4],
        }
    )


def test_count_multiple_columns(multi_column_dataframe: pd.DataFrame) -> None:
    """Test count multiple columns."""
    result = stats.get_stats(multi_column_dataframe)

    assert result.loc["count", "a"] == 8
    assert result.loc["count", "b"] == 8
    assert result.loc["count", "c"] == 8


def test_mean_multiple_columns(multi_column_dataframe: pd.DataFrame) -> None:
    """Test mean multiple columns."""
    result = stats.get_stats(multi_column_dataframe)

    assert result.loc["mean", "a"] == pytest.approx(4.875)
    assert result.loc["mean", "b"] == pytest.approx(45)
    assert result.loc["mean", "c"] == pytest.approx(1)


def test_minimum_multiple_columns(multi_column_dataframe: pd.DataFrame) -> None:
    """Test minimum multiple columns."""
    result = stats.get_stats(multi_column_dataframe)

    assert result.loc["min", "a"] == 1
    assert result.loc["min", "b"] == 10
    assert result.loc["min", "c"] == -10


def test_maximum_multiple_columns(multi_column_dataframe: pd.DataFrame) -> None:
    """Test maximum multiple columns."""
    result = stats.get_stats(multi_column_dataframe)

    assert result.loc["max", "a"] == 9
    assert result.loc["max", "b"] == 80
    assert result.loc["max", "c"] == 10


def test_etendue_multiple_columns(multi_column_dataframe: pd.DataFrame) -> None:
    """Test etendue multiple columns."""
    result = stats.get_stats(multi_column_dataframe)

    assert result.loc["etendue", "a"] == 8
    assert result.loc["etendue", "b"] == 70
    assert result.loc["etendue", "c"] == 20


def test_nb_nan_multiple_columns(multi_column_dataframe: pd.DataFrame) -> None:
    """Test number of multiple columns."""
    result = stats.get_stats(multi_column_dataframe)

    assert result.loc["nan", "a"] == 0
    assert result.loc["nan", "b"] == 0
    assert result.loc["nan", "c"] == 0


def test_quartiles_odd_number_of_values() -> None:
    """Test quartiles odd number of values."""
    df = pd.DataFrame({"values": [7, 1, 9, 3, 5]})

    result = stats.get_stats(df)

    assert result.loc["25%", "values"] == df["values"].quantile(0.25)
    assert result.loc["50%", "values"] == df["values"].quantile(0.50)
    assert result.loc["75%", "values"] == df["values"].quantile(0.75)


def test_quartiles_even_number_of_values() -> None:
    """Test quartiles even number of values."""
    df = pd.DataFrame({"values": [10, 2, 8, 4, 6, 12]})

    result = stats.get_stats(df)

    assert result.loc["25%", "values"] == df["values"].quantile(0.25)
    assert result.loc["50%", "values"] == df["values"].quantile(0.50)
    assert result.loc["75%", "values"] == df["values"].quantile(0.75)


def test_negative_and_decimal_values() -> None:
    """Test negative and decimal values."""
    values = [float("nan"), -10.5, 3.2, -2.7, float("nan"), 8.9, 0.0, 4.4, -1.1, float("nan")]
    df = pd.DataFrame({"values": values})

    expected = df["values"].describe()
    result = stats.get_stats(df)

    assert result.loc["count", "values"] == expected["count"]
    assert result.loc["mean", "values"] == expected["mean"]
    assert result.loc["std", "values"] == expected["std"]
    assert result.loc["min", "values"] == expected["min"]
    assert result.loc["25%", "values"] == expected["25%"]
    assert result.loc["50%", "values"] == expected["50%"]
    assert result.loc["75%", "values"] == expected["75%"]
    assert result.loc["max", "values"] == expected["max"]
    assert result.loc["etendue", "values"] == 19.4
    assert result.loc["nan", "values"] == 3


def test_duplicate_values() -> None:
    """Test duplicate values."""
    df = pd.DataFrame({"values": [5, 5, 5, 1, 1, 10, 10, 20]})

    expected = df["values"].describe()
    result = stats.get_stats(df)

    assert result.loc["count", "values"] == expected["count"]
    assert result.loc["mean", "values"] == expected["mean"]
    assert result.loc["std", "values"] == expected["std"]
    assert result.loc["min", "values"] == expected["min"]
    assert result.loc["25%", "values"] == expected["25%"]
    assert result.loc["50%", "values"] == expected["50%"]
    assert result.loc["75%", "values"] == expected["75%"]
    assert result.loc["max", "values"] == expected["max"]
    assert result.loc["etendue", "values"] == 19
    assert result.loc["nan", "values"] == 0
