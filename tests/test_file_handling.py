"""Our own test for file_handling.py."""

from pathlib import Path

import pandas as pd
from pandas import DataFrame
from pytest import CaptureFixture

from utils import load_csv, write_csv


def test_load_csv_success(tmp_path: Path) -> None:
    """Test for a valid CSV."""
    file_path = tmp_path / "valid_data.csv"
    file_path.write_text("col1,col2\n1,2\n3,4")
    dataset = load_csv(str(file_path))
    assert isinstance(dataset, pd.DataFrame)
    assert dataset.shape == (2, 2)
    assert list(dataset.columns) == ["col1", "col2"]


def test_load_csv_bad_extension(capsys: CaptureFixture[str]) -> None:
    """Test with a bad extension."""
    result = load_csv("data.txt")
    captured = capsys.readouterr()
    assert result is None
    assert "AssertionError: Files is not a .csv." in captured.out


def test_load_csv_file_not_found() -> None:
    """Test with a non-existent file."""
    result = load_csv("non_existent.csv")
    assert result is None


def test_load_csv_permission_denied(tmp_path: Path) -> None:
    """Test with no permission."""
    protected_file = tmp_path / "protected_data.csv"
    protected_file.write_text("col1,col2\n1,2")
    Path.chmod(protected_file, 0o000)

    try:
        result = load_csv(str(protected_file))
        assert result is None
    finally:
        Path.chmod(protected_file, 0o666)


def test_load_csv(tmp_path: Path) -> None:
    """Test load csv."""
    path = tmp_path / "ok_file.csv"
    path.write_text(
        "calories,duration\n420,50\n380,40\n390,45\n,72\n500,\n,\n",
        encoding="utf-8",
    )
    result = load_csv(str(path))
    excepted = DataFrame(
        {
            "calories": [420, 380, 390, float("nan"), 500, float("nan")],
            "duration": [50, 40, 45, 72, float("nan"), float("nan")],
        }
    )
    assert excepted.equals(result)


def test_load_csv_bad_column(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    """Test load csv bad column."""
    path = tmp_path / "bad_column.csv"
    path.write_text(
        "calories,duration\n420,50\n380,40\n390,45\n,72\n500,\n,,\n",
        encoding="utf-8",
    )
    result = load_csv(str(path))
    captured = capsys.readouterr()
    assert result is None
    assert "Exception: Error tokenizing data." in captured.out


def test_load_csv_empty_path(capsys: CaptureFixture[str]) -> None:
    """Test load csv empty path."""
    path = ""
    result = load_csv(path)
    captured = capsys.readouterr()
    assert result is None
    assert "AssertionError: Files is not a .csv.\n" in captured.out


def test_load_csv_bad_path(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    """Test load csv bad path."""
    path = tmp_path / "bad_path.csv"
    result = load_csv(str(path))
    captured = capsys.readouterr()
    assert result is None
    assert "Exception: [Errno 2] No such file or directory:" in captured.out


def test_load_csv_bad_extention(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    """Test load csv bad extension."""
    path = tmp_path / "bad_extension.txt"
    path.touch()
    result = load_csv(str(path))
    captured = capsys.readouterr()
    assert result is None
    assert "AssertionError: Files is not a .csv.\n" in captured.out


def test_load_csv_empty_file(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    """Test load csv empty file."""
    path = tmp_path / "empty.csv"
    path.touch()
    result = load_csv(str(path))
    captured = capsys.readouterr()
    assert result is None
    assert "Exception: No columns to parse from file\n" in captured.out


def test_load_csv_one_empty_line(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    """Test load csv one empty line."""
    path = tmp_path / "newline.csv"
    path.write_text("\n", encoding="utf-8")
    result = load_csv(str(path))
    captured = capsys.readouterr()
    assert result is None
    assert "Exception: No columns to parse from file\n" in captured.out


def test_load_csv_one_line(tmp_path: Path) -> None:
    """Test load csv one line."""
    path = tmp_path / "one_line.csv"
    path.write_text("calories,duration\n", encoding="utf-8")
    result = load_csv(str(path))
    assert result.empty
    assert list(result.columns) == ["calories", "duration"]


def test_write_csv_success(tmp_path: Path) -> None:
    """Test to write a valid CSV."""
    path = tmp_path / "ok_file.csv"
    df = DataFrame(
        {
            "calories": [420, 380, 390, float("nan"), 500, float("nan")],
            "duration": [50, 40, 45, 72, float("nan"), float("nan")],
        }
    )
    result = write_csv(path.name, df)
    assert result


def test_write_csv_empty_path(capsys: CaptureFixture[str]) -> None:
    """Test write csv empty path."""
    path = ""
    df = DataFrame(
        {
            "calories": [420, 380, 390, float("nan"), 500, float("nan")],
            "duration": [50, 40, 45, 72, float("nan"), float("nan")],
        }
    )
    result = write_csv(path, df)
    captured = capsys.readouterr()
    assert not result
    assert "AssertionError: Files is not a .csv.\n" in captured.out


def test_write_csv_bad_path(capsys: CaptureFixture[str]) -> None:
    """Test write csv bad path."""
    path = "bad_path_folder/file.csv"
    df = DataFrame(
        {
            "calories": [420, 380, 390, float("nan"), 500, float("nan")],
            "duration": [50, 40, 45, 72, float("nan"), float("nan")],
        }
    )
    result = write_csv(path, df)
    captured = capsys.readouterr()
    assert not result
    assert "Exception: Cannot save file into a non-existent directory:" in captured.out


def test_write_csv_bad_extention(capsys: CaptureFixture[str]) -> None:
    """Test write csv bad extension."""
    path = "file_handling.py"
    df = DataFrame(
        {
            "calories": [420, 380, 390, float("nan"), 500, float("nan")],
            "duration": [50, 40, 45, 72, float("nan"), float("nan")],
        }
    )
    result = write_csv(path, df)
    captured = capsys.readouterr()
    assert not result
    assert "AssertionError: Files is not a .csv.\n" in captured.out


def test_write_csv_no_extention(capsys: CaptureFixture[str]) -> None:
    """Test write csv no extension."""
    path = "file_no_extension"
    df = DataFrame(
        {
            "calories": [420, 380, 390, float("nan"), 500, float("nan")],
            "duration": [50, 40, 45, 72, float("nan"), float("nan")],
        }
    )
    result = write_csv(path, df)
    captured = capsys.readouterr()
    assert not result
    assert "AssertionError: Files is not a .csv.\n" in captured.out
