"""Our own test for utils.py."""

from pathlib import Path

import pandas as pd
from pytest import CaptureFixture

from utils import load_csv


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


def test_load_csv_file_not_found(capsys: CaptureFixture[str]) -> None:
    """Test with a non-existent file."""
    result = load_csv("non_existent.csv")
    captured = capsys.readouterr()
    assert result is None
    assert "AssertionError: Files not found." in captured.out


def test_load_csv_permission_denied(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """Test with no permission."""
    protected_file = tmp_path / "protected_data.csv"
    protected_file.write_text("col1,col2\n1,2")
    Path.chmod(protected_file, 0o000)

    try:
        result = load_csv(str(protected_file))
        captured = capsys.readouterr()
        assert result is None
        assert "AssertionError: Permissions dinied." in captured.out
    finally:
        Path.chmod(protected_file, 0o666)
