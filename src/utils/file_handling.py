"""Useful functions."""

import pandas as pd


def load_csv(path: str) -> pd.DataFrame | None:
    """Loads a .csv file into a DataFrame.

    Args:
        path (str): Path to the .csv file.

    Returns:
        pandas.DataFrame | None: The dataset, or None on error.
    """
    try:
        if not path.lower().endswith(".csv"):
            raise AssertionError("Files is not a .csv.")

        try:
            dataset = pd.read_csv(path)
        except PermissionError as exc:
            raise AssertionError("Permissions dinied.") from exc
        except FileNotFoundError as exc:
            raise AssertionError("Files not found.") from exc

        return dataset

    except AssertionError as error:
        print(f"{AssertionError.__name__}: {error}")
        return None


def write_csv(path: str, data: pd.DataFrame) -> bool:
    """Create a csv file from pd.DataFrame.

    Args:
        path (str): Path to the .csv file.
        data (pd.DataFrame): parameter.

    Returns:
        None.
    """
    try:
        if not path.lower().endswith(".csv"):
            raise AssertionError("Files is not a .csv.")
        data.to_csv(path, index=False, encoding="utf-8")
        return True
    except (PermissionError, AssertionError) as e:
        print("Error", e)
        return False
