"""Our own test for the training set feature selection."""

from pandas import DataFrame
from pytest import CaptureFixture

from model import create_multiclass_training_set, create_training_set_dict
from utils.variable import BETA0_NAME, HOUSES


def _make_dataset() -> DataFrame:
    return DataFrame(
        {
            "Hogwarts House": ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"],
            "Astronomy": [1.0, 2.0, 3.0, 4.0],
            "Herbology": [4.0, 3.0, 2.0, 1.0],
            "Charms": [1.0, 1.0, 2.0, 2.0],
        }
    )


def test_create_training_set_dict_selected_features() -> None:
    """Only the requested features must be kept."""
    training_set = create_training_set_dict(_make_dataset(), ["Astronomy", "Charms"])
    assert training_set is not None
    assert training_set[HOUSES[0]].features_name == [BETA0_NAME, "Astronomy", "Charms"]
    assert training_set[HOUSES[0]].features.shape[1] == 2


def test_create_multiclass_training_set_selected_features() -> None:
    """The softmax training set must follow the same feature selection."""
    training_set = create_multiclass_training_set(_make_dataset(), ["Herbology"])
    assert training_set is not None
    assert training_set.features_name == [BETA0_NAME, "Herbology"]
    assert training_set.features.shape[1] == 1


def test_create_training_set_dict_unknown_feature(capsys: CaptureFixture[str]) -> None:
    """An unknown feature must fail with an explicit message."""
    result = create_training_set_dict(_make_dataset(), ["Flying"])
    captured = capsys.readouterr()
    assert result is None
    assert "Feature Flying not defined in the dataset" in captured.out
