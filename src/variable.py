"""TODO."""

from typing import Literal, TypeAlias

import numpy as np

DATASET = "dataset_train.csv"
HOUSE_COLORS = {"Ravenclaw": "#222F5B", "Slytherin": "#1A472A", "Gryffindor": "#740001", "Hufflepuff": "#D89E15"}
COLS_TO_DROP = ["Index", "First Name", "Last Name", "Birthday", "Best Hand", "Hogwarts House"]
HOUSES = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
WEIGHTS_FILE = "weights.csv"
WEIGHTS_SOFTMAX_FILE = "weights_softmax.csv"
PREDICTION_FILE = "houses.csv"
BETA0_NAME = "beta0"

Optimizer: TypeAlias = Literal["batch", "sgd", "minibatch"]
DEFAULT_BATCH_SIZE = 32
EPSILON = 1e-12

Array1D: TypeAlias = np.typing.NDArray[np.float64]
Array2D: TypeAlias = np.typing.NDArray[np.float64]
ArrayIndex: TypeAlias = np.typing.NDArray[np.int64]
