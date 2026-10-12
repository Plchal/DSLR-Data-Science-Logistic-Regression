"""Multi-classifier using a logistic regression."""

from dataclasses import dataclass, field

import numpy as np
from pandas import DataFrame

from variable import Array1D, Array2D


@dataclass
class TrainingSet:
    """Dataclass used for the training.

    Args:
        features (Array2D): 2D array containing the observations.
        target (Array1D): 1D array containing the results.
        nb_taget (int): Number of result.
    """

    features_name: list[str]
    features: Array2D
    target: Array1D
    stats: DataFrame
    nb_taget: int = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the number of target."""
        self.nb_taget = self.target.size  # one dimension, so we can use size.


@dataclass
class HyperParameter:
    """Parameter of the training.

    Args:
        alpha (float): Learning rate.
        iteration (int): Number of times the gradient descent will be applied.
    """

    alpha: float = 0.1
    iteration: int = 10000


def logistic_regression(training_set: TrainingSet, parameter: HyperParameter | None = None) -> Array1D:
    """Use the logistic regresssion to calculate the weights that will be used for the prediction.

    Args:
        training_set (TrainingSet): Data used for the training.
        parameter (HyperParameter): Parameter of the training.

    Returns:
        Array1D: Parameter vector adjusted during gradient descent.
    """
    if parameter is None:
        parameter = HyperParameter()
    beta = np.zeros(training_set.features.shape[1] + 1)  # +1 for the design matrix.
    training_set.features = _create_design_matrix(training_set)
    # TODO: inverse standard score.
    return _gradient_descent(training_set, parameter, beta)


def _create_design_matrix(training_set: TrainingSet) -> Array2D:
    mean: Array1D = training_set.stats.loc["mean"].values
    std: Array1D = training_set.stats.loc["std"].values
    features = _standard_score(training_set.features, mean, std)
    one_value_col = np.ones(training_set.nb_taget)  # This column is used to estimate the y-intercept.
    return np.column_stack([one_value_col, features])


def _standard_score(features: Array2D, mean: Array1D, std: Array1D) -> Array2D:
    return (features - mean) / std  # TODO: check matrix 0 division ?


def _gradient_descent(training_set: TrainingSet, parameter: HyperParameter, beta: Array1D) -> Array1D:
    for _ in range(parameter.iteration):
        y_hat = _sigmoid(training_set.features @ beta)
        gradient = (1 / training_set.nb_taget) * training_set.features.T @ (y_hat - training_set.target)
        beta -= parameter.alpha * gradient
    return beta


def _sigmoid(z: Array1D) -> Array1D:
    """Converts any real number into a value between 0 and 1 to model probabilities."""
    return 1 / (1 + np.exp(-z))
