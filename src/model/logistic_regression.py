"""Multi-classifier using a logistic regression."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import numpy as np
from pandas import DataFrame

from utils.variable import DEFAULT_BATCH_SIZE, EPSILON, Array1D, Array2D, ArrayIndex, Optimizer

T = TypeVar("T", bound=np.ndarray)


@dataclass
class TrainingSet:
    """Dataclass used for the binary training.

    Args:
        features_name (list[str]): Names of the feature columns (intercept included).
        features (Array2D): 2D array containing the observations.
        target (Array1D): 1D array containing the results.
        stats (DataFrame): Statistics related to the training set.
        nb_target (int): Number of result.
    """

    features_name: list[str]
    features: Array2D
    target: Array1D
    stats: DataFrame
    nb_target: int = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the number of target."""
        self.nb_target = self.target.size  # one dimension, so we can use size.


@dataclass
class MulticlassTrainingSet:
    """Dataclass used for the multiclass (softmax) training.

    Args:
        features_name (list[str]): Names of the feature columns (intercept included).
        features (Array2D): 2D array containing the observations.
        one_hot (Array2D): 2D one-hot encoded target.
        stats (DataFrame): Statistics related to the training set.
        nb_sample (int): Number of observations.
        nb_class (int): Number of classes.
    """

    features_name: list[str]
    features: Array2D
    one_hot: Array2D
    stats: DataFrame
    nb_sample: int = field(init=False)
    nb_class: int = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the number of observations and classes."""
        self.nb_sample = self.features.shape[0]
        self.nb_class = self.one_hot.shape[1]


@dataclass
class HyperParameter:
    """Parameter of the training.

    Args:
        alpha (float): Learning rate.
        iteration (int): Number of times the gradient descent will be applied.
        optimizer (Optimizer): Optimisation algorithm ("batch", "sgd" or "minibatch").
        batch_size (int): Batch size used when the optimizer is "minibatch".
        seed (int | None): Seed used to make stochastic optimizers reproducible.
        track_loss (bool): Whether to store the loss at each iteration.
    """

    alpha: float = 0.1
    iteration: int = 10000
    optimizer: Optimizer = "batch"
    batch_size: int = DEFAULT_BATCH_SIZE
    seed: int | None = None
    track_loss: bool = True
    delta_early_stop: float = 1e-6
    watting: int = 1

    def __post_init__(self) -> None:
        """TODO."""
        if self.alpha <= 0:
            raise ValueError("alpha must be strictly positive")
        if self.iteration < 1:
            raise ValueError("iteration must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")


@dataclass
class TrainingResult(Generic[T]):
    """Result of a training.

    Args:
        weights (T): Parameter vector (binary) or matrix (multiclass).
        loss_history (list[float]): Loss value at each iteration.
    """

    weights: T
    loss_history: list[float]


@dataclass
class PredictionSet:
    """Dataclass used for the prediction.

    Args:
        weights (list[list[float]]): Weights define by the training.
        features (list[list[float]]): features of the data to be predicted.
        mean (list[list[float]]): Mean of each features during the training.
        std (list[list[float]]): Standard deviation of each features during the training.
    """

    weights: list[list[float]]
    features: list[list[float]]
    mean: list[list[float]]
    std: list[list[float]]


def logistic_regression(training_set: TrainingSet, parameter: HyperParameter | None = None) -> TrainingResult[Array1D]:
    """Use the logistic regresssion to calculate the weights that will be used for the prediction.

    Args:
        training_set (TrainingSet): Data used for the training.
        parameter (HyperParameter): Parameter of the training.

    Returns:
        TrainingResult[Array1D]: Parameter vector adjusted during gradient descent and loss history.
    """
    if parameter is None:
        parameter = HyperParameter()
    features = _build_design_matrix(training_set.features, training_set.stats)
    beta = np.zeros(features.shape[1])  # +1 for the design matrix.
    return _optimize_binary(features, training_set.target, beta, parameter)


def softmax_regression(
    training_set: MulticlassTrainingSet, parameter: HyperParameter | None = None
) -> TrainingResult[Array2D]:
    """Use the softmax regression to train a single multiclass classifier.

    Args:
        training_set (MulticlassTrainingSet): Data used for the training.
        parameter (HyperParameter): Parameter of the training.

    Returns:
        TrainingResult[Array2D]: Weight matrix and loss history.
    """
    if parameter is None:
        parameter = HyperParameter()
    features = _build_design_matrix(training_set.features, training_set.stats)
    weights = np.zeros((features.shape[1], training_set.nb_class))
    return _optimize_softmax(features, training_set.one_hot, weights, parameter)


def _build_design_matrix(features: Array2D, stats: DataFrame) -> Array2D:
    mean: Array1D = stats.loc["mean"].values
    std: Array1D = stats.loc["std"].values
    standardized = _standard_score(features, mean, std)
    one_value_col = np.ones(features.shape[0])  # This column is used to estimate the y-intercept.
    return np.column_stack([one_value_col, standardized])


def _standard_score(features: Array2D, mean: Array1D, std: Array1D) -> Array2D:
    safe_std = np.where(np.isfinite(std) & (std != 0), std, 1.0)
    return (features - mean) / safe_std


def _effective_batch_size(parameter: HyperParameter, nb_sample: int) -> int:
    if parameter.optimizer == "batch":
        return nb_sample
    if parameter.optimizer == "sgd":
        return 1
    return parameter.batch_size


def _iter_batches(nb_sample: int, batch_size: int, rng: np.random.Generator) -> Iterator[ArrayIndex]:
    indices = rng.permutation(nb_sample)
    for start in range(0, nb_sample, batch_size):
        yield indices[start : start + batch_size]


def _step_binary(features: Array2D, target: Array1D, beta: Array1D, alpha: float) -> None:
    probability = _sigmoid(features @ beta)
    gradient = (1 / features.shape[0]) * features.T @ (probability - target)
    beta -= alpha * gradient


def _step_softmax(features: Array2D, one_hot: Array2D, weights: Array2D, alpha: float) -> None:
    probability = _softmax(features @ weights)
    gradient = (1 / features.shape[0]) * features.T @ (probability - one_hot)
    weights -= alpha * gradient


def _optimize_binary(
    features: Array2D, target: Array1D, beta: Array1D, parameter: HyperParameter
) -> TrainingResult[Array1D]:
    best_loss = float("inf")
    best_weight = beta.copy()
    watting = 0

    rng = np.random.default_rng(parameter.seed)
    batch_size = _effective_batch_size(parameter, features.shape[0])
    loss_history: list[float] = []
    for _ in range(parameter.iteration):
        for indices in _iter_batches(features.shape[0], batch_size, rng):
            _step_binary(features[indices], target[indices], beta, parameter.alpha)
        loss = _binary_cross_entropy(_sigmoid(features @ beta), target)
        if parameter.track_loss:
            loss_history.append(loss)
        if loss < best_loss - parameter.delta_early_stop:
            best_loss, best_weight, watting = loss, beta.copy(), 0
        else:
            watting += 1
            if parameter.watting > 0 and watting >= parameter.watting:
                beta = best_weight
                break
    return TrainingResult[Array1D](beta, loss_history)


def _optimize_softmax(
    features: Array2D, one_hot: Array2D, weights: Array2D, parameter: HyperParameter
) -> TrainingResult[Array2D]:
    best_loss = float("inf")
    best_weights = weights.copy()
    watting = 0

    rng = np.random.default_rng(parameter.seed)
    batch_size = _effective_batch_size(parameter, features.shape[0])
    loss_history: list[float] = []
    for _ in range(parameter.iteration):
        for indices in _iter_batches(features.shape[0], batch_size, rng):
            _step_softmax(features[indices], one_hot[indices], weights, parameter.alpha)
        loss = _categorical_cross_entropy(_softmax(features @ weights), one_hot)
        if parameter.track_loss:
            loss_history.append(loss)
        if loss < best_loss - parameter.delta_early_stop:
            best_loss, best_weights, watting = loss, weights.copy(), 0
        else:
            watting += 1
            if parameter.watting > 0 and watting >= parameter.watting:
                weights = best_weights
                break
    return TrainingResult[Array2D](weights, loss_history)


def _binary_cross_entropy(probability: Array1D, target: Array1D) -> float:
    clipped = np.clip(probability, EPSILON, 1 - EPSILON)
    return float(-np.mean(target * np.log(clipped) + (1 - target) * np.log(1 - clipped)))


def _categorical_cross_entropy(probability: Array2D, one_hot: Array2D) -> float:
    clipped = np.clip(probability, EPSILON, 1 - EPSILON)
    return float(-np.mean(np.sum(one_hot * np.log(clipped), axis=1)))


def _sigmoid(z: Array1D) -> Array1D:
    """Converts any real number into a value between 0 and 1 to model probabilities."""
    z = np.clip(z, -500.0, 500.0)
    return 1 / (1 + np.exp(-z))


def _softmax(z: Array2D) -> Array2D:
    """Converts a matrix of raw scores into a probability distribution for each row."""
    z_max = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def create_prediction(prediction_set: PredictionSet) -> list[float]:
    """Use the weights and training data to predict the probability of belonging to class 1.

    Args:
        prediction_set (PredictionSet): Prediction set defined by weights, training data and the features of the data
        to be predicted.

    Returns:
        list[float]: Result of each predictions.
    """
    y_pred = []
    for i_row in range(len(prediction_set.features)):
        feature = np.array(prediction_set.features[i_row])
        mean = np.array(prediction_set.mean[i_row])
        std = np.array(prediction_set.std[i_row])
        standard_feature = _standard_score(feature, mean, std)
        standard_feature = np.array([1, *standard_feature])
        weights = np.array(prediction_set.weights[i_row])
        pred = _sigmoid(standard_feature @ weights)
        y_pred.append(float(pred))
    return y_pred
