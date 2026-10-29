"""Dataclasses used for the logistic regression."""

from dataclasses import dataclass

from pandas import DataFrame

from utils.variable import DEFAULT_BATCH_SIZE, Matrix, Optimizer


@dataclass
class TrainingSet:
    """Dataclass used for the logistic regression training.

    Args:
        features_name (list[str]): Names of the feature columns (intercept included).
        features (Matrix): 2D array containing the observations.
        outcome (Matrix): 2D array containing the result of observations.
        stats (DataFrame): Statistics related to the training set.
    """

    features_name: list[str]
    features: Matrix
    outcome: Matrix
    stats: DataFrame


@dataclass
class HyperParameter:
    """Parameter of the training.

    Args:
        alpha (float): Learning rate.
        iteration (int): Number of times the gradient descent will be applied.
        optimizer (Optimizer): Optimisation algorithm ("none", "batch", "sgd" or "minibatch").
        batch_size (int): Batch size used when the optimizer is "minibatch".
        seed (int | None): Seed used to make stochastic optimizers reproducible.
        track_loss (bool): Whether to store the loss at each iteration.
        delta_early_stop (float): The loss value at which training is considered complete and the steps are minimal.
        watting (int): Number of iterations expected once the delta early stop is reached.
    """

    alpha: float = 0.1
    iteration: int = 10000
    optimizer: Optimizer = "none"
    batch_size: int = DEFAULT_BATCH_SIZE
    seed: int | None = None
    track_loss: bool = True
    delta_early_stop: float = 1e-6
    watting: int = 1

    def __post_init__(self) -> None:
        """Verifiy the init value."""
        if self.alpha <= 0:
            raise ValueError("alpha must be strictly positive")
        if self.iteration < 1:
            raise ValueError("iteration must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")


@dataclass
class TrainingResult:
    """Result of a training.

    Args:
        weights (Matrix): Parameter vector (binary) or matrix (multiclass).
        loss_history (list[float]): Loss value at each iteration.
    """

    weights: Matrix
    loss_history: list[float]


@dataclass
class PredictionSet:
    """Dataclass used for the prediction.

    Args:
        design_matrix (Matrix): Standardized features with the intercept column.
        weights (Matrix): Weight matrix (features, outcome(s)).
    """

    design_matrix: Matrix
    weights: Matrix
