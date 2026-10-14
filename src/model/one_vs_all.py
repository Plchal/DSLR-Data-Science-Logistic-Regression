"""."""
from pandas import DataFrame

from .logistic_regression import HyperParameter
from utils.variable import Array1D, Array2D


class OneVsAll():
    """"TODO."""
    def optimize(features: Array2D, weight: Array1D, parameter: HyperParameter)










class TrainingSet:
    """Dataclass used for the binary training.

    Args:
        features_name (list[str]): Names of the feature columns (intercept included).
        features (Array2D): 2D array containing the observations.
        target (Array1D): 1D array containing the results.
        stats (DataFrame): Statistics related to the training set.
        nb_target (int): Number of result.
    """

    def _init__(self, features_name, features, targets, stats) -> None:
        """Initialize the number of target."""
        self.features_name = features_name
        self.features = features
        self.target = targets
        self.stats = stats
        self.nb_target = self.target.size

    def __post_init__(features: Array2D) -> None:
        self.weight = 

    def optimize(features: Array2D, weight: Array1D, parameter: HyperParameter)