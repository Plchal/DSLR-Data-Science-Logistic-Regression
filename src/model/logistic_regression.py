"""Implementation of a logistic regression."""

from collections.abc import Iterator

import numpy as np

from utils.variable import EPSILON, ArrayIndex, Matrix

from .binary_logistic_regression import BinaryLogisticRegression
from .dataclass import HyperParameter, PredictionSet, TrainingResult, TrainingSet
from .multinomial_logistic_regression import MultinomialLogisticRegression


def standard_score(features: Matrix, mean: Matrix, std: Matrix) -> Matrix:
    """Standardize features using their mean and standard deviation.

    Replaces invalid or zero standard deviations with 1 to avoid division by zero and non-finite results.
    """
    safe_std = np.where(np.isfinite(std) & (std != 0), std, 1.0)
    return (features - mean) / safe_std


class LogisticRegression:
    """Logistic regression model supporting binary and multi-class classification."""

    def __init__(self, model: BinaryLogisticRegression | MultinomialLogisticRegression) -> None:
        """Initialize logistic regression with the specified model."""
        self._model = model

    def train(self, training_set: TrainingSet, parameter: HyperParameter | None = None) -> TrainingResult:
        """Train the model by optimizing its weights on the training set.

        Initializes the weights and builds the design matrix before optimization.
        Uses default hyperparameters if none are provided.
        """
        if parameter is None:
            parameter = HyperParameter()
        weights = self._model.get_initial_weights(training_set)
        features = self._build_design_matrix(training_set)
        return self._optimize(parameter, training_set.outcome, features, weights)

    def _build_design_matrix(self, training_set: TrainingSet) -> Matrix:
        mean: Matrix = training_set.stats.loc["mean"].values
        std: Matrix = training_set.stats.loc["std"].values
        standardized = standard_score(training_set.features, mean, std)
        one_value_col = np.ones(training_set.features.shape[0])  # Used to estimate the y-intercept.
        return np.column_stack([one_value_col, standardized])

    def _optimize(
        self, parameter: HyperParameter, outcome: Matrix, features: Matrix, weights: Matrix
    ) -> TrainingResult:
        best_loss = float("inf")
        best_weights = weights.copy()
        watting = 0

        rng = None if parameter.optimizer == "none" else np.random.default_rng(parameter.seed)
        batch_size = self._effective_batch_size(parameter, features.shape[0])
        loss_history: list[float] = []
        for _ in range(parameter.iteration):
            for indices in self._iter_batches(features.shape[0], batch_size, rng):
                self._step(parameter, outcome[indices], features[indices], weights)
            probability = self._model.predict_proba(features @ weights)
            loss = self._cross_entropy(outcome, probability)
            if parameter.track_loss:
                loss_history.append(loss)
            if loss < best_loss - parameter.delta_early_stop:
                best_loss, best_weights, watting = loss, weights.copy(), 0
            else:
                watting += 1
                if parameter.watting > 0 and watting >= parameter.watting:
                    weights = best_weights
                    break
        return TrainingResult(weights, loss_history)

    def _iter_batches(self, nb_sample: int, batch_size: int, rng: np.random.Generator | None) -> Iterator[ArrayIndex]:
        indices = np.arange(batch_size) if rng is None else rng.permutation(nb_sample)
        for start in range(0, nb_sample, batch_size):
            yield indices[start : start + batch_size]

    def _effective_batch_size(self, parameter: HyperParameter, nb_sample: int) -> int:
        if parameter.optimizer in {"none", "batch"}:
            return nb_sample
        if parameter.optimizer == "sgd":
            return 1
        return parameter.batch_size

    def _step(self, parameter: HyperParameter, outcome: Matrix, features: Matrix, weights: Matrix) -> None:
        probability = self._model.predict_proba(features @ weights)
        gradient = (1 / features.shape[0]) * features.T @ (probability - outcome)
        weights -= parameter.alpha * gradient

    def _cross_entropy(self, outcome: Matrix, probability: Matrix) -> float:
        clipped = np.clip(probability, EPSILON, 1 - EPSILON)
        return self._model.cross_entropy(outcome, clipped)

    def create_prediction(self, prediction_set: PredictionSet) -> Matrix:
        """Use the prediction set to predict the probability of belonging to class 1, by using the specified model.

        Args:
            prediction_set (PredictionSet): Prediction set defined by weights, training data and the features of the
            data to be predicted.

        Returns:
            A matrix of predicted class probabilities.
        """
        return self._model.predict_proba(prediction_set.design_matrix @ prediction_set.weights)
