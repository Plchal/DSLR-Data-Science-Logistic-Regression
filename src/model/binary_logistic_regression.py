"""Implementation of a binary logistic regression model."""

import numpy as np

from utils.variable import Matrix

from .dataclass import TrainingSet


class BinaryLogisticRegression:
    """Binary logistic regression model for binary classification."""

    def get_initial_weights(self, training_set: TrainingSet) -> Matrix:
        """Initialize the model weights to zero.

        The weight vector includes one additional coefficient for the bias term.

        Args:
            training_set: Training set containing the input features and their corresponding outcomes.

        Returns:
            A zero-initialized weight vector of size n_features + 1.
        """
        return np.zeros(training_set.features.shape[1] + 1)

    def predict_proba(self, z: Matrix) -> Matrix:
        """Compute class probabilities from linear model outputs.

        Applies the sigmoid function to transform the input values into probabilities between 0 and 1.

        Args:
            z: Linear model outputs before applying the sigmoid function.

        Returns:
            Predicted probabilities for the positive class.
        """
        z = np.clip(z, -500.0, 500.0)
        return 1 / (1 + np.exp(-z))

    def cross_entropy(self, outcome: Matrix, clipped: Matrix) -> float:
        """Compute the binary cross-entropy loss.

        Measures the difference between the true binary outcomes and the predicted probabilities. The loss is averaged
        over all samples.

        Args:
            outcome: True binary labels, typically encoded as 0 or 1.
            clipped: Predicted probabilities clipped to avoid taking the logarithm of zero.

        Returns:
            The mean binary cross-entropy loss as a float.
        """
        return float(-np.mean(outcome * np.log(clipped) + (1 - outcome) * np.log(1 - clipped)))
