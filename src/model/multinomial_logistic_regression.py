"""Implementation of a multinomial logistic regression model."""

import numpy as np

from utils.variable import Matrix

from .dataclass import TrainingSet


class MultinomialLogisticRegression:
    """Multinomial logistic regression model for multi-class classification."""

    def get_initial_weights(self, training_set: TrainingSet) -> Matrix:
        """Initialize the model weights to zero.

        The weight matrix includes one additional row for the bias term. Each column corresponds to a target class.

        Args:
            training_set: Training set containing the input features and their corresponding class labels.

        Returns:
            A zero-initialized weight matrix of shape (n_features + 1, n_classes).
        """
        return np.zeros((training_set.features.shape[1] + 1, training_set.outcome.shape[1]))

    def predict_proba(self, z: Matrix) -> Matrix:
        """Compute class probabilities from linear model outputs.

        Applies the softmax function to convert the raw scores into a probability distribution over the target classes
        for each sample.

        Args:
            z: Matrix of linear model outputs, with one column per class.

        Returns:
            A matrix of predicted class probabilities, where each row contains probabilities that sum to 1.
        """
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cross_entropy(self, outcome: Matrix, clipped: Matrix) -> float:
        """Compute the categorical cross-entropy loss.

        Measures the difference between the true class labels and the predicted class probabilities. The loss is
        averaged over all samples.

        Args:
            outcome: One-hot encoded matrix of true class labels.
            clipped: Predicted class probabilities, clipped to avoid taking the logarithm of zero.

        Returns:
            The mean categorical cross-entropy loss as a float.
        """
        return float(-np.mean(np.sum(outcome * np.log(clipped), axis=1)))
