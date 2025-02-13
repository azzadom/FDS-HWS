from libs.models.logistic_regression import LogisticRegression
import numpy as np
from libs.math import softmax


class SoftmaxClassifier(LogisticRegression):
    def __init__(self, num_features: int, num_classes: int):
        self.parameters = np.random.normal(0, 1e-3, (num_features, num_classes))

    def predict(self, X: np.array) -> np.array:
        """
        Function to compute the 

        Args:
            X: it's the input data matrix. The shape is (N, H)

        Returns:
            scores: it's the matrix containing raw scores for each sample and each class. The shape is (N, K)
        """
        #START MY MODIFICATIONS HERE

        # Compute raw scores by taking the dot product of X and parameters
        scores = np.dot(X, self.parameters)

        # Apply softmax to convert scores to probabilities
        scores = softmax(scores)

        #END MY MODIFICATIONS HERE

        return scores

    def predict_labels(self, X: np.array) -> np.array:
        """
        Function to compute the predicted class for each sample.
        
        Args:
            X: it's the input data matrix. The shape is (N, H)
            
        Returns:
            preds: it's the predicted class for each sample. The shape is (N,)
        """
        # START MY MODIFICATIONS HERE

        # Compute probabilities
        probs = self.predict(X)
        
        # Take the class with the highest probability
        preds = np.argmax(probs, axis=1)

        # END MY MODIFICATIONS HERE

        return preds

    @staticmethod
    def likelihood(preds: np.array, y_onehot: np.array) -> float:
        """
        Function to compute the cross entropy loss from the predicted labels and the true labels.

        Args:
            preds: it's the matrix containing probability for each sample and each class. The shape is (N, K)
            y_onehot: it's the label array encoded as a one-hot vector. The shape is (N, K)

        Returns:
            loss: The scalar that is the mean error for each sample.
        """
        # START MY MODIFICATIONS HERE

        # Add a small epsilon to prevent log(0)
        epsilon = 1e-15
        preds = np.clip(preds, epsilon, 1 - epsilon)
        
        # Compute the cross-entropy loss
        loss = -np.sum(y_onehot * np.log(preds)) / len(y_onehot)

        # END MY MODIFICATIONS HERE

        return loss

    def update_theta(self, gradient: np.array, lr: float = 0.5):
        """
        Function to update the weights in-place.

        Args:
            gradient: the jacobian of the cross entropy loss.
            lr: the learning rate.

        Returns:
            None
        """
        # START MY MODIFICATIONS HERE

        # Update the parameters using gradient descent
        self.parameters -= lr * gradient

        # END MY MODIFICATIONS HERE
        pass

    @staticmethod
    def compute_gradient(X: np.array, y: np.array, preds: np.array) -> np.array:
        """
        Function to compute gradient of the cross entropy loss with respect the parameters. 

        Args:
            X: it's the input data matrix. The shape is (N, H)
            y: it's the label array encoded as a one-hot vector. The shape is (N, K)
            preds: it's the predicted labels. The shape is (N, K)

        Returns:
            jacobian: A matrix with the partial derivatives of the loss. The shape is (H, K)
        """
        # START MY MODIFICATIONS HERE

        # Compute the gradient as the dot product of X.T and (preds - y) divided by N
        jacobian = np.dot(X.T, (preds - y)) / len(y)

        # END MY MODIFICATIONS HERE
        return jacobian