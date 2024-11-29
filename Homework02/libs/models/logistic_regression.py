import numpy as np
from libs.math import sigmoid

class LogisticRegression:
    def __init__(self, num_features : int):
        self.parameters = np.random.normal(0, 0.01, num_features)
        
    def predict(self, x:np.array) -> np.array:
        """
        Method to compute the predictions for the input features.

        Args:
            x: it's the input data matrix.

        Returns:
            preds: the predictions of the input features.
        """
        #START MY MODIFICATIONS HERE

        # Compute the dot product between input features and parameters
        z = np.dot(x, self.parameters)
        
        # Apply the sigmoid (logistic) function
        preds = sigmoid(z)

        #END MY MODIFICATIONS HERE

        return preds
    
    @staticmethod
    def likelihood(preds, y : np.array) -> np.array:
        """
        Function to compute the log likelihood of the model parameters according to data x and label y.

        Args:
            preds: the predicted labels.
            y: the label array.

        Returns:
            log_l: the log likelihood of the model parameters according to data x and label y.
        """
        #START MY MODIFICATIONS HERE

        # Add small epsilon to prevent log(0) issues
        epsilon = 1e-15
        preds = np.clip(preds, epsilon, 1 - epsilon)
        
        # Compute the log likelihood
        log_l = np.sum(y * np.log(preds) + (1 - y) * np.log(1 - preds)) / len(y)

        #END MY MODIFICATIONS HERE

        return log_l
    
    def update_theta(self, gradient: np.array, lr : float = 0.5):
        """
        Function to update the weights in-place.

        Args:
            gradient: the gradient of the log likelihood.
            lr: the learning rate.

        Returns:
            None
        """
        #START MY MODIFICATIONS HERE

        # Update the parameters using the gradient and learning rate
        self.parameters = self.parameters + lr*gradient

        #END MY MODIFICATIONS HERE

        pass
        
    @staticmethod
    def compute_gradient(x : np.array, y: np.array, preds: np.array) -> np.array:
        """
        Function to compute the gradient of the log likelihood.

        Args:
            x: it's the input data matrix.
            y: the label array.
            preds: the predictions of the input features.

        Returns:
            gradient: the gradient of the log likelihood.
        """

        #START MY MODIFICATIONS HERE

        # Compute the gradient of the log likelihood
        gradient = np.dot((y - preds), x) / len(y)

        #END MY MODIFICATIONS HERE
        
        return gradient

