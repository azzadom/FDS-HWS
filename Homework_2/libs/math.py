import numpy as np


def sigmoid(x):
    """
    Function to compute the sigmoid of a given input x.

    Args:
        x: it's the input data matrix.

    Returns:
        g: The sigmoid of the input x
    """
    #START MY MODIFICATIONS HERE

    g = 1 / (1 + np.exp(-x))

    #END MY MODIFICATIONS HERE

    return g

def softmax(y):
    """
    Computes the softmax probabilities for each sample and each class.

    Args:
        y (np.ndarray): The predicted logits (scores) matrix of shape (N, K),
                        where N is the number of samples and K is the number of classes.

    Returns:
        np.ndarray: The matrix of softmax probabilities with shape (N, K).
    """
    #START MY MODIFICATIONS HERE

    # Subtract the maximum value in each row for numerical stability and prevent overflow
    y_stable = y - np.max(y, axis=1, keepdims=True)
    
    # Compute the exponentials
    exp_y = np.exp(y_stable)
    
    # Normalize by the sum of exponentials for each row
    softmax_scores = exp_y / np.sum(exp_y, axis=1, keepdims=True)

    #END MY MODIFICATIONS HERE
    
    return softmax_scores

