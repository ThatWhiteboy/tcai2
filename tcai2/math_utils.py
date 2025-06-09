"""Mathematical utility functions for AI/ML operations."""

import numpy as np
from typing import Union, List


def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Apply sigmoid activation function.
    
    Args:
        x: Input value(s)
        
    Returns:
        Sigmoid of input
    """
    # Clip x to prevent overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def relu(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Apply ReLU (Rectified Linear Unit) activation function.
    
    Args:
        x: Input value(s)
        
    Returns:
        ReLU of input
    """
    return np.maximum(0, x)


def softmax(x: Union[List, np.ndarray]) -> np.ndarray:
    """
    Apply softmax function to convert logits to probabilities.
    
    Args:
        x: Input logits
        
    Returns:
        Probability distribution
        
    Raises:
        ValueError: If input is empty
    """
    x = np.array(x)
    
    if x.size == 0:
        raise ValueError("Input cannot be empty")
    
    # Subtract max for numerical stability
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)


def mean_squared_error(predictions: Union[List, np.ndarray], 
                      targets: Union[List, np.ndarray]) -> float:
    """
    Calculate mean squared error between predictions and targets.
    
    Args:
        predictions: Predicted values
        targets: True target values
        
    Returns:
        Mean squared error
        
    Raises:
        ValueError: If predictions and targets have different lengths or are empty
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    if len(predictions) == 0:
        raise ValueError("Cannot calculate MSE for empty arrays")
        
    return np.mean((predictions - targets) ** 2)