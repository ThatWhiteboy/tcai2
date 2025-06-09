"""Mathematical utility functions for AI/ML operations."""

import numpy as np
from typing import Union, List, Optional, Tuple
from functools import lru_cache
import warnings


def sigmoid(x: Union[float, np.ndarray], derivative: bool = False) -> Union[float, np.ndarray]:
    """
    Apply sigmoid activation function with optimized numerical stability.
    
    Args:
        x: Input value(s)
        derivative: If True, return derivative of sigmoid
        
    Returns:
        Sigmoid output or its derivative
    """
    x = np.asarray(x, dtype=np.float64)
    
    if derivative:
        sig = sigmoid(x, derivative=False)
        return sig * (1 - sig)
    
    # Optimized sigmoid with better numerical stability
    # Use different formulations for positive and negative values
    positive_mask = x >= 0
    result = np.zeros_like(x, dtype=np.float64)
    
    # For positive values: 1 / (1 + exp(-x))
    exp_neg_x = np.exp(-np.clip(x[positive_mask], 0, 700))
    result[positive_mask] = 1 / (1 + exp_neg_x)
    
    # For negative values: exp(x) / (1 + exp(x))
    exp_x = np.exp(np.clip(x[~positive_mask], -700, 0))
    result[~positive_mask] = exp_x / (1 + exp_x)
    
    return result


def relu(x: Union[float, np.ndarray], derivative: bool = False, alpha: float = 0.0) -> Union[float, np.ndarray]:
    """
    Apply ReLU (Rectified Linear Unit) activation function with variants.
    
    Args:
        x: Input value(s)
        derivative: If True, return derivative of ReLU
        alpha: Slope for negative values (Leaky ReLU when > 0)
        
    Returns:
        ReLU output or its derivative
    """
    x = np.asarray(x, dtype=np.float64)
    
    if derivative:
        return np.where(x > 0, 1.0, alpha)
    
    return np.where(x > 0, x, alpha * x)


def softmax(x: Union[List, np.ndarray], axis: Optional[int] = None, temperature: float = 1.0) -> np.ndarray:
    """
    Apply softmax function to convert logits to probabilities with temperature scaling.
    
    Args:
        x: Input logits
        axis: Axis along which to apply softmax (None for global)
        temperature: Temperature parameter for scaling (higher = more uniform)
        
    Returns:
        Probability distribution
        
    Raises:
        ValueError: If input is empty or temperature <= 0
    """
    x = np.asarray(x, dtype=np.float64)
    
    if x.size == 0:
        raise ValueError("Input cannot be empty")
    
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    
    # Apply temperature scaling
    x_scaled = x / temperature
    
    # Subtract max for numerical stability
    x_shifted = x_scaled - np.max(x_scaled, axis=axis, keepdims=True)
    
    # Compute softmax
    exp_x = np.exp(np.clip(x_shifted, -700, 700))  # Prevent overflow
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


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


def tanh(x: Union[float, np.ndarray], derivative: bool = False) -> Union[float, np.ndarray]:
    """
    Apply hyperbolic tangent activation function.
    
    Args:
        x: Input value(s)
        derivative: If True, return derivative of tanh
        
    Returns:
        Tanh output or its derivative
    """
    x = np.asarray(x, dtype=np.float64)
    
    if derivative:
        tanh_x = tanh(x, derivative=False)
        return 1 - tanh_x ** 2
    
    return np.tanh(x)


def swish(x: Union[float, np.ndarray], beta: float = 1.0) -> Union[float, np.ndarray]:
    """
    Apply Swish activation function (x * sigmoid(beta * x)).
    
    Args:
        x: Input value(s)
        beta: Scaling parameter
        
    Returns:
        Swish output
    """
    x = np.asarray(x, dtype=np.float64)
    return x * sigmoid(beta * x)


def gelu(x: Union[float, np.ndarray], approximate: bool = False) -> Union[float, np.ndarray]:
    """
    Apply Gaussian Error Linear Unit (GELU) activation function.
    
    Args:
        x: Input value(s)
        approximate: Use approximate version for faster computation
        
    Returns:
        GELU output
    """
    x = np.asarray(x, dtype=np.float64)
    
    if approximate:
        # Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    else:
        # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        from scipy.special import erf
        return 0.5 * x * (1 + erf(x / np.sqrt(2)))


def cross_entropy_loss(predictions: Union[List, np.ndarray], 
                      targets: Union[List, np.ndarray],
                      epsilon: float = 1e-15) -> float:
    """
    Calculate cross-entropy loss for classification.
    
    Args:
        predictions: Predicted probabilities
        targets: True labels (one-hot encoded or class indices)
        epsilon: Small value to prevent log(0)
        
    Returns:
        Cross-entropy loss
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    
    # Clip predictions to prevent log(0)
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    
    if targets.ndim == 1:
        # Convert class indices to one-hot
        num_classes = int(np.max(targets)) + 1
        targets_onehot = np.eye(num_classes)[targets.astype(int)]
    else:
        targets_onehot = targets
    
    return -np.mean(np.sum(targets_onehot * np.log(predictions), axis=-1))


def huber_loss(predictions: Union[List, np.ndarray], 
               targets: Union[List, np.ndarray],
               delta: float = 1.0) -> float:
    """
    Calculate Huber loss (robust to outliers).
    
    Args:
        predictions: Predicted values
        targets: True target values
        delta: Threshold for switching between MSE and MAE
        
    Returns:
        Huber loss
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    
    residual = np.abs(predictions - targets)
    
    # Use MSE for small errors, MAE for large errors
    condition = residual <= delta
    squared_loss = 0.5 * residual ** 2
    linear_loss = delta * residual - 0.5 * delta ** 2
    
    return np.mean(np.where(condition, squared_loss, linear_loss))


@lru_cache(maxsize=32)
def _cached_matrix_multiply(a_shape: Tuple[int, ...], b_shape: Tuple[int, ...], 
                           a_hash: int, b_hash: int) -> np.ndarray:
    """Cache matrix multiplication results for repeated operations."""
    # This is a placeholder for demonstration - in practice, you'd store the actual matrices
    # For now, we'll just return the shape information
    return a_shape, b_shape


def batch_matrix_multiply(a: np.ndarray, b: np.ndarray, 
                         use_cache: bool = False) -> np.ndarray:
    """
    Optimized batch matrix multiplication.
    
    Args:
        a: First matrix/batch of matrices
        b: Second matrix/batch of matrices
        use_cache: Whether to use caching for repeated operations
        
    Returns:
        Matrix multiplication result
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    if use_cache:
        # Simple hash for caching (in practice, use more sophisticated hashing)
        a_hash = hash(a.tobytes())
        b_hash = hash(b.tobytes())
        _cached_matrix_multiply(a.shape, b.shape, a_hash, b_hash)
    
    return np.matmul(a, b)


def cosine_similarity(a: Union[List, np.ndarray], 
                     b: Union[List, np.ndarray]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity (-1 to 1)
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return np.dot(a, b) / (norm_a * norm_b)


def euclidean_distance(a: Union[List, np.ndarray], 
                      b: Union[List, np.ndarray]) -> float:
    """
    Calculate Euclidean distance between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Euclidean distance
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    
    return np.linalg.norm(a - b)