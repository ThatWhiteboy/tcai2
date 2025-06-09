"""TCAI2 - AI Utilities Library"""

__version__ = "0.2.0"

from .utils import (
    normalize_data, calculate_accuracy, split_data,
    calculate_metrics, split_data_advanced
)
from .math_utils import (
    sigmoid, relu, softmax, mean_squared_error,
    tanh, swish, gelu, cross_entropy_loss, huber_loss,
    batch_matrix_multiply, cosine_similarity, euclidean_distance
)

__all__ = [
    # Data utilities
    "normalize_data",
    "calculate_accuracy", 
    "calculate_metrics",
    "split_data",
    "split_data_advanced",
    
    # Activation functions
    "sigmoid",
    "relu", 
    "tanh",
    "swish",
    "gelu",
    "softmax",
    
    # Loss functions
    "mean_squared_error",
    "cross_entropy_loss",
    "huber_loss",
    
    # Math utilities
    "batch_matrix_multiply",
    "cosine_similarity",
    "euclidean_distance"
]