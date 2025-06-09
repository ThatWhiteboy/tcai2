"""TCAI2 - AI Utilities Library"""

__version__ = "0.1.0"

from .utils import normalize_data, calculate_accuracy, split_data
from .math_utils import sigmoid, relu, softmax, mean_squared_error

__all__ = [
    "normalize_data",
    "calculate_accuracy", 
    "split_data",
    "sigmoid",
    "relu",
    "softmax",
    "mean_squared_error"
]