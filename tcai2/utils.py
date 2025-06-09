"""Utility functions for data processing and model evaluation."""

import numpy as np
from typing import List, Tuple, Union


def normalize_data(data: Union[List[float], np.ndarray], method: str = "min_max") -> np.ndarray:
    """
    Normalize data using specified method.
    
    Args:
        data: Input data to normalize
        method: Normalization method ('min_max' or 'z_score')
        
    Returns:
        Normalized data as numpy array
        
    Raises:
        ValueError: If method is not supported or data is empty
    """
    if len(data) == 0:
        raise ValueError("Data cannot be empty")
        
    data_array = np.array(data)
    
    if method == "min_max":
        data_min = np.min(data_array)
        data_max = np.max(data_array)
        if data_max == data_min:
            return np.zeros_like(data_array)
        return (data_array - data_min) / (data_max - data_min)
    elif method == "z_score":
        mean = np.mean(data_array)
        std = np.std(data_array)
        if std == 0:
            return np.zeros_like(data_array)
        return (data_array - mean) / std
    else:
        raise ValueError(f"Unsupported normalization method: {method}")


def calculate_accuracy(predictions: Union[List, np.ndarray], 
                      targets: Union[List, np.ndarray]) -> float:
    """
    Calculate accuracy between predictions and targets.
    
    Args:
        predictions: Predicted values
        targets: True target values
        
    Returns:
        Accuracy as a float between 0 and 1
        
    Raises:
        ValueError: If predictions and targets have different lengths
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    if len(predictions) == 0:
        raise ValueError("Cannot calculate accuracy for empty arrays")
        
    return np.mean(predictions == targets)


def split_data(data: Union[List, np.ndarray], 
               train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.
    
    Args:
        data: Input data to split
        train_ratio: Ratio of data to use for training (0 < train_ratio < 1)
        
    Returns:
        Tuple of (train_data, test_data)
        
    Raises:
        ValueError: If train_ratio is not between 0 and 1 or data is empty
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
        
    data_array = np.array(data)
    
    if len(data_array) == 0:
        raise ValueError("Data cannot be empty")
    
    split_idx = int(len(data_array) * train_ratio)
    
    # Shuffle data before splitting
    shuffled_indices = np.random.permutation(len(data_array))
    shuffled_data = data_array[shuffled_indices]
    
    train_data = shuffled_data[:split_idx]
    test_data = shuffled_data[split_idx:]
    
    return train_data, test_data