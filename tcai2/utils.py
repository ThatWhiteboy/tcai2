"""Utility functions for data processing and model evaluation."""

import numpy as np
from typing import List, Tuple, Union, Optional, Literal
from functools import lru_cache
import warnings


def normalize_data(
    data: Union[List[float], np.ndarray], 
    method: Literal["min_max", "z_score", "robust", "unit_vector"] = "min_max",
    axis: Optional[int] = None,
    feature_range: Tuple[float, float] = (0, 1)
) -> np.ndarray:
    """
    Normalize data using specified method with optimized performance.
    
    Args:
        data: Input data to normalize
        method: Normalization method ('min_max', 'z_score', 'robust', 'unit_vector')
        axis: Axis along which to normalize (None for global normalization)
        feature_range: Target range for min_max normalization
        
    Returns:
        Normalized data as numpy array
        
    Raises:
        ValueError: If method is not supported or data is empty
    """
    if len(data) == 0:
        raise ValueError("Data cannot be empty")
        
    data_array = np.asarray(data, dtype=np.float64)
    
    if method == "min_max":
        data_min = np.min(data_array, axis=axis, keepdims=True)
        data_max = np.max(data_array, axis=axis, keepdims=True)
        
        # Handle constant data efficiently
        range_mask = (data_max - data_min) == 0
        if np.any(range_mask):
            warnings.warn("Constant features detected, setting to feature_range minimum")
            
        # Vectorized min-max scaling with custom range
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized = (data_array - data_min) / (data_max - data_min)
            normalized = np.where(range_mask, 0, normalized)
            
        # Scale to custom feature range
        min_val, max_val = feature_range
        return normalized * (max_val - min_val) + min_val
        
    elif method == "z_score":
        mean = np.mean(data_array, axis=axis, keepdims=True)
        std = np.std(data_array, axis=axis, keepdims=True, ddof=0)
        
        # Handle zero standard deviation
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized = (data_array - mean) / std
            normalized = np.where(std == 0, 0, normalized)
        return normalized
        
    elif method == "robust":
        # Robust scaling using median and IQR
        median = np.median(data_array, axis=axis, keepdims=True)
        q75 = np.percentile(data_array, 75, axis=axis, keepdims=True)
        q25 = np.percentile(data_array, 25, axis=axis, keepdims=True)
        iqr = q75 - q25
        
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized = (data_array - median) / iqr
            normalized = np.where(iqr == 0, 0, normalized)
        return normalized
        
    elif method == "unit_vector":
        # L2 normalization
        norm = np.linalg.norm(data_array, axis=axis, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized = data_array / norm
            normalized = np.where(norm == 0, 0, normalized)
        return normalized
        
    else:
        raise ValueError(f"Unsupported normalization method: {method}. "
                        f"Supported methods: 'min_max', 'z_score', 'robust', 'unit_vector'")


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


def calculate_metrics(predictions: Union[List, np.ndarray], 
                     targets: Union[List, np.ndarray],
                     average: Literal["binary", "macro", "micro", "weighted"] = "binary") -> dict:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        predictions: Predicted labels
        targets: True labels
        average: Averaging strategy for multi-class metrics
        
    Returns:
        Dictionary containing precision, recall, f1_score, and accuracy
    """
    predictions_array = np.asarray(predictions)
    targets_array = np.asarray(targets)
    
    if len(predictions_array) != len(targets_array):
        raise ValueError("Predictions and targets must have the same length")
    
    if len(predictions_array) == 0:
        raise ValueError("Cannot calculate metrics for empty arrays")
    
    # Get unique classes
    classes = np.unique(np.concatenate([predictions_array, targets_array]))
    
    if len(classes) == 2 and average == "binary":
        # Binary classification
        tp = np.sum((predictions_array == 1) & (targets_array == 1))
        fp = np.sum((predictions_array == 1) & (targets_array == 0))
        fn = np.sum((predictions_array == 0) & (targets_array == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
    else:
        # Multi-class classification
        precisions, recalls, f1s = [], [], []
        
        for cls in classes:
            tp = np.sum((predictions_array == cls) & (targets_array == cls))
            fp = np.sum((predictions_array == cls) & (targets_array != cls))
            fn = np.sum((predictions_array != cls) & (targets_array == cls))
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
        
        if average == "macro":
            precision = np.mean(precisions)
            recall = np.mean(recalls)
            f1 = np.mean(f1s)
        elif average == "micro":
            # Calculate global metrics
            tp_total = np.sum(predictions_array == targets_array)
            precision = recall = f1 = tp_total / len(predictions_array)
        elif average == "weighted":
            # Weight by support (number of true instances for each class)
            weights = [np.sum(targets_array == cls) for cls in classes]
            total_weight = sum(weights)
            
            precision = np.average(precisions, weights=weights) if total_weight > 0 else 0.0
            recall = np.average(recalls, weights=weights) if total_weight > 0 else 0.0
            f1 = np.average(f1s, weights=weights) if total_weight > 0 else 0.0
    
    accuracy = calculate_accuracy(predictions_array, targets_array)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


@lru_cache(maxsize=128)
def _cached_split_indices(data_length: int, train_ratio: float, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Cache split indices for repeated operations with same parameters."""
    if seed is not None:
        np.random.seed(seed)
    
    split_idx = int(data_length * train_ratio)
    shuffled_indices = np.random.permutation(data_length)
    
    return shuffled_indices[:split_idx], shuffled_indices[split_idx:]


def split_data_advanced(data: Union[List, np.ndarray], 
                       targets: Optional[Union[List, np.ndarray]] = None,
                       train_ratio: float = 0.8,
                       stratify: bool = False,
                       random_state: Optional[int] = None) -> Union[Tuple[np.ndarray, np.ndarray], 
                                                                   Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Advanced data splitting with stratification and reproducibility.
    
    Args:
        data: Input data to split
        targets: Target labels (required for stratification)
        train_ratio: Ratio of training data
        stratify: Whether to maintain class distribution
        random_state: Random seed for reproducibility
        
    Returns:
        Split data (and targets if provided)
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
        
    data_array = np.asarray(data)
    
    if len(data_array) == 0:
        raise ValueError("Data cannot be empty")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    if stratify and targets is not None:
        targets_array = np.asarray(targets)
        if len(data_array) != len(targets_array):
            raise ValueError("Data and targets must have the same length")
        
        # Stratified split
        unique_classes = np.unique(targets_array)
        train_indices = []
        test_indices = []
        
        for cls in unique_classes:
            cls_indices = np.where(targets_array == cls)[0]
            n_train = int(len(cls_indices) * train_ratio)
            
            if random_state is not None:
                np.random.seed(random_state + hash(str(cls)) % 1000)
            
            shuffled_cls_indices = np.random.permutation(cls_indices)
            train_indices.extend(shuffled_cls_indices[:n_train])
            test_indices.extend(shuffled_cls_indices[n_train:])
        
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        
        # Shuffle the final indices
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
        
        X_train, X_test = data_array[train_indices], data_array[test_indices]
        y_train, y_test = targets_array[train_indices], targets_array[test_indices]
        
        return X_train, X_test, y_train, y_test
    
    else:
        # Simple split
        train_indices, test_indices = _cached_split_indices(len(data_array), train_ratio, random_state)
        
        X_train, X_test = data_array[train_indices], data_array[test_indices]
        
        if targets is not None:
            targets_array = np.asarray(targets)
            y_train, y_test = targets_array[train_indices], targets_array[test_indices]
            return X_train, X_test, y_train, y_test
        
        return X_train, X_test