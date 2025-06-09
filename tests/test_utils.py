"""Tests for utils module."""

import pytest
import numpy as np
import warnings
from tcai2.utils import (
    normalize_data, calculate_accuracy, split_data,
    calculate_metrics, split_data_advanced
)


class TestNormalizeData:
    """Test cases for normalize_data function."""
    
    def test_min_max_normalization_basic(self):
        """Test basic min-max normalization."""
        data = [1, 2, 3, 4, 5]
        result = normalize_data(data, method="min_max")
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_min_max_normalization_single_value(self):
        """Test min-max normalization with single value."""
        data = [5]
        result = normalize_data(data, method="min_max")
        expected = np.array([0.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_min_max_normalization_identical_values(self):
        """Test min-max normalization with identical values."""
        data = [3, 3, 3, 3]
        result = normalize_data(data, method="min_max")
        expected = np.array([0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_z_score_normalization_basic(self):
        """Test basic z-score normalization."""
        data = [1, 2, 3, 4, 5]
        result = normalize_data(data, method="z_score")
        # Mean should be approximately 0, std should be approximately 1
        assert abs(np.mean(result)) < 1e-10
        assert abs(np.std(result) - 1.0) < 1e-10
    
    def test_z_score_normalization_zero_std(self):
        """Test z-score normalization with zero standard deviation."""
        data = [2, 2, 2, 2]
        result = normalize_data(data, method="z_score")
        expected = np.array([0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_numpy_array_input(self):
        """Test with numpy array input."""
        data = np.array([10, 20, 30, 40, 50])
        result = normalize_data(data, method="min_max")
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_negative_values(self):
        """Test normalization with negative values."""
        data = [-2, -1, 0, 1, 2]
        result = normalize_data(data, method="min_max")
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_empty_data_raises_error(self):
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError, match="Data cannot be empty"):
            normalize_data([])
    
    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported normalization method"):
            normalize_data([1, 2, 3], method="invalid")


class TestCalculateAccuracy:
    """Test cases for calculate_accuracy function."""
    
    def test_perfect_accuracy(self):
        """Test perfect accuracy case."""
        predictions = [1, 0, 1, 0, 1]
        targets = [1, 0, 1, 0, 1]
        result = calculate_accuracy(predictions, targets)
        assert result == 1.0
    
    def test_zero_accuracy(self):
        """Test zero accuracy case."""
        predictions = [1, 1, 1, 1, 1]
        targets = [0, 0, 0, 0, 0]
        result = calculate_accuracy(predictions, targets)
        assert result == 0.0
    
    def test_partial_accuracy(self):
        """Test partial accuracy case."""
        predictions = [1, 0, 1, 0, 1]
        targets = [1, 0, 0, 0, 1]
        result = calculate_accuracy(predictions, targets)
        assert result == 0.8  # 4 out of 5 correct
    
    def test_numpy_array_input(self):
        """Test with numpy array input."""
        predictions = np.array([1, 0, 1, 0])
        targets = np.array([1, 0, 1, 1])
        result = calculate_accuracy(predictions, targets)
        assert result == 0.75  # 3 out of 4 correct
    
    def test_single_element(self):
        """Test with single element arrays."""
        predictions = [1]
        targets = [1]
        result = calculate_accuracy(predictions, targets)
        assert result == 1.0
    
    def test_different_lengths_raises_error(self):
        """Test that different length arrays raise ValueError."""
        predictions = [1, 0, 1]
        targets = [1, 0]
        with pytest.raises(ValueError, match="must have the same length"):
            calculate_accuracy(predictions, targets)
    
    def test_empty_arrays_raises_error(self):
        """Test that empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="Cannot calculate accuracy for empty arrays"):
            calculate_accuracy([], [])


class TestSplitData:
    """Test cases for split_data function."""
    
    def test_basic_split(self):
        """Test basic data splitting."""
        np.random.seed(42)  # For reproducible results
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        train_data, test_data = split_data(data, train_ratio=0.8)
        
        assert len(train_data) == 8
        assert len(test_data) == 2
        assert len(train_data) + len(test_data) == len(data)
    
    def test_different_ratio(self):
        """Test splitting with different ratio."""
        np.random.seed(42)
        data = list(range(100))
        train_data, test_data = split_data(data, train_ratio=0.7)
        
        assert len(train_data) == 70
        assert len(test_data) == 30
    
    def test_numpy_array_input(self):
        """Test with numpy array input."""
        np.random.seed(42)
        data = np.array([1, 2, 3, 4, 5])
        train_data, test_data = split_data(data, train_ratio=0.6)
        
        assert len(train_data) == 3
        assert len(test_data) == 2
        assert isinstance(train_data, np.ndarray)
        assert isinstance(test_data, np.ndarray)
    
    def test_small_dataset(self):
        """Test with very small dataset."""
        np.random.seed(42)
        data = [1, 2]
        train_data, test_data = split_data(data, train_ratio=0.5)
        
        assert len(train_data) == 1
        assert len(test_data) == 1
    
    def test_invalid_ratio_raises_error(self):
        """Test that invalid ratios raise ValueError."""
        data = [1, 2, 3, 4, 5]
        
        with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
            split_data(data, train_ratio=0.0)
        
        with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
            split_data(data, train_ratio=1.0)
        
        with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
            split_data(data, train_ratio=1.5)
    
    def test_empty_data_raises_error(self):
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError, match="Data cannot be empty"):
            split_data([])
    
    def test_data_integrity(self):
        """Test that all original data is preserved after splitting."""
        np.random.seed(42)
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        train_data, test_data = split_data(data)
        
        combined = np.concatenate([train_data, test_data])
        combined_sorted = np.sort(combined)
        original_sorted = np.sort(data)
        
        np.testing.assert_array_equal(combined_sorted, original_sorted)


class TestCalculateMetrics:
    """Test cases for calculate_metrics function."""
    
    def test_binary_classification(self):
        """Test binary classification metrics."""
        predictions = [1, 0, 1, 1, 0]
        targets = [1, 0, 1, 0, 0]
        
        metrics = calculate_metrics(predictions, targets, average="binary")
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1
    
    def test_multiclass_macro(self):
        """Test multiclass classification with macro averaging."""
        predictions = [0, 1, 2, 1, 0, 2]
        targets = [0, 1, 2, 2, 0, 1]
        
        metrics = calculate_metrics(predictions, targets, average="macro")
        
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1
    
    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        predictions = [1, 0, 1, 0]
        targets = [1, 0, 1, 0]
        
        metrics = calculate_metrics(predictions, targets)
        
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0
    
    def test_empty_arrays_raise_error(self):
        """Test that empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="Cannot calculate metrics for empty arrays"):
            calculate_metrics([], [])


class TestSplitDataAdvanced:
    """Test cases for split_data_advanced function."""
    
    def test_basic_split(self):
        """Test basic data splitting."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        X_train, X_test = split_data_advanced(data, random_state=42)
        
        assert len(X_train) == 8
        assert len(X_test) == 2
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
    
    def test_with_targets(self):
        """Test splitting with targets."""
        data = np.array([1, 2, 3, 4, 5, 6])
        targets = np.array([0, 1, 0, 1, 0, 1])
        
        X_train, X_test, y_train, y_test = split_data_advanced(
            data, targets, random_state=42
        )
        
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert len(X_train) + len(X_test) == len(data)
    
    def test_stratified_split(self):
        """Test stratified splitting."""
        data = np.array(range(20))
        targets = np.array([0, 1] * 10)  # Balanced classes
        
        X_train, X_test, y_train, y_test = split_data_advanced(
            data, targets, stratify=True, random_state=42
        )
        
        # Check that class distribution is maintained
        train_class_ratio = np.mean(y_train)
        test_class_ratio = np.mean(y_test)
        original_ratio = np.mean(targets)
        
        assert abs(train_class_ratio - original_ratio) < 0.1
        assert abs(test_class_ratio - original_ratio) < 0.1
    
    def test_reproducibility(self):
        """Test that random_state ensures reproducibility."""
        data = np.array(range(100))
        
        X_train1, X_test1 = split_data_advanced(data, random_state=42)
        X_train2, X_test2 = split_data_advanced(data, random_state=42)
        
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)


class TestNormalizeDataAdvanced:
    """Test cases for advanced normalize_data features."""
    
    def test_robust_normalization(self):
        """Test robust normalization method."""
        data = [1, 2, 3, 4, 100]  # Data with outlier
        normalized = normalize_data(data, method="robust")
        
        assert isinstance(normalized, np.ndarray)
        assert len(normalized) == len(data)
        # Robust scaling should be less affected by outliers than min-max
        normalized_minmax = normalize_data(data, method="min_max")
        # The outlier should be less extreme in robust scaling
        assert abs(normalized[-1]) < abs(normalized_minmax[-1]) * 100  # Much less extreme
    
    def test_unit_vector_normalization(self):
        """Test unit vector normalization."""
        data = [3, 4]  # 3-4-5 triangle
        normalized = normalize_data(data, method="unit_vector")
        
        # Should have unit norm
        norm = np.linalg.norm(normalized)
        assert abs(norm - 1.0) < 1e-10
    
    def test_custom_feature_range(self):
        """Test min-max normalization with custom range."""
        data = [1, 2, 3, 4, 5]
        normalized = normalize_data(data, method="min_max", feature_range=(-1, 1))
        
        assert np.min(normalized) == -1.0
        assert np.max(normalized) == 1.0
    
    def test_axis_parameter(self):
        """Test normalization along specific axis."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        normalized = normalize_data(data, method="min_max", axis=0)
        
        # Each column should be normalized independently
        assert normalized.shape == data.shape
        assert np.allclose(np.min(normalized, axis=0), 0)
        assert np.allclose(np.max(normalized, axis=0), 1)
    
    def test_constant_features_warning(self):
        """Test warning for constant features."""
        data = [1, 1, 1, 1]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            normalized = normalize_data(data, method="min_max")
            
            assert len(w) == 1
            assert "Constant features detected" in str(w[0].message)
            assert np.allclose(normalized, 0)