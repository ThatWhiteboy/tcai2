"""Tests for utils module."""

import pytest
import numpy as np
from tcai2.utils import normalize_data, calculate_accuracy, split_data


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