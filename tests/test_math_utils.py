"""Tests for math_utils module."""

import pytest
import numpy as np
from tcai2.math_utils import sigmoid, relu, softmax, mean_squared_error


class TestSigmoid:
    """Test cases for sigmoid function."""
    
    def test_sigmoid_zero(self):
        """Test sigmoid at zero."""
        result = sigmoid(0)
        assert result == 0.5
    
    def test_sigmoid_positive(self):
        """Test sigmoid with positive values."""
        result = sigmoid(1)
        expected = 1 / (1 + np.exp(-1))
        assert abs(result - expected) < 1e-10
    
    def test_sigmoid_negative(self):
        """Test sigmoid with negative values."""
        result = sigmoid(-1)
        expected = 1 / (1 + np.exp(1))
        assert abs(result - expected) < 1e-10
    
    def test_sigmoid_array(self):
        """Test sigmoid with numpy array."""
        x = np.array([-2, -1, 0, 1, 2])
        result = sigmoid(x)
        expected = 1 / (1 + np.exp(-x))
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_sigmoid_large_positive(self):
        """Test sigmoid with large positive value (should approach 1)."""
        result = sigmoid(100)
        assert result > 0.99
        assert result <= 1.0
    
    def test_sigmoid_large_negative(self):
        """Test sigmoid with large negative value (should approach 0)."""
        result = sigmoid(-100)
        assert result < 0.01
        assert result >= 0.0
    
    def test_sigmoid_extreme_values(self):
        """Test sigmoid with extreme values to check overflow protection."""
        result_pos = sigmoid(1000)
        result_neg = sigmoid(-1000)
        
        assert 0 <= result_pos <= 1
        assert 0 <= result_neg <= 1
        assert not np.isnan(result_pos)
        assert not np.isnan(result_neg)


class TestRelu:
    """Test cases for relu function."""
    
    def test_relu_positive(self):
        """Test ReLU with positive values."""
        assert relu(5) == 5
        assert relu(0.5) == 0.5
        assert relu(100) == 100
    
    def test_relu_negative(self):
        """Test ReLU with negative values."""
        assert relu(-5) == 0
        assert relu(-0.5) == 0
        assert relu(-100) == 0
    
    def test_relu_zero(self):
        """Test ReLU with zero."""
        assert relu(0) == 0
    
    def test_relu_array(self):
        """Test ReLU with numpy array."""
        x = np.array([-3, -1, 0, 1, 3])
        result = relu(x)
        expected = np.array([0, 0, 0, 1, 3])
        np.testing.assert_array_equal(result, expected)
    
    def test_relu_float_array(self):
        """Test ReLU with float array."""
        x = np.array([-2.5, -0.1, 0.0, 0.1, 2.5])
        result = relu(x)
        expected = np.array([0.0, 0.0, 0.0, 0.1, 2.5])
        np.testing.assert_array_almost_equal(result, expected)


class TestSoftmax:
    """Test cases for softmax function."""
    
    def test_softmax_basic(self):
        """Test basic softmax functionality."""
        x = [1, 2, 3]
        result = softmax(x)
        
        # Check that probabilities sum to 1
        assert abs(np.sum(result) - 1.0) < 1e-10
        
        # Check that all values are positive
        assert np.all(result >= 0)
        
        # Check that larger inputs have larger probabilities
        assert result[2] > result[1] > result[0]
    
    def test_softmax_uniform(self):
        """Test softmax with uniform inputs."""
        x = [1, 1, 1, 1]
        result = softmax(x)
        expected = np.array([0.25, 0.25, 0.25, 0.25])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_softmax_numpy_array(self):
        """Test softmax with numpy array input."""
        x = np.array([0, 1, 2])
        result = softmax(x)
        
        assert abs(np.sum(result) - 1.0) < 1e-10
        assert np.all(result >= 0)
        assert isinstance(result, np.ndarray)
    
    def test_softmax_single_element(self):
        """Test softmax with single element."""
        x = [5]
        result = softmax(x)
        expected = np.array([1.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_softmax_large_values(self):
        """Test softmax with large values (numerical stability)."""
        x = [1000, 1001, 1002]
        result = softmax(x)
        
        assert abs(np.sum(result) - 1.0) < 1e-10
        assert np.all(result >= 0)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_softmax_negative_values(self):
        """Test softmax with negative values."""
        x = [-1, -2, -3]
        result = softmax(x)
        
        assert abs(np.sum(result) - 1.0) < 1e-10
        assert np.all(result >= 0)
        assert result[0] > result[1] > result[2]  # Less negative should have higher probability
    
    def test_softmax_empty_raises_error(self):
        """Test that empty input raises ValueError."""
        with pytest.raises(ValueError, match="Input cannot be empty"):
            softmax([])


class TestMeanSquaredError:
    """Test cases for mean_squared_error function."""
    
    def test_mse_perfect_predictions(self):
        """Test MSE with perfect predictions."""
        predictions = [1, 2, 3, 4, 5]
        targets = [1, 2, 3, 4, 5]
        result = mean_squared_error(predictions, targets)
        assert result == 0.0
    
    def test_mse_basic(self):
        """Test basic MSE calculation."""
        predictions = [1, 2, 3]
        targets = [2, 3, 4]
        result = mean_squared_error(predictions, targets)
        expected = np.mean([(1-2)**2, (2-3)**2, (3-4)**2])
        assert abs(result - expected) < 1e-10
    
    def test_mse_numpy_arrays(self):
        """Test MSE with numpy arrays."""
        predictions = np.array([0, 1, 2])
        targets = np.array([1, 1, 1])
        result = mean_squared_error(predictions, targets)
        expected = np.mean([(0-1)**2, (1-1)**2, (2-1)**2])  # [1, 0, 1] -> mean = 2/3
        assert abs(result - expected) < 1e-10
    
    def test_mse_single_element(self):
        """Test MSE with single element."""
        predictions = [5]
        targets = [3]
        result = mean_squared_error(predictions, targets)
        expected = (5 - 3) ** 2
        assert result == expected
    
    def test_mse_float_values(self):
        """Test MSE with float values."""
        predictions = [1.5, 2.5, 3.5]
        targets = [1.0, 2.0, 3.0]
        result = mean_squared_error(predictions, targets)
        expected = np.mean([0.25, 0.25, 0.25])  # All differences are 0.5, squared = 0.25
        assert abs(result - expected) < 1e-10
    
    def test_mse_negative_values(self):
        """Test MSE with negative values."""
        predictions = [-1, 0, 1]
        targets = [1, 0, -1]
        result = mean_squared_error(predictions, targets)
        expected = np.mean([(-1-1)**2, (0-0)**2, (1-(-1))**2])  # [4, 0, 4] -> mean = 8/3
        assert abs(result - expected) < 1e-10
    
    def test_mse_different_lengths_raises_error(self):
        """Test that different length arrays raise ValueError."""
        predictions = [1, 2, 3]
        targets = [1, 2]
        with pytest.raises(ValueError, match="must have the same length"):
            mean_squared_error(predictions, targets)
    
    def test_mse_empty_arrays_raises_error(self):
        """Test that empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="Cannot calculate MSE for empty arrays"):
            mean_squared_error([], [])
    
    def test_mse_large_values(self):
        """Test MSE with large values."""
        predictions = [1000, 2000, 3000]
        targets = [1001, 2001, 3001]
        result = mean_squared_error(predictions, targets)
        expected = 1.0  # All differences are 1, squared = 1
        assert abs(result - expected) < 1e-10