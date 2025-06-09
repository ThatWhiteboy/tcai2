"""Tests for math_utils module."""

import pytest
import numpy as np
from tcai2.math_utils import (
    sigmoid, relu, softmax, mean_squared_error,
    tanh, swish, gelu, cross_entropy_loss, huber_loss,
    batch_matrix_multiply, cosine_similarity, euclidean_distance
)


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


class TestTanh:
    """Test cases for tanh function."""
    
    def test_tanh_basic(self):
        """Test basic tanh functionality."""
        result = tanh(0)
        assert result == 0
        
        result = tanh([0, 1, -1])
        expected = np.tanh([0, 1, -1])
        np.testing.assert_allclose(result, expected)
    
    def test_tanh_derivative(self):
        """Test tanh derivative."""
        x = 0.5
        result = tanh(x, derivative=True)
        expected = 1 - np.tanh(x) ** 2
        assert abs(result - expected) < 1e-10


class TestSwish:
    """Test cases for swish function."""
    
    def test_swish_basic(self):
        """Test basic swish functionality."""
        x = 1.0
        result = swish(x)
        expected = x * sigmoid(x)
        assert abs(result - expected) < 1e-10
    
    def test_swish_with_beta(self):
        """Test swish with custom beta."""
        x = 1.0
        beta = 2.0
        result = swish(x, beta=beta)
        expected = x * sigmoid(beta * x)
        assert abs(result - expected) < 1e-10


class TestGelu:
    """Test cases for gelu function."""
    
    def test_gelu_approximate(self):
        """Test approximate GELU."""
        x = 1.0
        result = gelu(x, approximate=True)
        assert isinstance(result, (float, np.ndarray))
        assert result > 0  # GELU should be positive for positive input
    
    def test_gelu_exact(self):
        """Test exact GELU."""
        x = 1.0
        result = gelu(x, approximate=False)
        assert isinstance(result, (float, np.ndarray))
        assert result > 0


class TestCrossEntropyLoss:
    """Test cases for cross_entropy_loss function."""
    
    def test_cross_entropy_basic(self):
        """Test basic cross-entropy loss."""
        predictions = [[0.9, 0.1], [0.2, 0.8]]
        targets = [[1, 0], [0, 1]]
        
        result = cross_entropy_loss(predictions, targets)
        assert result > 0
        assert isinstance(result, float)
    
    def test_cross_entropy_class_indices(self):
        """Test cross-entropy with class indices."""
        predictions = [[0.9, 0.1], [0.2, 0.8]]
        targets = [0, 1]  # Class indices
        
        result = cross_entropy_loss(predictions, targets)
        assert result > 0
        assert isinstance(result, float)


class TestHuberLoss:
    """Test cases for huber_loss function."""
    
    def test_huber_loss_basic(self):
        """Test basic Huber loss."""
        predictions = [1, 2, 3]
        targets = [1.1, 2.1, 3.1]
        
        result = huber_loss(predictions, targets)
        assert result > 0
        assert isinstance(result, float)
    
    def test_huber_loss_with_outliers(self):
        """Test Huber loss with outliers."""
        predictions = [1, 2, 10]  # Large outlier
        targets = [1, 2, 3]
        
        result_huber = huber_loss(predictions, targets, delta=1.0)
        result_mse = mean_squared_error(predictions, targets)
        
        # Huber loss should be less affected by outliers
        assert result_huber < result_mse


class TestBatchMatrixMultiply:
    """Test cases for batch_matrix_multiply function."""
    
    def test_basic_multiplication(self):
        """Test basic matrix multiplication."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        
        result = batch_matrix_multiply(a, b)
        expected = np.matmul(a, b)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_with_caching(self):
        """Test matrix multiplication with caching."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        
        result1 = batch_matrix_multiply(a, b, use_cache=True)
        result2 = batch_matrix_multiply(a, b, use_cache=True)
        
        np.testing.assert_array_equal(result1, result2)


class TestCosineSimilarity:
    """Test cases for cosine_similarity function."""
    
    def test_identical_vectors(self):
        """Test cosine similarity of identical vectors."""
        a = [1, 2, 3]
        b = [1, 2, 3]
        
        result = cosine_similarity(a, b)
        assert abs(result - 1.0) < 1e-10
    
    def test_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors."""
        a = [1, 0]
        b = [0, 1]
        
        result = cosine_similarity(a, b)
        assert abs(result - 0.0) < 1e-10
    
    def test_opposite_vectors(self):
        """Test cosine similarity of opposite vectors."""
        a = [1, 2, 3]
        b = [-1, -2, -3]
        
        result = cosine_similarity(a, b)
        assert abs(result - (-1.0)) < 1e-10
    
    def test_zero_vector(self):
        """Test cosine similarity with zero vector."""
        a = [0, 0, 0]
        b = [1, 2, 3]
        
        result = cosine_similarity(a, b)
        assert result == 0.0
    
    def test_different_lengths_raises_error(self):
        """Test that different length vectors raise ValueError."""
        a = [1, 2, 3]
        b = [1, 2]
        
        with pytest.raises(ValueError, match="must have the same length"):
            cosine_similarity(a, b)


class TestEuclideanDistance:
    """Test cases for euclidean_distance function."""
    
    def test_identical_vectors(self):
        """Test Euclidean distance of identical vectors."""
        a = [1, 2, 3]
        b = [1, 2, 3]
        
        result = euclidean_distance(a, b)
        assert abs(result - 0.0) < 1e-10
    
    def test_simple_distance(self):
        """Test simple Euclidean distance."""
        a = [0, 0]
        b = [3, 4]
        
        result = euclidean_distance(a, b)
        expected = 5.0  # 3-4-5 triangle
        assert abs(result - expected) < 1e-10
    
    def test_negative_coordinates(self):
        """Test Euclidean distance with negative coordinates."""
        a = [-1, -1]
        b = [1, 1]
        
        result = euclidean_distance(a, b)
        expected = np.sqrt(8)  # sqrt((2)^2 + (2)^2)
        assert abs(result - expected) < 1e-10
    
    def test_different_lengths_raises_error(self):
        """Test that different length vectors raise ValueError."""
        a = [1, 2, 3]
        b = [1, 2]
        
        with pytest.raises(ValueError, match="must have the same length"):
            euclidean_distance(a, b)


class TestAdvancedActivations:
    """Test cases for advanced activation functions."""
    
    def test_sigmoid_derivative(self):
        """Test sigmoid derivative calculation."""
        x = 0.5
        result = sigmoid(x, derivative=True)
        sig_x = sigmoid(x, derivative=False)
        expected = sig_x * (1 - sig_x)
        assert abs(result - expected) < 1e-10
    
    def test_relu_leaky(self):
        """Test Leaky ReLU functionality."""
        x = [-1, 0, 1]
        alpha = 0.1
        result = relu(x, alpha=alpha)
        expected = [-0.1, 0, 1]
        np.testing.assert_allclose(result, expected)
    
    def test_relu_derivative(self):
        """Test ReLU derivative."""
        x = [-1, 0, 1]
        result = relu(x, derivative=True)
        expected = [0, 0, 1]
        np.testing.assert_array_equal(result, expected)
    
    def test_softmax_temperature(self):
        """Test softmax with temperature scaling."""
        x = [1, 2, 3]
        
        # Higher temperature should make distribution more uniform
        result_high_temp = softmax(x, temperature=10.0)
        result_low_temp = softmax(x, temperature=0.1)
        
        # High temperature should have lower max probability
        assert np.max(result_high_temp) < np.max(result_low_temp)
    
    def test_softmax_axis(self):
        """Test softmax along specific axis."""
        x = np.array([[1, 2], [3, 4]])
        result = softmax(x, axis=1)
        
        # Each row should sum to 1
        row_sums = np.sum(result, axis=1)
        np.testing.assert_allclose(row_sums, [1.0, 1.0])