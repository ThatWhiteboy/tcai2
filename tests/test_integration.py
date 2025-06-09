"""Integration tests for TCAI2 modules."""

import numpy as np
import pytest
from tcai2 import normalize_data, calculate_accuracy, sigmoid, relu, softmax, mean_squared_error


class TestIntegration:
    """Integration tests that use multiple modules together."""
    
    def test_ml_pipeline_simulation(self):
        """Test a simple ML pipeline using multiple utilities."""
        # Generate some sample data
        raw_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # Normalize the data
        normalized_data = normalize_data(raw_data, method="min_max")
        
        # Apply activation function
        activated_data = sigmoid(normalized_data)
        
        # Convert to binary predictions (threshold at 0.5)
        predictions = (activated_data > 0.5).astype(int)
        
        # Create some target values
        targets = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        
        # Calculate accuracy
        accuracy = calculate_accuracy(predictions, targets)
        
        # Verify the pipeline works
        assert isinstance(normalized_data, np.ndarray)
        assert isinstance(activated_data, np.ndarray)
        assert len(predictions) == len(targets)
        assert 0 <= accuracy <= 1
    
    def test_neural_network_simulation(self):
        """Test a simple neural network forward pass simulation."""
        # Input layer (3 neurons)
        inputs = [0.5, -0.2, 0.8]
        
        # Normalize inputs
        normalized_inputs = normalize_data(inputs, method="z_score")
        
        # Hidden layer with ReLU activation
        hidden_outputs = relu(normalized_inputs)
        
        # Output layer with softmax
        logits = [1.0, 2.0, 0.5]  # Simulated logits
        probabilities = softmax(logits)
        
        # Get prediction (argmax)
        prediction = np.argmax(probabilities)
        
        # Verify the forward pass
        assert len(hidden_outputs) == len(inputs)
        assert np.all(hidden_outputs >= 0)  # ReLU ensures non-negative
        assert abs(np.sum(probabilities) - 1.0) < 1e-10  # Probabilities sum to 1
        assert 0 <= prediction < len(logits)
    
    def test_model_evaluation_workflow(self):
        """Test a complete model evaluation workflow."""
        # Simulated model predictions and true targets
        predictions = [0.1, 0.8, 0.3, 0.9, 0.2, 0.7, 0.4, 0.6]
        targets = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        
        # Calculate MSE for regression-like evaluation
        mse = mean_squared_error(predictions, targets)
        
        # Convert to binary predictions for classification evaluation
        binary_predictions = (np.array(predictions) > 0.5).astype(int)
        binary_targets = np.array(targets, dtype=int)
        
        # Calculate accuracy
        accuracy = calculate_accuracy(binary_predictions, binary_targets)
        
        # Apply sigmoid to predictions to get probabilities
        sigmoid_predictions = sigmoid(predictions)
        
        # Verify evaluation metrics
        assert mse >= 0  # MSE is always non-negative
        assert 0 <= accuracy <= 1  # Accuracy is between 0 and 1
        assert np.all(0 <= sigmoid_predictions) and np.all(sigmoid_predictions <= 1)
    
    def test_data_preprocessing_pipeline(self):
        """Test a complete data preprocessing pipeline."""
        # Raw dataset
        raw_data = [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35]
        
        # Step 1: Min-max normalization
        minmax_normalized = normalize_data(raw_data, method="min_max")
        
        # Step 2: Z-score normalization of the min-max normalized data
        zscore_normalized = normalize_data(minmax_normalized, method="z_score")
        
        # Step 3: Apply different activation functions
        sigmoid_output = sigmoid(zscore_normalized)
        relu_output = relu(zscore_normalized)
        
        # Step 4: Create softmax probabilities from a subset
        subset_for_softmax = zscore_normalized[:3]
        softmax_probs = softmax(subset_for_softmax)
        
        # Verify the pipeline
        assert len(minmax_normalized) == len(raw_data)
        assert len(zscore_normalized) == len(raw_data)
        assert len(sigmoid_output) == len(raw_data)
        assert len(relu_output) == len(raw_data)
        assert len(softmax_probs) == 3
        
        # Check ranges
        assert np.all(0 <= minmax_normalized) and np.all(minmax_normalized <= 1)
        assert np.all(0 <= sigmoid_output) and np.all(sigmoid_output <= 1)
        assert np.all(relu_output >= 0)
        assert abs(np.sum(softmax_probs) - 1.0) < 1e-10
    
    def test_error_propagation(self):
        """Test that errors are properly propagated through the pipeline."""
        # Test empty data propagation
        with pytest.raises(ValueError):
            empty_data = []
            normalize_data(empty_data)
        
        # Test mismatched lengths propagation
        with pytest.raises(ValueError):
            predictions = [1, 2, 3]
            targets = [1, 2]
            calculate_accuracy(predictions, targets)
        
        # Test invalid method propagation
        with pytest.raises(ValueError):
            data = [1, 2, 3]
            normalize_data(data, method="invalid_method")