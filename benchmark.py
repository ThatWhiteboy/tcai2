#!/usr/bin/env python3
"""Performance benchmarks for TCAI2 optimizations."""

import time
import numpy as np
from tcai2 import (
    normalize_data, calculate_metrics, split_data_advanced,
    sigmoid, relu, tanh, swish, gelu, softmax,
    cross_entropy_loss, huber_loss, cosine_similarity, euclidean_distance
)


def benchmark_function(func, *args, iterations=1000, **kwargs):
    """Benchmark a function's execution time."""
    start_time = time.time()
    for _ in range(iterations):
        result = func(*args, **kwargs)
    end_time = time.time()
    return (end_time - start_time) / iterations, result


def run_benchmarks():
    """Run comprehensive benchmarks."""
    print("TCAI2 Performance Benchmarks")
    print("=" * 50)
    
    # Generate test data
    np.random.seed(42)
    small_data = np.random.randn(100)
    medium_data = np.random.randn(1000)
    large_data = np.random.randn(10000)
    
    # Classification data
    predictions = np.random.rand(1000, 3)  # 3-class probabilities
    predictions = predictions / predictions.sum(axis=1, keepdims=True)  # Normalize
    targets = np.random.randint(0, 3, 1000)
    
    # Vectors for similarity
    vec_a = np.random.randn(1000)
    vec_b = np.random.randn(1000)
    
    print("\n1. Data Normalization Benchmarks")
    print("-" * 30)
    
    methods = ["min_max", "z_score", "robust", "unit_vector"]
    for method in methods:
        avg_time, _ = benchmark_function(normalize_data, large_data, method=method, iterations=100)
        print(f"{method:12}: {avg_time*1000:.3f} ms")
    
    print("\n2. Activation Function Benchmarks")
    print("-" * 30)
    
    activations = [
        ("sigmoid", sigmoid),
        ("relu", relu),
        ("tanh", tanh),
        ("swish", swish),
        ("gelu_approx", lambda x: gelu(x, approximate=True)),
        ("gelu_exact", lambda x: gelu(x, approximate=False)),
    ]
    
    for name, func in activations:
        avg_time, _ = benchmark_function(func, large_data, iterations=100)
        print(f"{name:12}: {avg_time*1000:.3f} ms")
    
    print("\n3. Loss Function Benchmarks")
    print("-" * 30)
    
    # Binary predictions for some loss functions
    binary_pred = np.random.rand(1000)
    binary_targets = np.random.randint(0, 2, 1000)
    
    loss_functions = [
        ("cross_entropy", cross_entropy_loss, predictions, targets),
        ("huber_loss", huber_loss, binary_pred, binary_targets),
    ]
    
    for name, func, pred, targ in loss_functions:
        avg_time, _ = benchmark_function(func, pred, targ, iterations=100)
        print(f"{name:12}: {avg_time*1000:.3f} ms")
    
    print("\n4. Metrics Calculation Benchmarks")
    print("-" * 30)
    
    binary_pred_int = np.random.randint(0, 2, 1000)
    binary_targets_int = np.random.randint(0, 2, 1000)
    
    avg_time, metrics = benchmark_function(
        calculate_metrics, binary_pred_int, binary_targets_int, iterations=100
    )
    print(f"{'metrics':12}: {avg_time*1000:.3f} ms")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1-Score: {metrics['f1_score']:.3f}")
    
    print("\n5. Distance/Similarity Benchmarks")
    print("-" * 30)
    
    similarity_functions = [
        ("cosine_sim", cosine_similarity),
        ("euclidean", euclidean_distance),
    ]
    
    for name, func in similarity_functions:
        avg_time, _ = benchmark_function(func, vec_a, vec_b, iterations=100)
        print(f"{name:12}: {avg_time*1000:.3f} ms")
    
    print("\n6. Advanced Data Splitting Benchmarks")
    print("-" * 30)
    
    # Test stratified vs regular splitting
    data_2d = np.random.randn(10000, 10)
    labels = np.random.randint(0, 3, 10000)
    
    avg_time, _ = benchmark_function(
        split_data_advanced, data_2d, labels, stratify=False, iterations=10
    )
    print(f"{'regular':12}: {avg_time*1000:.1f} ms")
    
    avg_time, _ = benchmark_function(
        split_data_advanced, data_2d, labels, stratify=True, iterations=10
    )
    print(f"{'stratified':12}: {avg_time*1000:.1f} ms")
    
    print("\n7. Softmax with Temperature Scaling")
    print("-" * 30)
    
    logits = np.random.randn(1000, 10)
    
    temperatures = [0.1, 1.0, 10.0]
    for temp in temperatures:
        avg_time, result = benchmark_function(
            softmax, logits, axis=1, temperature=temp, iterations=100
        )
        print(f"temp={temp:4.1f}: {avg_time*1000:.3f} ms (max_prob: {np.max(result):.3f})")
    
    print("\n8. Memory Usage Comparison")
    print("-" * 30)
    
    # Test memory efficiency of vectorized operations
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    # Before large operation
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Large normalization operation
    huge_data = np.random.randn(100000)
    normalized = normalize_data(huge_data, method="z_score")
    
    # After operation
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Memory before: {mem_before:.1f} MB")
    print(f"Memory after:  {mem_after:.1f} MB")
    print(f"Memory delta:  {mem_after - mem_before:.1f} MB")
    
    print("\n" + "=" * 50)
    print("Benchmark completed successfully!")
    print(f"Total functions tested: {len(methods) + len(activations) + len(loss_functions) + len(similarity_functions) + 5}")


if __name__ == "__main__":
    run_benchmarks()