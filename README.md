# TCAI2 - AI Utilities Library

A collection of utility functions for AI and machine learning tasks.

## Features

- Data preprocessing utilities
- Mathematical helper functions
- Model evaluation tools

## Installation

```bash
pip install -r requirements.txt
```

## Testing

```bash
pytest
```

## Usage

```python
from tcai2.utils import normalize_data, calculate_accuracy
from tcai2.math_utils import sigmoid, relu

# Example usage
data = [1, 2, 3, 4, 5]
normalized = normalize_data(data)
print(normalized)
```