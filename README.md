# ESSA - Easy Singular Spectrum Analysis

A Python package for Singular Spectrum Analysis (SSA) of time series data.

## Installation

```bash
pip install essa
```

## Features

- Fast implementation of SSA using vectorized operations
- Support for both full SVD and randomized SVD for large datasets
- Support for Toeplitz SSA method for stationary series
- Simple API for decomposition and reconstruction
- Compatible with NumPy arrays

## Usage Example

```python
from essa import Decompose, reconstruct
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
t = np.linspace(0, 2*np.pi, 100)
series = np.sin(t) + 0.5*np.sin(3*t)

# Create decomposer with window size of 20
decomposer = Decompose(time_series=series, window_size=20)
decomposer.fit()  # Perform decomposition

# Reconstruct components
trend = reconstruct(decomposer, [[0]])[0]  # First component as trend
seasonal = reconstruct(decomposer, [[1, 2]])[0]  # 2nd and 3rd components as seasonality
noise = reconstruct(decomposer, [[3]])[0]  # 4th component as noise

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, series, label='Original Series')
plt.plot(t, trend, label='Trend')
plt.plot(t, seasonal, label='Seasonality')
plt.plot(t, noise, label='Noise')
plt.legend()
plt.title('SSA Decomposition')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()
```

## Toeplitz SSA Example

```python
# For stationary time series, Toeplitz SSA may provide better results
decomposer = Decompose(time_series=series, window_size=20, method="toeplitz")
decomposer.fit()

# Reconstruct components
trend = reconstruct(decomposer, [[0]])[0]
seasonal = reconstruct(decomposer, [[1, 2, 3]])[0]
# ... rest of analysis
```

## API Reference

### Decompose Class

```python
Decompose(time_series, window_size, method="basic", svd_method=None)
```

**Parameters:**

- `time_series` (np.ndarray): The time series data to analyze
- `window_size` (int): The embedding window length (L)
- `method` (str): SSA method to use - 'basic' (default) or 'toeplitz'
- `svd_method` (str): Only for basic method - 'full' for exact SVD or 'randomized' for approximate (default: 'full')

**Methods:**

- `fit()`: Perform decomposition and store components

### reconstruct Function

```python
reconstruct(decomposer, groups)
```

**Parameters:**

- `decomposer`: A fitted Decompose object
- `groups` (List[List[int]]): List of component groups to reconstruct

**Returns:**

- Array of reconstructed components for each group

## License

MIT License

## Citation

If you use this package in your research, please cite:

```Python
@software{essa2025,
  author = {Eugene Turov},
  title = {ESSA: Easy Singular Spectrum Analysis},
  year = {2025},
  url = {https://github.com/ProtonEvgeny/essa}
}
