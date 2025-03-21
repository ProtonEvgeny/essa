# ESSA - Easy Singular Spectrum Analysis

A Python package for Singular Spectrum Analysis (SSA) of time series data.

## Installation

```bash
pip install essa
```

## Features

- Fast implementation of SSA using vectorized operations
- Support for both full SVD and randomized SVD for large datasets
- Simple API for decomposition and reconstruction
- Compatible with NumPy arrays

## Usage Example

```python
from essa import SSA
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
t = np.linspace(0, 2*np.pi, 100)
series = np.sin(t) + 0.5*np.sin(3*t)

# Create SSA model with window size of 20
model = SSA(20)

# Decompose the time series
components = model.decompose(series)

# Reconstruct components
trend = model.reconstruct([[0]])
seasonal = model.reconstruct([[1, 2]])
noise = model.reconstruct([3])

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

## API Reference

### SSA Class

```python
SSA(window_size, svd_method='full', n_components=None)
```

**Parameters:**

- `window_size` (int): The embedding window length (L)
- `svd_method` (str): 'full' for exact SVD or 'randomized' for approximate (default: 'full')
- `n_components` (int): Number of components for randomized SVD (default: None)

**Methods:**

- `decompose(series)`: Decompose time series into components
- `reconstruct(groups)`: Reconstruct time series from selected components
- `ssa(series, groups)`: Perform decomposition and reconstruction in one step

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
```
