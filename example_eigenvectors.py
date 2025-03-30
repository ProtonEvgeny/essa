from essa import Decompose
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Generate noisy sinusoid time series
np.random.seed(1)
N = 100
sigma = 0.5  # Noise level
t = np.arange(1, N+1)
F = np.sin(2 * np.pi * t / 7) + sigma * np.random.normal(size=N)

# Window size for decomposition
window_size = 50

# --- Basic SSA ---
basic_decomposer = Decompose(time_series=F, window_size=window_size, method="basic")
basic_decomposer.fit()

# --- Toeplitz SSA ---
toeplitz_decomposer = Decompose(time_series=F, window_size=window_size, method="toeplitz")
toeplitz_decomposer.fit()

# Visualize original time series
plt.figure(figsize=(10, 4))
plt.plot(t, F)
plt.title('Noisy Sinusoid Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate percentage contribution for each eigenvector
basic_contrib = basic_decomposer.sigma**2 / np.sum(basic_decomposer.sigma**2) * 100
toeplitz_contrib = toeplitz_decomposer.sigma**2 / np.sum(toeplitz_decomposer.sigma**2) * 100

num_eigenvectors = 4

# Create figure for Toeplitz eigenvectors (top)
plt.figure(figsize=(12, 8))
plt.suptitle('Eigenvectors', fontsize=16)

# Top plot - Toeplitz method
gs_top = GridSpec(2, 4, height_ratios=[1, 1], hspace=0.4, wspace=0.1)
for i in range(num_eigenvectors):
    ax = plt.subplot(gs_top[0, i])
    ax.plot(toeplitz_decomposer.U[:, i], 'b-')
    ax.set_title(f'{i+1} ({toeplitz_contrib[i]:.2f}%)', fontsize=12)
    ax.grid(True)
    ax.set_xticks([])
    if i > 0:
        ax.set_yticks([])

# Bottom plot - Basic method
for i in range(num_eigenvectors):
    ax = plt.subplot(gs_top[1, i])
    ax.plot(basic_decomposer.U[:, i], 'b-')
    ax.set_title(f'{i+1} ({basic_contrib[i]:.2f}%)', fontsize=12)
    ax.grid(True)
    ax.set_xticks([])
    if i > 0:
        ax.set_yticks([])

plt.figtext(0.5, 0.01, 'Noisy sinusoid: 1D graphs of eigenvectors (top: Toeplitz method, bottom: Basic method)', 
           ha='center', fontsize=12)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()
