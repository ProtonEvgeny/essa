from essa import Decompose, reconstruct
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic time series
np.random.seed(42)
t = np.linspace(0, 10, 500)

# 1. Non-stationary series
trend = 0.5 * t
seasonal = 2 * np.sin(t) + 1.5 * np.sin(2 * t)
noise = 0.5 * np.random.randn(len(t))
non_stationary_series = trend + seasonal + noise

# 2. Stationary series
stationary_seasonal = 3 * np.sin(t) + np.sin(2 * t + 0.5) + 0.8 * np.sin(3 * t)
stationary_noise = 0.7 * np.random.randn(len(t))
stationary_series = stationary_seasonal + stationary_noise

# --- Basic SSA ---
non_stationary_decomposer = Decompose(time_series=non_stationary_series, window_size=60, svd_method="randomized")
non_stationary_decomposer.fit()

# Reconstruct decomposed series
ns_trend = reconstruct(non_stationary_decomposer, [[0, 1]])[0]
ns_seasonal = reconstruct(non_stationary_decomposer, [[2, 3, 4, 5]])[0]
ns_noise = reconstruct(non_stationary_decomposer, [[i for i in range(6, 20)]])[0]

# Visualize
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(t, non_stationary_series)
plt.title('Non-stationary time series')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t, ns_trend)
plt.title('Trend component')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t, ns_seasonal)
plt.title('Seasonal component')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(t, ns_noise)
plt.title('Noise component')
plt.grid(True)

plt.tight_layout()
plt.show()

# Visualize eigenvectors
plt.figure(figsize=(12, 8))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.plot(non_stationary_decomposer.U[:, i])
    plt.title(f'Eigenvector {i+1}')
    plt.grid(True)

plt.tight_layout()
plt.show()

# --- Toeplitz SSA ---
stationary_decomposer = Decompose(time_series=stationary_series, window_size=60, method="toeplitz")
stationary_decomposer.fit()

# Reconstruct decomposed series
s_trend = reconstruct(stationary_decomposer, [[0]])[0]
s_seasonal = reconstruct(stationary_decomposer, [[1, 2, 3, 4, 5]])[0]
s_noise = reconstruct(stationary_decomposer, [[i for i in range(6, 20)]])[0]

# Visualize
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(t, stationary_series)
plt.title('Stationary time series')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t, s_trend)
plt.title('Trend component')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t, s_seasonal)
plt.title('Seasonal component')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(t, s_noise)
plt.title('Noise component')
plt.grid(True)

plt.tight_layout()
plt.show()

# Visualize eigenvectors
plt.figure(figsize=(12, 8))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.plot(stationary_decomposer.U[:, i])
    plt.title(f"Eigenvector {i + 1}")
    plt.grid(True)

plt.tight_layout()
plt.show()
