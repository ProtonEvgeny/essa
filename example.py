from essa import SSA
import numpy as np
import matplotlib.pyplot as plt

# Example usage, synthetic data
t = np.linspace(0, 2*np.pi, 100)
series = np.sin(t) + 0.5*np.sin(3*t)

model = SSA(20)
components = model.decompose(series)

trend = model.reconstruct([[0]])
seasonal = model.reconstruct([[1, 2]])
noise = model.reconstruct([3])

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
