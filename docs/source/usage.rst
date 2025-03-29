Usage
=====

Basic Example
------------

Here's a basic example of how to use ESSA for time series decomposition:

.. code-block:: python

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
    plt.show()

Advanced Usage
-------------

Non-stationary Series Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a time series with a trend component (non-stationary):

.. code-block:: python

    # Non-stationary series with trend
    t = np.linspace(0, 10, 500)
    trend = 0.5 * t
    seasonal = 2 * np.sin(t) + 1.5 * np.sin(2 * t)
    noise = 0.5 * np.random.randn(len(t))
    non_stationary_series = trend + seasonal + noise

    # Basic SSA method
    decomposer = Decompose(time_series=non_stationary_series, window_size=60)
    decomposer.fit()

    # Reconstruct components
    # For trend, we typically use the first few components
    trend_component = reconstruct(decomposer, [[0, 1]])[0]
    # For seasonality, we use paired components
    seasonal_component = reconstruct(decomposer, [[2, 3, 4, 5]])[0]
    # Noise components are typically the higher frequency components
    noise_component = reconstruct(decomposer, [[i for i in range(6, 20)]])[0]

Stationary Series Analysis with Toeplitz Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a stationary time series, the Toeplitz method can provide better results:

.. code-block:: python

    # Stationary series (no trend)
    stationary_seasonal = 3 * np.sin(t) + np.sin(2 * t + 0.5) + 0.8 * np.sin(3 * t)
    stationary_noise = 0.7 * np.random.randn(len(t))
    stationary_series = stationary_seasonal + stationary_noise

    # Toeplitz SSA method
    decomposer = Decompose(time_series=stationary_series, window_size=60, method="toeplitz")
    decomposer.fit()

    # Reconstruct components
    trend = reconstruct(decomposer, [[0]])[0]
    seasonal = reconstruct(decomposer, [[1, 2, 3, 4, 5]])[0]
    noise = reconstruct(decomposer, [[i for i in range(6, 20)]])[0]
