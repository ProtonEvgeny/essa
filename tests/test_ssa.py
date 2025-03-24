import numpy as np
import pytest
from essa import Decompose, reconstruct

def test_basic_decompose_initialization():
    # Generate synthetic data
    t = np.linspace(0, 2*np.pi, 100)
    series = np.sin(t) + 0.5*np.sin(3*t)
    
    # Test valid initialization with default parameters
    decomposer = Decompose(time_series=series, window_size=10)
    assert decomposer.window_size == 10
    assert decomposer.svd_method == "full"
    
    # Test with randomized SVD
    decomposer = Decompose(time_series=series, window_size=10, svd_method="randomized")
    assert decomposer.window_size == 10
    assert decomposer.svd_method == "randomized"
    
    # Test invalid SVD method
    with pytest.raises(ValueError):
        Decompose(time_series=series, window_size=10, svd_method="invalid")

def test_toeplitz_decompose_initialization():
    # Generate synthetic data
    t = np.linspace(0, 2*np.pi, 100)
    series = np.sin(t) + 0.5*np.sin(3*t)
    
    # Test valid initialization
    decomposer = Decompose(time_series=series, window_size=10, method="toeplitz")
    assert decomposer.window_size == 10
    
    # Test that SVD method raises error for Toeplitz
    with pytest.raises(ValueError):
        Decompose(time_series=series, window_size=10, method="toeplitz", svd_method="full")
    
    # Test invalid method
    with pytest.raises(ValueError):
        Decompose(time_series=series, window_size=10, method="invalid")

def test_basic_decomposition_reconstruction():
    # Generate synthetic data
    t = np.linspace(0, 2*np.pi, 100)
    series = np.sin(t) + 0.5*np.sin(3*t)
    
    # Test basic decomposition
    decomposer = Decompose(time_series=series, window_size=20)
    decomposer.fit()
    
    # Check components attribute exists
    assert hasattr(decomposer, "components")
    
    # Check number of components
    assert len(decomposer.components) == np.linalg.matrix_rank(decomposer.trajectory_matrix)
    
    # Test reconstruction with single component
    reconstructed = reconstruct(decomposer, [[0]])
    assert reconstructed.shape[0] == 1  # One group
    assert reconstructed[0].shape[0] == len(series)
    
    # Test reconstruction with multiple components
    reconstructed = reconstruct(decomposer, [[0, 1]])
    assert reconstructed.shape[0] == 1  # One group
    assert reconstructed[0].shape[0] == len(series)
    
    # Test reconstruction with multiple groups
    reconstructed = reconstruct(decomposer, [[0], [1, 2]])
    assert reconstructed.shape[0] == 2  # Two groups
    assert reconstructed[0].shape[0] == len(series)
    assert reconstructed[1].shape[0] == len(series)

def test_toeplitz_decomposition_reconstruction():
    # Generate synthetic data
    t = np.linspace(0, 2*np.pi, 100)
    series = np.sin(t) + 0.5*np.sin(3*t)
    
    # Test Toeplitz decomposition
    decomposer = Decompose(time_series=series, window_size=20, method="toeplitz")
    decomposer.fit()
    
    # Check components attribute exists
    assert hasattr(decomposer, "components")
    
    # Check number of components equals window_size
    assert len(decomposer.components) == decomposer.window_size
    
    # Test reconstruction with single component
    reconstructed = reconstruct(decomposer, [[0]])
    assert reconstructed.shape[0] == 1  # One group
    assert reconstructed[0].shape[0] == len(series)
    
    # Test reconstruction with multiple components
    reconstructed = reconstruct(decomposer, [[0, 1]])
    assert reconstructed.shape[0] == 1  # One group
    assert reconstructed[0].shape[0] == len(series)
    
    # Test reconstruction with multiple groups
    reconstructed = reconstruct(decomposer, [[0], [1, 2]])
    assert reconstructed.shape[0] == 2  # Two groups
    assert reconstructed[0].shape[0] == len(series)
    assert reconstructed[1].shape[0] == len(series)

def test_reconstruct_validation():
    # Generate synthetic data
    t = np.linspace(0, 2*np.pi, 100)
    series = np.sin(t) + 0.5*np.sin(3*t)
    
    # Create decomposer without fitting
    decomposer = Decompose(time_series=series, window_size=20)
    
    # Test that reconstruct raises ValueError if decompose not called
    with pytest.raises(ValueError):
        reconstruct(decomposer, [[0]])
    
    # Now fit and ensure it works
    decomposer.fit()
    reconstructed = reconstruct(decomposer, [[0]])
    assert reconstructed.shape[0] == 1
