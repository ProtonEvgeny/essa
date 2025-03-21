import numpy as np
import pytest
from essa import SSA

def test_ssa_initialization():
    # Test valid initialization
    ssa = SSA(window_size=10)
    assert ssa.window_size == 10
    assert ssa.svd_method == "full"
    assert ssa.n_components == 10
    
    # Test with randomized SVD
    ssa = SSA(window_size=10, svd_method="randomized", n_components=5)
    assert ssa.window_size == 10
    assert ssa.svd_method == "randomized"
    assert ssa.n_components == 5
    
    # Test invalid SVD method
    with pytest.raises(ValueError):
        SSA(window_size=10, svd_method="invalid")
    
    # Test invalid n_components
    with pytest.raises(ValueError):
        SSA(window_size=10, n_components=11)

def test_decomposition_reconstruction():
    # Generate synthetic data
    t = np.linspace(0, 2*np.pi, 100)
    series = np.sin(t) + 0.5*np.sin(3*t)
    
    # Test decomposition
    ssa = SSA(window_size=20)
    components = ssa.decompose(series)
    
    # Check number of components
    assert len(components) == 20
    
    # Test reconstruction with single component
    reconstructed = ssa.reconstruct([0])
    assert reconstructed.shape == series.shape
    
    # Test reconstruction with multiple components
    reconstructed = ssa.reconstruct([0, 1])
    assert reconstructed.shape == series.shape
    
    # Test reconstruction with grouped components
    reconstructed = ssa.reconstruct([[0], [1, 2]])
    assert reconstructed.shape == series.shape

def test_ssa_method():
    # Generate synthetic data
    t = np.linspace(0, 2*np.pi, 100)
    series = np.sin(t) + 0.5*np.sin(3*t)
    
    # Test ssa method
    ssa = SSA(window_size=20)
    reconstructed = ssa.ssa(series, [[0], [1, 2]])
    
    # Check shape
    assert reconstructed.shape == series.shape
