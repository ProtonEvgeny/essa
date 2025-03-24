from .basic_decompose import BasicDecompose
from .toeplitz_decompose import ToeplitzDecompose
from typing import Union, List
import numpy as np

def reconstruct(decompose: Union[BasicDecompose, ToeplitzDecompose], groups: Union[List[int], List[List[int]]]) -> np.ndarray:
    """
    Reconstruct the data given the SSA decomposition and the desired grouping of the elementary components.

    Parameters
    ----------
    decompose : Union[BasicDecompose, ToeplitzDecompose]
        The decomposition object containing the elementary matrices.
    groups : Union[List[int], List[List[int]]]
        The groups of elementary matrices to be used for reconstruction.

    Returns
    -------
    np.ndarray
        The reconstructed time series.
    """
    
    def diagonal_averaging(matrix: np.ndarray) -> np.ndarray:
        m, n = matrix.shape
        reconstructed = np.zeros(m + n - 1)
        for k in range(-m + 1, n):
            diagonal = np.diagonal(matrix, offset=k)
            reconstructed[k + m - 1] = diagonal.mean()
        return reconstructed

    if not hasattr(decompose, "components"):
        raise ValueError("decompose time series before reconstruct")

    components = []
    for group in groups:
        X_group = np.sum([decompose.components[i] for i in group], axis=0)
        component = diagonal_averaging(X_group)
        components.append(component)
    return np.array(components)
