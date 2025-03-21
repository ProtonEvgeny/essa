import numpy as np
from scipy.linalg import svd as full_svd
from sklearn.utils.extmath import randomized_svd
from typing import List, Union

class SSA:
    """
    Singular Spectrum Analysis (SSA) for time series implementation.

    Parameters:
    - window_size (int): The embedding window length (L)
    - svd_method (str): 'full' for exact SVD or 'randomized' for approximate (default: 'full')
    - n_components (int): Number of components for randomized SVD (default: None)

    Example:
    >>> model = SSA(window_size=10)
    >>> components = model.decompose(series)
    >>> reconstructed = model.reconstruct(components)
    """
    def __init__(
        self, window_size: int, svd_method: str = "full", n_components: int = None
    ):
        self.window_size = window_size
        self.svd_method = svd_method
        self.n_components = n_components or window_size

        if svd_method not in ["full", "randomized"]:
            raise ValueError("svd_method must be 'full' or 'randomized'")

        if self.n_components > self.window_size:
            raise ValueError("n_components cannot exceed window_size")

    def _trajectory_matrix(self, series: np.ndarray) -> np.ndarray:
        """
        Build Hankel trajectory matrix from time series.
        """
        # n = len(series)
        # k = n - self.window_size + 1
        return np.lib.stride_tricks.sliding_window_view(series, self.window_size).T

    def _diagonal_averaging(self, matrix: np.ndarray) -> np.ndarray:
        """
        Vectorized diagonal averaging implementation.
        """
        m, n = matrix.shape
        reconstructed = np.zeros(m + n - 1)
        for k in range(-m + 1, n):
            diagonal = np.diagonal(matrix, offset=k)
            reconstructed[k + m - 1] = diagonal.mean()
        return reconstructed

    def decompose(self, series: np.ndarray) -> List[np.ndarray]:
        X = self._trajectory_matrix(series)
        if self.svd_method == "full":
            U, s, Vt = full_svd(X, full_matrices=False)
        else:
            U, s, Vt = randomized_svd(X, n_components=self.n_components)
        self.components_ = [s[i] * np.outer(U[:, i], Vt[i, :]) for i in range(len(s))]
        return self.components_

    def reconstruct(self, groups: Union[List[int], List[List[int]]]) -> np.ndarray:
        if not hasattr(self, "components_"):
            raise ValueError("decompose must be called before reconstruct")
        if all(isinstance(g, list) for g in groups):
            grouped_matrix = sum(sum(self.components_[i] for i in group) for group in groups)
        else:
            grouped_matrix = sum(self.components_[i] for i in groups)
        return self._diagonal_averaging(grouped_matrix)

    def ssa(self, series: np.ndarray, groups: List[List[int]]) -> np.ndarray:
        self.decompose(series)
        return self.reconstruct(groups)
