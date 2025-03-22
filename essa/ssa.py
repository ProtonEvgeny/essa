import numpy as np
from scipy.linalg import svd as full_svd
from sklearn.utils.extmath import randomized_svd
from typing import List, Union, Tuple

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
        self, window_size: int, ssa_method: str = "basic", svd_method: str = "full", n_components: int = None
    ):
        self.window_size = window_size
        self.ssa_method = ssa_method
        self.svd_method = svd_method
        self.n_components = n_components or window_size

        if ssa_method not in ["basic", "toeplitz"]:
            raise ValueError("ssa_method must be 'basic' or 'toeplitz'")

        if svd_method not in ["full", "randomized"]:
            raise ValueError("svd_method must be 'full' or 'randomized'")

        if self.n_components > self.window_size:
            raise ValueError("n_components cannot exceed window_size")

    def _trajectory_matrix(self, series: np.ndarray) -> np.ndarray:
        """
        Build Hankel trajectory matrix from time series.
        """
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

    def _svd(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.svd_method == "full":
            U, s, Vt = full_svd(matrix, full_matrices=False)
        elif self.svd_method == "randomized":
            U, s, Vt = randomized_svd(matrix, n_components=self.n_components)
        else:
            raise ValueError("svd_method must be 'full' or 'randomized'")
        return U, s, Vt

    def _toeplitz_covariance(self, series: np.ndarray) -> np.ndarray:
        L = self.window_size
        N = len(series)
        centered_series = series - np.mean(series)
        C_tilde = np.zeros((L, L))
        for k in range(L):
            if k >= N: 
                continue
            val = np.sum(centered_series[:N - k] * centered_series[k:]) / (N - k)
            for i in range(L - k):
                C_tilde[i, i + k] = val
                C_tilde[i + k, i] = val
        return C_tilde

    def decompose(self, series: np.ndarray) -> List[np.ndarray]:
        """
        Decompose the time series into components.
        """
        X = self._trajectory_matrix(series)
        if self.ssa_method == 'basic':
            U, s, Vt = self._svd(X)
            self.components_ = [s[i] * np.outer(U[:, i], Vt[i, :]) for i in range(len(s))]
        elif self.ssa_method == "toeplitz":
            C_tilde = self._toeplitz_covariance(series)
            eig_vals, eig_vecs = np.linalg.eigh(C_tilde)
            sigma = [
                np.linalg.norm(X.T @ eig_vecs[:, i]) for i in range(eig_vecs.shape[1])
            ]
            order = np.argsort(sigma)[::-1]
            self.components_ = []
            for idx in order:
                P = eig_vecs[:, idx]
                Q = (X.T @ P) / sigma[idx]
                component = sigma[idx] * np.outer(P, Q)
                self.components_.append(component)
        else:
            raise ValueError("ssa_method must be 'basic' or 'toeplitz'")
        return self.components_

    def reconstruct(self, groups: Union[List[int], List[List[int]]]) -> np.ndarray:
        if not hasattr(self, "components_"):
            raise ValueError("decompose must be called before reconstruct")
        grouped_matrix = sum(
            sum(self.components_[i] for i in group) if isinstance(group, list) else self.components_[group]
            for group in groups
        )
        return self._diagonal_averaging(grouped_matrix)

    def ssa(self, series: np.ndarray, groups: List[List[int]]) -> np.ndarray:
        self.decompose(series)
        return self.reconstruct(groups)
