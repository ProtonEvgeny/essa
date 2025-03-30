from .basic_decompose import BasicDecompose
import numpy as np
from typing import List

class ToeplitzDecompose(BasicDecompose):
    """
    ToeplitzDecompose performs SSA decomposition using a Toeplitz covariance matrix.

    This class extends the BasicDecompose class to perform Singular Spectrum Analysis (SSA)
    on a given time series using a Toeplitz covariance matrix.

    Attributes
    ----------
    time_series : np.ndarray
        The original time series data.
    window_size : int
        The size of the embedding window.
    time_series_centered : np.ndarray
        The centered version of the time series.
    ts_size : int
        The size of the time series.
    trajectory_matrix : np.ndarray
        The constructed trajectory matrix from the time series.
    d : int
        The rank of the trajectory matrix
    components : List[np.ndarray]
        List of elementary matrices constructed from the Toeplitz covariance matrix

    Methods
    -------
    fit() -> None
        Fits the Toeplitz SSA decomposition to the data.
    """

    def __init__(self, time_series: np.ndarray, window_size: int) -> None:
        """
        Initialize the ToeplitzDecompose class with a time series and window size.

        Parameters
        ----------
        time_series : np.ndarray
            The time series data to be analyzed.
        window_size : int
            The size of the window for trajectory matrix embedding.

        Returns
        -------
        None
        """
        super().__init__(time_series, window_size)
        self.time_series_centered = self.time_series - np.mean(self.time_series)
    
    def _toeplitz_matrix(self) -> np.ndarray:
        """
        Compute the Toeplitz matrix for the centered time series.

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray
            The Toeplitz matrix
        """
        L = self.window_size
        N = self.ts_size
        centered_series = self.time_series_centered
        covs = np.correlate(centered_series, centered_series, mode='full')[N - 1:]
        covs[: L] /= np.arange(N, N - L, -1)
        covs[L:] /= np.arange(N - L, 0, -1)
        return np.fromfunction(lambda i, j: covs[np.abs(i - j)], (L, L), dtype=int)

    def _decompose_toeplitz_matrix(self, trajectory_matrix: np.ndarray) -> List[np.ndarray]:
        """
        Decompose the trajectory matrix using the Toeplitz covariance matrix.

        Parameters
        ----------
        trajectory_matrix : np.ndarray
            The trajectory matrix

        Returns
        -------
        List[np.ndarray]
            A list of elementary matrices

        Notes
        -----
        This method uses the eigenvectors of the Toeplitz covariance matrix to
        decompose the trajectory matrix into its elementary matrices. The
        elementary matrices are then ordered in descending order of their
        importance, which is determined by their norm.
        """
        X = trajectory_matrix
        C_tilde = self._toeplitz_matrix()
        eigen_vals, eigen_vecs = np.linalg.eigh(C_tilde)
        order = np.argsort(eigen_vals)[::-1]
        elementary_matrices = []
        for idx in order:
            eigen_val = eigen_vals[idx]
            eigen_vec = eigen_vecs[:, idx]
            elementary_matrix = eigen_val * np.outer(eigen_vec, X.T @ eigen_vec)
            elementary_matrices.append(elementary_matrix)

        return elementary_matrices

    def fit(self) -> None:
        """
        Fit the Toeplitz SSA decomposition to the data.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This method sets the following attributes:

        - `self.trajectory_matrix`: The trajectory matrix of the time series
        - `self.d`: The rank of the trajectory matrix
        - `self.components`: The elementary matrices constructed from the
          Toeplitz covariance matrix
        """
        self.trajectory_matrix = self._trajectory_matrix()
        self.d = np.linalg.matrix_rank(self.trajectory_matrix)
        self.components = self._decompose_toeplitz_matrix(self.trajectory_matrix)
