from .basic_decompose import BasicDecompose
import numpy as np
from typing import List, Tuple

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
    U : np.ndarray
        Left singular vectors.
    sigma : np.ndarray
        Singular values.
    V : np.ndarray
        Right singular vectors.
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

    def _decompose_toeplitz_matrix(self, trajectory_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Decompose the trajectory matrix using the Toeplitz covariance matrix.

        Parameters
        ----------
        trajectory_matrix : np.ndarray
            The trajectory matrix

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]
            A tuple containing the sorted left singular vectors, sorted singular values,
            sorted right singular vectors, and a list of elementary matrices

        Notes
        -----
        The eigenvalues of the covariance matrix of the time series are used to
        compute the singular values of the trajectory matrix. The singular vectors
        are computed by projecting the columns of the trajectory matrix onto the
        eigenvectors of the covariance matrix. The elementary matrices are computed
        by taking the outer product of the left singular vectors with the right
        singular vectors.
        """
        X = trajectory_matrix
        C_tilde = self._toeplitz_matrix()
        eigen_vals, eigen_vecs = np.linalg.eigh(C_tilde)

        # Calculate the norm of the projection of X onto each eigenvector
        sigma = [np.linalg.norm(X.T @ eigen_vecs[:, i]) for i in range(self.window_size)]
        order = np.argsort(sigma)[::-1] # sort in descending order
        U_sorted = eigen_vecs[:, order]
        sigma_sorted = np.array(sigma)[order]

        V_columns = []
        elementary_matrices: List[np.ndarray] = []
        for idx in order:
            P = eigen_vecs[:, idx]
            proj = X.T @ P
            sigma_i = sigma[idx]
            V_i = proj / sigma_i # Scale the projection by the corresponding singular value
            V_columns.append(V_i)
            elementary_matrix = np.outer(P, proj)
            elementary_matrices.append(elementary_matrix)

        V_sorted = np.column_stack(V_columns)

        return U_sorted, sigma_sorted, V_sorted, elementary_matrices

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
        - `self.U`, `self.sigma`, `self.V`: The singular vectors and singular values
        - `self.components`: The elementary matrices constructed from the
          Toeplitz covariance matrix
        """
        self.trajectory_matrix = self._trajectory_matrix()
        self.U, self.sigma, self.V, self.components = self._decompose_toeplitz_matrix(self.trajectory_matrix)
        self.d = len(self.sigma)
