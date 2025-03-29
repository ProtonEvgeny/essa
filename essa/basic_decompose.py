import numpy as np
from typing import Tuple, List
from sklearn.utils.extmath import randomized_svd
from scipy.linalg import svd as full_svd

class BasicDecompose:
    """
    BasicDecompose performs Basic SSA decomposition on a given time series.

    SSA is a non-parametric spectral estimation technique used for time series
    decomposition, forecasting, and noise reduction. This class provides basic
    functionality to decompose a time series into its principal components using
    Singular Value Decomposition (SVD).

    Attributes
    ----------
    time_series : np.ndarray
        The time series data to be analyzed.
    window_size : int
        The size of the window for trajectory matrix embedding.
    svd_method : str
        The method for Singular Value Decomposition ('full' or 'randomized').
    ts_size : int
        The size of the time series.
    trajectory_matrix : np.ndarray
        The constructed trajectory matrix from the time series.
    U : np.ndarray
        Left singular vectors from SVD.
    sigma : np.ndarray
        Singular values from SVD.
    V : np.ndarray
        Right singular vectors from SVD.
    d : int
        Rank of the trajectory matrix.
    components : List[np.ndarray]
        List of elementary matrices derived from the SVD components.

    Methods
    -------
    fit() -> None
        Fits the SSA decomposition to the data.
    """

    def __init__(self, time_series: np.ndarray, window_size: int, svd_method: str = "full") -> None:
        """
        Initialize the BasicDecompose class with a time series, window size, and SVD method.

        Parameters
        ----------
        time_series : np.ndarray
            The time series data to be analyzed.
        window_size : int
            The size of the window for trajectory matrix embedding.
        svd_method : str, optional
            The method for Singular Value Decomposition ('full' or 'randomized'), by default 'full'.
        """
        self.time_series = time_series
        self.svd_method = "full" if svd_method is None else svd_method
        self.ts_size = len(time_series)
        self.window_size = window_size

    def _trajectory_matrix(self) -> np.ndarray:
        """
        Build Hankel trajectory matrix from time series.

        Returns
        -------
        np.ndarray
            Trajectory matrix of size (window_size, ts_size - window_size + 1)
        """
        X = np.zeros((self.window_size, self.ts_size - self.window_size + 1))
        for i in range(self.ts_size - self.window_size + 1):
            X[:, i] = self.time_series[i : i + self.window_size]

        return X

    def _svd(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute singular value decomposition (SVD) of given matrix.

        Parameters
        ----------
        matrix : np.ndarray
            The input matrix to be decomposed

        Returns
        -------
        U : np.ndarray
            Left singular vectors
        s : np.ndarray
            Singular values
        Vt : np.ndarray
            Right singular vectors

        Raises
        ------
        ValueError
            If svd_method is not 'full' or 'randomized'
        """
        if self.svd_method == "full":
            U, s, Vt = full_svd(matrix)
        elif self.svd_method == "randomized":
            U, s, Vt = randomized_svd(
                matrix,
                n_components=self.window_size - self.window_size // 3,
                n_oversamples=100,
                random_state=0,
                power_iteration_normalizer="LU",
                n_iter=15,
            )
        else:
            raise ValueError("svd_method must be 'full' or 'randomized'")

        return U, s, Vt

    def _decompose_trajectory_matrix(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Decompose trajectory matrix using SVD.

        Parameters
        ----------
        None

        Returns
        -------
        U : np.ndarray
            Left singular vectors
        s : np.ndarray
            Singular values
        V : np.ndarray
            Right singular vectors
        d : int
            Rank of the trajectory matrix

        Notes
        -----
        The rank of the trajectory matrix is determined using the matrix rank function
        from numpy.linalg.
        """
        U, s, Vt = self._svd(self.trajectory_matrix)
        d = np.linalg.matrix_rank(self.trajectory_matrix) if self.svd_method == "full" else self.window_size // 2 - 1
        V = Vt.T

        return U, s, V, d

    def _elementary_matrix(self) -> List[np.ndarray]:
        """
        Construct elementary matrices from SVD components.

        Returns
        -------
        List[np.ndarray]
            A list of elementary matrices

        Notes
        -----
        This method assumes that the decomposition has already been computed and
        that the attributes `self.U`, `self.sigma`, `self.V`, and `self.d` are
        initialized and available.
        """

        return [self.sigma[i] * np.outer(self.U[:, i], self.V[:, i]) for i in range(self.d)]

    def fit(self) -> None:
        """
        Fit the SSA decomposition to the data.

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
        - `self.U`, `self.sigma`, `self.V`, and `self.d`: The SVD components of the trajectory matrix
        - `self.components`: The elementary matrices constructed from the SVD components
        """
        self.trajectory_matrix = self._trajectory_matrix()
        self.U, self.sigma, self.V, self.d = self._decompose_trajectory_matrix()
        self.components = self._elementary_matrix()
