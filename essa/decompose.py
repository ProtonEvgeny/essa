from .basic_decompose import BasicDecompose
from .toeplitz_decompose import ToeplitzDecompose
from typing import Union
import numpy as np

class Decompose:
    """
    Class for performing SSA decomposition on a given time series.

    Parameters
    ----------
    time_series : np.ndarray
        The time series data.
    window_size : int
        The size of the window.
    method : str, optional
        The method for performing SSA decomposition. Options are 'basic' or
        'toeplitz'. Default is 'basic'.
    svd_method : str, optional
        The method for performing SVD on the trajectory matrix for the basic
        method. Options are 'full' or 'randomized'. Default is None, which
        results in full SVD.

    Returns
    -------
    Union[BasicDecompose, ToeplitzDecompose]
        The class object corresponding to the chosen method.
    """
    def __new__(
        cls,
        time_series: np.ndarray,
        window_size: int,
        method: str = "basic",
        svd_method: str = None
    ) -> Union[BasicDecompose, ToeplitzDecompose]:
        if method not in {"basic", "toeplitz"}:
            raise ValueError(f"Invalid method: {method}")

        if method == "toeplitz":
            if svd_method is not None:
                raise ValueError("SVD method is not supported for Toeplitz SSA")
            return ToeplitzDecompose(time_series, window_size)
        
        if svd_method not in {"full", "randomized", None}:
            raise ValueError("SVD method must be 'full' or 'randomized' for Basic SSA")
        
        return BasicDecompose(time_series, window_size, svd_method)