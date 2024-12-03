"""
存放工具类函数
"""

import numpy as np


def check_1d_array_length(var, var_name, expected_length):
    """Check length of 1D array.

    Parameters
    ----------
    var : array-like
        1D array to be checked.

    var_name : str
        Name of the variable. To be used in the error message.

    expected_length : int
        Expected length of the array.

    Raises
    ------
    ValueError
        If the length of the configured meta parameter is not correct.
    """
    actual_length = len(var)
    if actual_length != expected_length:
        s = "s" if expected_length > 1 else ""
        raise ValueError(
            f"Expected {var_name} with {expected_length} element{s}, "
            f"got {actual_length}.")


def ensure_1d_array(value, n_dims, var_name):
    """Process scalar or array-like input to ensure it is a 1D numpy array.  # 处理标量或类似数组的输入，确保其为 1D numpy 数组。

    Parameters
    ----------
    value : float or array-like, shape (n_dims,)
        Argument to be processed.  # 要处理的参数。

    n_dims : int
        Expected length of the 1d array.

    var_name : str
        Name of the variable in case an exception has to be raised.  # 发生异常时的变量名。

    Returns
    -------
    value : array, shape (n_dims,)
        1D numpy array with dtype float.  # dtype 为 float 的一维 numpy 数组。

    Raises
    ------
    ValueError
        If the argument is not compatible.  # 如果参数不兼容。
    """
    value = np.atleast_1d(value).astype(float)
    if value.ndim == 1 and value.shape[0] == 1:
        value = np.repeat(value, n_dims)
    if value.ndim > 1 or value.shape[0] != n_dims:
        raise ValueError(
            f"{var_name} has incorrect shape, expected ({n_dims},) "
            f"got {value.shape}")
    return value
