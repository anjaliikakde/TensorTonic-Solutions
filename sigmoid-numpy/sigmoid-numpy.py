import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    x= np.array(x)
    val = 1/ (1+np.exp(-x))
    return val