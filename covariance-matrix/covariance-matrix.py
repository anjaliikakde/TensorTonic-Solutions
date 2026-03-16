import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    if X is None or len(X)==0:
        return None
    X = np.array(X)
    
    if X.ndim != 2 or X.shape[0] <= 1:
        return None

    mu = np.mean(X, axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc)/(X.shape[0]-1)
   
    return cov