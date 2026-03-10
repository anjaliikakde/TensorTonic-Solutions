import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X)
    y = np.array(y)
    
    n, D = X.shape        # n = num samples, D = num features
    w = np.zeros(D)       # initialize weights as zeros
    b = 0.0               # initialize bias as zero

    for _ in range(steps):
        # Step 1: Predict
        z = X @ w + b           # @ means matrix multiply
        y_hat = _sigmoid(z)

        # Step 2: Gradients
        error = y_hat - y
        dw = (1/n) * X.T @ error
        db = (1/n) * np.sum(error)

        # Step 3: Update
        w = w - lr * dw
        b = b - lr * db

    return (w, b)