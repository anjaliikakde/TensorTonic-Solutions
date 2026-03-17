import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    w = np.array(w, dtype=float)
    g = np.array(g, dtype=float)
    s = np.array(s, dtype=float)

    # Step 1: Update running avg of squared gradients
    s = beta * s + (1 - beta) * (g ** 2)

    # Step 2: Update weights
    w = w - (lr * g) / (np.sqrt(s) + eps)

    return w.tolist(), s.tolist()