import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    w = np.array(w, dtype=float)
    m = np.array(m, dtype=float)
    v = np.array(v, dtype=float)
    grad = np.array(grad, dtype=float)

    # Step 1
    m = beta1 * m + (1 - beta1) * grad
    
    # Step 2
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    
    # Step 3
    w = w - lr * (beta1 * m + (1 - beta1) * grad) / (np.sqrt(v) + eps)
    
    return w.tolist(), m.tolist(), v.tolist()