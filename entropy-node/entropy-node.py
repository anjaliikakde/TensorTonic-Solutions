import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.array(y)

    values, counts = np.unique(y, return_counts=True)

    p = counts /counts.sum()

    p =p[p>0]

    entropy = -np.sum(p* np.log2(p))

    return float(entropy)