def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    a = set(set_a)
    b = set(set_b)

    intersection = a & b
    union = a | b

    if(len(union)==0):
        return 0.0
    return (len(intersection)/len(union))