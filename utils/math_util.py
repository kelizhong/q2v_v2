import numpy as np


def cos_distance(v1, v2):
    """ calculate cosine and returns cosine """
    n1 = get_norm_of_vector(v1)
    n2 = get_norm_of_vector(v2)
    ip = get_inner_product(v1, v2)
    return ip / (n1 * n2)


def get_inner_product(v1, v2):
    """ calculate inner product """
    return np.dot(v1, v2)


def get_norm_of_vector(v):
    """ calculate norm of vector """
    return np.linalg.norm(v)
