import numpy as np

def load_data(filename, shape):
    with open(filename, 'r') as file:
        return np.fromfile(file, sep=' ').reshape(shape)


def line_search(x_k, p_k, f, grad, alpha_k=1.0, c_=0.1, decrement_ratio=0.5, max_iters=1000):
    while f(x_k + alpha_k * p_k) > f(x_k) + c_ * alpha_k * np.dot(grad(x_k), p_k):
        alpha_k *= decrement_ratio
    return alpha_k

