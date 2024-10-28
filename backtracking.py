import numpy as np

def line_search(x_k, p_k, f, grad, alpha_k=1.0, c_=0.1, decrement_ratio=0.5, max_iters=1000):
    while f(x_k + alpha_k * p_k) > f(x_k) + c_ * alpha_k * np.dot(grad(x_k), p_k):
        alpha_k *= decrement_ratio
    return alpha_k


# def line_search(x_k, p_k, f, grad, alpha_k=1, c_=0.1, decrement_ratio=0.5, max_iters=100):
#     alpha_k = alpha_k

#     slope = c_ * p_k.T @ grad(x_k)

#     y_intercept = f(x_k)
#     y_line = lambda alpha_k: slope * alpha_k + y_intercept

#     for _ in range(max_iters):
#         x_k1 = x_k + alpha_k * p_k
#         if f(x_k1) < y_line(alpha_k):
#             break
#         else:
#             alpha_k *= decrement_ratio
#     return alpha_k


















# def backtracking_line_search(x_k, p_k, f, grad, alpha_k=1.0, decrement_ratio=0.5, c=0.1):
#     while f(x_k) + c * alpha_k * np.dot(grad, p_k) > f(x_k + alpha_k * p_k):
#         alpha_k *= decrement_ratio
#     return alpha_k
