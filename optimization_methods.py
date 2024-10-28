import numpy as np
import numpy.linalg as lalg
from utils import line_search


def gradient_descent(f, grad_f, x0, delta1=1e-3, delta2=1e-3, max_iter=1000):
    x_k = x0
    func_values = [f(x_k)]

    for k in range(max_iter):
        p_k = -grad_f(x_k)  # Descent direction
        alpha_k = line_search(x_k, p_k, f, grad_f) 
        
        x_k_next = x_k + alpha_k * p_k
        func_values.append(f(x_k_next))

        if lalg.norm(x_k_next - x_k) < delta1 or abs(f(x_k_next) - f(x_k)) < delta2:
            break

        x_k = x_k_next

    return x_k, func_values, k


def newton_method(f, grad_f, hessian_f, x0, delta1=1e-3, delta2=1e-3, max_iter=1000):
    x_k = x0
    func_values = [f(x_k)]

    for k in range(max_iter):
        grad_k = grad_f(x_k)
        hessian_k = hessian_f(x_k)
        p_k = -np.linalg.solve(hessian_k, grad_k)
        alpha_k = line_search(x_k, p_k, f, grad_f) 
        
        x_k_next = x_k + alpha_k * p_k
        func_values.append(f(x_k_next))

        if lalg.norm(x_k_next - x_k) < delta1 or abs(f(x_k_next) - f(x_k)) < delta2:
            break

        x_k = x_k_next

    return x_k, func_values, k


def quasi_newton_bfgs(f, grad_f, x0, delta1=1e-3, delta2=1e-3, max_iter=1000):
    x_k = x0
    func_values = [f(x_k)]
    B_k = np.eye(len(x0))  
    for k in range(max_iter):
        grad_k = grad_f(x_k)
        p_k = -np.linalg.solve(B_k, grad_k)
        alpha_k = line_search(x_k, p_k, f, grad_f) 
        
        x_k_next = x_k + alpha_k * p_k
        func_values.append(f(x_k_next))

        if lalg.norm(x_k_next - x_k) < delta1 or abs(f(x_k_next) - f(x_k)) < delta2:
            break

        s_k = (x_k_next - x_k).reshape(-1, 1) 
        y_k = (grad_f(x_k_next) - grad_k).reshape(-1, 1) 
        sy_k = float(y_k.T @ s_k) 
        if sy_k > 1e-10:
            B_k = B_k + (y_k @ y_k.T) / sy_k - (B_k @ s_k @ s_k.T @ B_k) / float(s_k.T @ B_k @ s_k)

        x_k = x_k_next

    return x_k, func_values, k



