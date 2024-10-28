import numpy as np
import numpy.linalg as lalg
from utils import line_search


def gradient_descent(f, grad_f, x0, delta1=1e-3, delta2=1e-3, max_iter=1000):
    x_k = x0
    func_values = [f(x_k)]

    for k in range(max_iter):
        p_k = -grad_f(x_k)
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


def adam_optimizer(f, x0, grad, max_iters=1000, initial_learning_rate=0.001, delta1=1e-4, delta2=1e-4):
    x_k = x0
    fx_ls = [f(x_k)]
    s = np.zeros_like(x_k)  # Initialize first moment vector
    r = np.zeros_like(x_k)  # Initialize second moment vector

    rho1 = 0.9  # Decay rate for first moment
    rho2 = 0.999  # Decay rate for second moment
    epsilon = 1e-8  # Small epsilon to avoid division by zero

    for k in range(max_iters):
        g = grad(x_k)

        s = rho1 * s + (1 - rho1) * g
        r = rho2 * r + (1 - rho2) * (g ** 2)

        s_hat = s / (1 - rho1 ** (k + 1))
        r_hat = r / (1 - rho2 ** (k + 1))

        learning_rate = initial_learning_rate * np.sqrt(1 - rho2 ** (k + 1)) / (1 - rho1 ** (k + 1))

        p_k = -learning_rate * s_hat / (np.sqrt(r_hat) + epsilon)
        x_k_next = x_k + p_k

        fx_next = f(x_k_next)
        fx_ls.append(fx_next)

        if np.linalg.norm(x_k_next - x_k) < delta1 or abs(fx_next - fx_ls[-2]) < delta2:
            break

        x_k = x_k_next

    return x_k, fx_ls, k + 1