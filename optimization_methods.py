import numpy as np
import numpy.linalg as lalg
from utils import line_search
# from backtracking import back_track_line


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

        # BFGS Update
        s_k = (x_k_next - x_k).reshape(-1, 1)  # Ensure s_k is a column vector
        y_k = (grad_f(x_k_next) - grad_k).reshape(-1, 1)  # Ensure y_k is a column vector
        sy_k = float(y_k.T @ s_k)  # Ensure sy_k is treated as a scalar
        if sy_k > 1e-10:  # Avoid division by zero
            B_k = B_k + (y_k @ y_k.T) / sy_k - (B_k @ s_k @ s_k.T @ B_k) / float(s_k.T @ B_k @ s_k)

        x_k = x_k_next

    return x_k, func_values, k


# def adam_optimizer(f, x0, grad, max_iters=1000, learning_rate=0.001, delta1=1e-6, delta2=1e-6, display=False, check_domain=False):
#     """
#     Adam optimizer with optional domain check to handle functions like log-barrier.
#     """
#     x_k = x0
#     fx_ls = [f(x_k)]
#     s = np.zeros_like(x_k)  # Initialize first moment vector
#     r = np.zeros_like(x_k)  # Initialize second moment vector

#     rho1 = 0.9  # Decay rate for first moment
#     rho2 = 0.999  # Decay rate for second moment
#     delta = 1e-8  # Small epsilon to avoid division by zero

#     def isin_domain(x):
#         """Checks if x remains within the domain of f."""
#         diff = b - A @ x  # Assuming b, A are global or passed as parameters
#         return np.all(diff > 0)

#     for k in range(max_iters):
#         g = grad(x_k)  # Compute gradient

#         # Update biased first and second moment estimates
#         s = rho1 * s + (1 - rho1) * g
#         r = rho2 * r + (1 - rho2) * g ** 2

#         # Bias-corrected moment estimates
#         s_hat = s / (1 - rho1 ** (k + 1))
#         r_hat = r / (1 - rho2 ** (k + 1))

#         # Adam update step
#         p_k = - learning_rate * s_hat / (np.sqrt(r_hat) + delta)
#         x_k_next = x_k + p_k

#         # Check domain validity if required
#         if check_domain and not isin_domain(x_k_next):
#             # Adjust learning rate or skip this update
#             learning_rate *= 0.5
#             continue  # Skip to next iteration without updating x_k

#         # Store function value
#         fx_ls.append(f(x_k_next))

#         if display:
#             print(f"Iteration {k+1}, x_k: {x_k}, f(x_k): {f(x_k)}, grad: {g}")

#         # Convergence checks
#         if np.linalg.norm(x_k_next - x_k) < delta1 or abs(fx_ls[-1] - fx_ls[-2]) < delta2:
#             print("Convergence reached.")
#             break

#         x_k = x_k_next

#     return x_k, fx_ls, k + 1



