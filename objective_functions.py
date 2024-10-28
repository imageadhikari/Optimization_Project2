import numpy as np

# Function 1
def f1(x):
    return np.sum(np.arange(1, len(x) + 1) * x**2)

def grad_f1(x):
    return 2 * np.arange(1, len(x) + 1) * x

def hessian_f1(x):
    return np.diag(2 * np.arange(1, len(x) + 1))

# # Function 2
# def f2(x, A, b, c):
#     return c.T @ x - np.sum(np.log(b - A @ x))

# def grad_f2(x, A, b, c):
#     return c + A.T @ (1 / (b - A @ x))

# def hessian_f2(x, A, b):
#     r = 1 / ((b - A @ x) ** 2)
#     return A.T @ np.diag(r.flatten()) @ A

# # Function 2
# def f2(x, A, b, c):
#     """
#     Log-barrier function for constrained optimization.
#     Assumes `back_track_line` handles the domain check.
#     """
#     x = x.reshape(-1, 1)  # Ensure x is a column vector
#     diff = b - A @ x
#     return float(c.T @ x - np.sum(np.log(diff)))

# Function 2
def f2(x, A, b, c):
    x = x.reshape(-1, 1)  # Ensure x is a column vector
    diff = b - A @ x
    if np.any(diff <= 0):
        return np.inf  # Assign a large value if log's argument is non-positive
    return float(c.T @ x - np.sum(np.log(diff)))

def grad_f2(x, A, b, c):
    x = x.reshape(-1, 1)  # Ensure x is a column vector
    diff = b - A @ x
    grad = c + A.T @ (1 / diff)
    return grad.flatten()  # Return as a flat array for compatibility

def hessian_f2(x, A, b, epsilon=1e-5):
    x = x.reshape(-1, 1)  # Ensure x is a column vector
    diff = b - A @ x
    r = 1 / (diff ** 2)
    hessian = A.T @ np.diag(r.flatten()) @ A
    return hessian + epsilon * np.eye(hessian.shape[0])  # Regularize with epsilon

# Function 3
def f3(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def grad_f3(x):
    return np.array([-400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]), 
                      200 * (x[1] - x[0]**2)])

def hessian_f3(x):
    return np.array([[1200 * x[0]**2 - 400 * x[1] + 2, -400 * x[0]], 
                     [-400 * x[0], 200]])
