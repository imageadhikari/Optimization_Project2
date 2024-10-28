import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data
from objective_functions import f1, grad_f1, hessian_f1, f2, grad_f2, hessian_f2, f3, grad_f3, hessian_f3
# from optimization_methods import gradient_descent, newton_method, quasi_newton_bfgs, adam_optimizer
from optimization_methods import gradient_descent, newton_method, quasi_newton_bfgs

# Load matrix data for Function 2
A = load_data('functions/fun2_A.txt', (500, 100))
c = load_data('functions/fun2_c.txt', (100, 1)).reshape(-1, 1)
b = load_data('functions/fun2_b.txt', (500, 1)).reshape(-1, 1)

# Initial guesses
x0_f1 = np.ones(100)
x0_f2 = np.ones(100) * 0.01
x0_f3 = np.array([10.0, 10.0])

os.makedirs('artifacts', exist_ok=True)

def save_convergence_plot(func_values, title, filename):
    for method, values in func_values.items():
        plt.plot(values, label=method)
    
    all_positive = all(all(val > 0 for val in values) for values in func_values.values())
    if all_positive:
        plt.yscale('log')
    
    plt.xlabel('Iterations')
    plt.ylabel('Function Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'artifacts/{filename}.png')
    plt.close()


def save_history_as_csv(values, filename, x_min=None):
    with open(f'artifacts/{filename}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration', 'Function Value'])
        
        for i, value in enumerate(values):
            writer.writerow([i, value])
        
        if x_min is not None:
            writer.writerow(['x_min', *x_min])


def plot_errors(values, title, filename):
    errors = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
    plt.plot(errors, label='Error (|f(x_k) - f(x_k-1)|)')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'artifacts/{filename}.png')
    plt.close()


methods = {
    "Gradient Descent": gradient_descent,
    "Newton's Method": newton_method,
    "Quasi-Newton (BFGS)": quasi_newton_bfgs,
    # "Adam Optimizer": adam_optimizer
}

for func, grad, hess, x0, title in [
        (f1, grad_f1, hessian_f1, x0_f1, "Function 1"),
        (lambda x: f2(x, A, b, c), lambda x: grad_f2(x, A, b, c), lambda x: hessian_f2(x, A, b), x0_f2, "Function 2"),
        (f3, grad_f3, hessian_f3, x0_f3, "Function 3")]:
    
    func_values = {}
    print(f"\nResults for {title}:")

    for method_name, method in methods.items():
        # if method_name == "Adam Optimizer":
        #     x_min, values, steps = method(func, x0, grad)
        if method_name in ["Quasi-Newton (BFGS)", "Gradient Descent"]:
            # Pass check_domain=True only for Function 2
            x_min, values, steps = method(func, grad, x0)
        else:
            x_min, values, steps = method(func, grad, hess, x0, 1e-2, 1e-2)
        
        func_values[method_name] = values
        min_value = values[-1]
        print(f"{method_name} - Steps: {steps}, Minimum f(x): {min_value}")
        
        csv_filename = f"{title.replace(' ', '_').lower()}_{method_name.replace(' ', '_').lower()}_history"

        if title == "Function 3":
            save_history_as_csv(values, csv_filename, x_min=x_min)
        else:
            save_history_as_csv(values, csv_filename)

        error_plot_filename = f"{title.replace(' ', '_').lower()}_{method_name.replace(' ', '_').lower()}_errors"
        plot_errors(values, f"Error vs Iterations for {title} - {method_name}", error_plot_filename)
    
    plot_filename = f"{title.replace(' ', '_').lower()}_convergence"
    save_convergence_plot(func_values, f"Convergence for {title}", plot_filename)

