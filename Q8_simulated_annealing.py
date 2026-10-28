import numpy as np
import matplotlib.pyplot as plt

def objective(x):
    return x**2 + 10 * np.sin(x)

def simulated_annealing(func, bounds, n_iter=1000, step_size=0.2, t_start=100.0, alpha=0.95):
    curr = bounds[0] + np.random.rand() * (bounds[1] - bounds[0])
    curr_eval = func(curr)
    best, best_eval = curr, curr_eval
    t = t_start
    history = [best_eval]

    for _ in range(n_iter):
        cand = curr + np.random.randn() * step_size
        cand = np.clip(cand, bounds[0], bounds[1])
        cand_eval = func(cand)

        diff = cand_eval - curr_eval
        if diff < 0 or np.random.rand() < np.exp(-diff / t):
            curr, curr_eval = cand, cand_eval
            if cand_eval < best_eval:
                best, best_eval = cand, cand_eval

        history.append(best_eval)
        t *= alpha
        if t < 1e-8:
            break

    return best, best_eval, history

if __name__ == '__main__':
    np.random.seed(42)
    bounds = [-10.0, 10.0]
    best_x, best_y, history = simulated_annealing(objective, bounds)

    print(f"Optimal x: {best_x:.4f}")
    print(f"Minimum f(x): {best_y:.4f}")

    x_range = np.linspace(bounds[0], bounds[1], 1000)
    y_range = objective(x_range)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(x_range, y_range, label='f(x) = x^2 + 10*sin(x)', color='blue')
    axes[0].scatter(best_x, best_y, color='red', s=70, zorder=5, label=f'Best: ({best_x:.2f}, {best_y:.2f})')
    axes[0].set_title("Function Landscape & Found Minimum")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history, color='green')
    axes[1].set_title("Simulated Annealing Convergence")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Best Objective Value")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
