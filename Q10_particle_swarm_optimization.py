import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(pos):
    x, y = pos[0], pos[1]
    return (1.0 - x)**2 + 100.0 * (y - x**2)**2

def particle_swarm_optimization(func, bounds, n_particles=50, n_iter=150, w=0.7, c1=1.5, c2=1.5):
    dim = len(bounds)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])

    positions = lower + np.random.rand(n_particles, dim) * (upper - lower)
    velocities = (np.random.rand(n_particles, dim) - 0.5) * (upper - lower) * 0.1
    pbest = positions.copy()
    pbest_scores = np.array([func(p) for p in positions])

    gbest_idx = np.argmin(pbest_scores)
    gbest = pbest[gbest_idx].copy()
    gbest_score = pbest_scores[gbest_idx]
    history = [gbest_score]

    for _ in range(n_iter):
        for i in range(n_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            velocities[i] = (w * velocities[i] + 
                             c1 * r1 * (pbest[i] - positions[i]) + 
                             c2 * r2 * (gbest - positions[i]))
            
            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], lower, upper)

            score = func(positions[i])
            if score < pbest_scores[i]:
                pbest[i] = positions[i].copy()
                pbest_scores[i] = score
                if score < gbest_score:
                    gbest = positions[i].copy()
                    gbest_score = score

        history.append(gbest_score)

    return gbest, gbest_score, history

if __name__ == '__main__':
    np.random.seed(42)
    bounds = [(-2.0, 2.0), (-1.0, 3.0)]
    best_pos, best_val, history = particle_swarm_optimization(rosenbrock, bounds)

    print(f"Optimal Position (x, y): {best_pos[0]:.4f}, {best_pos[1]:.4f}")
    print(f"Minimum Value: {best_val:.6f}")

    x = np.linspace(bounds[0][0], bounds[0][1], 200)
    y = np.linspace(bounds[1][0], bounds[1][1], 200)
    X, Y = np.meshgrid(x, y)
    Z = (1.0 - X)**2 + 100.0 * (Y - X**2)**2

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    contour = axes[0].contourf(X, Y, np.log1p(Z), levels=35, cmap='viridis')
    axes[0].plot(1, 1, 'w*', markersize=12, label='True Min (1, 1)')
    axes[0].plot(best_pos[0], best_pos[1], 'ro', markersize=8, label=f'Found Min: ({best_pos[0]:.2f}, {best_pos[1]:.2f})')
    axes[0].set_title("Rosenbrock Contour & PSO Solution")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()
    fig.colorbar(contour, ax=axes[0], label='log(1 + f(x, y))')

    axes[1].plot(history, color='crimson')
    axes[1].set_yscale('log')
    axes[1].set_title("PSO Convergence (Log Scale)")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Global Best Cost")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
