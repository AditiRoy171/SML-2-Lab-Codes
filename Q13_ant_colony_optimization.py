import numpy as np
import matplotlib.pyplot as plt

def generate_graph(n_nodes=10):
    pos = {
        0: (0, 3),
        1: (2, 5), 2: (2, 3), 3: (2, 1),
        4: (5, 5), 5: (5, 3), 6: (5, 1),
        7: (8, 4), 8: (8, 2),
        9: (10, 3)
    }
    edges = [
        (0, 1, 3.5), (0, 2, 2.5), (0, 3, 3.0),
        (1, 4, 4.0), (1, 5, 3.0),
        (2, 4, 3.2), (2, 5, 2.0), (2, 6, 2.8),
        (3, 5, 3.5), (3, 6, 2.2),
        (4, 7, 3.0), (4, 8, 4.5),
        (5, 7, 2.5), (5, 8, 2.5),
        (6, 7, 4.0), (6, 8, 2.0),
        (7, 9, 2.5), (8, 9, 2.0)
    ]
    adj = {i: [] for i in range(n_nodes)}
    weights = {}
    for u, v, w in edges:
        adj[u].append(v)
        weights[(u, v)] = w
    return pos, adj, weights

def ant_colony_shortest_path(adj, weights, start=0, target=9, n_ants=25, n_iter=60, alpha=1.0, beta=2.0, rho=0.2, Q=10.0):
    pheromones = {edge: 1.0 for edge in weights}
    best_path = None
    best_len = float('inf')
    history = []

    for _ in range(n_iter):
        iter_paths = []
        iter_lengths = []

        for _ in range(n_ants):
            curr = start
            path = [curr]
            length = 0.0

            while curr != target:
                neighbors = adj[curr]
                if not neighbors:
                    break
                probs = []
                for nxt in neighbors:
                    tau = pheromones[(curr, nxt)] ** alpha
                    eta = (1.0 / weights[(curr, nxt)]) ** beta
                    probs.append(tau * eta)
                total = sum(probs)
                if total == 0:
                    break
                probs = [p / total for p in probs]
                nxt = np.random.choice(neighbors, p=probs)
                path.append(nxt)
                length += weights[(curr, nxt)]
                curr = nxt

            if curr == target:
                iter_paths.append(path)
                iter_lengths.append(length)
                if length < best_len:
                    best_len = length
                    best_path = path

        history.append(best_len)

        for edge in pheromones:
            pheromones[edge] *= (1.0 - rho)

        for path, length in zip(iter_paths, iter_lengths):
            deposit = Q / length
            for i in range(len(path) - 1):
                pheromones[(path[i], path[i+1])] += deposit

    return best_path, best_len, history

if __name__ == '__main__':
    np.random.seed(42)
    pos, adj, weights = generate_graph(n_nodes=10)
    best_path, best_len, history = ant_colony_shortest_path(adj, weights, start=0, target=9)

    print(f"Shortest Path: {' -> '.join(map(str, best_path))}")
    print(f"Shortest Path Length: {best_len:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for (u, v), w in weights.items():
        axes[0].plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 'gray', linestyle='--', alpha=0.6)
        mid_x = (pos[u][0] + pos[v][0]) / 2
        mid_y = (pos[u][1] + pos[v][1]) / 2
        axes[0].text(mid_x, mid_y, f"{w}", color='black', fontsize=8)

    path_edges = list(zip(best_path[:-1], best_path[1:]))
    for u, v in path_edges:
        axes[0].plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color='red', linewidth=3, zorder=3)

    for node, (x, y) in pos.items():
        color = 'green' if node == 0 else ('gold' if node == 9 else 'skyblue')
        axes[0].scatter(x, y, s=400, color=color, edgecolors='black', zorder=4)
        axes[0].text(x, y, str(node), ha='center', va='center', weight='bold')

    axes[0].set_title(f"ACO Shortest Path: {' -> '.join(map(str, best_path))} (Length: {best_len:.2f})")
    axes[0].axis('off')

    axes[1].plot(history, color='darkorange', linewidth=2)
    axes[1].set_title("ACO Convergence Curve")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Best Path Length")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
