import numpy as np
import matplotlib.pyplot as plt

def compute_distances(cities):
    n = len(cities)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(cities[i] - cities[j])
    return dist_matrix

def tour_distance(tour, dist_matrix):
    dist = sum(dist_matrix[tour[i], tour[i+1]] for i in range(len(tour)-1))
    dist += dist_matrix[tour[-1], tour[0]]
    return dist

def ordered_crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(np.random.choice(size, 2, replace=False))
    child = [-1] * size
    child[a:b] = parent1[a:b]
    
    p2_idx = 0
    for i in range(size):
        if child[i] == -1:
            while parent2[p2_idx] in child:
                p2_idx += 1
            child[i] = parent2[p2_idx]
    return child

def mutate(tour, mutation_rate=0.05):
    tour = tour.copy()
    if np.random.rand() < mutation_rate:
        i, j = np.random.choice(len(tour), 2, replace=False)
        tour[i], tour[j] = tour[j], tour[i]
    return tour

def genetic_algorithm_tsp(cities, pop_size=60, generations=300, mutation_rate=0.1):
    n_cities = len(cities)
    dist_matrix = compute_distances(cities)
    
    population = [np.random.permutation(n_cities).tolist() for _ in range(pop_size)]
    best_tour = None
    best_dist = float('inf')
    history = []

    for gen in range(generations):
        distances = [tour_distance(ind, dist_matrix) for ind in population]
        fitness = [1.0 / d for d in distances]
        
        gen_min_idx = np.argmin(distances)
        if distances[gen_min_idx] < best_dist:
            best_dist = distances[gen_min_idx]
            best_tour = population[gen_min_idx]
        history.append(best_dist)
        
        probs = np.array(fitness) / sum(fitness)
        new_population = [best_tour]
        
        while len(new_population) < pop_size:
            idx1, idx2 = np.random.choice(pop_size, size=2, p=probs)
            child = ordered_crossover(population[idx1], population[idx2])
            child = mutate(child, mutation_rate)
            new_population.append(child)
            
        population = new_population

    return best_tour, best_dist, history

if __name__ == '__main__':
    np.random.seed(42)
    n_cities = 15
    cities = np.random.rand(n_cities, 2) * 100

    best_tour, best_dist, history = genetic_algorithm_tsp(cities, pop_size=80, generations=300, mutation_rate=0.15)
    print(f"Optimal Tour: {best_tour}")
    print(f"Shortest Distance: {best_dist:.3f}")

    tour_coords = cities[best_tour + [best_tour[0]]]
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(tour_coords[:, 0], tour_coords[:, 1], 'o-', color='crimson', markersize=7)
    for idx, (x, y) in enumerate(cities):
        axes[0].text(x + 1, y + 1, str(idx), fontsize=10, weight='bold')
    axes[0].set_title(f"Optimal TSP Tour (Total Dist: {best_dist:.2f})")
    axes[0].grid(True)

    axes[1].plot(history, color='blue')
    axes[1].set_title("GA Convergence Curve")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Shortest Route Distance")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
