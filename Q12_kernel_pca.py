import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA, KernelPCA

if __name__ == '__main__':
    np.random.seed(42)
    X, y = make_circles(n_samples=500, factor=0.3, noise=0.06, random_state=42)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=7)
    X_kpca = kpca.fit_transform(X)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].scatter(X[y == 0, 0], X[y == 0, 1], color='red', alpha=0.7, label='Class 0')
    axes[0].scatter(X[y == 1, 0], X[y == 1, 1], color='blue', alpha=0.7, label='Class 1')
    axes[0].set_title("Original Non-Linear Data")
    axes[0].set_xlabel("X1")
    axes[0].set_ylabel("X2")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color='red', alpha=0.7, label='Class 0')
    axes[1].scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], color='blue', alpha=0.7, label='Class 1')
    axes[1].set_title("Standard Linear PCA (Overlapping)")
    axes[1].set_xlabel("PC 1")
    axes[1].set_ylabel("PC 2")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1], color='red', alpha=0.7, label='Class 0')
    axes[2].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1], color='blue', alpha=0.7, label='Class 1')
    axes[2].set_title("Kernel PCA with RBF (Linearly Separable)")
    axes[2].set_xlabel("KPCA Component 1")
    axes[2].set_ylabel("KPCA Component 2")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()
