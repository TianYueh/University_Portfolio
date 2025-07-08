import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans as SKKMeans, AgglomerativeClustering as SKAgglo
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)

# === k-means++ Initialization Function ===
def kmeans_plus_plus_init(X, k, random_state=None):
    rng = np.random.RandomState(random_state)
    n_samples = X.shape[0]
    centers = []
    # 1st center
    idx = rng.randint(n_samples)
    centers.append(X[idx])
    # Remaining centers
    for _ in range(1, k):
        d2 = np.min([np.sum((X - c)**2, axis=1) for c in centers], axis=0)
        probs = d2 / d2.sum()
        idx = rng.choice(n_samples, p=probs)
        centers.append(X[idx])
    return np.array(centers)

# === Custom KMeans with k-means++ and multi-init ===
class MyKMeans:
    def __init__(self, n_clusters=3, n_init=10, max_iters=100, tol=1e-4, random_state=None):
        self.k = n_clusters
        self.n_init = n_init
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

    def fit_predict(self, X):
        best_inertia = np.inf
        best_labels = None
        rng = np.random.RandomState(self.random_state)
        for i in range(self.n_init):
            centers = kmeans_plus_plus_init(X, self.k, random_state=rng.randint(1e9))
            for _ in range(self.max_iters):
                dists = np.linalg.norm(X[:, None] - centers[None, :], axis=2)
                labels = np.argmin(dists, axis=1)
                new_centers = np.array([
                    X[labels == j].mean(axis=0) if np.any(labels == j)
                    else X[rng.randint(len(X))]
                    for j in range(self.k)
                ])
                if np.linalg.norm(new_centers - centers) < self.tol:
                    break
                centers = new_centers
            inertia = np.sum((X - centers[labels])**2)
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
        return best_labels

# === Custom Single-Linkage Agglomerative ===
class MyAggloSingle:
    def __init__(self, n_clusters=3):
        self.k = n_clusters

    def fit_predict(self, X):
        n = X.shape[0]
        clusters = [[i] for i in range(n)]
        dist_mat = np.linalg.norm(X[:, None] - X[None, :], axis=2)
        np.fill_diagonal(dist_mat, np.inf)
        while len(clusters) > self.k:
            min_val, pair = np.inf, (None, None)
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):
                    d = dist_mat[np.ix_(clusters[i], clusters[j])].min()
                    if d < min_val:
                        min_val, pair = d, (i, j)
            i, j = pair
            clusters[i] += clusters[j]
            del clusters[j]
        labels = np.empty(n, dtype=int)
        for idx, cl in enumerate(clusters):
            labels[cl] = idx
        return labels

# === Data Preparation ===
iris = load_iris()
X_iris = StandardScaler().fit_transform(iris.data)
X_iris_2d = PCA(n_components=2, random_state=42).fit_transform(X_iris)
y_iris = iris.target

X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)
X_blobs = StandardScaler().fit_transform(X_blobs)

X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=42)
X_moons = StandardScaler().fit_transform(X_moons)

datasets = {
    'Iris': (X_iris_2d, y_iris, 3),
    'Blobs': (X_blobs, y_blobs, 4),
    'Moons': (X_moons, y_moons, 2)
}

# === Evaluation and Visualization ===
results = []
for name, (X, y_true, k) in datasets.items():
    models = [
        ('MyKMeans', MyKMeans(n_clusters=k, n_init=10, random_state=42)),
        ('SKKMeans', SKKMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)),
        ('MyAggloSingle', MyAggloSingle(n_clusters=k)),
        ('SKAgglo', SKAgglo(n_clusters=k, linkage='single'))
    ]
    for model_name, model in models:
        labels = model.fit_predict(X)
        # Visualization
        plt.figure(figsize=(4, 4))
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, edgecolor='k')
        plt.title(f'{model_name} on {name}')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.tight_layout()
        plt.show()
        # Metrics
        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else np.nan
        db = davies_bouldin_score(X, labels) if len(set(labels)) > 1 else np.nan
        ch = calinski_harabasz_score(X, labels) if len(set(labels)) > 1 else np.nan
        ari = adjusted_rand_score(y_true, labels)
        nmi = normalized_mutual_info_score(y_true, labels)
        results.append({
            'Dataset': name,
            'Model': model_name,
            'Clusters': len(set(labels)),
            'Silhouette': round(sil, 3) if not np.isnan(sil) else None,
            'Davies-Bouldin': round(db, 3) if not np.isnan(db) else None,
            'Calinski-Harabasz': round(ch, 3) if not np.isnan(ch) else None,
            'ARI': round(ari, 3),
            'NMI': round(nmi, 3)
        })

# Output results as markdown table
df = pd.DataFrame(results)
print(df.to_markdown(index=False))

