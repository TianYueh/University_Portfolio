import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN,
    SpectralClustering
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.neighbors import NearestNeighbors

# Prepare datasets with true labels
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

# Define clustering models
def get_models(k):
    return [
        ('K-Means++', KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)),
        ('Agglomerative (Ward)', AgglomerativeClustering(n_clusters=k, linkage='ward')),
        ('GMM (full)', GaussianMixture(n_components=k, covariance_type='full', random_state=42)),
        ('Spectral (RBF)', SpectralClustering(n_clusters=k, affinity='rbf', gamma=1.0, random_state=42)),
        ('DBSCAN', DBSCAN(eps=0.2, min_samples=5))
    ]

# Collect evaluation results
results = []
for name, (X, y_true, k) in datasets.items():
    for model_name, model in get_models(k):
        if model_name == 'DBSCAN' and name != 'Moons':
            continue
        if hasattr(model, 'fit_predict'):
            labels = model.fit_predict(X)
        else:
            labels = model.fit(X).predict(X)
        if len(set(labels)) <= 1:
            sil, db, ch = (np.nan, np.nan, np.nan)
        else:
            sil = silhouette_score(X, labels)
            db = davies_bouldin_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
        ari = adjusted_rand_score(y_true, labels)
        nmi = normalized_mutual_info_score(y_true, labels)
        results.append({
            'Dataset': name,
            'Model': model_name,
            'Clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'Silhouette': round(sil, 3) if not np.isnan(sil) else None,
            'Davies-Bouldin': round(db, 3) if not np.isnan(db) else None,
            'Calinski-Harabasz': round(ch, 3) if not np.isnan(ch) else None,
            'ARI': round(ari, 3),
            'NMI': round(nmi, 3)
        })

df_results = pd.DataFrame(results)
print(df_results.to_markdown(index=False))
