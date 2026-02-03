"""
Basic Clustering Example
========================

This example demonstrates basic clustering using Nalyst.

Topics covered:
- KMeans clustering
- DBSCAN clustering
- Hierarchical clustering
- Evaluating clustering results
"""

import numpy as np
from nalyst.clustering import (
    KMeansCluster,
    DBSCANCluster,
    HierarchicalCluster,
    GaussianMixture
)
from nalyst.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score
)
from nalyst.datasets import load_iris, make_blobs

# Example 1: KMeans Clustering

# print a simple banner so outputs are grouped clearly
print("=" * 60)
print("Example 1: KMeans Clustering")
print("=" * 60)

# create a toy blob dataset so the math stays intuitive
X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)
print(f"Dataset shape: {X.shape}")
print(f"True clusters: {len(np.unique(y_true))}")

# run kmeans to get crisp assignments
kmeans = KMeansCluster(n_clusters=4, random_state=42)
labels = kmeans.train_infer(X)

# collect a few common clustering diagnostics
silhouette = silhouette_score(X, labels)
calinski = calinski_harabasz_score(X, labels)
davies = davies_bouldin_score(X, labels)
ari = adjusted_rand_score(y_true, labels)

print(f"\nKMeans Results:")
print(f"  Silhouette Score:      {silhouette:.4f} (higher is better, max=1)")
print(f"  Calinski-Harabasz:     {calinski:.2f} (higher is better)")
print(f"  Davies-Bouldin:        {davies:.4f} (lower is better)")
print(f"  Adjusted Rand Index:   {ari:.4f} (vs true labels)")

print(f"\nCluster sizes:")
for i in range(4):
    print(f"  Cluster {i}: {(labels == i).sum()} samples")

print(f"\nCluster centers:")
print(kmeans.cluster_centers_)

# Example 2: Finding Optimal Number of Clusters

# move into a quick elbow search for k selection
print("\n" + "=" * 60)
print("Example 2: Elbow Method - Finding Optimal K")
print("=" * 60)

inertias = []
silhouettes = []
k_range = range(2, 10)

for k in k_range:
    kmeans = KMeansCluster(n_clusters=k, random_state=42)
    labels = kmeans.train_infer(X)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X, labels))

print(f"\n{'K':<5} {'Inertia':<15} {'Silhouette':<12}")
print("-" * 35)
for k, inertia, sil in zip(k_range, inertias, silhouettes):
    print(f"{k:<5} {inertia:<15.2f} {sil:<12.4f}")

best_k = list(k_range)[np.argmax(silhouettes)]
print(f"\nOptimal K based on Silhouette: {best_k}")

# Example 3: DBSCAN Clustering

# demonstrate density based clustering for irregular shapes
print("\n" + "=" * 60)
print("Example 3: DBSCAN Clustering")
print("=" * 60)

# DBSCAN can find clusters of arbitrary shape and detect outliers
dbscan = DBSCANCluster(eps=0.5, min_samples=5)
labels = dbscan.train_infer(X)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()

print(f"\nDBSCAN Results:")
print(f"  Number of clusters: {n_clusters}")
print(f"  Number of outliers: {n_noise}")

if n_clusters > 1:
    # Exclude noise points for silhouette calculation
    mask = labels != -1
    if mask.sum() > 0:
        sil = silhouette_score(X[mask], labels[mask])
        print(f"  Silhouette Score:   {sil:.4f}")

# Example 4: Hierarchical Clustering

# give a taste of linkage based clustering styles
print("\n" + "=" * 60)
print("Example 4: Hierarchical Clustering")
print("=" * 60)

# Different linkage methods
linkages = ['ward', 'complete', 'average', 'single']

print(f"\n{'Linkage':<12} {'Silhouette':<12} {'Calinski-Harabasz':<18}")
print("-" * 45)

for linkage in linkages:
    hc = HierarchicalCluster(n_clusters=4, linkage=linkage)
    labels = hc.train_infer(X)

    sil = silhouette_score(X, labels)
    cal = calinski_harabasz_score(X, labels)

    print(f"{linkage:<12} {sil:<12.4f} {cal:<18.2f}")

# Example 5: Gaussian Mixture Models

# showcase a probabilistic alternative with mixture models
print("\n" + "=" * 60)
print("Example 5: Gaussian Mixture Models")
print("=" * 60)

gmm = GaussianMixture(n_components=4, random_state=42)
labels = gmm.train_infer(X)

# gmm provides soft assignments so we can inspect confidence per class
proba = gmm.infer_proba(X)

print(f"\nGMM Results:")
print(f"  Number of components: {gmm.n_components}")
print(f"  Silhouette Score:     {silhouette_score(X, labels):.4f}")

print(f"\nFirst 5 samples - Cluster probabilities:")
print(f"{'Sample':<8} {'Cluster 0':<12} {'Cluster 1':<12} {'Cluster 2':<12} {'Cluster 3':<12}")
print("-" * 60)
for i in range(5):
    print(f"{i:<8} {proba[i, 0]:<12.3f} {proba[i, 1]:<12.3f} {proba[i, 2]:<12.3f} {proba[i, 3]:<12.3f}")

# Example 6: Clustering Real Data (Iris)

# finish by running the same algorithms on a real dataset
print("\n" + "=" * 60)
print("Example 6: Clustering Iris Dataset")
print("=" * 60)

X, y_true = load_iris(return_X_y=True)

# Try different algorithms
algorithms = {
    "KMeans": KMeansCluster(n_clusters=3, random_state=42),
    "Hierarchical": HierarchicalCluster(n_clusters=3, linkage='ward'),
    "GMM": GaussianMixture(n_components=3, random_state=42),
}

print(f"\n{'Algorithm':<15} {'Silhouette':<12} {'ARI (vs true)':<15}")
print("-" * 45)

for name, algo in algorithms.items():
    labels = algo.train_infer(X)
    sil = silhouette_score(X, labels)
    ari = adjusted_rand_score(y_true, labels)
    print(f"{name:<15} {sil:<12.4f} {ari:<15.4f}")

print("\n Clustering examples completed!")
