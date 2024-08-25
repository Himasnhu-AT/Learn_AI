# Clustering Algorithms

## Introduction

Clustering is a fundamental unsupervised learning technique used to group data points into clusters based on their similarity. This means we do not have predefined labels for the data, and the algorithm aims to discover meaningful patterns and structures within the dataset. Clustering algorithms find applications in various domains, including:

* **Customer segmentation:** Grouping customers based on their purchasing behavior, demographics, or preferences.
* **Image segmentation:** Identifying different objects or regions within an image.
* **Document clustering:** Organizing text documents into clusters based on their topics or themes.
* **Anomaly detection:** Identifying outliers or unusual data points that deviate significantly from the rest of the data.

In this tutorial, we will explore two popular clustering algorithms: K-Means and Hierarchical clustering.

## K-Means Clustering

### Algorithm Overview

K-Means clustering is a simple yet effective algorithm that partitions data points into *k* clusters, where *k* is a predetermined number. The algorithm works iteratively by following these steps:

1. **Initialization:** Randomly choose *k* data points as initial cluster centroids.
2. **Assignment:** Assign each data point to the cluster whose centroid is closest to it.
3. **Update:** Recalculate the centroid of each cluster based on the assigned data points.
4. **Repeat steps 2 and 3 until convergence:** The algorithm stops when the cluster assignments no longer change significantly between iterations.

**Key Concepts:**

* **Centroid:** The mean of all data points within a cluster.
* **Distance metric:** Used to calculate the distance between data points and centroids (e.g., Euclidean distance, Manhattan distance).
* **Convergence:** The algorithm stops when the cluster assignments stabilize, indicating that the clusters have been well-defined.

### Implementation (Python)

Sample Python Code: 

```{language}
import numpy as np
from sklearn.cluster import KMeans

# Sample data
data = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# Initialize KMeans with k=2
kmeans = KMeans(n_clusters=2, random_state=0)

# Fit the model to the data
kmeans.fit(data)

# Get cluster labels for each data point
labels = kmeans.labels_

# Print cluster assignments
print("Cluster labels:", labels)

# Get cluster centroids
centroids = kmeans.cluster_centers_

# Print cluster centroids
print("Cluster centroids:", centroids)
```

### Strengths and Weaknesses

**Strengths:**

* Relatively simple to understand and implement.
* Efficient for large datasets.

**Weaknesses:**

* Requires specifying the number of clusters *k* beforehand.
* Sensitive to the initial choice of centroids.
* Can be affected by outliers in the data.

### Evaluating K-Means Performance

* **Elbow method:** Plot the within-cluster sum of squares (WCSS) for different values of *k*. The "elbow" point on the plot often indicates the optimal number of clusters.
* **Silhouette score:** Measures how similar each data point is to its own cluster compared to other clusters. A high silhouette score indicates good clustering.

## Hierarchical Clustering

### Algorithm Overview

Hierarchical clustering is a method that builds a hierarchy of clusters. It can be either agglomerative (bottom-up) or divisive (top-down):

* **Agglomerative clustering:** Starts with each data point as its own cluster and iteratively merges the closest clusters until a single cluster remains.
* **Divisive clustering:** Starts with all data points in a single cluster and iteratively splits the cluster into smaller clusters until each data point is in its own cluster.

### Implementation (Python)

Sample Python Code: 

```{language}
import numpy as np
from sklearn.cluster import AgglomerativeClustering

# Sample data
data = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# Initialize AgglomerativeClustering with linkage='ward'
hierarchical = AgglomerativeClustering(n_clusters=2, linkage='ward')

# Fit the model to the data
hierarchical.fit(data)

# Get cluster labels for each data point
labels = hierarchical.labels_

# Print cluster assignments
print("Cluster labels:", labels)
```

### Strengths and Weaknesses

**Strengths:**

* Does not require specifying the number of clusters beforehand.
* Can handle data with complex shapes and structures.
* Provides a hierarchical representation of clusters.

**Weaknesses:**

* Can be computationally expensive for large datasets.
* Sensitive to the choice of linkage method.

### Evaluating Hierarchical Clustering Performance

* **Dendrogram:** A graphical representation of the cluster hierarchy. It helps visualize the merging or splitting of clusters at different levels.
* **Silhouette score:** Similar to K-Means, a high silhouette score indicates good clustering.

## Assignment

1. **K-Means Clustering:**
   - Implement K-Means clustering on the Iris dataset (available in scikit-learn) with different values of *k*.
   - Plot the WCSS for different *k* values using the elbow method to find the optimal number of clusters.
   - Calculate the silhouette score for the optimal number of clusters.
2. **Hierarchical Clustering:**
   - Implement hierarchical clustering on the Iris dataset using different linkage methods (e.g., 'ward', 'average').
   - Visualize the dendrogram for each linkage method.
   - Calculate the silhouette score for each linkage method and compare the results.

## Conclusion

Clustering algorithms are powerful tools for discovering hidden patterns and structures in data. By understanding the different types of algorithms and their strengths and weaknesses, we can effectively apply them to solve a wide range of problems in various domains.
