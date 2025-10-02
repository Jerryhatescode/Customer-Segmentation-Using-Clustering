
Customer Segmentation with K-Means Clustering
=============================================

This project demonstrates customer segmentation using K-Means clustering 
on a synthetic dataset. It highlights data preprocessing, cluster evaluation, 
visualization, and profiling of customer segments.

Features
--------
- Generate synthetic dataset with:
  - Age
  - Annual Income (k$)
  - Spending Score (1–100)
- Standardize features using StandardScaler
- Determine optimal number of clusters with:
  - Elbow Method (inertia)
  - Silhouette Score
- Apply K-Means clustering
- Dimensionality reduction with PCA (2D visualization)
- 3D scatter plot visualization (Age, Income, Spending Score)
- Cluster profiling and descriptive labeling:
  - High-value
  - Bargain-seekers
  - Low-value
  - Cautious spenders

Requirements
------------
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Workflow
--------
1. Data generation
2. Preprocessing (scaling features)
3. Cluster selection (Elbow Method, Silhouette Score)
4. K-Means clustering
5. Dimensionality reduction (PCA)
6. Visualization (2D & 3D plots)
7. Cluster profiling and segment labeling

Outputs
-------
- Inertia plot for Elbow Method
- Silhouette scores for multiple clusters
- 2D PCA scatter plot of customer segments
- 3D scatter plot of Age, Income, Spending Score
- Cluster summary table with mean values
- Segment counts by descriptive labels

Notes
-----
- Synthetic dataset is used for demonstration.
- Replace with real-world customer data for practical use.
- PCA is applied only for visualization, not for clustering.
"""
