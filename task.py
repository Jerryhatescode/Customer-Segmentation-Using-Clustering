import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D



np.random.seed(42)
n_customers = 200

df = pd.DataFrame({
    'CustomerID': range(1, n_customers+1),
    'Age': np.random.randint(18, 70, size=n_customers),
    'Annual Income (k$)': np.random.randint(15, 150, size=n_customers),
    'Spending Score (1-100)': np.random.randint(1, 100, size=n_customers)
})

print("First 5 rows of synthetic dataset:")
print(df.head())



features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()


for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"Silhouette score for {k} clusters: {score}")


optimal_clusters = 4  
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:,0]
df['PCA2'] = X_pca[:,1]

plt.figure(figsize=(8,6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='Set2', s=80)
plt.title("Customer Segments (PCA 2D)")
plt.show()


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df['Age'], df['Annual Income (k$)'], df['Spending Score (1-100)'],
                     c=df['Cluster'], cmap='Set2', s=60)
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
plt.title("Customer Segments (3D)")
plt.show()


cluster_summary = df.groupby('Cluster')[features].mean()
print("\nCluster Summary:")
print(cluster_summary)


def label_segments(row):
    if row['Cluster'] == 0:
        return 'High-value'
    elif row['Cluster'] == 1:
        return 'Bargain-seekers'
    elif row['Cluster'] == 2:
        return 'Low-value'
    else:
        return 'Cautious spenders'

df['Segment'] = df.apply(label_segments, axis=1)
print("\nSegment counts:")
print(df['Segment'].value_counts())
