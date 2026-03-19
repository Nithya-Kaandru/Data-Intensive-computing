# ========== Step 0: Setup ==========
from pyspark.sql import SparkSession
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.functions import col, when, from_unixtime, isnan, count, avg, regexp_replace, year, to_date, size, split

# Set styles
sns.set(style="whitegrid")

# Create output folders if not exists
output_folder = "/opt/hadoop/output/clustering"
os.makedirs(output_folder, exist_ok=True)

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("DICProject-ReaderSegmentation") \
    .config("spark.sql.session.timeZone", "America/New_York") \
    .getOrCreate()

# Load the merged parquet
merged_df = spark.read.parquet("/opt/hadoop/output/merged_df_parquet")
print("\nLoaded merged_df for clustering.")

# ========== Step 1: Aggregate User Features ==========

user_features_df = merged_df.groupBy("User_id").agg(
    avg("review/score").alias("avg_review_score"),
    count("review/score").alias("num_reviews"),
    avg("review_length").alias("avg_review_length")
)

user_features_pd = user_features_df.toPandas()

print("\nAggregated user features successfully.")

# ========== Step 2: Feature Scaling ==========

features_to_scale = ['avg_review_score', 'num_reviews', 'avg_review_length']

scaler = StandardScaler()
scaled_features = scaler.fit_transform(user_features_pd[features_to_scale])

print("\nFeatures scaled successfully.")

# ========== Step 3: Clustering Using K-Means ==========

kmeans = KMeans(n_clusters=4, random_state=42)
user_features_pd['cluster'] = kmeans.fit_predict(scaled_features)

print("\nClustering done.")

# Cluster size
cluster_sizes = user_features_pd['cluster'].value_counts()
print("\nCluster Size Distribution:")
print(cluster_sizes)

# Cluster profile summary
cluster_profiles = user_features_pd.groupby('cluster').agg({
    'avg_review_score': 'mean',
    'num_reviews': 'mean',
    'avg_review_length': 'mean',
    'User_id': 'count'
}).rename(columns={'User_id': 'num_users'})

# Save cluster profile to text
summary_file_path = f"{output_folder}/cluster_summary.txt"
with open(summary_file_path, "w") as f:
    f.write("=" * 30 + "\n")
    f.write("Reader Segmentation Cluster Summary\n")
    f.write("=" * 30 + "\n\n")

    f.write("Cluster Size Distribution:\n")
    for cluster_id, size in cluster_sizes.items():
        f.write(f"Cluster {cluster_id}: {size} users\n")

    f.write("\nCluster Profiles (Averages):\n")
    f.write(cluster_profiles.round(2).to_string())
    f.write("\n")

print(f"\nCluster profiles saved to {summary_file_path}")

# ========== Step 4: Elbow Method for Optimal k ==========

inertia = []
k_values = list(range(2, 8))

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled_features)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method to Choose Optimal k', fontsize=16)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (WCSS)')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_folder}/elbow_method.png")
plt.close()

print("\nElbow method plot saved.")

# ========== Step 5: Visualize Clusters Using PCA ==========

pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

pca_df = pd.DataFrame(pca_features, columns=['PC1', 'PC2'])
pca_df['cluster'] = user_features_pd['cluster']

plt.figure(figsize=(10, 8))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cluster', palette='Set2', alpha=0.6)
plt.title('User Clusters after PCA', fontsize=16)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig(f"{output_folder}/user_clusters_pca.png")
plt.close()

print("\nCluster visualization plot saved.")

# ========== Step 6: Clean Exit ==========
spark.stop()
print("\n Spark session stopped cleanly.")
