# ========== Step 0: Setup ==========
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, from_unixtime, isnan, count, year, to_date, size, split, avg
from pyspark import StorageLevel
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sns.set(style="whitegrid")

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("DICProject-Phase2-Merge-Clean-EDA") \
    .config("spark.sql.session.timeZone", "America/New_York") \
    .config("spark.sql.files.maxPartitionBytes", 134217728) \
    .getOrCreate()

# Define output folders
output_folder = "/opt/hadoop/project_phase2/output"
summary_file_path = f"{output_folder}/summaries/merged_cleaning_summary.txt"
plot_folder = f"{output_folder}/plots/"
os.makedirs(os.path.dirname(summary_file_path), exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)
open(summary_file_path, "w").close()  # fresh

# Utility functions
def append_text_block(text, filepath, title=""):
    with open(filepath, "a") as f:
        if title:
            f.write("\n" + "="*30 + f"\n{title}\n" + "="*30 + "\n\n")
        f.write(text + "\n\n")

def append_dataframe(df, filepath, title=""):
    rows = df.collect()
    with open(filepath, "a") as f:
        if title:
            f.write("\n" + "="*30 + f"\n{title}\n" + "="*30 + "\n\n")
        for row in rows:
            row_dict = row.asDict()
            for key, value in row_dict.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

# ========== Step 1: Load Datasets ==========

books_df = spark.read.parquet("hdfs:///user/n1/parquet/cleaned_books_final_new/")
reviews_df = spark.read.parquet("hdfs:///user/n1/parquet/cleaned_reviews_final_new/")
print("Loaded cleaned books_df and reviews_df.")

# ========== Step 2: Merge Datasets ==========
merged_df = reviews_df.join(
    books_df,
    on="Title",
    how="left"
)

print(" Merged datasets using LEFT JOIN on Title.")

# Save initial schema
append_text_block(merged_df._jdf.schema().treeString(), summary_file_path, title="Merged Dataset Initial Schema")

# ========== Step 3: Data Cleaning and Feature Engineering ==========

# Type conversions
merged_df = merged_df.withColumn("review/score", col("review/score").cast("double"))
merged_df = merged_df.withColumn("ratingsCount", col("ratingsCount").cast("double"))

# Drop unwanted columns if exist
drop_columns = ["review/helpfulness", "image", "previewLink", "infoLink"]
merged_df = merged_df.drop(*[c for c in drop_columns if c in merged_df.columns])

# Create reviewYear
merged_df = merged_df.withColumn(
    "reviewYear",
    year(to_date(col("reviewTimeReadable")))
)

# Compute review_length
merged_df = merged_df.withColumn(
    "review_length",
    size(split(col("review/text"), " "))
)

append_text_block(merged_df._jdf.schema().treeString(), summary_file_path, title="Merged Dataset Schema After Cleaning")

print(" Data formatting completed.")

# ========== Step 4: Save Merged Cleaned Data ==========
save_path = "hdfs:///user/n1/parquet/cleaned_merged_books_reviews_final"
merged_df.repartition(10).write.mode("overwrite").parquet(save_path)
print(f" Cleaned merged_df saved successfully at {save_path}")

# ========== Step 5: Outlier Handling ==========

# Extract User Level Features
user_features_df = merged_df.groupBy("User_id").agg(
    count("review/score").alias("num_reviews"),
    avg("review/score").alias("avg_review_score"),
    avg("review_length").alias("avg_review_length")
)

user_features_pd = user_features_df.toPandas()

# Clip outliers
user_features_pd['avg_review_score'] = user_features_pd['avg_review_score'].clip(0.0, 5.0)

num_reviews_99th = np.percentile(user_features_pd['num_reviews'], 99)
user_features_pd['num_reviews'] = np.where(user_features_pd['num_reviews'] > num_reviews_99th, num_reviews_99th, user_features_pd['num_reviews'])

avg_review_length_99th = np.percentile(user_features_pd['avg_review_length'], 99)
user_features_pd['avg_review_length'] = np.where(user_features_pd['avg_review_length'] > avg_review_length_99th, avg_review_length_99th, user_features_pd['avg_review_length'])

append_text_block(f"Outlier Handling done on avg_review_score (0-5), num_reviews (cap {num_reviews_99th}), avg_review_length (cap {avg_review_length_99th})", summary_file_path)

# Save Outlier Verification Plots
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(user_features_pd['avg_review_score'], bins=20, kde=True, ax=axs[0], color='skyblue')
axs[0].set_title('Avg Review Score (Post-Capping)')

sns.histplot(user_features_pd['num_reviews'], bins=30, kde=True, ax=axs[1], color='orchid')
axs[1].set_title('Number of Reviews (Post-Capping)')

sns.histplot(user_features_pd['avg_review_length'], bins=30, kde=True, ax=axs[2], color='lightgreen')
axs[2].set_title('Avg Review Length (Post-Capping)')

plt.tight_layout()
plt.savefig(f"{plot_folder}/outlier_verification.png")
plt.close()

print(" Outlier detection and handling completed.")

# ========== Step 6: Exploratory Data Analysis (Plots) ==========

merged_pd = merged_df.select(
    "review/score", "ratingsCount", "review_length",
    "reviewYear", "categories", "authors", "Title"
).toPandas()

# Genre Distribution (Pie)
plt.figure(figsize=(8, 8))
top_categories = merged_pd['categories'].value_counts().head(10)
labels = top_categories.index
plt.pie(top_categories.values, explode=[0.1]+[0]*9, labels=labels, autopct='%1.1f%%', shadow=True, textprops={'fontsize': 10})
plt.title('Distribution of Books Based on Genre')
plt.tight_layout()
plt.savefig(f"{plot_folder}/genre_distribution_pie.png")
plt.close()

# Distribution of Review Scores
plt.figure(figsize=(10, 6))
sns.histplot(merged_pd['review/score'].dropna(), bins=10, kde=True, color='teal')
plt.title("Distribution of Review Scores")
plt.tight_layout()
plt.savefig(f"{plot_folder}/review_score_distribution.png")
plt.close()

# Average Review Score Over Time
plt.figure(figsize=(10, 6))
merged_pd.groupby('reviewYear')['review/score'].mean().plot(marker='o')
plt.title('Average Review Score Over Time')
plt.tight_layout()
plt.savefig(f"{plot_folder}/avg_review_score_over_time.png")
plt.close()

# Reviews Per Year
plt.figure(figsize=(10, 6))
merged_pd.groupby('reviewYear')['review/score'].count().plot(kind='bar')
plt.title('Number of Reviews Per Year')
plt.tight_layout()
plt.savefig(f"{plot_folder}/reviews_per_year.png")
plt.close()

# Top Authors
plt.figure(figsize=(10, 6))
merged_pd['authors'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Authors by Number of Reviews')
plt.tight_layout()
plt.savefig(f"{plot_folder}/top_authors_reviews.png")
plt.close()

# Top Books
plt.figure(figsize=(10, 6))
merged_pd['Title'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Books by Number of Reviews')
plt.tight_layout()
plt.savefig(f"{plot_folder}/top_books_reviews.png")
plt.close()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(merged_pd[['review/score', 'ratingsCount', 'review_length']].dropna().corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Between Numeric Features')
plt.tight_layout()
plt.savefig(f"{plot_folder}/correlation_heatmap.png")
plt.close()

# PCA Visualization
scaler = StandardScaler()
scaled_data = scaler.fit_transform(merged_pd[['review/score', 'ratingsCount', 'review_length']].dropna())
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
pca_merged_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_merged_df, x='PC1', y='PC2')
plt.title('PCA of Review Metrics')
plt.tight_layout()
plt.savefig(f"{plot_folder}/pca_reviews.png")
plt.close()

# Review Length vs Review Score
plt.figure(figsize=(10, 6))
sns.violinplot(x='review/score', y='review_length', data=merged_pd, palette='Pastel1')
plt.title('Distribution of Review Length by Score', fontsize=16)
plt.xlabel('Rating')
plt.ylabel('Review Length')
plt.tight_layout()
plt.savefig(f"{output_folder}//review_length_violinplot.png")
plt.close()

# Review Volume Over Time (Monthly)
plt.figure(figsize=(12, 6))
merged_pd['reviewTimeReadable'] = pd.to_datetime(merged_pd['reviewYear'], format='%Y', errors='coerce')
merged_pd.set_index('reviewTimeReadable').resample('M')['review/score'].count().plot()
plt.title('Review Volume Over Time (Monthly)', fontsize=16)
plt.xlabel('Time')
plt.ylabel('Number of Reviews')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_folder}//review_volume_monthly.png")
plt.close()

print(" All EDA plots generated successfully.")

# ========== Step 7: Clean Exit ==========
spark.stop()
print("\n Spark session stopped cleanly.")
