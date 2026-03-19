# ========== Step 0: Setup ==========
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, from_unixtime, isnan, count, avg, regexp_replace, year, to_date, size, split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
from textblob import TextBlob
from nltk.corpus import stopwords
from wordcloud import WordCloud
import re
from collections import Counter
import nltk

# Output Folder Setup
output_folder = "/content/DA"
os.makedirs(output_folder, exist_ok=True)

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("DICProject-Phase2-Merge-Clean-EDA") \
    .config("spark.sql.session.timeZone", "America/New_York") \
    .config("spark.sql.files.maxPartitionBytes", 134217728) \
    .getOrCreate()

# ========== Step 1: Load Merged Dataset ==========
merged_df = spark.read.parquet("/opt/hadoop/output/cleaned_merged_books_reviews_final")

# ========== Step 2: Data Formatting ==========
merged_df = merged_df.withColumn(
    "categories",
    regexp_replace("categories", r"[\[\]']", "")
)

# ========== Objective 1: Top-Rated and Most-Reviewed Books ==========

book_stats_df = merged_df.groupBy("Title").agg(
    count("review/score").alias("review_count"),
    avg("review/score").alias("avg_review_score")
)

most_reviewed_books = book_stats_df.orderBy(col("review_count").desc())
top_rated_books = book_stats_df.filter(col("review_count") > 50).orderBy(col("avg_review_score").desc())

most_reviewed_pd = most_reviewed_books.limit(10).toPandas()
top_rated_pd = top_rated_books.limit(10).toPandas()

# Plot 1 - Most Reviewed Books
plt.figure(figsize=(12, 6))
sns.barplot(x="review_count", y="Title", data=most_reviewed_pd, palette="Blues_d")
plt.title("Top 10 Most-Reviewed Books", fontsize=16)
plt.tight_layout()
plt.savefig(f"{output_folder}/most_reviewed_books.png")
plt.close()

# Plot 2 - Top Rated Books
plt.figure(figsize=(12, 6))
sns.barplot(x="avg_review_score", y="Title", data=top_rated_pd, palette="Greens_d")
plt.title("Top 10 Top-Rated Books (Min 50 Reviews)", fontsize=16)
plt.tight_layout()
plt.savefig(f"{output_folder}/top_rated_books.png")
plt.close()

# ========== Objective 2: Extract Insights from Review Sentiments ==========

# Ensure necessary download
nltk.download('stopwords')

# Create output folder
output_folder = "/opt/hadoop/output/DA"
os.makedirs(output_folder, exist_ok=True)

# Define sentiment summary file
sentiment_summary_file = f"{output_folder}/sentiment_summary.txt"
open(sentiment_summary_file, "w").close()  # Clean old file

# Helper functions
def append_text_block(text, filepath, title=""):
    with open(filepath, "a") as f:
        if title:
            f.write(f"\n{'='*30}\n{title}\n{'='*30}\n\n")
        f.write(text)
        f.write("\n\n")

def append_list_block(items, filepath, title=""):
    with open(filepath, "a") as f:
        if title:
            f.write(f"\n{'='*30}\n{title}\n{'='*30}\n\n")
        for word, count in items:
            f.write(f"{word}: {count}\n")
        f.write("\n")

# Step 1: Sample and Classify Sentiments
review_sample_df = merged_df.sample(fraction=0.1)
review_sample_pd = review_sample_df.select("review/text").dropna().toPandas()

def classify_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.1:
        return "Positive"
    elif analysis.sentiment.polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

review_sample_pd["sentiment"] = review_sample_pd["review/text"].apply(classify_sentiment)

# Save Sentiment Distribution
sentiment_counts = review_sample_pd["sentiment"].value_counts().to_dict()

append_text_block(
    "Sampled review/text ready for sentiment analysis.\n\nSentiment classification done.\n\nSentiment counts:",
    sentiment_summary_file,
    title="Sentiment Analysis Summary"
)

for sentiment, count in sentiment_counts.items():
    append_text_block(f"{sentiment}: {count}", sentiment_summary_file)

# Step 2: Clean Texts and Extract Top Keywords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def get_top_keywords(reviews):
    words = []
    for text in reviews:
        cleaned = clean_text(text)
        for word in cleaned.split():
            if word not in stop_words:
                words.append(word)
    return Counter(words).most_common(20)

positive_reviews = review_sample_pd[review_sample_pd["sentiment"] == "Positive"]["review/text"]
negative_reviews = review_sample_pd[review_sample_pd["sentiment"] == "Negative"]["review/text"]

top_positive_keywords = get_top_keywords(positive_reviews)
top_negative_keywords = get_top_keywords(negative_reviews)

# Save Top Keywords
append_list_block(top_positive_keywords, sentiment_summary_file, title="Top Positive Keywords")
append_list_block(top_negative_keywords, sentiment_summary_file, title="Top Negative Keywords")

# Step 3: Generate WordClouds
positive_text = " ".join(positive_reviews)
negative_text = " ".join(negative_reviews)

def generate_wordcloud(text, title, output_path):
    wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="Greens").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

generate_wordcloud(positive_text, "Positive Reviews WordCloud", f"{output_folder}/positive_wordcloud.png")
generate_wordcloud(negative_text, "Negative Reviews WordCloud", f"{output_folder}/negative_wordcloud.png")

print("\nSentiment summary and wordclouds saved successfully!")

# ========== Objective 3: Impact of Popularity (ratingsCount) on Review Score ==========

sample_df = merged_df.select("ratingsCount", "review/score").dropna()
sample_pd = sample_df.toPandas()

plt.figure(figsize=(10, 6))
sns.scatterplot(x="ratingsCount", y="review/score", data=sample_pd, alpha=0.6, edgecolor="black")
plt.title("Impact of Ratings Count on Review Score")
plt.tight_layout()
plt.savefig(f"{output_folder}/ratingscount_vs_reviewscore.png")
plt.close()

# ========== Objective 4: Analyze Reading Trends Across Categories ==========

category_review_counts = merged_df.groupBy("categories").agg(
    count("review/score").alias("review_count")
).orderBy(col("review_count").desc())

top_categories_pd = category_review_counts.limit(10).toPandas()

plt.figure(figsize=(10, 6))
sns.barplot(y=top_categories_pd['categories'], x=top_categories_pd['review_count'], palette="Blues_d")
plt.title("Top 10 Most-Reviewed Categories")
plt.tight_layout()
plt.savefig(f"{output_folder}/top_categories_reviews.png")
plt.close()

# Smoothed Trend Plot
trend_df = merged_df.groupBy("categories", "reviewYear").agg(
    count("review/score").alias("review_count")
)
trend_pd = trend_df.toPandas()

year_range = list(range(1995, 2025))
trend_pivot = trend_pd.pivot_table(index="categories", columns="reviewYear", values="review_count", fill_value=0).reindex(columns=year_range, fill_value=0).reset_index()

plt.figure(figsize=(14, 8))
for category in top_categories_pd["categories"]:
    if category in trend_pivot["categories"].values:
        row = trend_pivot[trend_pivot["categories"] == category]
        years = year_range
        counts = row.iloc[0, 1:].values
        plt.plot(years, counts, label=category)

plt.title("Review Trends Over Years for Top Categories (Smoothed)")
plt.xlabel("Year")
plt.ylabel("Number of Reviews")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_folder}/category_trend_over_time_smoothed.png")
plt.close()

# ========== Objective 5: Understand User Engagement Patterns ==========

user_review_counts = merged_df.groupBy("User_id").agg(
    count("review/score").alias("num_reviews")
)

user_review_pd = user_review_counts.select("num_reviews").sample(fraction=0.1).toPandas()

plt.figure(figsize=(10, 6))
sns.histplot(user_review_pd["num_reviews"], bins=30, kde=True, color="purple")
plt.title("Distribution of Number of Reviews per User")
plt.xlim(0, 200)
plt.tight_layout()
plt.savefig(f"{output_folder}/user_reviews_distribution.png")
plt.close()

review_length_pd = merged_df.select("review_length").dropna().sample(fraction=0.1).toPandas()

plt.figure(figsize=(10, 6))
sns.histplot(review_length_pd["review_length"], bins=50, kde=True, color="green")
plt.title("Distribution of Review Lengths (Words)")
plt.xlim(0, 2000)
plt.tight_layout()
plt.savefig(f"{output_folder}/review_length_distribution.png")
plt.close()

category_engagement_pd = category_review_counts.limit(10).toPandas()

plt.figure(figsize=(10, 6))
sns.barplot(y=category_engagement_pd['categories'], x=category_engagement_pd['review_count'], palette="Blues_r")
plt.title("Top 10 Categories by Number of Reviews")
plt.tight_layout()
plt.savefig(f"{output_folder}/category_engagement.png")
plt.close()

author_review_counts = merged_df.groupBy("authors").agg(
    count("review/score").alias("num_reviews")
).orderBy(col("num_reviews").desc())

top_authors_pd = author_review_counts.limit(10).toPandas()
plt.figure(figsize=(10, 6))
sns.barplot(y=top_authors_pd['authors'], x=top_authors_pd['num_reviews'], palette="Purples_r")
plt.title("Top 10 Authors by Number of Reviews")
plt.tight_layout()
plt.savefig(f"{output_folder}/author_engagement.png")
plt.close()

# ========== Step Final: Clean Exit ==========
spark.stop()
print("\n Spark session stopped cleanly.")
