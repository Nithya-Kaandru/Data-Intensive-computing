# ========== Step 0: Setup ==========
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, count, regexp_replace
from pyspark.sql.types import IntegerType
from pyspark import StorageLevel
import os

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("DICProject-Phase2-BooksEDA-Cleaning") \
    .config("spark.sql.session.timeZone", "America/New_York") \
    .getOrCreate()

# Define output folders
output_folder = "/opt/hadoop/project_phase2/output"
os.makedirs(output_folder, exist_ok=True)
summary_file_path = f"{output_folder}/summaries/books_eda_cleaning_summary.txt"
open(summary_file_path, "w").close()  # fresh overwrite

# ========== Step 1: Utility Functions ==========

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

# ========== Step 2: Load Books Dataset ==========

books_df = spark.read.parquet("hdfs:///user/n1/parquet/cleaned_books")
books_df.persist(StorageLevel.MEMORY_AND_DISK)
_ = books_df.count()
print("Books dataset loaded and cached.")

# ========== Step 3: Initial EDA ==========

# Save Schema
schema_string_books = books_df._jdf.schema().treeString()
append_text_block(schema_string_books, summary_file_path, title="Books Dataset Initial Schema")

# Top 5 Rows
top5_rows = books_df.select("Title", "authors", "categories", "ratingsCount").limit(5).collect()
with open(summary_file_path, "a") as f:
    f.write("\n" + "="*30 + "\nBooks Dataset Top 5 Rows\n" + "="*30 + "\n\n")
    for row in top5_rows:
        f.write(str(row.asDict()) + "\n")

# Summary Statistics (DESCRIBE)
describe_rows = books_df.describe().collect()
with open(summary_file_path, "a") as f:
    f.write("\n" + "="*30 + "\nBooks Dataset Summary Statistics (DESCRIBE)\n" + "="*30 + "\n\n")
    for row in describe_rows:
        f.write(str(row.asDict()) + "\n")

# Detailed Summary (SUMMARY)
summary_rows = books_df.summary().collect()
with open(summary_file_path, "a") as f:
    f.write("\n" + "="*30 + "\nBooks Dataset Detailed Summary (SUMMARY)\n" + "="*30 + "\n\n")
    for row in summary_rows:
        f.write(str(row.asDict()) + "\n")

# Missing Values
books_missing = books_df.select([
    count(when(col(c).isNull() | isnan(c), c)).alias(c) for c in books_df.columns
])
append_dataframe(books_missing, summary_file_path, title="Books Missing Values (Before Cleaning)")

# Top 20 Authors
top_authors = books_df.groupBy("authors").count().orderBy("count", ascending=False)
append_dataframe(top_authors.limit(20), summary_file_path, title="Top 20 Authors")

# Top 20 Categories
top_categories = books_df.groupBy("categories").count().orderBy("count", ascending=False)
append_dataframe(top_categories.limit(20), summary_file_path, title="Top 20 Categories")

# ========== Step 4: Data Cleaning and Type Conversion ==========

# Convert ratingsCount to IntegerType
books_df = books_df.withColumn("ratingsCount", col("ratingsCount").cast(IntegerType()))
append_text_block(books_df._jdf.schema().treeString(), summary_file_path, title="Schema After Type Conversion")

# Drop rows where Title is missing
books_df = books_df.dropna(subset=["Title"])

# Fill other missing fields
books_df = books_df.withColumn("description", when(col("description").isNull(), "No Description Available").otherwise(col("description"))) \
    .withColumn("authors", when(col("authors").isNull(), "Unknown Author").otherwise(col("authors"))) \
    .withColumn("image", when(col("image").isNull(), "https://example.com/default_image.jpg").otherwise(col("image"))) \
    .withColumn("previewLink", when(col("previewLink").isNull(), "No Preview Available").otherwise(col("previewLink"))) \
    .withColumn("publisher", when(col("publisher").isNull(), "Unknown Publisher").otherwise(col("publisher"))) \
    .withColumn("publishedDate", when(col("publishedDate").isNull(), "Unknown Date").otherwise(col("publishedDate"))) \
    .withColumn("infoLink", when(col("infoLink").isNull(), "No Info Available").otherwise(col("infoLink"))) \
    .withColumn("categories", when(col("categories").isNull(), "Uncategorized").otherwise(col("categories"))) \
    .withColumn("ratingsCount", when(col("ratingsCount").isNull(), 0).otherwise(col("ratingsCount")))

# Missing values after cleaning
books_missing_after = books_df.select([
    count(when(col(c).isNull() | isnan(c), c)).alias(c) for c in books_df.columns
])
append_dataframe(books_missing_after, summary_file_path, title="Books Missing Values (After Cleaning)")

print("Missing value handling completed.")

# ========== Step 5: Outlier Detection ==========

quantiles = books_df.approxQuantile("ratingsCount", [0.01, 0.99], 0.0)
lower_bound, upper_bound = quantiles
append_text_block(f"RatingsCount Outlier Bounds: 1% = {lower_bound}, 99% = {upper_bound}", summary_file_path, title="Outlier Bounds")

# Winsorize ratingsCount
books_df = books_df.withColumn(
    "ratingsCount",
    when(col("ratingsCount") < lower_bound, lower_bound)
    .when(col("ratingsCount") > upper_bound, upper_bound)
    .otherwise(col("ratingsCount"))
)

print("Outlier handling completed.")

# ========== Step 6: Duplicate Handling ==========

before_count = books_df.count()
books_df = books_df.dropDuplicates()
after_count = books_df.count()

append_text_block(f"Duplicate Rows Removed: {before_count - after_count}", summary_file_path, title="Duplicate Removal Summary")

# ========== Step 7: Final Verification ==========

final_sample = books_df.select("Title", "authors", "categories", "ratingsCount").limit(10)
append_dataframe(final_sample, summary_file_path, title="Final Sample After Cleaning")

append_text_block(books_df._jdf.schema().treeString(), summary_file_path, title="Final Schema After Cleaning")

print("All cleaning steps completed and saved.")

# ========== Step 8: Save Cleaned Books DataFrame to HDFS ==========

save_path = "hdfs:///user/n1/parquet/cleaned_books_final_new"
books_df.write.mode("overwrite").parquet(save_path)

print(f"Cleaned books_df saved successfully at: {save_path}")

# ========== Step 9: Clean Exit ==========

spark.stop()
print("Spark session stopped cleanly.")