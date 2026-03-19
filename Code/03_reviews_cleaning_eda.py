# ========== Step 0: Setup ==========
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, from_unixtime, isnan, count
from pyspark.sql.types import IntegerType
from pyspark import StorageLevel
import os

# Initialize Spark session
spark = SparkSession.builder \
    .appName("DICProject-Phase2-ReviewsEDA-Cleaning") \
    .config("spark.sql.files.maxPartitionBytes", 134217728) \
    .config("spark.sql.session.timeZone", "America/New_York") \
    .getOrCreate()

# Output folders
output_folder = "/opt/hadoop/project_phase2/output"
os.makedirs(output_folder, exist_ok=True)
summary_file_path = f"{output_folder}/summaries/reviews_eda_cleaning_summary.txt"
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

# ========== Step 2: Load Reviews Dataset ==========

reviews_df = spark.read.parquet("hdfs:///user/n1/parquet/cleaned_reviews")
print("Reviews dataset loaded.")

# ========== Step 3: Initial EDA ==========

# Save Schema
schema_string_reviews = reviews_df._jdf.schema().treeString()
append_text_block(schema_string_reviews, summary_file_path, title="Reviews Dataset Initial Schema")

# Top 5 Rows
top5_reviews = reviews_df.select("User_id", "Title", "review/score", "review/text").limit(5).collect()
with open(summary_file_path, "a") as f:
    f.write("\n" + "="*30 + "\nReviews Dataset Top 5 Rows\n" + "="*30 + "\n\n")
    for row in top5_reviews:
        f.write(str(row.asDict()) + "\n")

# Summary Statistics (review/score)
describe_reviews_rows = reviews_df.select("review/score").describe().collect()
with open(summary_file_path, "a") as f:
    f.write("\n" + "="*30 + "\nReviews Dataset Summary Statistics (review/score)\n" + "="*30 + "\n\n")
    for row in describe_reviews_rows:
        f.write(str(row.asDict()) + "\n")

# Missing Values
reviews_missing = reviews_df.select([
    count(when(col(c).isNull() | isnan(c), c)).alias(c) for c in reviews_df.columns
])
append_dataframe(reviews_missing, summary_file_path, title="Reviews Missing Values (Before Cleaning)")

# Top 20 Users
top_users = reviews_df.groupBy("User_id").count().orderBy("count", ascending=False)
append_dataframe(top_users.limit(20), summary_file_path, title="Top 20 Users (Most Reviews)")

# Top 20 Books
top_books = reviews_df.groupBy("Title").count().orderBy("count", ascending=False)
append_dataframe(top_books.limit(20), summary_file_path, title="Top 20 Books (Most Reviews)")

# ========== Step 4: Data Cleaning ==========

# Drop unwanted columns if they exist
drop_columns = ["Price", "review/helpfulness"]
existing_cols = [c for c in drop_columns if c in reviews_df.columns]
if existing_cols:
    reviews_df = reviews_df.drop(*existing_cols)
    append_text_block(f"Dropped Columns: {existing_cols}", summary_file_path, title="Dropped Columns Info")

# Convert 'review/time' Unix timestamp to readable datetime
if "review/time" in reviews_df.columns:
    reviews_df = reviews_df.withColumn(
        "reviewTimeReadable",
        from_unixtime(col("review/time").cast(IntegerType()))
    )
    append_text_block("Converted 'review/time' to 'reviewTimeReadable'", summary_file_path)

# Drop rows with missing Title or User_id
reviews_df = reviews_df.dropna(subset=["Title", "User_id"])
append_text_block("Dropped rows with missing Title/User_id.", summary_file_path)

# Fill missing profileName
if "profileName" in reviews_df.columns:
    reviews_df = reviews_df.withColumn(
        "profileName", when(col("profileName").isNull(), "Anonymous").otherwise(col("profileName"))
    )

# Fill missing review/summary
if "review/summary" in reviews_df.columns:
    reviews_df = reviews_df.withColumn(
        "review/summary", when(col("review/summary").isNull(), "No Summary Provided").otherwise(col("review/summary"))
    )

# Fill missing review/text
reviews_df = reviews_df.withColumn(
    "review/text", when(col("review/text").isNull(), "No Review Available").otherwise(col("review/text"))
)

print("Missing value handling completed.")

# ========== Step 5: Missing Values After Cleaning ==========

reviews_missing_after = reviews_df.select([
    count(when(col(c).isNull() | isnan(c), c)).alias(c) for c in reviews_df.columns
])

print("\nMissing Values After Cleaning:")
reviews_missing_after.show(truncate=False)

# ========== Step 6: Duplicate Detection and Removal ==========
before_count = reviews_df.count()
reviews_df = reviews_df.dropDuplicates()
after_count = reviews_df.count()

print(f"\nDuplicate Rows Removed: {before_count - after_count}")

# ========== Step 7: Final Verification ==========
print("\nFinal cleaned sample (5 rows):")
reviews_df.select("User_id", "Title", "review/score", "review/text").limit(5).show(truncate=False, vertical=True)

print("\nFinal Schema After Cleaning:")
reviews_df.printSchema()

print("All cleaning steps completed and saved.")

# ========== Step 8: Save Cleaned Reviews DataFrame to HDFS ==========

save_path = "hdfs:///user/n1/parquet/cleaned_reviews_final_new"
reviews_df = reviews_df.repartition(30)
reviews_df.write.mode("overwrite").parquet(save_path)

print(f"Cleaned reviews_df saved successfully at: {save_path}")

# ========== Step 9: Clean Exit ==========

spark.stop()
print("Spark session stopped cleanly.")
