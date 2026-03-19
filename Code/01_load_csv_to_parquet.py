from pyspark.sql import SparkSession

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("DICProject") \
    .config("spark.sql.session.timeZone", "America/New_York") \
    .getOrCreate()

# Load from CSV
books_df = spark.read.csv("hdfs:///user/n1/books_data.csv", header=True, inferSchema=True)
reviews_df = spark.read.csv("hdfs:///user/n1/Books_rating.csv", header=True, inferSchema=True)

# Save as Parquet
books_df.write.mode("overwrite").parquet("hdfs:///user/n1/parquet/cleaned_books")
reviews_df.write.mode("overwrite").parquet("hdfs:///user/n1/parquet/cleaned_reviews")

print("Saved CSVs as Parquet successfully!")

spark.stop()
print("Spark session stopped.")