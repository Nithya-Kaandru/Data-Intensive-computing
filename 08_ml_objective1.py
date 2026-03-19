# ========== Review Score Prediction (Regression) ==========

# Step 0: Setup
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import os

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("DICProject-ReviewScorePrediction") \
    .config("spark.sql.session.timeZone", "America/New_York") \
    .getOrCreate()

output_folder = "/opt/hadoop/output/review_score_prediction"
os.makedirs(output_folder, exist_ok=True)

# Step 1: Load the Merged Dataset
merged_df = spark.read.parquet("/opt/hadoop/output/merged_df_parquet")
print("\nLoaded merged_df for review score prediction.")

# Step 2: Feature Engineering
feature_cols = ["ratingsCount", "review_length"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
final_df = assembler.transform(merged_df).select("features", "review/score")

# Step 3: Train-Test Split
train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=42)

# Step 4: Model Training
lr = LinearRegression(labelCol="review/score", featuresCol="features")
lr_model = lr.fit(train_df)

# Save the trained model
lr_model.write().overwrite().save("/opt/hadoop/output/review_score_model")
print("\nReview score regression model trained and saved.")

# Step 5: Model Evaluation
predictions = lr_model.transform(test_df)

evaluator_rmse = RegressionEvaluator(labelCol="review/score", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="review/score", predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print(f"\nRoot Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-Squared (R2 Score): {r2:.4f}")

# Save evaluation results
with open(f"{output_folder}/review_score_regression_results.txt", "w") as f:
    f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
    f.write(f"R-Squared (R2 Score): {r2:.4f}\n")

# Step 6: Clean Exit
spark.stop()
print("\nSpark session stopped cleanly.")


# ========== Personalized Book Recommendation (ALS) ==========

# Step 0: Setup
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
import os

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("DICProject-BookRecommendationALS") \
    .config("spark.sql.session.timeZone", "America/New_York") \
    .getOrCreate()

output_folder = "/opt/hadoop/output/book_recommendation_als"
os.makedirs(output_folder, exist_ok=True)

# Step 1: Load the Merged Dataset
merged_df = spark.read.parquet("/opt/hadoop/output/merged_df_parquet")
print("\nLoaded merged_df for recommendation system.")

# Step 2: Data Preparation
merged_df = merged_df.withColumn("rating", col("review/score").cast("double"))

# Index users and books
user_indexer = StringIndexer(inputCol="User_id", outputCol="userIndex")
book_indexer = StringIndexer(inputCol="Id", outputCol="bookIndex")

merged_df_indexed = user_indexer.fit(merged_df).transform(merged_df)
merged_df_indexed = book_indexer.fit(merged_df_indexed).transform(merged_df_indexed)

ratings_df = merged_df_indexed.select("userIndex", "bookIndex", "rating").dropna()

# Step 3: Train-Test Split
train_df, test_df = ratings_df.randomSplit([0.8, 0.2], seed=42)

# Step 4: Model Training
als = ALS(
    maxIter=5,
    regParam=0.1,
    rank=10,
    userCol="userIndex",
    itemCol="bookIndex",
    ratingCol="rating",
    coldStartStrategy="drop",
    nonnegative=True
)

als_model = als.fit(train_df)

# Save the ALS model
als_model.write().overwrite().save("/opt/hadoop/output/als_recommendation_model")
print("\nALS recommendation model trained and saved.")

# Step 5: Model Evaluation
predictions = als_model.transform(test_df)

evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

rmse = evaluator.evaluate(predictions)

print(f"\nALS Model Root Mean Squared Error (RMSE): {rmse:.4f}")

# Save evaluation results
with open(f"{output_folder}/als_evaluation_results.txt", "w") as f:
    f.write(f"ALS Model RMSE: {rmse:.4f}\n")

# Step 6: Generate Recommendations
user_recs = als_model.recommendForAllUsers(5)
user_recs.show(5, truncate=False)

# Save recommendations
user_recs.write.mode("overwrite").parquet("/opt/hadoop/output/user_recommendations")

# Step 7: Clean Exit
spark.stop()
print("\nSpark session stopped cleanly.")
