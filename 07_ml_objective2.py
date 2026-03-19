# ========== Step 0: Setup ==========
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import os
from pyspark.sql.functions import col, when, from_unixtime, isnan, count, avg, regexp_replace, year, to_date, size, split

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("DICProject-SentimentClassification") \
    .config("spark.sql.session.timeZone", "America/New_York") \
    .getOrCreate()

output_folder = "/opt/hadoop/output/sentiment_classification"
os.makedirs(output_folder, exist_ok=True)

# ========== Step 1: Load the Merged Dataset ==========
merged_df = spark.read.parquet("/opt/hadoop/output/merged_df_parquet")

print("\nLoaded merged_df for sentiment classification.")

# ========== Step 2: Label Preparation ==========
# Create binary sentiment label based on review score
merged_df = merged_df.withColumn(
    "sentiment_label",
    when(col("review/score") >= 4, "positive")
    .otherwise("negative")
)

# ========== Step 3: NLP Pipeline ==========
tokenizer = Tokenizer(inputCol="review/text", outputCol="tokens")
remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
hashingTF = HashingTF(inputCol="filtered_tokens", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features")

label_indexer = StringIndexer(inputCol="sentiment_label", outputCol="label")

lr = LogisticRegression(maxIter=20, regParam=0.01)

pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, label_indexer, lr])

# ========== Step 4: Train-Test Split ==========
train_df, test_df = merged_df.randomSplit([0.8, 0.2], seed=42)

# ========== Step 5: Model Training ==========
model = pipeline.fit(train_df)

# Save the model
model.write().overwrite().save("/opt/hadoop/output/sentiment_model")

print("\nSentiment model trained and saved.")

# ========== Step 6: Model Evaluation ==========
predictions = model.transform(test_df)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"\nSentiment Classification Accuracy: {accuracy:.4f}")

with open(f"{output_folder}/sentiment_classification_results.txt", "w") as f:
    f.write(f"Accuracy on Test Set: {accuracy:.4f}\n")

# ========== Step 7: Clean Exit ==========
spark.stop()
print("\nSpark session stopped cleanly.")