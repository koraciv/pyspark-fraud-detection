from pyspark.sql import SparkSession
import pandas as pd

def load_and_balance_data(file_path):
    print("Initializing PySpark Session for Ingestion...")
    spark = SparkSession.builder \
        .appName("FraudDetection_Ingest") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    try:
        print("Loading transaction data...")
        df = spark.read.csv(file_path, header=True, inferSchema=True)

        print("Balancing classes via distributed under-sampling...")
        fraud_df = df.filter(df["Class"] == 1)
        legit_df = df.filter(df["Class"] == 0)
        
        fraud_count = fraud_df.count()
        sampled_legit_df = legit_df.sample(withReplacement=False, fraction=fraud_count / legit_df.count(), seed=42)
        
        balanced_df = fraud_df.union(sampled_legit_df)

        print("Transforming to Pandas for downstream processing...")
        pandas_df = balanced_df.toPandas()
        
        return pandas_df

    finally:
        spark.stop()
