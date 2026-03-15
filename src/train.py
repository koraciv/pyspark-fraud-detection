import argparse
from pyspark.sql import SparkSession
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

def main():
    print("Initializing PySpark Session...")
    spark = SparkSession.builder \
        .appName("FraudDetectionPipeline") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    try:
        # 1. Data Ingestion
        print("Loading transaction data...")
        df = spark.read.csv("../data/creditcard.csv", header=True, inferSchema=True)

        # 2. Distributed Data Balancing
        print("Balancing classes via distributed under-sampling...")
        fraud_df = df.filter(df["Class"] == 1)
        legit_df = df.filter(df["Class"] == 0)
        
        fraud_count = fraud_df.count()
        sampled_legit_df = legit_df.sample(withReplacement=False, fraction=fraud_count / legit_df.count(), seed=42)
        
        balanced_df = fraud_df.union(sampled_legit_df)

        # 3. Handoff to Local Memory (Pandas)
        print("Transforming to Pandas for scikit-learn processing...")
        pandas_df = balanced_df.toPandas()

        X = pandas_df.drop('Class', axis=1)
        y = pandas_df['Class']

        # 4. Feature Scaling & Split
        print("Scaling features and splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 5. Model Training
        print("Training Random Forest Classifier...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # 6. Evaluation
        predictions = model.predict(X_test_scaled)
        
        print("\n================ PIPELINE RESULTS ================")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))

    except Exception as e:
        print(f"\nPipeline Error: {e}")
        print("Ensure 'creditcard.csv' is placed in the '../data/' directory.")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
