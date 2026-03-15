# PySpark Financial Risk Engine: Fraud Detection

## 📌 Project Overview
Credit card fraud costs financial institutions billions annually. The core challenge in building a fraud detection model is the massive class imbalance—legitimate transactions vastly outnumber fraudulent ones. 

This project demonstrates a scalable data engineering and machine learning pipeline. It utilizes **PySpark** to process a highly imbalanced dataset of anonymized credit card transactions, engineering a balanced dataset to train a **scikit-learn** Random Forest classifier. 

**Business Value:** This pipeline simulates a financial risk decision model optimized for high recall, ensuring maximum fraud detection while managing false positives.

## 🛠️ Tech Stack
* **Data Engineering & Distributed Processing:** Apache Spark (PySpark), Pandas
* **Machine Learning:** scikit-learn (RandomForestClassifier)
* **Environment:** Python 3.x, Jupyter Notebook / Google Colab

## 🏗️ Architecture & Workflow
1. **Data Ingestion:** Loaded 284,000+ transaction records using PySpark.
2. **Distributed Data Balancing:** Executed under-sampling on the majority class (legitimate transactions) across the Spark cluster to match the minority class (fraud), preventing model bias.
3. **Feature Scaling:** Standardized transaction amounts and time features to ensure uniform model weighting.
4. **Model Training:** Transitioned the balanced, optimized dataset to Pandas for training a scikit-learn Random Forest model.
5. **Evaluation:** Evaluated model performance using Recall, Precision, and a Confusion Matrix to align with risk management objectives.

## 📂 Project Structure
```text
├── data/                   # Data directory (Dataset excluded via .gitignore)
├── notebooks/              # Jupyter notebooks for EDA and prototyping
│   └── 01_pyspark_pipeline.ipynb
├── src/                    # Production Python scripts
│   ├── ingest.py           # PySpark data loading and sampling
│   ├── train.py            # Model training and scaling
│   └── evaluate.py         # Evaluation metrics and reporting
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignored files and directories
└── README.md               # Project documentation
