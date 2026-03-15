import os
from ingest import load_and_balance_data
from trainer i  mport train_model
from evaluate import evaluate_model

def run_pipeline():
    data_path = "../data/creditcard.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Could not find data file at {data_path}")
        return

    try:
        # Step 1: Ingest
        clean_data = load_and_balance_data(data_path)
        
        # Step 2: Train
        model, X_test_scaled, y_test = train_model(clean_data)
        
        # Step 3: Evaluate
        evaluate_model(model, X_test_scaled, y_test)
        
        print("\nModular Pipeline executed successfully.")

    except Exception as e:
        print(f"\nPipeline failed: {e}")

if __name__ == "__main__":
    run_pipeline()
