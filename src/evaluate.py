from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test_scaled, y_test):
    print("Generating predictions on test data...")
    predictions = model.predict(X_test_scaled)
    
    print("\n================ PIPELINE RESULTS ================")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
