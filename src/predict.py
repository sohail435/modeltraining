import joblib
import numpy as np

def make_prediction(new_data):
    # 1. LOAD THE SAVED MODEL
    model_path = "../models/random_forest_v1.joblib"
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # 2. Make a prediction
    prediction = model.predict(new_data)
    return prediction

if __name__ == "__main__":
    # Simulating brand new data coming from a user or database (1 row, 4 columns)
    dummy_new_data = np.array([[0.5, -1.2, 3.4, 0.1]])
    
    result = make_prediction(dummy_new_data)
    print(f"Prediction for the new data point: Class {result[0]}")