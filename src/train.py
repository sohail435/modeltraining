import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from data_prep import get_and_split_data

def train_and_save_model():
    # 1. Get the cleaned data
    X_train, X_test, y_train, y_test = get_and_split_data()
    
    # 2. Set up the Random Search
    print("3. Tuning the model...")
    rf = RandomForestClassifier(random_state=42)
    param_distributions = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20]}
    
    search = RandomizedSearchCV(rf, param_distributions, n_iter=3, cv=3, random_state=42)
    search.fit(X_train, y_train)
    
    # 3. Extract the winner
    best_model = search.best_estimator_
    print(f"4. Best hyperparameters found: {search.best_params_}")
    
    # 4. SAVE THE MODEL
    os.makedirs("models", exist_ok=True)
    model_path = "models/random_forest_v1.joblib"
    joblib.dump(best_model, model_path)
    print(f"5. Model saved successfully to {model_path}")

if __name__ == "__main__":
    train_and_save_model()