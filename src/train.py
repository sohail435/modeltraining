import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from data_prep import get_and_split_data, get_processing_pipeline

def train_and_save_pipeline():
    # 1. Fetch data arrays
    X_train, X_test, y_train, y_test = get_and_split_data()
    
    # 2. Get the preprocessing lanes
    preprocessor = get_processing_pipeline()
    
    # 3. Create the unified master pipeline
    # This combines features engineering + model into a single engine
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', HistGradientBoostingClassifier(random_state=42))
    ])
    
    # 4. Set up hyperparameter tuning for components INSIDE the pipeline
    # Note the syntax: componentName__parameterName
    print("3. Optimization engine spinning up...")
    param_distributions = {
        'classifier__max_iter': [50, 100, 150],
        'classifier__max_depth': [3, 5, 10],
        'classifier__learning_rate': [0.01, 0.1, 0.2]
    }
    
    search = RandomizedSearchCV(
        full_pipeline, 
        param_distributions, 
        n_iter=5, 
        cv=3, 
        scoring='roc_auc', 
        random_state=42
    )
    
    print("4. Fitting unified pipeline across search space...")
    search.fit(X_train, y_train)
    
    best_pipeline = search.best_estimator_
    print(f"5. Optimized hyper-parameters found: {search.best_params_}")
    print(f"   Best Cross-Validation ROC-AUC Score: {search.best_score_:.4f}")
    
    # 5. Save the entire operational structure safely
    os.makedirs("models", exist_ok=True)
    pipeline_path = "models/saas_churn_pipeline_v1.joblib"
    joblib.dump(best_pipeline, pipeline_path)
    print(f"6. Full production pipeline successfully saved to: {pipeline_path}")

if __name__ == "__main__":
    train_and_save_pipeline()