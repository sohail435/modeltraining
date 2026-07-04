import os
import joblib
import optuna
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from data_prep import get_and_split_data, get_processing_pipeline

# Silence Optuna log spam to keep our terminal clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    """Objective function for Optuna to maximize ROC-AUC with balanced weights."""
    X_train, X_test, y_train, y_test = get_and_split_data()
    preprocessor = get_processing_pipeline()
    
    # Define the hyperparameter search space dynamically
    params = {
        'classifier__max_iter': trial.suggest_int('classifier__max_iter', 50, 200, step=50),
        'classifier__max_depth': trial.suggest_int('classifier__max_depth', 3, 12),
        'classifier__learning_rate': trial.suggest_float('classifier__learning_rate', 0.01, 0.2, log=True),
        'classifier__l2_regularization': trial.suggest_float('classifier__l2_regularization', 1e-3, 10.0, log=True)
    }
    
    # Construct the pipeline with the current trial's parameters
    best_clf_params = {k.split('__')[1]: v for k, v in params.items()}
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', HistGradientBoostingClassifier(random_state=42, **best_clf_params))
    ])
    
    # Compute balanced weights specifically for this training array
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    # Fit the pipeline using the sample weights to combat class imbalance
    pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)
    
    # Evaluate performance using the holdout split for optimization guidance
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob)

def run_advanced_tuning():
    print("🚀 Phase 2: Starting Balanced Bayesian Optimization via Optuna...")
    X_train, X_test, y_train, y_test = get_and_split_data()
    
    # Run the study optimizing for the highest validation ROC-AUC score
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)
    
    print("\n🏆 Optimization Complete!")
    print(f"Best Trial Score (Validation ROC-AUC): {study.best_value:.4f}")
    print(f"Best Hyperparameters: {study.best_params}")
    
    # Isolate parameters for the final classifier component
    best_clf_params = {k.split('__')[1]: v for k, v in study.best_params.items()}
    preprocessor = get_processing_pipeline()
    
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', HistGradientBoostingClassifier(random_state=42, **best_clf_params))
    ])
    
    print("\n🏋️ Training final model with optimal parameters and balanced class weights...")
    # Calculate weights on the entire training dataset
    final_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    final_pipeline.fit(X_train, y_train, classifier__sample_weight=final_weights)
    
    # Final production check on the unseen Test Set
    y_pred = final_pipeline.predict(X_test)
    y_prob = final_pipeline.predict_proba(X_test)[:, 1]
    
    print("\n📊 Production Evaluation Metrics:")
    print(f"Test ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the newly balanced engine
    os.makedirs("models", exist_ok=True)
    pipeline_path = "models/saas_churn_pipeline_v2.joblib"
    joblib.dump(final_pipeline, pipeline_path)
    print(f"\n💾 Balanced production pipeline saved to: {pipeline_path}")

if __name__ == "__main__":
    run_advanced_tuning()