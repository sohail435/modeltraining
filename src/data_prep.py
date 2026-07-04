import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_realistic_saas_data():
    """Simulates a messy database pull of SaaS customer metrics."""
    print("1. Simulating real-world SaaS customer dataset...")
    np.random.seed(42)
    n_samples = 2000
    
    usage = np.random.normal(loc=40, scale=15, size=n_samples).clip(1, 120)
    tickets = np.random.poisson(lam=2, size=n_samples)
    spend = np.random.uniform(low=19, high=299, size=n_samples)
    tiers = np.random.choice(['Basic', 'Pro', 'Enterprise'], size=n_samples, p=[0.5, 0.3, 0.2])
    
    # Intentionally inject a few missing values to simulate real data corruption
    usage[np.random.choice(n_samples, size=20, replace=False)] = np.nan
    
    churn_prob = 0.1 + (tickets * 0.15) - (np.nan_to_num(usage, nan=40) * 0.005)
    churn_prob = np.clip(churn_prob, 0, 1)
    churn = np.random.binomial(n=1, p=churn_prob)
    
    df = pd.DataFrame({
        'usage_hours_per_month': usage,
        'support_tickets': tickets,
        'monthly_spend': spend,
        'subscription_tier': tiers,
        'churn': churn
    })
    return df

def get_processing_pipeline():
    """Builds an isolated professional preprocessing blueprint."""
    numeric_features = ['usage_hours_per_month', 'support_tickets', 'monthly_spend']
    categorical_features = ['subscription_tier']

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

def get_and_split_data():
    df = create_realistic_saas_data()
    X = df.drop(columns=['churn'])
    y = df['churn']
    print("2. Splitting dataset into stratified arrays...")
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)