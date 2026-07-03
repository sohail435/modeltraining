from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def get_and_split_data():
    print("1. Generating synthetic dataset...")
    # Creates a fake dataset with 1,000 rows and 4 columns
    X, y = make_classification(n_samples=1000, n_features=4, random_state=42)
    
    print("2. Splitting into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test