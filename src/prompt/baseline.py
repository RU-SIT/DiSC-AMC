# %%
from data_processing import load_processed_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import time
from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd

# Function to train and evaluate models
def train_and_evaluate_models(train_data, test_data):
    # Define classifiers to compare
    classifiers = {
        'SVM': SVC(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = {}
    
    # Extract features and labels
    X_train = train_data['stats']
    y_train = train_data['label']
    X_test = test_data['stats']
    y_test = test_data['label']
    
    for name, clf in classifiers.items():
        print(f"Training {name}...")
        start_time = time.time()
        
        # Train the model
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        train_time = time.time() - start_time
        
        results[name] = {
            'accuracy': accuracy,
            'model': clf,
            'predictions': y_pred,
            'training_time': train_time
        }
        
        print(f"{name} - Accuracy: {accuracy:.4f}, Time: {train_time:.2f} seconds")
    
    return results

# %%
if __name__ == "__main__":
    # Load the processed data
    train_data = load_processed_data('../../data/RadioML/train_data.pkl')
    test_data = load_processed_data('../../data/RadioML/test_data.pkl')
    
    # Get modulation types (class names)
    class_names = np.unique(train_data['label'])
    
    # Train and evaluate models
    results = train_and_evaluate_models(train_data, test_data)
# %%
