"""
Machine learning models for passenger traffic analysis.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                            accuracy_score, classification_report, confusion_matrix,
                            silhouette_score)

class TrafficModels:
    """Implement all required ML models."""
    
    def __init__(self, df, features, target):
        """
        Initialize models with data.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Preprocessed data
        features : list
            List of feature column names
        target : str
            Target column name
        """
        self.df = df
        self.features = features
        self.target = target
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare train-test split."""
        print("Preparing train-test split...")
        
        X = self.df[self.features].copy()
        y = self.df[self.target].copy()
        
        # For classification, create binary target
        if y.nunique() > 10:  # If continuous, create classification target
            self.classification_target = 'traffic_level'
            threshold = y.quantile(0.75)
            y_class = (y > threshold).astype(int)
        else:
            self.classification_target = self.target
            y_class = y
        
        # Split for regression
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Split for classification
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
            X, y_class, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_reg_scaled = self.scaler.fit_transform(X_train_reg)
        X_test_reg_scaled = self.scaler.transform(X_test_reg)
        X_train_clf_scaled = self.scaler.fit_transform(X_train_clf)
        X_test_clf_scaled = self.scaler.transform(X_test_clf)
        
        self.data = {
            'regression': {
                'X_train': X_train_reg_scaled, 'X_test': X_test_reg_scaled,
                'y_train': y_train_reg, 'y_test': y_test_reg
            },
            'classification': {
                'X_train': X_train_clf_scaled, 'X_test': X_test_clf_scaled,
                'y_train': y_train_clf, 'y_test': y_test_clf
            },
            'original': {
                'X': X, 'y': y, 'y_class': y_class
            }
        }
        
        return self.data
    
    def linear_regression(self):
        """Implement Linear Regression."""
        print("Training Linear Regression...")
        
        data = self.data['regression']
        model = LinearRegression()
        model.fit(data['X_train'], data['y_train'])
        
        # Predictions
        y_pred = model.predict(data['X_test'])
        
        # Metrics
        mse = mean_squared_error(data['y_test'], y_pred)
        mae = mean_absolute_error(data['y_test'], y_pred)
        r2 = r2_score(data['y_test'], y_pred)
        
        self.models['linear_regression'] = model
        self.results['linear_regression'] = {
            'model': model,
            'predictions': y_pred,
            'metrics': {'MSE': mse, 'MAE': mae, 'R2': r2},
            'coefficients': dict(zip(self.features, model.coef_))
        }
        
        print(f"Linear Regression Results:")
        print(f"  MSE: {mse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  R²: {r2:.4f}")
        
        return self.results['linear_regression']
    
    def logistic_regression(self):
        """Implement Logistic Regression."""
        print("Training Logistic Regression...")
        
        data = self.data['classification']
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(data['X_train'], data['y_train'])
        
        # Predictions
        y_pred = model.predict(data['X_test'])
        y_pred_proba = model.predict_proba(data['X_test'])[:, 1]
        
        # Metrics
        accuracy = accuracy_score(data['y_test'], y_pred)
        report = classification_report(data['y_test'], y_pred, output_dict=True)
        cm = confusion_matrix(data['y_test'], y_pred)
        
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'metrics': {'accuracy': accuracy},
            'classification_report': report,
            'confusion_matrix': cm
        }
        
        print(f"Logistic Regression Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        
        return self.results['logistic_regression']
    
    def svm_classification(self, kernel='rbf'):
        """Implement SVM for classification."""
        print(f"Training SVM with {kernel} kernel...")
        
        data = self.data['classification']
        model = SVC(kernel=kernel, probability=True, random_state=42)
        model.fit(data['X_train'], data['y_train'])
        
        # Predictions
        y_pred = model.predict(data['X_test'])
        y_pred_proba = model.predict_proba(data['X_test'])[:, 1]
        
        # Metrics
        accuracy = accuracy_score(data['y_test'], y_pred)
        report = classification_report(data['y_test'], y_pred, output_dict=True)
        cm = confusion_matrix(data['y_test'], y_pred)
        
        self.models[f'svm_{kernel}'] = model
        self.results[f'svm_{kernel}'] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'metrics': {'accuracy': accuracy},
            'classification_report': report,
            'confusion_matrix': cm
        }
        
        print(f"SVM ({kernel}) Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        
        return self.results[f'svm_{kernel}']
    
    def kmeans_clustering(self, n_clusters=3):
        """Implement K-means clustering."""
        print(f"Training K-means with {n_clusters} clusters...")
        
        X = self.data['original']['X']
        X_scaled = self.scaler.fit_transform(X)
        
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = model.fit_predict(X_scaled)
        
        # Metrics
        silhouette = silhouette_score(X_scaled, clusters)
        
        # Analyze clusters
        self.df['cluster'] = clusters
        cluster_stats = self.df.groupby('cluster')[self.features + [self.target]].mean()
        
        self.models['kmeans'] = model
        self.results['kmeans'] = {
            'model': model,
            'clusters': clusters,
            'metrics': {'silhouette_score': silhouette},
            'cluster_statistics': cluster_stats,
            'cluster_counts': self.df['cluster'].value_counts().to_dict()
        }
        
        print(f"K-means Results:")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Cluster distribution: {self.results['kmeans']['cluster_counts']}")
        
        return self.results['kmeans']
    
    def run_all_models(self):
        """Run all required models."""
        print("=" * 50)
        print("Running all machine learning models...")
        print("=" * 50)
        
        self.prepare_data()
        
        results = {}
        results['linear_regression'] = self.linear_regression()
        print("-" * 30)
        
        results['logistic_regression'] = self.logistic_regression()
        print("-" * 30)
        
        results['svm_rbf'] = self.svm_classification(kernel='rbf')
        print("-" * 30)
        
        results['kmeans'] = self.kmeans_clustering(n_clusters=3)
        print("=" * 50)
        
        print("All models completed!")
        return results