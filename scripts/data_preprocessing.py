"""
Data preprocessing module for HK immigration passenger traffic data.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import holidays

class DataPreprocessor:
    """Preprocess HK immigration passenger traffic data."""
    
    def __init__(self, data_path):
        """
        Initialize the preprocessor.
        
        Parameters:
        -----------
        data_path : str
            Path to the raw data file
        """
        self.data_path = data_path
        self.hk_holidays = holidays.HK()
        self.df = None
        
    def load_data(self):
        """Load raw data from CSV."""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path, encoding='utf-8')
        print(f"Data shape: {self.df.shape}")
        return self.df
    
    def clean_data(self):
        """Clean the loaded data."""
        print("Cleaning data...")
        
        # Basic cleaning
        self.df.dropna(subset=['passenger_count'], inplace=True)
        self.df['passenger_count'] = pd.to_numeric(self.df['passenger_count'], errors='coerce')
        
        # Convert date if exists
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
            self.df['date'].fillna(method='ffill', inplace=True)
        
        # Remove duplicates
        self.df.drop_duplicates(inplace=True)
        
        return self.df
    
    def engineer_features(self):
        """Create new features from the data."""
        print("Engineering features...")
        
        if 'date' in self.df.columns:
            # Temporal features
            self.df['year'] = self.df['date'].dt.year
            self.df['month'] = self.df['date'].dt.month
            self.df['day'] = self.df['date'].dt.day
            self.df['day_of_week'] = self.df['date'].dt.dayofweek  # Monday=0, Sunday=6
            self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
            
            # Holiday features
            self.df['is_holiday'] = self.df['date'].apply(
                lambda x: x in self.hk_holidays
            ).astype(int)
            
            # Seasonal features
            self.df['quarter'] = self.df['date'].dt.quarter
            self.df['day_of_year'] = self.df['date'].dt.dayofyear
            
            # Rolling statistics
            self.df['rolling_7day_mean'] = self.df.groupby('immigration_point')['passenger_count']\
                .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
        
        return self.df
    
    def prepare_for_ml(self, target_col='passenger_count'):
        """Prepare data for machine learning."""
        print("Preparing for ML...")
        
        # Create target variable for classification
        traffic_threshold = self.df[target_col].quantile(0.75)
        self.df['traffic_level'] = (self.df[target_col] > traffic_threshold).astype(int)
        
        # Select features for modeling
        numeric_features = ['month', 'day', 'day_of_week', 'is_weekend', 
                           'is_holiday', 'quarter', 'day_of_year']
        
        # Keep only features that exist in the dataframe
        available_features = [f for f in numeric_features if f in self.df.columns]
        
        return self.df, available_features, target_col
    
    def run_pipeline(self):
        """Run the complete preprocessing pipeline."""
        self.load_data()
        self.clean_data()
        self.engineer_features()
        df, features, target = self.prepare_for_ml()
        
        print("Preprocessing completed!")
        print(f"Final shape: {df.shape}")
        print(f"Features: {features}")
        
        return df, features, target