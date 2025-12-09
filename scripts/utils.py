"""
Utility functions for the project.
"""
import os
import json
import yaml
import pickle
from datetime import datetime
import pandas as pd

def create_project_structure():
    """Create the project directory structure."""
    directories = [
        'data/raw',
        'data/processed',
        'notebooks',
        'scripts',
        'reports/figures',
        'models',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    print("Project structure created successfully!")

def save_results(results, filename):
    """Save model results to file."""
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    with open(filename, 'w') as f:
        json.dump(results, f, default=convert_for_json, indent=2)
    
    print(f"Results saved to {filename}")

def load_config(config_file='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging():
    """Setup basic logging for the project."""
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/project_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)