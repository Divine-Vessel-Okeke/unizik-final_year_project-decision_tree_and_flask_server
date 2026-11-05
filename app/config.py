# /app/config.py

import os

class Config:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    
    MODEL_FILE_NAME = 'unizik_fraud_decision_tree.joblib'
    MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILE_NAME)
    
    # Ensure directories exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)