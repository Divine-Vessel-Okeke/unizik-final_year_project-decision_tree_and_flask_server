# app/config.py

import os

class Config:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    TRAINING_DATA_FILE_NAME = 'synthetic_fraud_features.csv'
    TRAINING_DATA_FILE_PATH = os.path.join(DATA_DIR, TRAINING_DATA_FILE_NAME)

    MODEL_FILE_NAME = 'fraud_decision_tree.joblib'
    MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILE_NAME)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)