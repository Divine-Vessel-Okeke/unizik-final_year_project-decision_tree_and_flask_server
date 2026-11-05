# app/train_model.py
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
  accuracy_score,
  precision_score,
  recall_score,
  f1_score,
  classification_report,
  confusion_matrix
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.config import Config

def train_and_evaluate_model():
  print("="*60)
  print("INITIALIZING FRAUD DECISION TREE TRAINING")
  print("="*60)

    # 1. Load the synthetic feature dataset
  data_path = Config.DATA_FILE_PATH
  if not os.path.exists(data_path):
    print(f"[FATAL ERROR] Dataset not found at: {data_path}")
    print("Please execute 'python app/data_generator.py' before running this script.")
    sys.exit(1)

  df = pd.read_csv(data_path)
  print(f"[INFO] Loaded {len(df):,} rows.")

  # 2. Extract feature matrix (X) and target vector (y)
  feature_columns = [
    'device_student_count_24h',
    'page_dwell_time_seconds',
    'is_high_risk_asn',
    'failed_attempts_1h',
    'session_hardware_mismatch',
    'is_off_peak_hour'
  ]

  X = df[feature_columns]
  y = df['target_label']

  # 3. Execute Stratified 80/20 Train-Test Split
  X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
  )
  print(f"[INFO] Train-Test Split Complete -> Training Set: {len(X_train):,} | Testing Set: {len(X_test):,}")

  # 4. Initialize Decision Tree Classifier with depth and sample-size limits
  model = DecisionTreeClassifier(
    criterion='gini',     # Use Gini Impurity to calculate optimal node splits
    max_depth=5,       # Cap tree depth to prevent overfitting to statistical noise
    min_samples_split=10,   # Require at least 10 samples to consider splitting a node
    min_samples_leaf=5,    # Require at least 5 samples to form a terminal classification leaf
    random_state=42      # Fixed seed for reproducible results
  )

  # 5. Fit the model to the training dataset
  print("[INFO] Training DecisionTreeClassifier on the 6 extracted features...")
  model.fit(X_train, y_train)
  print("[SUCCESS] Training complete.")

  # 6. Execute predictions against the unseen testing dataset
  y_pred = model.predict(X_test)

  # 7. Calculate core statistical evaluation metrics
  acc = accuracy_score(y_test, y_pred)
  prec = precision_score(y_test, y_pred)
  rec = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  conf_matrix = confusion_matrix(y_test, y_pred)

  print("\n" + "="*60)
  print("MODEL EVALUATION RESULTS (ON 1,030 UNSEEN TEST RECORDS)")
  print("="*60)
  print(f"Overall Accuracy : {acc * 100:.2f}%")
  print(f"Precision (Fraud) : {prec * 100:.2f}% (When flagged as fraud, how often is it correct?)")
  print(f"Recall (Fraud)  : {rec * 100:.2f}% (Out of all real fraud, how much did we catch?)")
  print(f"F1-Score     : {f1 * 100:.2f}% (Harmonic mean of precision and recall)")
  print("="*60)

  print("\nCONFUSION MATRIX:")
  print(f"True Negatives (Legit predicted Legit)  : {conf_matrix[0][0]:,} | False Positives (Legit predicted Fraud): {conf_matrix[0][1]:,}")
  print(f"False Negatives (Fraud predicted Legit) : {conf_matrix[1][0]:,}  | True Positives (Fraud predicted Fraud) : {conf_matrix[1][1]:,}")

  print("\nDETAILED CLASSIFICATION REPORT:")
  print(classification_report(y_test, y_pred, target_names=['Legitimate (0)', 'Fraudulent (1)']))
  print("="*60)

  # 8. Extract and output mathematical Feature Importance rankings
  importances = model.feature_importances_
  feature_importance_df = pd.DataFrame({
    'Feature_Variable': feature_columns,
    'Gini_Importance': importances
  }).sort_values(by='Gini_Importance', ascending=False).reset_index(drop=True)

  print("\nFEATURE IMPORTANCE RANKINGS (GINI IMPURITY REDUCTION):")
  for idx, row in feature_importance_df.iterrows():
    print(f"#{idx+1} | {row['Feature_Variable'].ljust(26)} : {row['Gini_Importance']*100:.2f}%")
  print("="*60)

  # 9. Serialize and save the trained model binary to disk using joblib
  os.makedirs(Config.MODELS_DIR, exist_ok=True)
  joblib.dump(model, Config.MODEL_PATH, compress=3)
  print(f"\n[SUCCESS] Trained Decision Tree binary exported to: {Config.MODEL_PATH}")
  print("Trained model saved.\n")

if __name__ == '__main__':
  train_and_evaluate_model()