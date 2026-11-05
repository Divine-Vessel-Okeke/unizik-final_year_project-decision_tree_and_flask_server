# /app/data_generator.py

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.config import Config

def generate_unizik_dataset() -> pd.DataFrame:
    np.random.seed(42)
    print("Initializing synthetic data generation with deep multi-variable overlap...")
    
    # 1. Class 0A: Standard Individual Students (3,000 rows)
    # Fast autofill students can finish in 8s; 8% use VPNs for privacy; up to 3 network declines allowed
    n_student = 3000
    student_data = {
        'device_student_count_24h': np.random.choice([1, 2, 3, 4], size=n_student, p=[0.75, 0.15, 0.07, 0.03]),
        'page_dwell_time_seconds': np.round(np.random.normal(loc=60.0, scale=25.0, size=n_student), 1),
        'is_high_risk_asn': np.random.choice([0, 1], size=n_student, p=[0.92, 0.08]),
        'failed_attempts_1h': np.random.choice([0, 1, 2, 3], size=n_student, p=[0.70, 0.20, 0.07, 0.03]),
        'session_hardware_mismatch': np.random.choice([0, 1], size=n_student, p=[0.97, 0.03]),
        'is_off_peak_hour': np.random.choice([0, 1], size=n_student, p=[0.94, 0.06]),
        'target_label': np.zeros(n_student, dtype=int)
    }
    # Allow dwell times down to 8.0 seconds to overlap with slow fraudsters
    student_data['page_dwell_time_seconds'] = np.clip(student_data['page_dwell_time_seconds'], 8.0, 180.0)
    df_student = pd.DataFrame(student_data)
    
    # 2. Class 0B: Legitimate Campus Cybercafe Operators (650 rows)
    # High velocity (10 to 120); rapid experienced typing (down to 10s); occasional network declines (up to 4 fails)
    n_cafe = 650
    cafe_data = {
        'device_student_count_24h': np.random.randint(low=10, high=120, size=n_cafe),
        'page_dwell_time_seconds': np.round(np.random.uniform(low=10.0, high=50.0, size=n_cafe), 1),
        'is_high_risk_asn': np.random.choice([0, 1], size=n_cafe, p=[0.94, 0.06]),
        'failed_attempts_1h': np.random.choice([0, 1, 2, 3, 4], size=n_cafe, p=[0.55, 0.25, 0.12, 0.06, 0.02]),
        'session_hardware_mismatch': np.random.choice([0, 1], size=n_cafe, p=[0.93, 0.07]),
        'is_off_peak_hour': np.random.choice([0, 1], size=n_cafe, p=[0.90, 0.10]),
        'target_label': np.zeros(n_cafe, dtype=int)
    }
    df_cafe = pd.DataFrame(cafe_data)

    # 3. Class 1: Fraudulent Syndicates / Stolen Card Testers (1,500 rows)
    # Velocity spans 2 to 120; dwell time spans 2s to 35s (overlapping legit!); 35% use local SIMs; fails span 0 to 7
    n_fraud = 1500
    fraud_data = {
        'device_student_count_24h': np.random.randint(low=2, high=120, size=n_fraud),
        'page_dwell_time_seconds': np.round(np.random.uniform(low=2.0, high=35.0, size=n_fraud), 1),
        'is_high_risk_asn': np.random.choice([0, 1], size=n_fraud, p=[0.35, 0.65]),
        'failed_attempts_1h': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], size=n_fraud, p=[0.10, 0.15, 0.15, 0.20, 0.15, 0.15, 0.06, 0.04]),
        'session_hardware_mismatch': np.random.choice([0, 1], size=n_fraud, p=[0.30, 0.70]),
        'is_off_peak_hour': np.random.choice([0, 1], size=n_fraud, p=[0.40, 0.60]),
        'target_label': np.ones(n_fraud, dtype=int)
    }
    df_fraud = pd.DataFrame(fraud_data)
    
    # 4. Concatenate, shuffle random row order, and reset index
    df_full = pd.concat([df_student, df_cafe, df_fraud], ignore_index=True)
    df_shuffled = df_full.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    return df_shuffled

def run_pipeline():
    df = generate_unizik_dataset()
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    export_path = os.path.join(Config.DATA_DIR, 'unizik_transactions.csv')
    df.to_csv(export_path, index=False)
    
    print("\n" + "="*60)
    print("UNIZIK MATRIX GENERATED WITH DEEP MULTI-VARIABLE OVERLAP")
    print("="*60)
    print(f"Total Records    : {len(df):,}")
    print(f"Legitimate (0)   : {len(df[df['target_label'] == 0]):,} rows ({len(df[df['target_label'] == 0])/len(df)*100:.1f}%)")
    print(f"Fraudulent (1)   : {len(df[df['target_label'] == 1]):,} rows ({len(df[df['target_label'] == 1])/len(df)*100:.1f}%)")
    print("="*60)
    print("\nFeature Class Averages (Validation Check):")
    print(df.groupby('target_label').mean().round(2))
    print("="*60 + "\n")

if __name__ == '__main__':
    run_pipeline()