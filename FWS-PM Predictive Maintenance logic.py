# maintenance_ts.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def generate_harmonic_data():
    """Simulates feature extraction from machine vibration."""
    # Create sample time-series features (Frequency peaks, RMS, Kurtosis, etc.)
    data = {
        'Machine_ID': [1, 2, 3, 4, 5],
        'Harmonic_Peak': [0.1, 0.2, 0.8, 0.15, 0.95], # High 'Harmonic_Peak' often indicates wear.
        'RMS_Vibration': [1.5, 1.8, 4.2, 1.6, 5.0],
        'Status': ['Normal', 'Normal', 'Warning', 'Normal', 'Critical']
    }
    return pd.DataFrame(data)

def train_maintenance_model(df):
    """Trains a simple classifier for Predictive Maintenance."""
    df['Status_Code'] = df['Status'].map({'Normal': 0, 'Warning': 1, 'Critical': 2})
    
    X = df[['Harmonic_Peak', 'RMS_Vibration']]
    y = df['Status_Code']
    
    # Simple ML model (Classification for prediction)
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    return model

def predict_machine_health(model, machine_id=3):
    """Predicts health of a specific machine based on its features."""
    df = generate_harmonic_data()
    sample = df[df['Machine_ID'] == machine_id].iloc[0]
    
    # Extract features for prediction
    X_pred = np.array([sample['Harmonic_Peak'], sample['RMS_Vibration']]).reshape(1, -1)
    
    prediction_code = model.predict(X_pred)[0]
    status_map = {0: 'Normal', 1: 'Warning', 2: 'Critical'}
    
    # Output a predictive status
    print(f"[P7/Predictive Maintenance] Machine {machine_id} features analyzed.")
    return status_map[prediction_code]
