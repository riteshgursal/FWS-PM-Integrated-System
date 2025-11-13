# main.py
import os
import numpy as np
import pandas as pd
from safety_pose import (
    simulate_pose_detection, generate_occluded_data, check_safety, visualize_safety
)
from maintenance_ts import (
    generate_harmonic_data, train_maintenance_model, predict_machine_health
)

def run_fws_pm_system():
    os.makedirs('output', exist_ok=True)
    print("--- FWS-PM SYSTEM START ---")
    
    # --- 1. P7: Predictive Maintenance (Simulated Machine Failure Prediction) ---
    print("\n[STEP 1: P7] Running Predictive Maintenance Model...")
    harmonic_data = generate_harmonic_data()
    pm_model = train_maintenance_model(harmonic_data)
    
    health_status = predict_machine_health(pm_model, machine_id=3)
    print(f"Machine Health Status: {health_status}")
    print("-------------------------")

    # --- 2. P3: Generative Data Robustness (Set Safety Threshold) ---
    print("\n[STEP 2: P3] Setting Robust CV Threshold via Synthetic Data...")
    safety_threshold = generate_occluded_data()
    print(f"Safety Zone established: X > {safety_threshold:.1f}")
    print("-------------------------")

    # --- 3. P4: Real-Time Worker Safety (Simulated CV Inference) ---
    print("\n[STEP 3: P4] Running Real-Time Worker Safety Check...")
    
    # Scenario A: Safe Posture
    wrist_safe = simulate_pose_detection(is_unsafe=False)
    is_alert_A, msg_A = check_safety(wrist_safe, safety_threshold)
    print(f"Scenario A (Safe): {msg_A}")
    visualize_safety(wrist_safe, is_alert_A).save("output/scenario_A_safe.png")
    
    # Scenario B: Unsafe Posture
    wrist_unsafe = simulate_pose_detection(is_unsafe=True)
    is_alert_B, msg_B = check_safety(wrist_unsafe, safety_threshold)
    print(f"Scenario B (Unsafe): {msg_B}")
    visualize_safety(wrist_unsafe, is_alert_B).save("output/scenario_B_unsafe.png")
    
    print("--- FWS-PM SYSTEM COMPLETE ---")
    print("Output visualizations saved in the 'output/' folder.")
    print("This integrated system demonstrates core knowledge from P3, P4, and P7.")

if __name__ == "__main__":
    run_fws_pm_system()