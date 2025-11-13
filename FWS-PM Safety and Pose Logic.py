# safety_pose.py
import numpy as np
import random
from PIL import Image, ImageDraw

def simulate_pose_detection(is_unsafe=False):
    """Simulates real-time 2D pose detection (P4 knowledge)."""
    # Simulate a single worker's left wrist (x, y) coordinates
    if is_unsafe:
        # Unsafe posture: wrist too close to machine boundary (e.g., x < 50)
        wrist_x = random.randint(10, 45)
    else:
        # Safe posture
        wrist_x = random.randint(100, 300)
    
    # Standard output: (keypoint, x_coord, y_coord)
    return np.array([wrist_x, 200])

def generate_occluded_data(threshold=50):
    """Simulates generative augmentation (P3 knowledge)."""
    # P3 Concept: Use synthetic data (e.g., occluded views) to set a robust threshold.
    # We simulate training on 100 noisy/occluded samples to determine a safety boundary.
    simulated_synthetic_data = np.random.normal(loc=50, scale=10, size=100)
    
    # Calculate a conservative, robust threshold (e.g., 95th percentile)
    robust_threshold = np.percentile(simulated_synthetic_data, 95)
    
    print(f"[P3/Generative CV] Synthetically robust boundary set at X={robust_threshold:.1f}")
    return robust_threshold

def check_safety(wrist_pos, threshold):
    """Checks if the simulated posture is unsafe based on the robust threshold."""
    if wrist_pos[0] < threshold:
        return True, f"ALERT: Wrist position ({wrist_pos[0]}) violates P3 threshold."
    return False, "Worker posture is SAFE."

def visualize_safety(img_path, wrist_pos, is_alert):
    """Generates a simple visualization of the safety zone."""
    img_size = (400, 250)
    img = Image.new('RGB', img_size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw the simulated machine (red zone)
    draw.rectangle([(0, 0), (60, 250)], fill='red')
    
    # Draw safety zone (green zone)
    draw.rectangle([(60, 0), (400, 250)], fill='lightgreen')
    
    # Draw the wrist keypoint
    color = 'red' if is_alert else 'blue'
    draw.ellipse([(wrist_pos[0]-5, wrist_pos[1]-5), (wrist_pos[0]+5, wrist_pos[1]+5)], fill=color)
    
    # Save the image to the output folder
    img.save(img_path)
