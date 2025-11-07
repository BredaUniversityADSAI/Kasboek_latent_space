import json
import os
import numpy as np

# Define folder paths
CALIBRATION_DIR = './calibrations'
SAVED_IMAGES_DIR = './saved_images'
CALIBRATION_JSON = os.path.join(CALIBRATION_DIR, 'calibration.json')

def ensure_directories():
    """Ensure required directories exist"""
    if not os.path.exists(CALIBRATION_DIR):
        os.makedirs(CALIBRATION_DIR)
    if not os.path.exists(SAVED_IMAGES_DIR):
        os.makedirs(SAVED_IMAGES_DIR)

def save_calibration(data, path=None):
    """Save calibration data to JSON file"""
    ensure_directories()
    if path is None:
        path = CALIBRATION_JSON
    
    # Convert numpy arrays to lists if present
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Calibration saved to {path}")

def load_calibration(path=None):
    """Load calibration data from JSON file"""
    ensure_directories()
    if path is None:
        path = CALIBRATION_JSON
    
    if not os.path.exists(path):
        return None
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        print(f"Calibration loaded from {path}")
        return data
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return None

def norm_to_pixel(norm, w, h):
    """Convert normalized coordinates (0..1) to pixel coordinates"""
    x = int(np.clip(norm[0], 0.0, 1.0) * w)
    y = int(np.clip(norm[1], 0.0, 1.0) * h)
    return x, y

def pixel_to_norm(pixel, w, h):
    """Convert pixel coordinates to normalized (0..1) coordinates"""
    x = np.clip(pixel[0] / float(w), 0.0, 1.0)
    y = np.clip(pixel[1] / float(h), 0.0, 1.0)
    return x, y

def save_timestamp_log(filename, data):
    """Save timestamped log data"""
    ensure_directories()
    filepath = os.path.join(CALIBRATION_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    return filepath