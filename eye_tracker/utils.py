import json
import os
import numpy as np

CALIBRATION_JSON = './calibration.json'

def save_calibration(data, path=CALIBRATION_JSON):
    with open(path, 'w') as f:
        json.dump(data, f)

def load_calibration(path=CALIBRATION_JSON):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None

# convert normalized (0..1) to pixel coordinates
def norm_to_pixel(norm, w, h):
    x = int(np.clip(norm[0], 0.0, 1.0) * w)
    y = int(np.clip(norm[1], 0.0, 1.0) * h)
    return x, y