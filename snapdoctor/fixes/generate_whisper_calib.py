# save as generate_whisper_calib.py, or inline in a quick script
import numpy as np
import os

os.makedirs("data/calibration_whisper", exist_ok=True)
for i in range(20):
    sample = np.random.randn(1, 80, 3000).astype(np.float32)
    np.save(f"data/calibration_whisper/sample_{i}.npy", sample)
print("Generated 20 calibration samples")