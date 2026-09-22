import json
import os

data = {
    "model": "vit_b16_tanhpow_qdq.onnx",
    "device": "Snapdragon X Elite CRD",
    "compile_status": "SUCCESS",
    "profile_status": "SUCCESS",
    "estimated_inference_time_us": 52628,
    "mean_inference_time_us": 52746,
    "first_load_time_us": 30416840,
    "warm_load_time_us": 1231141,
    "first_load_peak_memory_bytes": 626294784,
    "baseline_unquantized_tanhpow_us": 72000
}

os.makedirs("reports", exist_ok=True)
with open("reports/vit_b16_tanhpow_qdq_profile.json", "w") as f:
    json.dump(data, f, indent=2)

print("Saved to reports/vit_b16_tanhpow_qdq_profile.json")