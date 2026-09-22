"""
Compile + profile a quantized model on a real Snapdragon device via Qualcomm AI Hub.
"""

import argparse
import os
import sys
import qai_hub as hub


def profile_on_device(onnx_path: str, device_name: str = "Snapdragon X Elite CRD"):
    if not os.path.isfile(onnx_path):
        sys.exit(f"ERROR: model file not found: {onnx_path}")

    print(f"Model file to submit : {os.path.abspath(onnx_path)}")
    print(f"File size            : {os.path.getsize(onnx_path) / 1e6:.2f} MB")
    print(f"Device               : {device_name}")

    device = hub.Device(device_name)

    print(f"\nSubmitting compile job for {onnx_path} on {device_name}...")
    compile_job = hub.submit_compile_job(
        model=onnx_path,
        device=device,
        options="--target_runtime qnn_dlc",
    )
    compile_job.wait()
    status = compile_job.get_status()
    print(f"Compile job status: {status}")
    print(f"Compile job URL   : {compile_job.url}")

    if status.code != "SUCCESS":
        print("\n=== Compile job FAILED — pulling converter log for detail ===")
        log_dir = "data/logs/compile_fail"
        try:
            log_files = compile_job.download_job_logs(log_dir)
            print(f"Downloaded {len(log_files)} log file(s):")
            for lf in log_files:
                print(f"\n--- {lf} ---")
                with open(lf, "r", encoding="utf-8", errors="replace") as f:
                    print(f.read())
        except Exception as e:
            print(f"Could not auto-download log ({e}). Open manually: {compile_job.url}")
        sys.exit("Compile failed — see log above / job URL. Not submitting profile job.")

    target_model = compile_job.get_target_model()

    print("\nSubmitting profile job...")
    profile_job = hub.submit_profile_job(
        model=target_model,
        device=device,
    )
    profile_job.wait()
    profile_status = profile_job.get_status()
    print(f"Profile job status: {profile_status}")
    print(f"Profile job URL   : {profile_job.url}")

    if profile_status.code != "SUCCESS":
        print("\n=== Profile job FAILED — pulling log for detail ===")
        log_dir = "data/logs/profile_fail"
        try:
            log_files = profile_job.download_job_logs(log_dir)
            print(f"Downloaded {len(log_files)} log file(s):")
            for lf in log_files:
                print(f"\n--- {lf} ---")
                with open(lf, "r", encoding="utf-8", errors="replace") as f:
                    print(f.read())
        except Exception as e:
            print(f"Could not auto-download log ({e}). Open manually: {profile_job.url}")
        sys.exit("Profile failed — see log above / job URL.")

    profile_data = profile_job.download_profile()
    return compile_job, profile_job, profile_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("qdq_onnx", help="Path to the quantized (QDQ) ONNX model to profile")
    parser.add_argument("--device", default="Snapdragon X Elite CRD")
    args = parser.parse_args()

    compile_job, profile_job, profile_data = profile_on_device(args.qdq_onnx, args.device)

    print("\n--- Profile summary ---")
    print(profile_data.get("execution_summary", {}))

    import json
    import os
    os.makedirs("reports", exist_ok=True)
    model_stem = os.path.basename(args.qdq_onnx).replace(".onnx", "")
    raw_path = f"reports/{model_stem}_profile_raw.json"
    with open(raw_path, "w") as f:
        json.dump(profile_data, f, indent=2)
    print(f"\nFull raw profile (with execution_detail) saved to: {raw_path}")