"""Compiles and profiles the quantized model on a real Snapdragon X Elite
via Qualcomm AI Hub cloud device farm."""
import qai_hub as hub

DEVICE_NAME = "Snapdragon X Elite CRD"
MODEL_PATH = "data/models/resnet18_fresh_qdq.onnx"

def main():
    device = hub.Device(DEVICE_NAME)
    print(f"Submitting compile job for {MODEL_PATH} on {DEVICE_NAME}...")

    compile_job = hub.submit_compile_job(
        model=MODEL_PATH,
        device=device,
        options="--target_runtime precompiled_qnn_onnx",
    )
    compile_job.wait()
    print("Compile job status:", compile_job.get_status())

    target_model = compile_job.get_target_model()

    print("Submitting profile job...")
    profile_job = hub.submit_profile_job(
        model=target_model,
        device=device,
    )
    profile_job.wait()
    print("Profile job status:", profile_job.get_status())

    profile_data = profile_job.download_profile()
    print("\n--- Profile summary ---")
    print(profile_data)

if __name__ == "__main__":
    main()