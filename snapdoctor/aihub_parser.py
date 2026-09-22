"""
Compile + profile a quantized model on a real Snapdragon device
via Qualcomm AI Hub, then render the profiling result through
SnapDoctor's report format.
"""

import argparse
import qai_hub as hub

from .. import report


DEVICE_NAME = "Snapdragon X Elite CRD"


def profile_on_device(
    onnx_path: str,
    device_name: str = DEVICE_NAME,
):
    """
    Compile and profile a quantized ONNX model using Qualcomm AI Hub.

    Returns:
        compile_job
        profile_job
        profile_data
    """

    device = hub.Device(device_name)

    print(f"Submitting compile job for {onnx_path}")
    print(f"Device: {device_name}")

    compile_job = hub.submit_compile_job(
        model=onnx_path,
        device=device,
        options="--target_runtime qnn_context_binary",
    )

    compile_job.wait()

    print(f"Compile job status: {compile_job.get_status()}")

    target_model = compile_job.get_target_model()

    print("\nSubmitting profile job...")

    profile_job = hub.submit_profile_job(
        model=target_model,
        device=device,
    )

    profile_job.wait()

    print(f"Profile job status: {profile_job.get_status()}")

    profile_data = profile_job.download_profile()

    return compile_job, profile_job, profile_data


def main():
    parser = argparse.ArgumentParser(
        description="Profile a quantized ONNX model using Qualcomm AI Hub."
    )

    parser.add_argument(
        "qdq_onnx",
        help="Path to the quantized QDQ ONNX model",
    )

    parser.add_argument(
        "--device",
        default=DEVICE_NAME,
        help="Qualcomm AI Hub device name",
    )

    args = parser.parse_args()

    compile_job, profile_job, profile_data = profile_on_device(
        args.qdq_onnx,
        args.device,
    )

    print("\nCompile job URL:", compile_job.url)
    print("Profile job URL:", profile_job.url)

    print("\nGenerating SnapDoctor report...\n")

    report.print_aihub_report(profile_data)


if __name__ == "__main__":
    main()