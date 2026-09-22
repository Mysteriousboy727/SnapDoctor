"""Turns diagnosis results into human-readable reports — both from
local static analysis and from real Qualcomm AI Hub hardware profiles."""
from .fallback_detector import Diagnosis, residency_summary, check_quantization_requirement
from typing import List


def print_report(diagnoses: List[Diagnosis], model_path: str = None):
    summary = residency_summary(diagnoses)
    print("=" * 50)
    print(f"Op-level NPU compatibility: {summary['residency_pct']}% "
          f"({summary['resident_ops']}/{summary['total_ops']} ops)")
    print("=" * 50)
    if summary["fallback_ops"]:
        print("\nUnsupported ops (blocking full NPU residency):")
        for d in summary["fallback_ops"]:
            print(f"  - {d.op_name} [{d.op_type}]: {d.reason}")
    else:
        print("\nAll ops are QNN-supported types.")

    if model_path:
        q = check_quantization_requirement(model_path)
        print("\n" + "-" * 50)
        print(f"HTP (NPU) runnable: {q['htp_runnable']}")
        print(f"  {q['note']}")


def print_aihub_report(profile_data: dict):
    from .aihub_parser import parse_aihub_profile, summarize_timing

    diagnoses = parse_aihub_profile(profile_data)
    summary = residency_summary(diagnoses)
    timing = summarize_timing(profile_data)

    print("=" * 50)
    print("REAL HARDWARE PROFILE — Snapdragon X Elite (via Qualcomm AI Hub)")
    print("=" * 50)
    print(f"NPU Residency (verified on-device): {summary['residency_pct']}% "
          f"({summary['resident_ops']}/{summary['total_ops']} layers)")
    print(f"Estimated inference time: {timing['estimated_inference_time_us']} µs")
    print(f"Warm load time: {timing['warm_load_time_us']} µs")
    if timing['peak_memory_bytes']:
        print(f"Peak inference memory: {timing['peak_memory_bytes'] / 1024 / 1024:.2f} MB")

    if summary["fallback_ops"]:
        print("\nLayers NOT on NPU:")
        for d in summary["fallback_ops"]:
            print(f"  - {d.op_name}: {d.reason}")
    else:
        print("\nAll layers confirmed running on NPU (real hardware).")


if __name__ == "__main__":
    from .graph_parser import load_graph, extract_ops
    from .qnn_log_parser import parse_log
    from .fallback_detector import diagnose
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else "data/models/unet.onnx"
    log_path = sys.argv[2] if len(sys.argv) > 2 else "data/logs/sample.log"

    ops = extract_ops(load_graph(model_path))
    events = parse_log(log_path)
    diagnoses = diagnose(ops, events)
    print_report(diagnoses, model_path=model_path)