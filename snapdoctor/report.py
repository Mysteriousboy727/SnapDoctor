
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