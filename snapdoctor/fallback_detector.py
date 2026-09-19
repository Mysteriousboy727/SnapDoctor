
from dataclasses import dataclass
from typing import List, Set
from .graph_parser import OpNode
from .qnn_log_parser import FallbackEvent

# NOTE: this is a starter list — extend it as you verify against real QNN EP docs
QNN_SUPPORTED_OPS: Set[str] = {
    "Conv", "MatMul", "Gemm", "Relu", "Sigmoid", "Add", "Mul", "Reshape",
    "Transpose", "Softmax", "LayerNormalization", "Resize", "Concat",
}


@dataclass
class Diagnosis:
    op_name: str
    op_type: str
    resident: bool
    reason: str


def diagnose(ops: List[OpNode], fallback_events: List[FallbackEvent] = None) -> List[Diagnosis]:
    fallback_events = fallback_events or []
    fallback_names = {e.node_name for e in fallback_events}

    results = []
    for op in ops:
        if op.name in fallback_names:
            results.append(Diagnosis(op.name, op.op_type, False, "runtime fallback logged by QNN EP"))
        elif op.op_type not in QNN_SUPPORTED_OPS:
            results.append(Diagnosis(op.name, op.op_type, False, f"'{op.op_type}' not in known QNN-supported op set"))
        else:
            results.append(Diagnosis(op.name, op.op_type, True, "resident"))
    return results


def residency_summary(diagnoses: List[Diagnosis]) -> dict:
    total = len(diagnoses)
    resident = sum(1 for d in diagnoses if d.resident)
    return {
        "total_ops": total,
        "resident_ops": resident,
        "residency_pct": round(100 * resident / total, 1) if total else 0.0,
        "fallback_ops": [d for d in diagnoses if not d.resident],
    }