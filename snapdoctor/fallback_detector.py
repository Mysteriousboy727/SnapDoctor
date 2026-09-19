from dataclasses import dataclass
from typing import List, Set
from .graph_parser import OpNode
from .qnn_log_parser import FallbackEvent

# Sourced from official ONNX Runtime QNN EP docs:
# https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
QNN_SUPPORTED_OPS: Set[str] = {
    "Abs", "Add", "And", "ArgMax", "ArgMin", "Asin", "Atan", "AveragePool",
    "BatchNormalization", "Cast", "Clip", "Concat", "Conv", "ConvTranspose",
    "Cos", "DepthToSpace", "DequantizeLinear", "Div", "Elu", "Equal", "Exp",
    "Expand", "Flatten", "Floor", "Gather", "Gelu", "Gemm", "GlobalAveragePool",
    "Greater", "GreaterOrEqual", "GridSample", "HardSwish", "InstanceNormalization",
    "LRN", "LayerNormalization", "LeakyRelu", "Less", "LessOrEqual", "Log",
    "LogSoftmax", "LpNormalization", "MatMul", "Max", "MaxPool", "Min", "Mul",
    "Neg", "Not", "Or", "Prelu", "Pad", "Pow", "QuantizeLinear", "ReduceMax",
    "ReduceMean", "ReduceMin", "ReduceProd", "ReduceSum", "Relu", "Resize",
    "Round", "Sigmoid", "Sign", "Sin", "Slice", "Softmax", "SpaceToDepth",
    "Split", "Sqrt", "Squeeze", "Sub", "Tanh", "Tile", "TopK", "Transpose",
    "Unsqueeze", "Where",
}

# NOT supported by QNN EP at all (control flow) — always fallback regardless of context
QNN_UNSUPPORTED_OPS: Set[str] = {"Loop", "If"}


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


def check_quantization_requirement(model_path: str) -> dict:
    """HTP (NPU) backend only runs quantized models. A model with all-float32
    weights/activations will NOT actually execute on NPU regardless of op support."""
    import onnx
    model = onnx.load(model_path)

    float_initializers = 0
    quant_ops = 0
    for init in model.graph.initializer:
        if init.data_type == onnx.TensorProto.FLOAT:
            float_initializers += 1

    for node in model.graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
            quant_ops += 1

    is_quantized = quant_ops > 0
    return {
        "is_quantized": is_quantized,
        "float_initializer_count": float_initializers,
        "qdq_op_count": quant_ops,
        "htp_runnable": is_quantized,
        "note": (
            "Model appears fully float32 — HTP backend requires a quantized "
            "(QDQ int8/int16) model. Run quantize_model.py before deployment."
            if not is_quantized else
            "Model contains QuantizeLinear/DequantizeLinear ops — quantized, HTP-eligible."
        ),
    }