
import onnx
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class OpNode:
    name: str
    op_type: str
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, Any] = field(default_factory=dict)


def load_graph(model_path: str, strict_check: bool = False) -> onnx.ModelProto:
    model = onnx.load(model_path)
    if strict_check:
        onnx.checker.check_model(model)
    else:
        try:
            onnx.checker.check_model(model)
        except Exception as e:
            print(f"[warning] onnx checker validation issue (non-fatal, continuing): {e}")
    return model


def extract_ops(model: onnx.ModelProto) -> List[OpNode]:
    ops = []
    for node in model.graph.node:
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        ops.append(OpNode(
            name=node.name or f"{node.op_type}_{len(ops)}",
            op_type=node.op_type,
            inputs=list(node.input),
            outputs=list(node.output),
            attrs=attrs,
        ))
    return ops


def op_type_histogram(ops: List[OpNode]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for op in ops:
        hist[op.op_type] = hist.get(op.op_type, 0) + 1
    return hist


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/models/unet.onnx"
    m = load_graph(path)
    ops = extract_ops(m)
    print(f"Loaded {len(ops)} ops")
    for op_type, count in sorted(op_type_histogram(ops).items(), key=lambda x: -x[1]):
        print(f"  {op_type}: {count}")