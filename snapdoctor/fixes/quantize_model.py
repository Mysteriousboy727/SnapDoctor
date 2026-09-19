"""Quantizes a float32 ONNX model into QDQ format for QNN HTP execution."""
import numpy as np
import onnxruntime
from onnxruntime.quantization import QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model
from onnxruntime.quantization import CalibrationDataReader


class RandomDataReader(CalibrationDataReader):
    """TODO: replace random data with real representative inputs for accuracy."""
    def __init__(self, model_path: str, num_samples: int = 10):
        session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        self.data_list = []
        for _ in range(num_samples):
            sample = {}
            for inp in inputs:
                shape = [d if isinstance(d, int) else 1 for d in inp.shape]
                sample[inp.name] = np.random.random(shape).astype(np.float32)
            self.data_list.append(sample)
        self.enum_data = None

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.data_list)
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


def quantize_for_qnn(input_path: str, output_path: str):
    reader = RandomDataReader(input_path)
    preproc_path = input_path.replace(".onnx", ".preproc.onnx")
    changed = qnn_preprocess_model(input_path, preproc_path)
    model_to_quantize = preproc_path if changed else input_path

    qnn_config = get_qnn_qdq_config(
    model_to_quantize, reader,
    activation_type=QuantType.QUInt8,   # changed from QUInt16
    weight_type=QuantType.QUInt8,
)
    quantize(model_to_quantize, output_path, qnn_config)

    # Force single-file save (embed all weight data, no external .data file)
    import onnx
    model = onnx.load(output_path, load_external_data=True)
    onnx.save(model, output_path, save_as_external_data=False)

    print(f"Saved quantized model (single-file): {output_path}")


if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "data/models/resnet18_fresh.onnx"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/models/resnet18_fresh_qdq.onnx"
    quantize_for_qnn(inp, out)