"""
Quantize an ONNX model to uint8 QDQ (static quantization) for Snapdragon HTP.
Handles models with external data (.onnx.data), like vit_b16.
Usage: python -m snapdoctor.fixes.quantize_model <input.onnx> <output_qdq.onnx> [--calib-dir DIR]
"""

import argparse
import os
import numpy as np
import onnx
from onnxruntime.quantization import (
    quantize_static,
    QuantType,
    QuantFormat,
    CalibrationDataReader,
)


class ImageCalibrationDataReader(CalibrationDataReader):
    def __init__(
        self,
        calib_dir: str,
        input_name: str,
        input_shape: tuple,
        limit: int = 200,
    ):
        from PIL import Image

        self.input_name = input_name
        self.input_shape = input_shape

        self.files = [
            os.path.join(calib_dir, f)
            for f in os.listdir(calib_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ][:limit]

        self._iter = iter(self.files)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.mean = mean.reshape(1, 3, 1, 1)
        self.std = std.reshape(1, 3, 1, 1)

    def _preprocess(self, path):
        from PIL import Image

        h, w = self.input_shape[2], self.input_shape[3]

        img = Image.open(path).convert("RGB").resize((w, h))

        arr = np.array(img).astype(np.float32) / 255.0

        arr = arr.transpose(2, 0, 1)[np.newaxis, ...]

        arr = (arr - self.mean) / self.std

        return arr.astype(np.float32)

    def get_next(self):
        path = next(self._iter, None)

        if path is None:
            return None

        return {
            self.input_name: self._preprocess(path)
        }


def get_model_input_info(input_path: str):
    """
    Load just the graph (with external data linked, load_external_data=True is
    the default and required so onnx can actually read initializer shapes/values
    for a model like vit_b16 that split weights into a .data file).
    """

    model = onnx.load(
        input_path,
        load_external_data=True,
    )

    graph_input = model.graph.input[0]

    input_name = graph_input.name

    dims = [
        d.dim_value
        for d in graph_input.type.tensor_type.shape.dim
    ]

    input_shape = (
        tuple(dims)
        if all(dims)
        else (1, 3, 224, 224)
    )

    return input_name, input_shape


def quantize(
    input_path: str,
    output_path: str,
    calib_dir: str | None = None,
):
    input_name, input_shape = get_model_input_info(input_path)

    print(
        f"Input tensor: {input_name}, "
        f"shape: {input_shape}"
    )

    external_data_file = input_path + ".data"

    if os.path.exists(external_data_file):
        print(
            f"External data detected: "
            f"{external_data_file}"
        )

    if calib_dir:
        calib_reader = ImageCalibrationDataReader(
            calib_dir,
            input_name,
            input_shape,
        )

    else:

        class RandomCalibrationDataReader(CalibrationDataReader):
            def __init__(self, n=50):
                self.n = n
                self.count = 0

            def get_next(self):
                if self.count >= self.n:
                    return None

                self.count += 1

                return {
                    input_name: np.random.randn(
                        *input_shape
                    ).astype(np.float32)
                }

        calib_reader = RandomCalibrationDataReader()

    # quantize_static reads model_input from disk itself
    # (handles external data correctly as long as the
    # .data file sits alongside the .onnx file, which
    # it does here) — we do NOT pass it a pre-loaded
    # onnx.ModelProto.

    quantize_static(
        model_input=input_path,
        model_output=output_path,
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        use_external_data_format=False,
        op_types_to_quantize=["Conv", "MatMul", "Gemm", "Tanh", "Pow"],
               extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": True,
        },
    )

    print(
        f"Quantized model written to "
        f"{output_path}"
    )

    if os.path.exists(output_path + ".data"):
        print(
            f"External data written to "
            f"{output_path}.data"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_onnx")
    parser.add_argument("output_onnx")
    parser.add_argument("--calib-dir", default=None)

    args = parser.parse_args()

    quantize(
        args.input_onnx,
        args.output_onnx,
        args.calib_dir,
    )