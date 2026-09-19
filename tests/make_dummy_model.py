import onnx
from onnx import helper, TensorProto

# A tiny graph: Conv -> Relu -> Reshape (Reshape isn't in our supported list, so it'll show as a fallback)
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [4, 3, 3, 3])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 6, 6])

conv = helper.make_node("Conv", ["X", "W"], ["conv_out"], name="conv1")
relu = helper.make_node("Relu", ["conv_out"], ["relu_out"], name="relu1")
reshape = helper.make_node("Reshape", ["relu_out", "shape"], ["Y"], name="reshape1")

shape_init = helper.make_tensor("shape", TensorProto.INT64, [2], [1, -1])

graph = helper.make_graph([conv, relu, reshape], "dummy_graph", [X, W], [Y], initializer=[shape_init])
model = helper.make_model(graph, producer_name="snapdoctor-test")
model.opset_import[0].version = 13

onnx.save(model, "data/models/dummy.onnx")
print("Saved data/models/dummy.onnx")