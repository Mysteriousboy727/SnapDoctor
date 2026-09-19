import torch
import torchvision

model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy_input, "data/models/resnet18_fresh.onnx",
    input_names=["input"], output_names=["output"],
    opset_version=17,
    dynamic_axes=None,
)
print("Exported resnet18_fresh.onnx at opset 17")