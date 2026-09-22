"""Exports a Vision Transformer (attention-based) model to ONNX —
tests SnapDoctor's pipeline against attention/LayerNorm ops, not just CNN ops."""
import torch
import torchvision

model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy_input, "data/models/vit_b16.onnx",
    input_names=["input"], output_names=["output"],
    opset_version=17,
    dynamic_axes=None,
)
print("Exported vit_b16.onnx")