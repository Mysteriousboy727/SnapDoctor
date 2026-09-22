"""
Re-export torchvision ViT-B/16 with tanh-approximate GELU instead of exact
(erf-based) GELU, to avoid the Erf op QAIRT's converter can't layout-infer.
"""

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


def replace_gelu_with_tanh_approx(model: nn.Module):
    for name, module in model.named_children():
        if isinstance(module, nn.GELU):
            setattr(model, name, nn.GELU(approximate="tanh"))
        else:
            replace_gelu_with_tanh_approx(module)  # recurse into children
    return model


def export():
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.eval()

    model = replace_gelu_with_tanh_approx(model)

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        "data/models/vit_b16_tanhgelu.onnx",
        input_names=["input"],
        output_names=["output_0"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print("Exported data/models/vit_b16_tanhgelu.onnx (tanh-approx GELU, no Erf ops)")


if __name__ == "__main__":
    export()