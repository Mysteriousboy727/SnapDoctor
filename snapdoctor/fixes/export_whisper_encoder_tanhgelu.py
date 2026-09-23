# snapdoctor/fixes/export_whisper_encoder_tanhgelu.py
import torch
import torch.nn.functional as F
from transformers import WhisperConfig, WhisperModel

# tanh-approximation GELU, same formula as your ViT workaround
def gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))

config = WhisperConfig.from_pretrained("openai/whisper-base")
config.activation_function = "gelu_new"  # fixes per-layer blocks

model = WhisperModel.from_pretrained("openai/whisper-base", config=config)
encoder = model.encoder
encoder.eval()

# patch the conv-stem GELU calls directly
F.gelu = gelu_tanh  # global monkey-patch, catches any direct F.gelu(...) call

dummy_input = torch.randn(1, 80, 3000)

torch.onnx.export(
    encoder,
    dummy_input,
    "data/models/whisper_base_encoder_tanhgelu.onnx",
    input_names=["input_features"],
    output_names=["last_hidden_state"],
    opset_version=17,
    dynamo=False,
)
print("Exported: data/models/whisper_base_encoder_tanhgelu.onnx")