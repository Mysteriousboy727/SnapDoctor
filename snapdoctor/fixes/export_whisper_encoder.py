# export_whisper_encoder.py
import torch
from transformers import WhisperModel

model = WhisperModel.from_pretrained("openai/whisper-base")
encoder = model.encoder
encoder.eval()

# Whisper encoder expects log-mel spectrogram input: (batch, n_mels, frames)
# whisper-base: n_mels=80, frames=3000 (30s audio, fixed-length by design — this works in your favor, it's already a static shape)
dummy_input = torch.randn(1, 80, 3000)

torch.onnx.export(
    encoder,
    dummy_input,
    "data/models/whisper_base_encoder.onnx",
    input_names=["input_features"],
    output_names=["last_hidden_state"],
    opset_version=17,
    dynamo=False,  # keep consistent with your ViT export approach
)