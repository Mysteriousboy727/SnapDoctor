# 🩺 SnapDoctor

### AI Model Diagnostics for Snapdragon NPU Deployment

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-lightgrey?logo=onnx)
![Qualcomm](https://img.shields.io/badge/Qualcomm-Snapdragon%20X%20Elite-red?logo=qualcomm)
![Status](https://img.shields.io/badge/status-active%20development-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

> **"Does my model actually run on the NPU — or just claim to?"**
> SnapDoctor diagnoses why AI models underperform on Snapdragon NPUs, fixes the issues it finds, and verifies the fix on real hardware.

---

## 🎯 The Problem

Developers deploying AI models on Snapdragon devices hit a wall of silent failures:

```mermaid
flowchart LR
    A[👨‍💻 Developer] --> B[📦 model.onnx]
    B --> C{Enable QNN EP}
    C --> D[🤔 Does it work?]
    D --> E["❓ Unsupported op?"]
    D --> F["❓ Needs quantization?"]
    D --> G["❓ Silent CPU fallback?"]
    D --> H["❓ Cryptic error code?"]
    style D fill:#ffcc00,stroke:#333
    style E fill:#ff6b6b,color:#fff
    style F fill:#ff6b6b,color:#fff
    style G fill:#ff6b6b,color:#fff
    style H fill:#ff6b6b,color:#fff
```

Developers **think** they're getting NPU acceleration. Reality often looks like:

| Assumption | Reality |
|---|---|
| ✅ "I enabled QNNExecutionProvider" | ⚠️ Silent fallback to CPU for unsupported ops |
| ✅ "All my ops are standard ONNX ops" | ⚠️ HTP backend **requires quantization** — fp32 models never touch NPU |
| ✅ "It compiled fine" | ⚠️ Compiles ≠ runs efficiently ≠ runs on NPU at all |

---

## 🚀 What SnapDoctor Does

SnapDoctor closes the loop: **diagnose → fix → verify on real hardware.**

```mermaid
flowchart TD
    A[📥 Input: model.onnx] --> B[🔍 Graph Parser]
    B --> C[📊 Op-level Analysis]
    C --> D{Op-support check}
    D -->|✅ Supported| E[Quantization Check]
    D -->|❌ Unsupported| F[🚩 Flag fallback op]

    E --> G{Is it quantized?}
    G -->|❌ No| H[⚠️ HTP requires QDQ format]
    G -->|✅ Yes| I[✅ HTP-runnable]

    H --> J[🛠️ Auto-quantize<br/>uint8 QDQ]
    J --> K[☁️ Submit to Qualcomm AI Hub]
    I --> K
    F --> L[📄 Diagnostic Report]

    K --> M[💻 Real Snapdragon X Elite<br/>Compile + Profile]
    M --> N[📈 Real NPU Residency %<br/>Real Latency ms]
    N --> L

    style A fill:#4dabf7,color:#fff
    style L fill:#51cf66,color:#fff
    style M fill:#e64980,color:#fff
    style N fill:#51cf66,color:#fff
```

---

## 🏗️ Architecture

```mermaid
flowchart TB
    subgraph core["🩺 snapdoctor/ — Diagnostic Engine"]
        GP[graph_parser.py<br/>📖 Reads ONNX graph]
        QLP[qnn_log_parser.py<br/>📜 Parses QNN runtime logs]
        FD[fallback_detector.py<br/>🚦 Op-support + quantization checks]
        RPT[report.py<br/>📋 Human-readable diagnosis]
        GP --> FD
        QLP --> FD
        FD --> RPT
    end

    subgraph fixes["🛠️ snapdoctor/fixes/ — Remediation"]
        QM[quantize_model.py<br/>⚙️ QDQ quantization]
        EXP[export_resnet.py / export_vit.py<br/>📤 Clean ONNX export]
        AIH[aihub_profile.py<br/>☁️ Real hardware verification]
    end

    subgraph hw["💻 Qualcomm AI Hub"]
        DEV[Snapdragon X Elite CRD<br/>Real cloud-hosted device]
    end

    core -.diagnoses.-> fixes
    fixes --> AIH
    AIH --> DEV
    DEV --> AIH
    AIH --> RPT

    style core fill:#1971c2,color:#fff
    style fixes fill:#e8590c,color:#fff
    style hw fill:#2f9e44,color:#fff
```

---

## ✅ Proven So Far

| Stage | Status | Evidence |
|---|---|---|
| 🔍 Graph parsing | ✅ Working | Parses real ONNX models, extracts op histogram |
| 🚦 Op-support diagnosis | ✅ Working | Full QNN EP op list, sourced from [official docs](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html) |
| 🧪 Quantization-requirement check | ✅ Working | Correctly flags fp32 models as non-HTP-runnable |
| 🛠️ Quantization fix pipeline | ✅ Working | uint8 QDQ quantization via ONNX Runtime tools |
| ☁️ Real hardware verification | ✅ **Proven on real Snapdragon X Elite** | ResNet18 → **100% NPU residency**, ~350µs inference, reproduced across 3 independent runs |

### 📊 Real Hardware Result (ResNet18, quantized)

```
==================================================
REAL HARDWARE PROFILE — Snapdragon X Elite (via Qualcomm AI Hub)
==================================================
NPU Residency (verified on-device): 100.0% (36/36 layers)
Estimated inference time: 348 µs
Warm load time: 345,853 µs
Peak inference memory: ~11.2 MB

All layers confirmed running on NPU (real hardware).
```

> Every single layer executed on `compute_unit: NPU` — not estimated, not simulated. Measured on real Qualcomm AI Hub cloud-hosted Snapdragon X Elite silicon.

---

## 🚧 In Progress

```mermaid
flowchart LR
    A[✅ ResNet18<br/>CNN baseline] --> B[🔄 ViT-B/16<br/>Attention + LayerNorm]
    B --> C[⏳ ControlNet block<br/>Diffusion-specific ops]
    C --> D[⏳ SnapGen Control<br/>Full demo app]

    style A fill:#51cf66,color:#fff
    style B fill:#ffd43b
    style C fill:#adb5bd
    style D fill:#adb5bd
```

- [x] CNN model pipeline (ResNet18) — diagnosed → quantized → hardware-verified
- [ ] Attention-based model pipeline (ViT-B/16) — testing op-support against LayerNorm/Softmax/Gelu
- [ ] Diffusion-specific ops (ControlNet-shaped blocks)
- [ ] SnapGen Control — the flagship visual demo (Stable Diffusion + ControlNet on NPU)

---

## 📂 Project Structure

```
snapdoctor/
├── snapdoctor/              # 🩺 Core diagnostic engine
│   ├── graph_parser.py      #    Reads ONNX graphs
│   ├── qnn_log_parser.py    #    Parses QNN runtime logs
│   ├── fallback_detector.py #    Op-support + quantization checks
│   ├── report.py            #    Human-readable + AI Hub reports
│   └── fixes/
│       ├── quantize_model.py    # QDQ quantization
│       ├── export_resnet.py     # Clean ONNX export (CNN baseline)
│       ├── export_vit.py        # Clean ONNX export (attention model)
│       └── aihub_profile.py     # Real Snapdragon hardware verification
├── snapgen/                 # 🎨 ControlNet demo app (in progress)
├── data/
│   ├── models/               # ONNX model files
│   └── logs/                 # QNN logs + AI Hub profiles
└── configs/
    └── model_config.yaml
```

---

## ⚡ Quick Start

```bash
# Clone and set up
git clone https://github.com/Mysteriousboy727/Snapit.git
cd snapit
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Run diagnosis on a model
python -m snapdoctor.report data/models/your_model.onnx data/logs/your_log.log

# Quantize a model for NPU deployment
python -m snapdoctor.fixes.quantize_model data/models/your_model.onnx data/models/your_model_qdq.onnx

# Verify on real Snapdragon hardware (requires Qualcomm AI Hub API token)
qai-hub configure --api_token YOUR_TOKEN
python -m snapdoctor.fixes.aihub_profile
```

---

## 🔑 Why This Matters

```mermaid
flowchart LR
    subgraph before["❌ Before SnapDoctor"]
        B1[Model] --> B2["QNN Error 14001"] --> B3["😐 What does this mean?"]
    end
    subgraph after["✅ After SnapDoctor"]
        A1[Model] --> A2[🩺 Diagnosis] --> A3["Op 'Resize' unsupported<br/>→ suggested fix"]
    end
    style before fill:#ffe3e3
    style after fill:#e3fce3
```

We're not claiming to fix Qualcomm's hardware. We're closing the gap between **"Snapdragon has an NPU"** and **"a developer can reliably understand, deploy, and verify their model on it."**

---

## 🛠️ Tech Stack

![ONNX Runtime](https://img.shields.io/badge/-ONNX%20Runtime-005CED?style=flat-square)
![QNN EP](https://img.shields.io/badge/-QNN%20Execution%20Provider-CC0000?style=flat-square)
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Qualcomm AI Hub](https://img.shields.io/badge/-Qualcomm%20AI%20Hub-3253DC?style=flat-square)

---

<div align="center">



</div>
