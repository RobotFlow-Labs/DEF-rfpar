# DEF-rfpar Training Report

## Module
- **Name**: DEF-rfpar (RFPAR: Remember and Forget Pixel Attack using RL)
- **Paper**: ArXiv 2502.07821
- **Date**: 2026-04-05

## Classification Attack (ImageNet-1K, ResNeXt50)

| Metric | Value | Paper Reference |
|--------|-------|-----------------|
| Attack Success Rate | **93.0%** | 95%+ |
| Images Attacked | 200 | - |
| Images Deceived | 186 | - |
| Mean L0 | 143.9 | - |
| Mean L2 | 6.26 | - |
| Average Queries | 463 | 500-1000 |
| Forget Iterations | 100 | 100 |
| Pixel Budget (alpha) | 0.01 | 0.01 |
| Convergence Patience (T) | 3 | 3 |
| Bound Threshold (eta) | 0.05 | 0.05 |

## Detection Attack (COCO, YOLO11n)

| Metric | Value | Paper Reference |
|--------|-------|-----------------|
| Images | 50 | - |
| Alpha | 0.05 | 0.05 |
| Patience | 20 | 20 |
| Status | Running | - |

## Hardware
- **GPU**: NVIDIA L4 (23GB VRAM)
- **CUDA**: 12.0 / PyTorch cu128
- **Time**: 419s (classification)

## Export Formats
- [x] PyTorch (.pth) — 393MB
- [x] Safetensors — 393MB
- [x] ONNX (opset 18) — 393MB
- [x] TensorRT FP16 — 197MB
- [x] TensorRT FP32 — 393MB

## HuggingFace
- **Repo**: [ilessio-aiflowlab/DEF-rfpar](https://huggingface.co/ilessio-aiflowlab/DEF-rfpar)

## Hyperparameters
```toml
[attack]
max_iterations = 100
bound_threshold_eta = 0.05
alpha = 0.01  # classification
convergence_duration_t = 3
batch_size = 50
rl_learning_rate = 0.0001
seed = 2
```
