# DEF-rfpar — RL Pixel Attack on Object Detection

**Wave 8 Defense Module**

**Repo**: https://github.com/KAU-QuantumAILab/RFPAR

**Domain**: RL Adversarial Attack

**Stack**: ATLAS (fleet AV) / ORACLE (through-wall)

## Status: ⬜ Not Started

## Context
RFPAR implements reinforcement learning-based pixel attacks on object detection systems (YOLOv8, Argoverse). Evaluates adversarial vulnerability of autonomous vehicle perception using RL-optimized perturbations. Critical for understanding attack surfaces and designing robust fleet perception systems.

## Build Requirements
- [ ] Clone repo and verify builds
- [ ] Create tasks/ PRD breakdown
- [ ] Implement CUDA kernel optimizations (following /anima-optimize-cuda-pipeline)
- [ ] Implement MLX equivalent
- [ ] Run benchmark suite (latency, throughput, memory)
- [ ] Dual-compute validation (MLX + CUDA)

## CUDA Kernel Targets
- [ ] Identify bottleneck operations (profile first)
- [ ] RL environment interaction kernels (parallel sampling)
- [ ] Fused gradient computation for RL policy optimization
- [ ] INT8/FP16 quantized attack generation pipeline
- [ ] Save kernels to /mnt/forge-data/shared_infra/cuda_extensions/

## Package Manager: uv (NEVER pip)
## Python: >= 3.10
