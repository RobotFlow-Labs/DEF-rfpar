# Tasks — DEF-rfpar: Remember & Forget Pixel Attack using RL

PRD breakdown for RL-based black-box pixel adversarial attack optimization.

## Build Plan

| # | Task | Description | Status | Est. Hours |
|---|------|-------------|--------|-----------|
| PRD-001 | Environment + Dataset Setup | Clone repo, uv sync, download ImageNet-1K and Argoverse | ⬜ | 6h |
| PRD-002 | Baseline Reproduction (CUDA) | Reproduce YOLOv8 baseline on Argoverse, ImageNet classifier | ⬜ | 8h |
| PRD-003 | RL Environment Implementation | Pixel selection space, one-step RL, reward function | ⬜ | 10h |
| PRD-004 | Remember-Forget Process | Track successful pixels, prune ineffective ones | ⬜ | 8h |
| PRD-005 | Parallel Pixel Sampling CUDA Kernel | Sample multiple candidates, compute confidence in parallel | ⬜ | 10h |
| PRD-006 | Batch Reward Computation Kernel | Fused RL reward calculation with detector forward pass | ⬜ | 8h |
| PRD-007 | MLX Port + Dual-Compute | Port RL environment to MLX, validate parity | ⬜ | 12h |
| PRD-008 | Query Efficiency Optimization | Minimize query count (target: <500 queries) | ⬜ | 6h |
| PRD-009 | Quantization + Edge Deployment | INT8 optimization for Jetson Orin NX | ⬜ | 6h |
| PRD-010 | Comprehensive Benchmarking | Query efficiency, attack success, latency across both backends | ⬜ | 6h |

## PRD-001: Environment Setup & Dataset Download

**Objective**: Prepare development environment and download ImageNet-1K and Argoverse datasets.

**Tasks**:
- Clone https://github.com/KAU-QuantumAILab/RFPAR
- Run `uv sync` to install dependencies (PyTorch, YOLOv8, gym)
- Download ImageNet-1K validation set (50K images) → `/mnt/forge-data/shared_infra/datasets/imagenet1k/`
- Download Argoverse dataset (autonomous driving) → `/mnt/forge-data/shared_infra/datasets/argoverse/`
- Install YOLOv8 weights
- Verify RL environment library (gym-like interface)

**Acceptance**:
- [ ] Both datasets downloaded and checksummed
- [ ] YOLOv8 weights available
- [ ] RL environment dependencies installed

## PRD-002: Baseline Reproduction on CUDA

**Objective**: Reproduce clean baseline detection performance.

**Tasks**:
- Load YOLOv8 pretrained model on Argoverse
- Measure baseline mAP on vehicle/pedestrian/cyclist detection
- Profile inference latency per image
- Document baseline confidence scores on clean images
- Measure memory usage

**Acceptance**:
- [ ] Baseline mAP documented for Argoverse
- [ ] Inference latency ≤30ms per image
- [ ] Confidence distribution logged

## PRD-003: RL Environment Implementation

**Objective**: Implement one-step RL environment for pixel selection.

**Tasks**:
- Design pixel selection environment (action space: (x, y, magnitude))
- Implement state space (image features around candidate pixel)
- Implement reward function: reward = -confidence_drop (minimize detector confidence)
- Implement one-step RL policy (epsilon-greedy or similar)
- Test on single image, validate convergence toward high-reward pixels
- Profile environment step latency

**Acceptance**:
- [ ] Environment initializes and steps without errors
- [ ] Reward function differentiable and decreasing over iterations
- [ ] Policy selects pixels that reduce detector confidence
- [ ] Step latency <100ms per pixel sample

## PRD-004: Remember-Forget Process Implementation

**Objective**: Implement pixel tracking and pruning logic.

**Tasks**:
- Implement "remember" process: Track successful pixels (high reward)
- Implement "forget" process: Prune low-reward pixels from candidate set
- Design pruning heuristic (threshold-based or probabilistic)
- Minimize redundant pixel perturbations
- Test on single image, validate attack effectiveness

**Acceptance**:
- [ ] Successful pixels tracked and reused
- [ ] Unsuccessful pixels pruned (reduces future exploration)
- [ ] Attack success rate ≥90% with <100 pixels
- [ ] No regression in effectiveness vs baseline RL

## PRD-005: Parallel Pixel Sampling CUDA Kernel

**Objective**: Implement high-performance kernel for parallel pixel candidate evaluation.

**Tasks**:
- Design kernel to sample multiple pixel candidates in parallel
- Compute perturbed image for each candidate
- Evaluate detector confidence for each candidate
- Return top-k candidates by reward
- Validate output against sequential implementation
- Benchmark: target 5-10x speedup

**Acceptance**:
- [ ] Kernel compiles without warnings
- [ ] Output matches sequential reference (atol=1e-4)
- [ ] Processes 8+ candidates in parallel
- [ ] Achieves ≥5x speedup vs sequential evaluation

## PRD-006: Batch Reward Computation CUDA Kernel

**Objective**: Implement fused kernel for RL reward calculation.

**Tasks**:
- Design kernel combining pixel perturbation and detector forward pass
- Compute confidence scores for batch of perturbed images
- Calculate RL rewards from confidence drop
- Fuse with remember-forget update logic
- Validate output against PyTorch reference
- Benchmark: target 3x speedup

**Acceptance**:
- [ ] Kernel fuses perturbation and detector call
- [ ] Output matches reference (atol=1e-4)
- [ ] Handles batch processing (4-8 images)
- [ ] ≥3x speedup vs separate operations

## PRD-007: MLX Port & Dual-Compute Validation

**Objective**: Port RL environment to MLX for Apple Silicon.

**Tasks**:
- Implement MLX version of RL environment step function
- Port pixel sampling and reward computation to MLX
- Implement MLX equivalents of CUDA kernels
- Validate CUDA ↔ MLX numerical parity (atol=1e-4)
- Test attack generation on Mac Studio M-series
- Benchmark relative performance

**Acceptance**:
- [ ] MLX implementation runs without errors
- [ ] Numerical parity validated across 10 test images
- [ ] Attack success rate ≥95% on both backends
- [ ] Query count similar on both backends

## PRD-008: Query Efficiency Optimization

**Objective**: Minimize total queries to achieve ≥95% attack success.

**Tasks**:
- Implement adaptive pixel selection (focus on high-reward regions)
- Implement early stopping criteria (confidence drop threshold)
- Measure query count needed for target success rate
- Compare with state-of-the-art black-box methods
- Target: <500 queries for 95%+ success on ImageNet-1K
- Document query efficiency metrics

**Acceptance**:
- [ ] Attack success rate ≥95% with ≤500 queries
- [ ] Query efficiency compared to baselines
- [ ] Transferability validated (YOLO attacks on other models ≥75%)

## PRD-009: Quantization & Edge Deployment

**Objective**: Quantize for edge deployment on Jetson Orin NX.

**Tasks**:
- Quantize YOLOv8 to INT8 using TensorRT
- Optimize pixel sampling kernel for edge hardware
- Test inference on Jetson Orin NX (target: ≥60 FPS)
- Measure memory footprint (target: <1.5GB)
- Validate attack success rate on edge device
- Benchmark energy consumption

**Acceptance**:
- [ ] Quantized model loads on Jetson
- [ ] Inference speed ≥60 FPS
- [ ] Memory <1.5GB
- [ ] Attack success rate ≥90% on edge device

## PRD-010: Comprehensive Benchmarking

**Objective**: Final validation of query efficiency and attack effectiveness.

**Tasks**:
- Measure query count needed for different success rates (80%, 90%, 95%)
- Profile attack generation speed on both CUDA and MLX
- Benchmark pixel sampling kernel throughput
- Validate cross-model transferability on diverse detectors
- Document robustness to transformations (rotation, brightness, scale)
- Create comprehensive benchmark report

**Acceptance**:
- [ ] Query efficiency metrics documented
- [ ] Attack success rate ≥95% with ≤500 queries on both backends
- [ ] Transferability ≥75% on unseen models
- [ ] Report includes comparison with state-of-the-art black-box methods
