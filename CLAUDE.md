# DEF-rfpar — Remember and Forget Pixel Attack using Reinforcement Learning (RFPAR)
# Wave 8 Defense Module
# Paper: ArXiv 2502.07821
# Authors: Dongsu Song, Daehwa Ko, Jay Hoon Jung (Korea Aerospace University)
# Repo: https://github.com/KAU-QuantumAILab/RFPAR
# Domain: RL-based Black-box Pixel Adversarial Attacks on Detection
# Product Stack: ATLAS / ORACLE

## Status: Classification DONE (93% ASR), Detection running, Exports DONE, HF pushed

## Paper Summary
RFPAR proposes a novel reinforcement learning-based pixel attack method that overcomes randomness in query-based attacks. Unlike patch-based approaches, RFPAR selects individual pixels for perturbation using a one-step RL algorithm. The key innovation is the "Remember and Forget" process: remembering successful pixel perturbations (positive rewards) and forgetting unsuccessful ones (negative rewards). Tested on ImageNet-1K for classification and extends to object detection (YOLOv8, Argoverse dataset). Outperforms state-of-the-art query-based pixel attacks.

## Architecture
**Core Components:**
1. **RL-based Pixel Selection**:
   - Environment: Per-pixel perturbation space (sparse: limited pixels modified)
   - Policy: One-step RL using reward function based on confidence drop
   - Action space: Select pixel coordinates (x,y) and perturbation magnitude
   - Reward: -confidence_score (minimize detector confidence)

2. **Remember & Forget Process**:
   - Remember: Track successful perturbations, build perturbation set
   - Forget: Prune ineffective pixels, focus optimization on high-reward regions
   - Mitigates randomness in pixel selection

3. **Query Efficiency**:
   - Black-box setting: Only queries detector output, no gradients
   - Reduced queries vs. other black-box methods
   - Works on classification and object detection

4. **Test Datasets**:
   - ImageNet-1K (classification)
   - Argoverse (object detection with YOLOv8)

## Key Results
- **Classification (ImageNet-1K)**:
  - Attack success rate: 95%+ with limited pixel budget (50-100 pixels)
  - Query efficiency: 30-40% fewer queries than state-of-the-art
  - Black-box transferability: 75%+ success when transferred to different models

- **Object Detection (Argoverse, YOLOv8)**:
  - Confidence drop: 50-70% mAP reduction with sparse perturbations
  - Pixel budget: <1% of image pixels modified
  - Real-world applicability: Adversarial patches on road signs or vehicle surfaces

- **Performance**:
  - Query budget: ~500-1000 queries for successful attack
  - Robustness: Attacks remain effective with transformations (rotation, brightness)

## Datasets Used
1. **ImageNet-1K** (Classification)
   - 1M training, 50K validation images
   - 1000 object classes
   - Download: https://www.image-net.org/

2. **Argoverse** (Autonomous Driving)
   - High-resolution tracking data from Miami and Silicon Valley
   - Object detection: cars, pedestrians, cyclists
   - Download: https://www.argoverse.org/

## Dependencies
- PyTorch (cu128)
- YOLOv8 (Ultralytics)
- NumPy, Pillow for image processing
- RL environment implementation (gym-like interface)

## Build Requirements
- [ ] Clone repo and `uv sync`
- [ ] Download ImageNet-1K or subset for testing
- [ ] Download Argoverse dataset
- [ ] Install YOLOv8 weights
- [ ] Implement RL environment (pixel selection, reward function)
- [ ] Profile RL policy optimization (one-step RL per query)
- [ ] Implement CUDA kernels for parallel pixel sampling
- [ ] Port to MLX (tensor operations for RL)
- [ ] Benchmark query efficiency and attack speed
- [ ] Dual-compute validation

## CUDA Kernel Targets
1. **Parallel Pixel Sampling Kernel**
   - Sample multiple pixel candidates in parallel
   - Compute confidence for each perturbation
   - Target: 5-10x speedup

2. **Batch Reward Computation Kernel**
   - Compute RL rewards for batch of pixel candidates
   - Fuse with detector forward pass
   - Target: 3x speedup

3. **Remember-Forget Update Kernel**
   - Update reward history and pruning decisions
   - Target: 2x speedup

## Defense Marketplace Value
Sparse pixel attacks (only few pixels modified) are harder to detect than patch-based attacks. RFPAR demonstrates that RL can optimize which pixels matter most, relevant to ATLAS and ORACLE where pixel-level imperceptibility is crucial for stealth attacks.

## Package Manager: uv (NEVER pip)
## Python: >= 3.10
## Torch: cu128 index
