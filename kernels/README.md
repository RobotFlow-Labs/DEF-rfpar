# Custom Kernels — DEF-rfpar: Remember & Forget Pixel Attack using RL

This folder contains custom CUDA and MLX Metal kernels for efficient RL-based black-box pixel adversarial attacks.

## Kernel Targets (from arxiv 2502.07821)

### 1. Parallel Pixel Sampling Kernel (parallel_pixel_sampling)
**Purpose**: Sample multiple pixel candidates in parallel and evaluate detector confidence.

**Inputs**:
- Original image (3×H×W)
- Current perturbation state (pixel coordinates and magnitudes)
- Detector model weights
- Batch of candidate pixels to evaluate (K pixels)

**Operations**:
- For each candidate pixel (x, y):
  - Create perturbed image copy
  - Apply perturbation magnitude [0, 255]
  - Forward pass through YOLOv8 detector
  - Compute confidence score
- Return top-k candidates by reward (confidence drop)

**Target**: 5-10x speedup vs sequential evaluation
**Baseline latency**: 100ms per 8 candidates → **Target**: 10-20ms
**Throughput**: Evaluate 100+ candidates per second

### 2. Batch Reward Computation Kernel (batch_reward_computation)
**Purpose**: Fused RL reward calculation with detector forward pass.

**Inputs**:
- Batch of perturbed images (B×3×H×W)
- Detector forward pass output (bboxes, confidences)
- Current best perturbation state

**Operations**:
- Forward through detection head (parallel for batch)
- Compute confidence for each image
- Calculate reward: reward = -confidence_drop
- Identify high-reward pixels
- Update best perturbation state

**Target**: 3x speedup over sequential forward passes
**Baseline latency**: 40ms per batch of 8 → **Target**: 13ms
**Throughput**: Process 8+ images in parallel

### 3. Remember-Forget Update Kernel (remember_forget_update)
**Purpose**: Track successful pixels and prune unsuccessful ones.

**Inputs**:
- Current perturbation history (pixels tried so far)
- Reward history (confidence drops for each pixel)
- Pruning threshold (e.g., reward > 0.1)

**Operations**:
- Identify successful pixels (high reward)
- Mark for reuse in future iterations
- Identify unsuccessful pixels (low reward)
- Prune from candidate set
- Update memory efficiently

**Target**: 2x speedup vs CPU tracking
**Baseline latency**: 5ms per update → **Target**: 2.5ms

## Performance Targets

| Kernel | Baseline | Target | Est. Speedup |
|--------|----------|--------|--------------|
| parallel_pixel_sampling | 100ms (8 cand.) | 15ms | 6.7x |
| batch_reward_computation | 40ms (batch 8) | 13ms | 3x |
| remember_forget_update | 5ms | 2.5ms | 2x |
| **Total per RL step** | **145ms** | **30.5ms** | **4.75x** |

## Query Efficiency Optimization

**Attack Budget**: Target <500 queries for 95%+ success

```
Total queries = N_iterations × N_candidates_per_iteration
Target: 50 iterations × 10 candidates = 500 queries
```

**Baseline**: ~1000 queries (sequential pixel search)
**Optimized**: ~500 queries (parallel + remember-forget)

## Build Configuration

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_ARCH="80;86;89" \
  -DYOLO_VERSION="v8" \
  -DENABLE_MLX=ON \
  ..
make -j$(nproc)
```

## MLX Metal Equivalents

- `mlx/pixel_sampling_metal.mm` — Parallel pixel evaluation on Metal
- `mlx/reward_computation_metal.mm` — Fused reward calculation
- `mlx/remember_forget_metal.mm` — Pixel tracking and pruning on Metal

## Integration Points

**PyTorch**:
```python
from anima_cuda.rfpar import parallel_pixel_sampling, batch_reward_computation, remember_forget_update
```

**MLX**:
```python
from anima_mlx.rfpar import parallel_pixel_sampling, batch_reward_computation, remember_forget_update
```

## RL Environment Performance

**Single Attack (95%+ success)**:
- Query budget: <500 queries
- Latency: ~15 seconds (CUDA optimized)
- Pixel count: <100 pixels modified (<1% of image)

**Batch Processing** (10 images):
- Total queries: <5000
- Latency: ~150 seconds parallel
- Throughput: 60 attacks per GPU hour

## IP Note

Custom kernels are proprietary ANIMA IP:
- **Parallel pixel sampling** reduces attack generation time
- **Remember-forget** process minimizes redundant exploration
- **Query efficiency** critical for real-world black-box attacks
- Used for assessing detector robustness on autonomous systems
