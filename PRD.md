# PRD — DEF-rfpar (Wave 8 Defense Module)

> Module: DEF-rfpar | Domain: RL adversarial robustness (pixel-level black-box attacks)
> Stack: ATLAS (fleet AV) / ORACLE (through-wall)
> Source Paper: arXiv:2502.07821 (NeurIPS 2024 poster)
> Status: In Progress (planning + scaffolding)

## Objective
Build a production-grade ANIMA implementation of RFPAR that reproduces key paper outcomes, adds CUDA/MLX execution backends, and provides benchmarkable robustness evaluations for object detectors (YOLOv8/DDQ) and classifiers.

## Problem Statement
Current module state includes paper metadata and upstream reference code, but lacks:
- an ANIMA-standard architecture and package layout,
- task-level execution plan tied to paper sections,
- reproducibility harness and production serving interfaces,
- kernel/backend scaffolding for CUDA + MLX parity.

## Paper-grounded Scope
- Method core: Remember process + Forget process with one-step REINFORCE (§2.2).
- Classification evaluation: ImageNet-1K with attack sparsity and query efficiency (§3.2).
- Detection evaluation: MS-COCO and Argoverse with mAP/RM/query trade-offs (§3.3, §3.4).
- Key hyperparameters: max iterations=100, eta=0.05, T={3,20}, alpha={0.01..0.05} (§3.1, Appendix C/D).

## Success Criteria
- Reproducibility:
  - Recreate paper-like trends for query efficiency and perturbation sparsity.
  - Match or approach reported RD/RM/query scale on YOLOv8.
- Engineering:
  - Standard ANIMA 7-PRD structure and tracked tasks.
  - Python package scaffold under `src/anima_rfpar/` with configs/tests.
  - CUDA and MLX backend interfaces with shared contract.
- Operations:
  - Benchmark harness covering latency, throughput, memory, query count.
  - Production-facing API/ROS2 plan staged behind PRDs 5 and 6.

## Constraints
- Package manager: `uv` only.
- Python: 3.11 (project-pinned).
- Do not modify upstream reference logic until parity tests are in place.
- Keep dataset paths and model paths explicit via config.

## Deliverables (this autopilot pass)
- `ASSETS.md` generated from local paper/repo and online verification.
- Full PRD suite in `prds/`.
- Granular task slices in `tasks/` + build index.
- Initial scaffold in `src/`, `configs/`, `tests/`.

## Build Phases
1. Foundation: package/config/data contracts.
2. Core algorithm integration: Remember/Forget with reference parity.
3. Inference + attack pipelines.
4. Evaluation harness and benchmark reports.
5. API serving + deployment surfaces.
6. ROS2/ANIMA integration.
7. Production hardening and export pipeline.

## Dependencies
- PyTorch, torchvision, ultralytics, numpy, pillow.
- Optional: mmdetection stack for DDQ.
- Optional backend: Apple MLX.

## Risks
- Query-based attacks are expensive; without strict budget control, benchmarks become non-comparable.
- Argoverse scale and object density can mask mAP reduction despite high object removal (§3.4).
- Upstream code has implicit globals/path assumptions; adapter boundary is required.

## Non-goals (current slice)
- Full end-to-end training from scratch.
- Immediate DDQ production integration.
- Kernel micro-optimizations before baseline parity.
