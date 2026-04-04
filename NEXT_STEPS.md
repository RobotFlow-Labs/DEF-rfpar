# NEXT_STEPS — DEF-rfpar: Remember & Forget Pixel Attack using RL
## Last Updated: 2026-04-04
## Status: DOCUMENTATION COMPLETE — Ready for implementation
## MVP Readiness: 15% (documentation), 0% (code)

## What Was Completed This Session

✅ **CLAUDE.md** — Paper enriched with arxiv 2502.07821 (RL pixel attacks)
✅ **tasks/README.md** — 10 actionable PRD tasks
✅ **kernels/README.md** — 3 CUDA kernel specifications
✅ **benchmarks/README.md** — Paper metrics + implementation targets

## Implementation Phases (Est. 68 hours total)

| Phase | Tasks | Hours | Status |
|-------|-------|-------|--------|
| 1. Environment & Baseline | PRD-001, 002 | 14h | TODO |
| 2. RL Environment | PRD-003, 004 | 18h | TODO |
| 3. CUDA Kernels | PRD-005, 006 | 18h | TODO |
| 4. MLX Port & Dual-Compute | PRD-007 | 14h | TODO |
| 5. Query Optimization | PRD-008 | 6h | TODO |
| 6. Edge Deployment | PRD-009 | 12h | TODO |
| 7. Benchmarking | PRD-010 | 6h | TODO |

## Immediate Next Actions (Phase 1)

**Environment Setup**:
- [ ] Clone https://github.com/KAU-QuantumAILab/RFPAR
- [ ] Run `uv sync` (RL environment, gym)
- [ ] Download datasets (46.5GB total):
  - ImageNet-1K validation (50K, 6.5GB)
  - Argoverse (autonomous driving, 40GB)

**Baseline**:
- [ ] Load YOLOv8 model
- [ ] Measure mAP on Argoverse
- [ ] Profile baseline attack latency

## Models/Datasets

| Item | Size | Location |
|------|------|----------|
| ImageNet-1K | 6.5GB | `/mnt/forge-data/.../imagenet1k/` |
| Argoverse | 40GB | `/mnt/forge-data/.../argoverse/` |
| YOLOv8 | 100-200MB | Ultralytics |

## Blockers
None — ready to start.

## Key Paper Results
- Attack success: 95%+ with <100 pixels (<1% of image)
- Query efficiency: 30-40% reduction vs baseline
- mAP reduction: 50-70% on Argoverse
- Transferability: 75%+ to unseen models
