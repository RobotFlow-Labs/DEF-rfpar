# PRD-07: Production Hardening and Export

> Module: DEF-rfpar | Priority: P2
> Depends on: PRD-04
> Status: ⬜ Not started

## Objective
Harden the module for reproducible operation, error containment, backend parity, and deployment artifacts.

## Context (from paper)
Method effectiveness is tightly linked to query accounting and robust evaluation practice; production must guard those invariants.

## Acceptance Criteria
- [ ] Structured error model and retry semantics added.
- [ ] CUDA/MLX parity report generated for core operations.
- [ ] Export and packaging path documented (weights/config/reports).
- [ ] Final report includes achieved vs paper metrics.

## Files
- `docs/production_runbook.md`
- `reports/TRAINING_REPORT.md`
- `reports/PARITY_REPORT.md`

## Risks
- Version skew across Ultralytics/PyTorch/MMDetection can drift metrics.
