# PRD-04: Evaluation and Benchmark Harness

> Module: DEF-rfpar | Priority: P1
> Depends on: PRD-03
> Status: ⬜ Not started

## Objective
Implement a benchmark harness that computes paper-aligned metrics (success rate, L0, query count, mAP reduction, RM/ATA).

## Context (from paper)
Paper compares against prior methods by success, L0, query count, and object detection RD/mAP metrics (Tables 1–4; §3.2–§3.4).

## Acceptance Criteria
- [ ] Evaluation runner emits structured JSON/CSV summaries.
- [ ] Supports ImageNet-like and COCO/Argoverse-like evaluation modes.
- [ ] Tracks metric deltas against paper targets in `ASSETS.md`.

## Files
- `src/anima_rfpar/benchmarks.py`
- `benchmarks/rfpar_benchmark_spec.md`
- `tests/test_benchmark_schema.py`

## Risks
- Argoverse object density can reduce mAP drop sensitivity despite high RM.
