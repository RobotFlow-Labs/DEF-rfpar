# PRD-02: Core Attack Model (Remember + Forget)

> Module: DEF-rfpar | Priority: P0
> Depends on: PRD-01
> Status: ⬜ Not started

## Objective
Implement the core RFPAR loop as reusable pipeline primitives while preserving behavior parity with the reference repo.

## Context (from paper)
RFPAR combines one-step REINFORCE, memory retention of best perturbations, and periodic forgetting/reset to reduce random drift and improve query efficiency (§2.2).

Paper references:
- §2.2 Remember process
- §2.2 Forget process
- Convergence condition using eta and T (§2.2 equations around reward bound)

## Acceptance Criteria
- [ ] Pipeline supports classification and detection modes.
- [ ] Convergence/forget reset logic is parameterized.
- [ ] Adapter can execute reference-style action sampling and reward update.
- [ ] Smoke run works on small sample image batch.

## Files
- `src/anima_rfpar/reference.py`
- `src/anima_rfpar/pipeline.py`
- `tests/test_pipeline_contract.py`

## Risks
- Upstream implementation mixes global path and mutable env state.
- Action shape logic differs for `attack_pixel == 1` vs multi-pixel.
