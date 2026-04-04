# PRD-03: Inference and Attack Execution Pipeline

> Module: DEF-rfpar | Priority: P0
> Depends on: PRD-02
> Status: ⬜ Not started

## Objective
Provide runnable entry points for classification and detection attack execution with reproducible input/output artifacts.

## Context (from paper)
Experiments evaluate both classification and detection tasks under constrained pixel budgets and query limits (§3.2–§3.4).

## Acceptance Criteria
- [ ] CLI entrypoint supports `classification` and `detection` modes.
- [ ] Outputs include adversarial images, deltas, and per-sample metrics.
- [ ] Query counters and attack stats are serialized.

## Files
- `src/anima_rfpar/cli.py`
- `src/anima_rfpar/pipeline.py`
- `tests/test_cli_contract.py`

## Risks
- Detector outputs vary by backend/version, requiring stable post-processing.
