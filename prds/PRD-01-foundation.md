# PRD-01: Foundation, Config, and Data Contracts

> Module: DEF-rfpar | Priority: P0
> Depends on: None
> Status: ⬜ Not started

## Objective
Create the ANIMA-compliant base package, configuration system, and dataset ingress contracts needed by all later PRDs.

## Context (from paper)
RFPAR applies one-step RL attack loops across ImageNet-1K and object detection datasets (MS-COCO, Argoverse), with strict hyperparameter controls (§3.1).

Paper references:
- §2.1, §2.2 for task formulation and Remember process.
- §3.1 for datasets, metrics, and attack hyperparameters.

## Acceptance Criteria
- [ ] `src/anima_rfpar/` package exists with typed config + data contracts.
- [ ] `configs/default.toml`, `configs/paper.toml`, `configs/debug.toml` exist.
- [ ] Dataset path resolver handles ImageNet/COCO/Argoverse roots.
- [ ] Basic config and data unit tests exist.

## Files
- `pyproject.toml`
- `src/anima_rfpar/config.py`
- `src/anima_rfpar/types.py`
- `src/anima_rfpar/data.py`
- `configs/default.toml`
- `configs/paper.toml`
- `configs/debug.toml`
- `tests/test_config.py`
- `tests/test_data.py`

## Risks
- Dataset layouts differ between sample zips and production mirrors.
- Upstream reference scripts assume working directory side effects.
