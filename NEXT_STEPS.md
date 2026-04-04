# NEXT_STEPS — DEF-rfpar (RFPAR)
## Last Updated: 2026-04-04
## Status: PRD + TASK + SCAFFOLD COMPLETE
## MVP Readiness: 35% (planning+scaffold), 0% (baseline parity), 0% (kernel optimization)

## Completed in this Session

- Generated full module asset manifest from local paper/repo + online references.
- Generated full top-level PRD and the 7-PRD suite.
- Re-sliced build work into granular task files with dependency order.
- Scaffolded executable package under `src/anima_rfpar/`.
- Added config profiles (`default`, `paper`, `debug`) and baseline tests.
- Added service/docker/ros2/docs/reports placeholders to unblock later PRDs.

## Current Build Frontier

Focus on **PRD-01** and **PRD-02** execution:

- [ ] Create `uv` environment with Python 3.11
- [ ] PRD-0101 — bootstrap and dependency lock with `uv`
- [ ] PRD-0102 — validate config loading against all profiles
- [ ] PRD-0103 — harden dataset contract checks for real dataset layouts
- [ ] PRD-0104 — run and fix baseline tests
- [ ] PRD-0201 — wire real reference module loading and smoke import checks
- [ ] PRD-0202 — implement first executable Remember/Forget loop using reference logic

## Blockers

- Production-scale datasets are not available locally (Argoverse missing).
- DDQ stack is not wired into current scaffold yet.

## Key Paths

- PRD root: `PRD.md`
- PRD suite: `prds/`
- Task index: `tasks/INDEX.md`
- Package scaffold: `src/anima_rfpar/`
- Configs: `configs/`
- Paper: `papers/2502.07821.pdf`
- Reference implementation: `repositories/RFPAR/`
