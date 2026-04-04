# PRD-05: API + Docker Serving

> Module: DEF-rfpar | Priority: P1
> Depends on: PRD-03
> Status: ⬜ Not started

## Objective
Expose attack and evaluation primitives through a service boundary suitable for ANIMA orchestration.

## Context (from paper)
Query-limited black-box behavior is central to deployment realism; service must meter queries and enforce budgets.

## Acceptance Criteria
- [ ] FastAPI app with `/health`, `/ready`, `/attack`, `/benchmark`.
- [ ] Request schema includes query budget and attack alpha.
- [ ] Docker serve profile created with deterministic startup.

## Files
- `src/anima_rfpar/service/app.py`
- `docker/Dockerfile.serve`
- `docker-compose.serve.yml`
- `.env.serve.example`

## Risks
- GPU/MLX heterogeneity requires backend capability discovery.
