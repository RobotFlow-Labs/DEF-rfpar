# DEF-rfpar Task Index — 21 Tasks

## Build Order

| Task | Title | Depends | Status |
|---|---|---|---|
| PRD-0101 | Bootstrap package and pyproject | None | ⬜ |
| PRD-0102 | Implement typed configuration loader | PRD-0101 | ⬜ |
| PRD-0103 | Implement dataset/path contracts | PRD-0102 | ⬜ |
| PRD-0104 | Add foundation tests | PRD-0102, PRD-0103 | ⬜ |

| PRD-0201 | Build reference adapter boundary | PRD-0103 | ⬜ |
| PRD-0202 | Implement Remember/Forget pipeline skeleton | PRD-0201 | ⬜ |
| PRD-0203 | Add action/reward contract tests | PRD-0202 | ⬜ |

| PRD-0301 | Add CLI attack entrypoint | PRD-0202 | ⬜ |
| PRD-0302 | Implement output artifact writer | PRD-0301 | ⬜ |
| PRD-0303 | Add attack smoke-flow test harness | PRD-0302 | ⬜ |

| PRD-0401 | Define benchmark result schema | PRD-0302 | ⬜ |
| PRD-0402 | Implement metric calculators (L0/L2/RM/RD) | PRD-0401 | ⬜ |
| PRD-0403 | Add benchmark schema and aggregate tests | PRD-0402 | ⬜ |

| PRD-0501 | Scaffold FastAPI service | PRD-0301 | ⬜ |
| PRD-0502 | Add docker serving profile | PRD-0501 | ⬜ |
| PRD-0503 | Add service contract tests | PRD-0501 | ⬜ |

| PRD-0601 | Scaffold ROS2 node contract | PRD-0501 | ⬜ |
| PRD-0602 | Add ROS2 launch/config stubs | PRD-0601 | ⬜ |
| PRD-0603 | Add anima_module.yaml skeleton | PRD-0602 | ⬜ |

| PRD-0701 | Add production runbook and parity checklist | PRD-0402 | ⬜ |
| PRD-0702 | Add release checklist and failure-mode table | PRD-0701 | ⬜ |
| PRD-0703 | Add final benchmark report template | PRD-0702 | ⬜ |

## Notes
- Ordered by PRD sequence from `prds/README.md`.
- Each PRD now has 3-6 tasks as required by `/anima-create-prd` playbook.
- Use this file as the canonical build queue for `/anima-autopilot`.
