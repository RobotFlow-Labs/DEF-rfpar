# PRD-06: ROS2 + ANIMA Integration

> Module: DEF-rfpar | Priority: P1
> Depends on: PRD-05
> Status: ⬜ Not started

## Objective
Provide ROS2 node and ANIMA manifest bindings for online robustness probing in ATLAS/ORACLE pipelines.

## Context (from paper)
Method is relevant for camera-based perception robustness under sparse perturbation assumptions.

## Acceptance Criteria
- [ ] ROS2 node wraps attack evaluation for incoming image streams.
- [ ] Query limits and attack profiles configurable by topic/service params.
- [ ] ANIMA module manifest references service and topic contracts.

## Files
- `ros2/rfpar_node.py`
- `ros2/launch/rfpar.launch.py`
- `anima_module.yaml`

## Risks
- Real-time throughput constraints on edge platforms.
