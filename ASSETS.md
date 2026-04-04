# DEF-rfpar — Asset Manifest

## Paper
- Title: Amnesia as a Catalyst for Enhancing Black Box Pixel Attacks in Image Classification and Object Detection
- ArXiv: 2502.07821 (v1, submitted 2025-02-10)
- OpenReview: NeurIPS 2024 poster (published 2024-09-25)
- Authors: Dongsu Song, Daehwa Ko, Jay Hoon Jung

## Status
ALMOST — reference code and paper are present; production-scale datasets and reproducibility harness are still missing.

## Runtime Baseline
- Python: 3.11
- Package manager: `uv`

## Source Verification
- Local paper: `papers/2502.07821.pdf`
- Local reference repo: `repositories/RFPAR/`
- Online sources:
  - arXiv: https://arxiv.org/abs/2502.07821
  - OpenReview: https://openreview.net/forum?id=NTkYSWnVjl
  - Code: https://github.com/KAU-QuantumAILab/RFPAR

## Pretrained Weights
| Model | Size | Source | Local Path | Status |
|---|---:|---|---|---|
| YOLOv8n | ~6 MB | Ultralytics / reference repo | `repositories/RFPAR/yolov8n.pt` | DONE |
| ResNeXt50-32x4d | ~95 MB | torchvision | external (runtime pull) | MISSING |
| DDQ DETR-4scale | ~200+ MB | MMDetection | external (runtime pull) | MISSING |

## Datasets
| Dataset | Scope | Source | Local Path | Target Infra Path | Status |
|---|---|---|---|---|---|
| ImageNet-1K (sample zip) | classification sample | repo bundle | `repositories/RFPAR/ImageNet.zip` | `/mnt/forge-data/shared_infra/datasets/imagenet1k/` | PARTIAL |
| MS-COCO (sample zip) | object detection sample | repo bundle | `repositories/RFPAR/COCO.zip` | `/mnt/forge-data/shared_infra/datasets/coco2017/` | PARTIAL |
| Argoverse-1.1 val | large-scale AV benchmark | Argoverse | not present | `/mnt/forge-data/shared_infra/datasets/argoverse/` | MISSING |

## Hyperparameters (paper-aligned)
| Param | Value | Source |
|---|---|---|
| max_iterations | 100 | §3.1 + Appendix C/D |
| bound_threshold_eta | 0.05 | §3.1 + Appendix C/D |
| attack_rate_alpha (classification) | 0.01 | §3.1 + Appendix C |
| attack_rate_alpha (detection) | 0.01 to 0.05 | §3.1 + Appendix D |
| convergence_duration_T (classification) | 3 | §3.1 + Appendix C |
| convergence_duration_T (detection) | 20 | §3.1 + Appendix D |
| yolo_conf_threshold | 0.5 | §2.1 |

## Expected Metrics (paper)
| Benchmark | Metric | Paper Value | Our Target |
|---|---|---:|---:|
| ImageNet-1K | Query reduction vs SOTA | 26.0% | >= 20% |
| ImageNet-1K | L0 reduction vs SOTA | 41.1% | >= 30% |
| MS-COCO (YOLO) | mAP reduction (RD) | 0.29 | >= 0.25 |
| MS-COCO (YOLO) | Avg query count | 1270 | <= 1400 |
| Argoverse (YOLO) | RM | 0.94 | >= 0.90 |
| Argoverse (YOLO) | ATA | 0.10% | <= 0.15% |

## Gating Rules
- No production benchmark claim until Argoverse and full ImageNet/COCO assets are present.
- Keep reference code frozen in `repositories/RFPAR/`; do not patch upstream scripts directly.
- Place custom CUDA extension artifacts under `/mnt/forge-data/shared_infra/cuda_extensions/`.
