from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetPaths:
    reference_repo: Path
    imagenet_root: Path
    coco_root: Path
    argoverse_root: Path


@dataclass(frozen=True)
class AttackRunSummary:
    mode: str
    backend: str
    dry_run: bool
    query_budget: int
    output_dir: Path
