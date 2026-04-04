from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from .types import DatasetPaths

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore


_VALID_BACKENDS = {"cpu", "cuda", "mlx"}
_VALID_MODES = {"classification", "detection"}


@dataclass(frozen=True)
class AttackConfig:
    module_name: str
    seed: int
    backend: str
    mode: str
    dry_run: bool
    output_dir: Path
    paths: DatasetPaths
    max_iterations: int
    bound_threshold_eta: float
    alpha: float
    convergence_duration_t: int
    query_budget: int
    yolo_conf_threshold: float


def _read_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _required(section: Dict[str, Any], key: str, file_path: Path) -> Any:
    if key not in section:
        raise ValueError(f"Missing key '{key}' in section from {file_path}")
    return section[key]


def load_attack_config(path: str | Path) -> AttackConfig:
    file_path = Path(path)
    raw = _read_toml(file_path)

    module = raw.get("module", {})
    execution = raw.get("execution", {})
    paths_raw = raw.get("paths", {})
    attack = raw.get("attack", {})

    backend = str(_required(execution, "backend", file_path)).lower()
    mode = str(_required(execution, "mode", file_path)).lower()

    if backend not in _VALID_BACKENDS:
        raise ValueError(f"Invalid backend '{backend}', expected one of {_VALID_BACKENDS}")
    if mode not in _VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}', expected one of {_VALID_MODES}")

    dataset_paths = DatasetPaths(
        reference_repo=Path(_required(paths_raw, "reference_repo", file_path)),
        imagenet_root=Path(_required(paths_raw, "imagenet_root", file_path)),
        coco_root=Path(_required(paths_raw, "coco_root", file_path)),
        argoverse_root=Path(_required(paths_raw, "argoverse_root", file_path)),
    )

    return AttackConfig(
        module_name=str(module.get("name", "DEF-rfpar")),
        seed=int(module.get("seed", 2)),
        backend=backend,
        mode=mode,
        dry_run=bool(execution.get("dry_run", True)),
        output_dir=Path(execution.get("output_dir", "outputs")),
        paths=dataset_paths,
        max_iterations=int(_required(attack, "max_iterations", file_path)),
        bound_threshold_eta=float(_required(attack, "bound_threshold_eta", file_path)),
        alpha=float(_required(attack, "alpha", file_path)),
        convergence_duration_t=int(_required(attack, "convergence_duration_t", file_path)),
        query_budget=int(_required(attack, "query_budget", file_path)),
        yolo_conf_threshold=float(_required(attack, "yolo_conf_threshold", file_path)),
    )
