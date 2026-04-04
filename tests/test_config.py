from pathlib import Path

from anima_rfpar.config import load_attack_config


def test_load_attack_config_defaults() -> None:
    cfg = load_attack_config("configs/default.toml")
    assert cfg.module_name == "DEF-rfpar"
    assert cfg.mode in {"classification", "detection"}
    assert cfg.backend in {"cpu", "cuda", "mlx"}
    assert cfg.max_iterations > 0


def test_load_attack_config_paths_are_pathlike() -> None:
    cfg = load_attack_config("configs/default.toml")
    assert isinstance(cfg.paths.reference_repo, Path)
    assert isinstance(cfg.paths.imagenet_root, Path)
