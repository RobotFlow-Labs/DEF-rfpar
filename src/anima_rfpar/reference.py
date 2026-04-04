from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict

_REFERENCE_REQUIRED = [
    "Adversarial_RL_simple",
    "Environment",
    "config",
    "utils",
]


def load_reference_modules(reference_repo: str | Path = "repositories/RFPAR") -> Dict[str, ModuleType]:
    repo = Path(reference_repo)
    if not repo.exists():
        raise FileNotFoundError(f"Reference repo is missing: {repo}")

    if str(repo.resolve()) not in sys.path:
        sys.path.insert(0, str(repo.resolve()))

    loaded: Dict[str, ModuleType] = {}
    for module_name in _REFERENCE_REQUIRED:
        loaded[module_name] = importlib.import_module(module_name)
    return loaded
