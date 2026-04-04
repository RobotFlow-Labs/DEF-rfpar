from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

from PIL import Image


_DEFAULT_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def list_image_files(root: str | Path, suffixes: Sequence[str] = _DEFAULT_SUFFIXES) -> List[Path]:
    path = Path(root)
    if not path.exists():
        raise FileNotFoundError(f"Image root does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Image root is not a directory: {path}")

    files = [p for p in sorted(path.iterdir()) if p.suffix.lower() in suffixes]
    return files


def load_rgb_image(path: str | Path) -> Image.Image:
    img = Image.open(path)
    return img.convert("RGB")


def count_images(root: str | Path, suffixes: Sequence[str] = _DEFAULT_SUFFIXES) -> int:
    return len(list_image_files(root, suffixes=suffixes))


def iter_existing(paths: Iterable[Path]) -> List[Path]:
    return [p for p in paths if p.exists()]
