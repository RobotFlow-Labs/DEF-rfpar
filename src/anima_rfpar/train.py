"""RFPAR attack runner — main entry point.

Usage:
    python -m anima_rfpar.train --config configs/paper.toml
    python -m anima_rfpar.train --config configs/debug.toml --mode classification
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .attack import AttackResult, run_classification_attack, run_detection_attack
from .config import AttackConfig, load_attack_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("anima_rfpar.train")


def _load_classification_data(
    imagenet_root: Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load ImageNet subset from reference repo format."""
    img_dir = imagenet_root / "images"
    label_path = imagenet_root / "label.pt"

    if not img_dir.exists():
        raise FileNotFoundError(f"ImageNet images not found at {img_dir}")

    list_images = sorted(glob.glob(str(img_dir / "*.png")))
    if not list_images:
        list_images = sorted(glob.glob(str(img_dir / "*.jpg")))
    if not list_images:
        raise FileNotFoundError(f"No images found in {img_dir}")

    labels = torch.load(label_path, map_location=device, weights_only=True)

    img_list = []
    for name in list_images:
        img = np.array(Image.open(name))
        img_list.append(img)

    images = torch.tensor(np.array(img_list)).permute(0, 3, 1, 2).float() / 255.0
    logger.info(f"Loaded {len(images)} ImageNet images, shape={images.shape}")
    return images, labels


def _load_detection_data(
    coco_root: Path,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Load COCO images for detection attack."""
    list_images = sorted(glob.glob(str(coco_root / "*.jpg")))
    if not list_images:
        raise FileNotFoundError(f"No COCO images found in {coco_root}")

    img_list = []
    hw_list = []
    for path in list_images:
        img = np.array(Image.open(path))
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        hw_list.append([img.shape[0], img.shape[1]])
        img_list.append(img)

    logger.info(f"Loaded {len(img_list)} COCO images for detection attack")
    return img_list, np.array(hw_list)


def _load_classifier(device: torch.device) -> torch.nn.Module:
    """Load ResNeXt50 pretrained classifier."""
    import torchvision.models as models

    model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
    model = model.to(device)
    model.eval()
    logger.info("Loaded ResNeXt50_32x4d classifier")
    return model


def _load_detector(model_path: str | None = None):
    """Load YOLO detector."""
    from ultralytics import YOLO

    if model_path and Path(model_path).exists():
        model = YOLO(model_path)
    elif Path("/mnt/forge-data/models/yolo11n.pt").exists():
        model = YOLO("/mnt/forge-data/models/yolo11n.pt")
    elif Path("repositories/RFPAR/yolov8n.pt").exists():
        model = YOLO("repositories/RFPAR/yolov8n.pt")
    else:
        model = YOLO("yolov8n.pt")

    logger.info(f"Loaded YOLO detector: {model.model_name}")
    return model


def save_agent_checkpoint(
    result: AttackResult, output_dir: Path, config: AttackConfig
) -> Path:
    """Save the RL agent weights as a checkpoint."""
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "agent_state_dict": result.agent_state,
        "metrics": {
            "success_rate": result.metrics.success_rate,
            "mean_l0": result.metrics.mean_l0,
            "mean_l2": result.metrics.mean_l2,
            "mean_queries": result.metrics.mean_queries,
            "mode": result.metrics.mode,
        },
        "config": {
            "mode": config.mode,
            "alpha": config.alpha,
            "max_iterations": config.max_iterations,
            "convergence_duration_t": config.convergence_duration_t,
            "bound_threshold_eta": config.bound_threshold_eta,
            "query_budget": config.query_budget,
        },
    }

    path = ckpt_dir / "best.pth"
    torch.save(ckpt, path)
    logger.info(f"Saved agent checkpoint to {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="RFPAR Attack Runner")
    parser.add_argument("--config", default="configs/paper.toml", help="Config file")
    parser.add_argument(
        "--mode", choices=["classification", "detection"], default=None,
        help="Override mode from config"
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Override max iterations")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    cfg = load_attack_config(args.config)
    mode = args.mode or cfg.mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(cfg.output_dir) / mode
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log config
    logger.info(f"[CONFIG] {args.config}")
    logger.info(f"[MODE] {mode}")
    logger.info(f"[DEVICE] {device}")
    logger.info(f"[GPU] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"[OUTPUT] {output_dir}")

    max_iter = args.max_steps or cfg.max_iterations

    if mode == "classification":
        model = _load_classifier(device)
        images, labels = _load_classification_data(Path(cfg.paths.imagenet_root), device)

        result = run_classification_attack(
            model=model,
            images=images,
            labels=labels,
            device=device,
            output_dir=output_dir,
            max_iterations=max_iter,
            alpha=cfg.alpha,
            patience=cfg.convergence_duration_t,
            limit=cfg.bound_threshold_eta,
            batch_size=50,
            lr=1e-4,
            seed=cfg.seed,
        )

    elif mode == "detection":
        model = _load_detector()
        coco_root = Path(cfg.paths.coco_root)
        if not coco_root.exists():
            # Fallback to COCO val2017
            coco_root = Path("/mnt/forge-data/datasets/coco/val2017")
        images_np, hw_array = _load_detection_data(coco_root)

        result = run_detection_attack(
            model=model,
            images=images_np,
            hw_array=hw_array,
            device=device,
            output_dir=output_dir,
            max_iterations=max_iter,
            alpha=cfg.alpha,
            patience=cfg.convergence_duration_t,
            limit=cfg.bound_threshold_eta,
            batch_size=50,
            lr=1e-4,
            conf=cfg.yolo_conf_threshold,
            seed=cfg.seed,
        )
    else:
        logger.error(f"Unknown mode: {mode}")
        return 1

    # Save checkpoint
    save_agent_checkpoint(result, output_dir, cfg)

    # Print final report
    m = result.metrics
    logger.info("=" * 60)
    logger.info(f"RFPAR Attack Complete — {m.mode}")
    logger.info(f"  Images:      {m.total_images}")
    logger.info(f"  Deceived:    {m.total_deceived}")
    logger.info(f"  Success:     {m.success_rate:.1%}")
    logger.info(f"  Mean L0:     {m.mean_l0:.1f}")
    logger.info(f"  Mean L2:     {m.mean_l2:.2f}")
    logger.info(f"  Avg queries: {m.mean_queries:.0f}")
    logger.info(f"  Forget iter: {m.forget_iterations}")
    logger.info(f"  Time:        {m.elapsed_sec:.1f}s")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
