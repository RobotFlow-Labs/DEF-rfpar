"""RFPAR serving module — AnimaNode subclass for inference."""
from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .agent import REINFORCEAgent

logger = logging.getLogger(__name__)


class RFPARNode:
    """RFPAR inference node — generates adversarial perturbation suggestions."""

    def __init__(self):
        self.agent: REINFORCEAgent | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector_mode = False
        self.weights_loaded = False

    def setup_inference(self, checkpoint_path: str | Path | None = None):
        """Load agent weights for inference."""
        if checkpoint_path is None:
            checkpoint_path = Path("/data/weights/best.pth")

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            return

        # weights_only=False required: checkpoint contains config dicts alongside state_dict
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        config = ckpt.get("config", {})
        self.detector_mode = config.get("mode", "classification") == "detection"

        self.agent = REINFORCEAgent(
            img_h=224,
            img_w=224,
            channels=3,
            detector_mode=self.detector_mode,
        ).to(self.device)
        self.agent.load_state_dict(ckpt["agent_state_dict"])
        self.agent.eval()
        self.weights_loaded = True
        logger.info(f"RFPAR agent loaded from {ckpt_path}")

    @torch.no_grad()
    def predict(self, image_bytes: bytes) -> dict:
        """Predict pixel perturbation actions for an input image.

        Returns dict with action means and stds for adversarial pixel selection.
        """
        if self.agent is None:
            return {"error": "Agent not loaded"}

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)

        mean, std = self.agent(tensor)
        actions = torch.sigmoid(mean)

        return {
            "action_mean": mean[0].cpu().tolist(),
            "action_std": std[0].cpu().tolist(),
            "pixel_x": int(actions[0, 0].item() * 224),
            "pixel_y": int(actions[0, 1].item() * 224),
            "color_r": float(actions[0, 2].item()),
            "color_g": float(actions[0, 3].item()),
            "color_b": float(actions[0, 4].item()),
            "mode": "detection" if self.detector_mode else "classification",
        }

    def get_status(self) -> dict:
        return {
            "agent_loaded": self.weights_loaded,
            "device": str(self.device),
            "mode": "detection" if self.detector_mode else "classification",
        }
