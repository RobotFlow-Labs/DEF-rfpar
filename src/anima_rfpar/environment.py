"""RFPAR Environment — applies pixel perturbations and computes rewards.

Handles both classification (softmax confidence) and detection (YOLO box counts).
"""
from __future__ import annotations

import torch
import numpy as np


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def normalize_imagenet(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean = IMAGENET_MEAN.view(1, 3, 1, 1).to(device)
    std = IMAGENET_STD.view(1, 3, 1, 1).to(device)
    return (images - mean) / std


def apply_pixel_perturbations(
    images: torch.Tensor,
    actions: torch.Tensor,
    n_pixels: int,
    detector_mode: bool = False,
) -> torch.Tensor:
    """Apply pixel perturbations from actions to images.

    Args:
        images: (B, C, H, W) original images
        actions: (B*n_pixels, 5) — sigmoid(x, y, r, g, b)
        n_pixels: number of pixels to perturb per image
        detector_mode: if True, color values are 0/255 binary
    """
    B, C, H, W = images.shape
    actions = torch.sigmoid(actions)

    x = (actions[:, 0] * H - 1).long().clamp(0, H - 1)
    y = (actions[:, 1] * W - 1).long().clamp(0, W - 1)

    if detector_mode:
        r = (actions[:, 2] > 0.5).float() * 255
        g = (actions[:, 3] > 0.5).float() * 255
        b = (actions[:, 4] > 0.5).float() * 255
    else:
        r = (actions[:, 2] > 0.5).float()
        g = (actions[:, 3] > 0.5).float()
        b = (actions[:, 4] > 0.5).float()

    result = []
    for i in range(B):
        changed = images[i].clone()
        if n_pixels == 1:
            changed[0, x[i], y[i]] = r[i]
            changed[1, x[i], y[i]] = g[i]
            changed[2, x[i], y[i]] = b[i]
        else:
            idx = torch.arange(n_pixels, device=images.device) * B + i
            idx = idx[idx < len(x)]
            if detector_mode:
                changed[0, x[idx], y[idx]] = r[idx].to(torch.uint8).float()
                changed[1, x[idx], y[idx]] = g[idx].to(torch.uint8).float()
                changed[2, x[idx], y[idx]] = b[idx].to(torch.uint8).float()
            else:
                changed[0, x[idx], y[idx]] = r[idx]
                changed[1, x[idx], y[idx]] = g[idx]
                changed[2, x[idx], y[idx]] = b[idx]
        result.append(changed.unsqueeze(0))

    return torch.cat(result, 0)


class ClassificationEnv:
    """Environment for classification attacks (ResNeXt50, etc.)."""

    def __init__(self, model: torch.nn.Module, device: torch.device, n_pixels: int = 1):
        self.model = model
        self.device = device
        self.n_pixels = n_pixels
        self.ori_prob: torch.Tensor | None = None

    @torch.no_grad()
    def step(
        self,
        images: torch.Tensor,
        actions: torch.Tensor,
        labels: torch.Tensor,
        ori_prob: torch.Tensor | None = None,
        init: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute one attack step.

        Args:
            ori_prob: original confidence for this batch (used when not init)

        Returns: (rewards, change_list, changed_images, changed_preds)
        """
        changed = apply_pixel_perturbations(images, actions, self.n_pixels, detector_mode=False)
        labels = labels.to(self.device)

        if init:
            orig_out = torch.softmax(
                self.model(normalize_imagenet(images.to(self.device), self.device)), dim=1
            )
            self.ori_prob = orig_out[np.arange(labels.shape[0]), labels].clone()
            ori_prob = self.ori_prob

        changed_out = torch.softmax(
            self.model(normalize_imagenet(changed.to(self.device), self.device)), dim=1
        )
        changed_preds_val, changed_preds_idx = torch.max(changed_out, dim=1)

        change_list = (labels != changed_preds_idx).to(self.device)
        rewards = ori_prob - changed_out[np.arange(labels.shape[0]), labels]

        return rewards, change_list, changed.to(self.device), changed_preds_idx


class DetectionEnv:
    """Environment for detection attacks (YOLO)."""

    def __init__(self, model, device: torch.device, n_pixels: int = 1, conf: float = 0.5):
        self.model = model
        self.device = device
        self.n_pixels = n_pixels
        self.conf = conf
        self.ori_prob: list = []
        self.ori_cls: list = []
        self.ori_box_num: list = []

    @torch.no_grad()
    def step(
        self,
        images: torch.Tensor,
        actions: torch.Tensor,
        bt: int,
        init: bool = False,
        labels: list | None = None,
        probs: list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Detection attack step — compare box counts before/after perturbation.

        Returns: (rewards, dif_list, changed_images_np)
        """
        changed = apply_pixel_perturbations(images.to(self.device), actions, self.n_pixels, True)

        if init:
            labels = []
            probs = []
            for img in images:
                result = self.model(
                    img.detach().cpu().numpy().transpose(1, 2, 0),
                    imgsz=640,
                    conf=self.conf,
                    verbose=False,
                )
                probs.append(result[0].boxes.conf)
                labels.append(result[0].boxes.cls)
            self.ori_prob.extend(probs)
            self.ori_cls.extend(labels)

        changed_np = changed.detach().cpu().numpy().transpose(0, 2, 3, 1)
        prob_list = []
        cls_list = []

        for img_np in changed_np:
            results = self.model(img_np, conf=self.conf, verbose=False)
            prob_list.append(results[0].boxes.conf)
            cls_list.append(results[0].boxes.cls)

        rewards = []
        dif_list = []
        batch_size = images.shape[0]

        for i in range(len(cls_list)):
            size = max(
                torch.bincount(labels[i].long()).shape[0],
                torch.bincount(cls_list[i].long()).shape[0],
            )
            temp = (
                torch.bincount(labels[i].long(), minlength=size)
                - torch.bincount(cls_list[i].long(), minlength=size)
            )
            dif = temp.sum()
            dif_list.append(dif)

            if (temp != 0).any():
                pos_idx = [j for j in range(len(temp)) if temp[j] > 0]
                pos_val = [temp[j].item() for j in pos_idx]
                for idx_val, count in zip(pos_idx, pos_val):
                    add_cls = torch.full((count,), idx_val, dtype=torch.long, device=self.device)
                    add_prob = torch.zeros(count, device=self.device)
                    cls_list[i] = torch.cat((cls_list[i].long(), add_cls), 0)
                    prob_list[i] = torch.cat((prob_list[i], add_prob), 0)

                neg_idx = [j for j in range(len(temp)) if temp[j] < 0]
                neg_val = [abs(temp[j].item()) for j in neg_idx]
                for idx_val, count in zip(neg_idx, neg_val):
                    add_cls = torch.full((count,), idx_val, dtype=torch.long, device=self.device)
                    add_prob = torch.zeros(count, device=self.device)
                    labels[i] = torch.cat((labels[i].long(), add_cls), 0)
                    probs[i] = torch.cat((probs[i], add_prob), 0)

            reward = (
                probs[i][labels[i].sort()[1]].cpu()
                - prob_list[i][cls_list[i].sort()[1]].cpu()
            ).sum() + dif
            rewards.append(reward)

        return (
            torch.tensor(rewards, device=self.device),
            torch.tensor(dif_list),
            changed_np,
        )
