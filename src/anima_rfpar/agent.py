"""REINFORCE agent for RFPAR pixel selection.

Paper: ArXiv 2502.07821 — one-step RL with learnable mean+std for pixel actions.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class REINFORCEAgent(nn.Module):
    """Policy network that outputs (x, y, r, g, b) action distributions."""

    def __init__(
        self,
        img_h: int = 224,
        img_w: int = 224,
        channels: int = 3,
        lr: float = 1e-4,
        detector_mode: bool = False,
    ):
        super().__init__()
        self.detector_mode = detector_mode

        self.conv1 = nn.Conv2d(channels, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.relu = nn.ReLU()

        fc_in = (img_h // 4) * (img_w // 4) * 64
        self.fc = nn.Linear(fc_in, 512)
        self.action_mean = nn.Linear(512, 5)
        self.action_logstd = nn.Linear(512, 5)

        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.relu(self.conv1(state))
        x = F.max_pool2d(x, 2)
        x = self.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc(x))

        mean = self.action_mean(x)
        logstd = self.action_logstd(x)

        if self.detector_mode:
            std = 2 * torch.sigmoid(logstd)
        else:
            std = torch.exp(logstd)
        return mean, std

    def train_step(self, log_probs: torch.Tensor, rewards: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        loss = (-log_probs * rewards).sum()
        loss.backward()
        self.optimizer.step()
        return loss.item()


def sample_actions(
    means: torch.Tensor,
    stds: torch.Tensor,
    n_pixels: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample pixel perturbation actions from Gaussian policy.

    Returns:
        actions: (n_pixels, batch, 5) if n_pixels > 1, else (batch, 5)
        log_probs: corresponding log probabilities
    """
    stds = torch.clamp(stds, 0.1, 10)
    cov = torch.diag_embed(stds.pow(2))
    distribution = dist.MultivariateNormal(means, cov)

    if n_pixels == 1:
        actions = distribution.sample()
    else:
        actions = distribution.sample((n_pixels,))

    log_probs = distribution.log_prob(actions)
    return actions, log_probs
