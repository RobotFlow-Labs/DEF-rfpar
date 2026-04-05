"""RFPAR Attack — Remember and Forget process.

Classification attack on ImageNet with ResNeXt50.
Detection attack on COCO with YOLO.
Closely follows reference: github.com/KAU-QuantumAILab/RFPAR
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.distributions as dist
from PIL import Image

from .agent import REINFORCEAgent

logger = logging.getLogger(__name__)


@dataclass
class AttackMetrics:
    mode: str = ""
    total_images: int = 0
    total_deceived: int = 0
    success_rate: float = 0.0
    mean_l0: float = 0.0
    mean_l2: float = 0.0
    mean_queries: float = 0.0
    forget_iterations: int = 0
    elapsed_sec: float = 0.0


@dataclass
class AttackResult:
    metrics: AttackMetrics = field(default_factory=AttackMetrics)
    agent_state: dict | None = None


def _l0_norm(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    return [torch.count_nonzero(a - b).cpu().item()]


def _l2_norm(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    return [torch.norm((a.float() - b.float()), p=2).cpu().item()]


def _early_stopping(delta: float, stop_count: int, limit: float, patience: int):
    if delta <= limit:
        stop_count += 1
        return stop_count, stop_count >= patience
    return 0, False


_IMAGENET_MEAN: torch.Tensor | None = None
_IMAGENET_STD: torch.Tensor | None = None


def _normalize_imagenet(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    global _IMAGENET_MEAN, _IMAGENET_STD
    if _IMAGENET_MEAN is None or _IMAGENET_MEAN.device != device:
        _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        _IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (x - _IMAGENET_MEAN) / _IMAGENET_STD


def _sample_action(means, stds, n_pixels, device):
    stds = torch.clamp(stds, 0.1, 10)
    cov = torch.diag_embed(stds.pow(2)).to(device)
    distribution = dist.MultivariateNormal(means, cov)
    if n_pixels == 1:
        actions = distribution.sample()
    else:
        actions = distribution.sample((n_pixels,))
    log_probs = distribution.log_prob(actions)
    return actions, log_probs


def _make_transformed_cls(original_images, actions, n_pixels):
    """Apply pixel perturbations for classification (values in [0,1])."""
    dev = original_images.device
    B, C, H, W = original_images.shape
    actions = torch.sigmoid(actions)
    x = (actions[:, 0] * H - 1).long().clamp(0, H - 1)
    y = (actions[:, 1] * W - 1).long().clamp(0, W - 1)
    r = (actions[:, 2] > 0.5).float()
    g = (actions[:, 3] > 0.5).float()
    b = (actions[:, 4] > 0.5).float()

    arr = []
    for i in range(B):
        changed = original_images[i].clone()
        if n_pixels == 1:
            changed[0, x[i], y[i]] = r[i]
            changed[1, x[i], y[i]] = g[i]
            changed[2, x[i], y[i]] = b[i]
        else:
            idx = (torch.arange(n_pixels, device=dev) * B + i)
            idx = idx[idx < len(x)]
            changed[0, x[idx], y[idx]] = r[idx]
            changed[1, x[idx], y[idx]] = g[idx]
            changed[2, x[idx], y[idx]] = b[idx]
        arr.append(changed.unsqueeze(0))
    return torch.cat(arr, 0)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y

    def __getitem__(self, i):
        return self.x_data[i], self.y_data[i]

    def __len__(self):
        return self.x_data.shape[0]


def run_classification_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    output_dir: Path,
    max_iterations: int = 100,
    alpha: float = 0.01,
    patience: int = 3,
    limit: float = 0.05,
    batch_size: int = 50,
    lr: float = 1e-4,
    seed: int = 2,
) -> AttackResult:
    """Run RFPAR classification attack — faithful to reference main_cls.py."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    N, C, H, W = images.shape
    n_pixels = max(1, int((H + W) / 2 * alpha))

    adv_dir = output_dir / "adv_images"
    delta_dir = output_dir / "delta_images"
    adv_dir.mkdir(parents=True, exist_ok=True)
    delta_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    train_data = _Dataset(images.clone().to(device), labels.to(device))

    update_images = images.clone().to(device)
    metric_images = images.clone().to(device)
    idx_list = torch.arange(N, device=device)
    init_ori_prob: list[torch.Tensor] = []

    total_deceived = 0
    L0: list[float] = []
    L2: list[float] = []
    query_count = 0
    save_labels = torch.zeros(N)
    remember_step = 0
    ori_prob_full = torch.tensor([], device=device)
    update_rewards = torch.zeros(update_images.shape[0], device=device)

    logger.info(
        f"[RFPAR-CLS] N={N}, pixels={n_pixels}, alpha={alpha}, "
        f"patience={patience}, limit={limit}, bs={batch_size}"
    )

    for p in range(max_iterations):
        # Forget: reinitialize agent
        agent = REINFORCEAgent(H, W, C, lr=lr, detector_mode=False).to(device)

        if p != 0:
            init_ori_prob = [ori_prob_full - update_rewards]
        else:
            init_ori_prob = []

        ori_prob_full = (
            torch.cat(init_ori_prob, 0) if init_ori_prob else torch.tensor([], device=device)
        )

        flag = False
        stop_count = 0
        deceived_count = 0
        succes_list = []
        succes_list_idx = []
        succes_labels_list = []
        delta_list = []
        update_rewards = torch.zeros(update_images.shape[0], device=device)

        while True:
            loader = torch.utils.data.DataLoader(
                train_data, batch_size=batch_size, shuffle=False
            )
            remember_step += 1

            train_x_parts = []
            train_y_parts = []
            change_train_x_parts = []
            dip = 0
            total_rewards_list: list[float] = []
            total_change_list = torch.tensor([], device=device)
            init_step = len(ori_prob_full) == 0

            batch_ori_probs: list[torch.Tensor] = []

            for bt, (s, lbl) in enumerate(loader):
                s = s.to(device)
                lbl = lbl.to(device)

                # Forward through agent
                action_means, action_stds = agent(s)
                actions, log_probs = _sample_action(action_means, action_stds, n_pixels, device)

                if n_pixels != 1:
                    actions = actions.view(-1, 5)
                    log_probs = log_probs.sum(axis=0)

                # Determine ori_prob for this batch
                if init_step:
                    # First time: compute original confidence
                    with torch.no_grad():
                        orig_out = torch.softmax(
                            model(_normalize_imagenet(s.float(), device)), dim=1
                        )
                        batch_op = orig_out[np.arange(lbl.shape[0]), lbl]
                        batch_ori_probs.append(batch_op.clone())
                        ori_prob_batch = batch_op
                else:
                    start = bt * batch_size
                    end = start + len(lbl)
                    ori_prob_batch = ori_prob_full[start:end]

                # Apply perturbation and get reward
                changed = _make_transformed_cls(s.float(), actions, n_pixels)
                query_count += len(lbl)

                with torch.no_grad():
                    changed_out = torch.softmax(
                        model(_normalize_imagenet(changed.to(device), device)), dim=1
                    )
                    changed_preds_val, changed_preds_idx = torch.max(changed_out, dim=1)
                    change_list = (lbl != changed_preds_idx).to(device)
                    rewards = ori_prob_batch - changed_out[np.arange(lbl.shape[0]), lbl]

                # Collect results
                train_x_parts.append(s[change_list == 0])
                train_y_parts.append(lbl[change_list == 0])
                succes_list.append(changed[change_list == 1].to(device))
                succes_list_idx.append(
                    idx_list[batch_size * bt : batch_size * bt + len(lbl)][change_list == 1]
                )
                succes_labels_list.append(changed_preds_idx[change_list == 1])

                change_train_x_parts.append(changed.to(device))
                total_change_list = torch.cat(
                    (total_change_list, change_list.float()), dim=0
                ).long()
                total_rewards_list.extend(rewards.tolist())
                dip += change_list.sum().item()

                # Train RL
                agent.train_step(log_probs, rewards + change_list.float())

            # After first full pass, set ori_prob_full
            if init_step and batch_ori_probs:
                ori_prob_full = torch.cat(batch_ori_probs, 0)

            if not train_x_parts or all(t.shape[0] == 0 for t in train_x_parts):
                break

            change_train_x = torch.cat(change_train_x_parts, 0)
            train_x = torch.cat([t for t in train_x_parts if t.shape[0] > 0], 0)
            train_y = torch.cat([t for t in train_y_parts if t.shape[0] > 0], 0)

            deceived_count += dip

            # Compute metrics for deceived images
            if total_change_list.sum() > 0:
                for idx_i in range(len(total_change_list)):
                    if total_change_list[idx_i] == 1:
                        L0.extend(_l0_norm(metric_images[idx_i], change_train_x[idx_i]))
                        L2.extend(_l2_norm(metric_images[idx_i], change_train_x[idx_i]))
                        delta_list.append(
                            abs(metric_images[idx_i] - change_train_x[idx_i])
                        )

            # Prune deceived images from working sets
            mask = total_change_list == 0
            metric_images = metric_images[mask]
            change_train_x = change_train_x[mask]
            idx_list = idx_list[mask]
            ori_prob_full = ori_prob_full[mask]

            # Memory update
            update_rewards = update_rewards[mask]
            update_images = update_images[mask]
            standard = update_rewards.mean()

            total_rewards_t = torch.tensor(total_rewards_list, device=device)[mask]
            stacked = torch.stack([update_rewards, total_rewards_t], dim=0)
            best_idx = torch.max(stacked, axis=0).indices
            arange = torch.arange(total_rewards_t.shape[0], device=device)
            update_rewards = stacked[best_idx, arange]

            update_rewards_sum = update_rewards.mean() + total_change_list.sum()

            stacked_imgs = torch.stack([update_images, change_train_x], dim=0)
            update_images = stacked_imgs[best_idx, arange]

            # Convergence check
            delta_val = ((update_rewards_sum - standard) / (standard + 1e-8)).item()
            stop_count, flag = _early_stopping(delta_val, stop_count, limit, patience)

            if flag:
                train_data = _Dataset(update_images, train_y)
                break
            else:
                train_data = _Dataset(train_x, train_y)

        total_deceived += deceived_count

        # Save adversarial images
        nonempty_succ = [s for s in succes_list if s.numel() > 0]
        nonempty_idx = [s for s in succes_list_idx if s.numel() > 0]
        nonempty_labels = [s for s in succes_labels_list if s.numel() > 0]
        if nonempty_succ:
            all_succ = torch.cat(nonempty_succ, 0)
            all_idx = torch.cat(nonempty_idx, 0)
            all_labels = torch.cat(nonempty_labels, 0)

            for n in range(len(all_succ)):
                img_idx = int(all_idx[n])
                img = Image.fromarray(
                    (all_succ[n].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                )
                img.save(adv_dir / f"adv_{img_idx:04d}.png")
                save_labels[img_idx] = all_labels[n].item()

                if n < len(delta_list):
                    d = Image.fromarray(
                        (255 - delta_list[n].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                    )
                    d.save(delta_dir / f"delta_{img_idx:04d}.png")

        logger.info(
            f"Forget:{p + 1}, deceived={deceived_count}, "
            f"total={total_deceived}, remaining={len(train_data)}"
        )

        if len(train_data) == 0:
            logger.info("All images deceived")
            break

    elapsed = time.time() - t0
    metrics = AttackMetrics(
        mode="classification",
        total_images=N,
        total_deceived=total_deceived,
        success_rate=total_deceived / N if N > 0 else 0.0,
        mean_l0=float(np.mean(L0)) if L0 else 0.0,
        mean_l2=float(np.mean(L2)) if L2 else 0.0,
        mean_queries=query_count / N if N > 0 else 0.0,
        forget_iterations=p + 1,
        elapsed_sec=elapsed,
    )

    torch.save(save_labels, output_dir / "adv_labels.pt")
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(asdict(metrics), f, indent=2)

    logger.info(f"[RFPAR-CLS] Done: {metrics.success_rate:.1%} success, {elapsed:.1f}s")
    return AttackResult(metrics=metrics, agent_state=agent.state_dict())


def _make_transformed_det(original_images, actions, n_pixels):
    """Apply pixel perturbations for detection (values 0 or 255)."""
    dev = original_images.device
    B, C, H, W = original_images.shape
    actions = torch.sigmoid(actions)
    x = (actions[:, 0] * H - 1).long().clamp(0, H - 1)
    y = (actions[:, 1] * W - 1).long().clamp(0, W - 1)
    r = (actions[:, 2] > 0.5).float() * 255
    g = (actions[:, 3] > 0.5).float() * 255
    b = (actions[:, 4] > 0.5).float() * 255

    arr = []
    for i in range(B):
        changed = original_images[i].clone()
        if n_pixels == 1:
            changed[0, x[i], y[i]] = r[i].to(torch.uint8).float()
            changed[1, x[i], y[i]] = g[i].to(torch.uint8).float()
            changed[2, x[i], y[i]] = b[i].to(torch.uint8).float()
        else:
            idx = (torch.arange(n_pixels, device=dev) * B + i)
            idx = idx[idx < len(x)]
            changed[0, x[idx], y[idx]] = r[idx].to(torch.uint8).float()
            changed[1, x[idx], y[idx]] = g[idx].to(torch.uint8).float()
            changed[2, x[idx], y[idx]] = b[idx].to(torch.uint8).float()
        arr.append(changed.unsqueeze(0))
    return torch.cat(arr, 0)


def run_detection_attack(
    model,
    images: list[np.ndarray],
    hw_array: np.ndarray,
    device: torch.device,
    output_dir: Path,
    max_iterations: int = 100,
    alpha: float = 0.05,
    patience: int = 20,
    limit: float = 0.05,
    batch_size: int = 50,
    lr: float = 1e-4,
    conf: float = 0.50,
    seed: int = 2,
) -> AttackResult:
    """Run RFPAR detection attack — faithful to reference main_od.py."""
    import torchvision.transforms as transforms

    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    N = len(images)
    h_max, w_max = int(hw_array.max(axis=0)[0]), int(hw_array.max(axis=0)[1])
    shape_unity = bool((hw_array[:, 0] == h_max).all() and (hw_array[:, 1] == w_max).all())
    n_pixels = max(1, int((h_max + w_max) / 2 * alpha))

    adv_dir = output_dir / "adv_images"
    delta_dir = output_dir / "delta_images"
    adv_dir.mkdir(parents=True, exist_ok=True)
    delta_dir.mkdir(parents=True, exist_ok=True)

    if not shape_unity:
        for idx in range(N):
            images[idx] = np.pad(
                images[idx],
                ((0, h_max - hw_array[idx, 0]), (0, w_max - hw_array[idx, 1]), (0, 0)),
                "constant",
            )

    logger.info(
        f"[RFPAR-DET] N={N}, pixels={n_pixels}, alpha={alpha}, "
        f"shape_unity={shape_unity}, patience={patience}"
    )

    # Get initial detections
    ori_prob_list = []
    ori_cls_list = []
    ori_box_num = []
    for n, img in enumerate(images):
        target = img if shape_unity else img[: hw_array[n, 0], : hw_array[n, 1]]
        results = model(target, conf=conf, verbose=False)
        ori_prob_list.append(results[0].boxes.conf)
        ori_cls_list.append(results[0].boxes.cls)
        ori_box_num.append(results[0].boxes.shape[0])

    avg_boxes = float(np.mean(ori_box_num))
    logger.info(f"Average detected objects: {avg_boxes:.1f}")

    update_images = torch.tensor(np.array(images), device=device).clone()
    metric_images = torch.tensor(np.array(images)).clone()
    yolo_list = torch.tensor(np.array(images)).clone()

    torchvision_transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.CenterCrop(224)]
    )
    train_data = _Dataset(torch.tensor(np.array(images)), torch.zeros(N))

    it = 0
    iteration = torch.zeros(N, device=device)
    box_count = torch.zeros(N)
    query_count = 0

    for p in range(max_iterations):
        agent = REINFORCEAgent(224, 224, 3, lr=lr, detector_mode=True).to(device)

        flag = False
        stop_count = 0
        prev_change_list = torch.zeros(update_images.shape[0], device=device)
        update_rewards = torch.zeros(update_images.shape[0], device=device)

        # Current labels/probs for this Forget iteration
        cur_cls = list(ori_cls_list)
        cur_prob = list(ori_prob_list)
        need_init = True

        while True:
            bts = math.ceil(train_data.x_data.shape[0] / batch_size)
            change_train_x = []
            it += 1
            total_rewards_list: list[float] = []
            total_change_list = torch.tensor([], device=device)

            for bt in range(bts):
                start = bt * batch_size
                end = min(start + batch_size, train_data.x_data.shape[0])
                s = train_data.x_data[start:end]

                if not need_init:
                    labels = cur_cls[start:end]
                    probs = cur_prob[start:end]

                s_perm = s.permute(0, 3, 1, 2).float()
                s_norm = torchvision_transform(s_perm.to(device)) / 255
                action_means, action_stds = agent(s_norm)
                actions, log_probs = _sample_action(action_means, action_stds, n_pixels, device)

                if n_pixels != 1:
                    actions = actions.view(-1, 5)
                    log_probs = log_probs.sum(axis=0)

                changed = _make_transformed_det(s_perm.to(device), actions, n_pixels)
                query_count += len(s)

                with torch.no_grad():
                    if need_init:
                        labels = []
                        probs = []
                        for img_t in s_perm:
                            res = model(
                                img_t.cpu().numpy().transpose(1, 2, 0),
                                conf=conf, verbose=False,
                            )
                            probs.append(res[0].boxes.conf)
                            labels.append(res[0].boxes.cls)
                        cur_prob[start:end] = probs
                        cur_cls[start:end] = labels

                    changed_np = changed.cpu().numpy().transpose(0, 2, 3, 1)
                    prob_list = []
                    cls_list = []
                    for img_np in changed_np:
                        res = model(img_np, conf=conf, verbose=False)
                        prob_list.append(res[0].boxes.conf)
                        cls_list.append(res[0].boxes.cls)

                    # Compute rewards (box count change + confidence diff)
                    rewards = []
                    dif_list = []
                    for ii in range(len(cls_list)):
                        lab_cpu = labels[ii].long().cpu()
                        cls_cpu = cls_list[ii].long().cpu()
                        sz = max(
                            torch.bincount(lab_cpu).shape[0] if lab_cpu.numel() > 0 else 1,
                            torch.bincount(cls_cpu).shape[0] if cls_cpu.numel() > 0 else 1,
                        )
                        orig_bc = (
                            torch.bincount(lab_cpu, minlength=sz)
                            if lab_cpu.numel() > 0 else torch.zeros(sz)
                        )
                        new_bc = (
                            torch.bincount(cls_cpu, minlength=sz)
                            if cls_cpu.numel() > 0 else torch.zeros(sz)
                        )
                        temp = orig_bc - new_bc
                        dif = temp.sum()
                        dif_list.append(dif)

                        # Match labels/probs for reward
                        lab_i = labels[ii].clone()
                        prob_i = probs[ii].clone()
                        cls_i = cls_list[ii].clone()
                        plist_i = prob_list[ii].clone()

                        if (temp != 0).any():
                            for j in range(len(temp)):
                                cnt = int(temp[j])
                                if cnt > 0:
                                    pad = torch.full(
                                        (cnt,), j, dtype=torch.long, device=device
                                    )
                                    cls_i = torch.cat((cls_i.long().to(device), pad))
                                    zeros = torch.zeros(cnt, device=device)
                                    plist_i = torch.cat((plist_i.to(device), zeros))
                                elif cnt < 0:
                                    pad = torch.full(
                                        (-cnt,), j, dtype=torch.long, device=device
                                    )
                                    lab_i = torch.cat((lab_i.long().to(device), pad))
                                    zeros = torch.zeros(-cnt, device=device)
                                    prob_i = torch.cat((prob_i.to(device), zeros))

                        reward = (
                            prob_i[lab_i.sort()[1]].cpu() - plist_i[cls_i.sort()[1]].cpu()
                        ).sum() + dif
                        rewards.append(reward)

                rewards_t = torch.tensor(rewards, device=device)

                for img_np in changed_np:
                    change_train_x.append(img_np)
                total_change_list = torch.cat(
                    (total_change_list, torch.tensor(dif_list, device=device).float()), dim=0
                ).long()
                total_rewards_list.extend(rewards_t.tolist())

                agent.train_step(log_probs, rewards_t)

            need_init = False

            # Memory update
            standard = update_rewards.mean()
            total_rewards_t = torch.tensor(total_rewards_list, device=device)

            stacked = torch.stack([update_rewards, total_rewards_t], dim=0)
            best_idx = torch.max(stacked, axis=0).indices
            arange = torch.arange(len(total_rewards_t))
            update_rewards = stacked[best_idx, arange]

            change_t = torch.tensor(np.array(change_train_x), device=device)
            stacked_imgs = torch.stack([update_images, change_t], dim=0)
            update_images = stacked_imgs[best_idx, arange]

            temp = torch.max(
                torch.stack([prev_change_list, total_change_list.float()], dim=0), axis=0
            ).values
            delta_box = (temp - prev_change_list).cpu()
            prev_change_list = temp.clone()

            if delta_box.sum() > 0:
                for ci in range(len(delta_box)):
                    if delta_box[ci] > 0:
                        yolo_list[ci] = torch.tensor(change_train_x[ci])
                        iteration[ci] = it
            box_count += delta_box

            update_sum = update_rewards.mean()
            delta_val = (
                ((update_sum - standard) / (standard + 1e-8)).item() + delta_box.sum().item()
            )
            stop_count, flag = _early_stopping(delta_val, stop_count, limit, patience)

            if flag:
                # Re-detect for next Forget iteration
                ori_cls_list = []
                ori_prob_list = []
                for ci in range(update_images.shape[0]):
                    img_np = update_images[ci].cpu().numpy()
                    res = model(img_np, conf=conf, verbose=False)
                    ori_prob_list.append(res[0].boxes.conf)
                    ori_cls_list.append(res[0].boxes.cls)
                train_data = _Dataset(update_images.cpu(), torch.zeros(len(update_images)))
                break
            else:
                # Re-detect current images
                cur_cls = []
                cur_prob = []
                for ci in range(len(change_train_x)):
                    res = model(change_train_x[ci], conf=conf, verbose=False)
                    cur_prob.append(res[0].boxes.conf)
                    cur_cls.append(res[0].boxes.cls)

        total_elim = box_count.mean().item()
        logger.info(f"Forget:{p + 1}, eliminated_boxes_avg={total_elim:.1f}")

        if avg_boxes > 0 and avg_boxes - total_elim <= 0:
            logger.info("All objects eliminated")
            break

    # Compute final metrics
    L0: list[float] = []
    L2: list[float] = []
    for idx in range(N):
        if shape_unity:
            L0.extend(_l0_norm(metric_images[idx], yolo_list[idx]))
            L2.extend(_l2_norm(metric_images[idx].float(), yolo_list[idx].float()))
        else:
            h, w = hw_array[idx, 0], hw_array[idx, 1]
            L0.extend(_l0_norm(metric_images[idx, :h, :w], yolo_list[idx, :h, :w]))
            L2.extend(_l2_norm(metric_images[idx, :h, :w].float(), yolo_list[idx, :h, :w].float()))

    # Save adversarial images
    for idx in range(N):
        if shape_unity:
            img = Image.fromarray(yolo_list[idx].numpy().astype("uint8"))
        else:
            h, w = hw_array[idx, 0], hw_array[idx, 1]
            img = Image.fromarray(yolo_list[idx, :h, :w].numpy().astype("uint8"))
        img.save(adv_dir / f"adv_{idx + 1:04d}.png")

    elapsed = time.time() - t0
    metrics = AttackMetrics(
        mode="detection",
        total_images=N,
        total_deceived=int(box_count.sum().item()),
        success_rate=total_elim / avg_boxes if avg_boxes > 0 else 0.0,
        mean_l0=float(np.mean(L0)) if L0 else 0.0,
        mean_l2=float(np.mean(L2)) if L2 else 0.0,
        mean_queries=query_count / N if N > 0 else 0.0,
        forget_iterations=p + 1,
        elapsed_sec=elapsed,
    )

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(asdict(metrics), f, indent=2)

    logger.info(f"[RFPAR-DET] Done: {metrics.success_rate:.1%} box elimination, {elapsed:.1f}s")
    return AttackResult(metrics=metrics, agent_state=agent.state_dict())
