"""RFPAR Attack — Remember and Forget process.

Classification attack on ImageNet with ResNeXt50.
Detection attack on COCO with YOLO.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .agent import REINFORCEAgent, sample_actions
from .environment import ClassificationEnv, DetectionEnv

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
    adv_images: list = field(default_factory=list)


def _l0_norm(orig: torch.Tensor, changed: torch.Tensor) -> float:
    return torch.count_nonzero(orig - changed).item()


def _l2_norm(orig: torch.Tensor, changed: torch.Tensor) -> float:
    return torch.norm((orig.float() - changed.float()), p=2).item()


def _early_stopping(delta: float, stop_count: int, limit: float, patience: int):
    if delta <= limit:
        stop_count += 1
        if stop_count >= patience:
            return stop_count, True
    else:
        stop_count = 0
    return stop_count, False


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

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
    """Run RFPAR classification attack (Remember & Forget loop).

    Args:
        model: victim classifier (eval mode)
        images: (N, C, H, W) normalized to [0, 1]
        labels: (N,) ground truth labels
        device: CUDA device
        output_dir: save adversarial images here
        max_iterations: max Forget iterations (paper: 100)
        alpha: pixel ratio (paper: 0.01 for classification)
        patience: convergence duration T (paper: 3)
        limit: bound threshold eta (paper: 0.05)
        batch_size: RL batch size (paper: 50)
        lr: RL learning rate (paper: 1e-4)
        seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = time.time()
    N, C, H, W = images.shape
    n_pixels = max(1, int((H + W) / 2 * alpha))

    adv_dir = output_dir / "adv_images"
    delta_dir = output_dir / "delta_images"
    adv_dir.mkdir(parents=True, exist_ok=True)
    delta_dir.mkdir(parents=True, exist_ok=True)

    env = ClassificationEnv(model, device, n_pixels)
    train_data = MyDataset(images.clone().to(device), labels.to(device))

    update_images = images.clone().to(device)
    metric_images = images.clone().to(device)
    idx_list = torch.arange(N, device=device)
    init_ori_prob: list[torch.Tensor] = []

    total_deceived = 0
    all_iterations: list[int] = []
    all_l0: list[float] = []
    all_l2: list[float] = []
    query_count = 0
    save_labels = torch.zeros(N)

    logger.info(
        f"[RFPAR-CLS] N={N}, pixels={n_pixels}, alpha={alpha}, "
        f"patience={patience}, limit={limit}, batch_size={batch_size}"
    )

    for p in range(max_iterations):
        agent = REINFORCEAgent(
            img_h=H, img_w=W, channels=C, lr=lr, detector_mode=False
        ).to(device)

        if p != 0:
            init_ori_prob_t = env.ori_prob - update_rewards
        else:
            init_ori_prob_t = []

        if isinstance(init_ori_prob_t, list):
            env.ori_prob = init_ori_prob_t
        else:
            env.ori_prob = init_ori_prob_t

        flag = False
        stop_count = 0
        deceived_count = 0
        succes_list = []
        succes_list_idx = []
        succes_labels = []
        delta_list = []
        update_rewards = torch.zeros(update_images.shape[0], device=device)
        init_step = p == 0
        remember_iter = 0

        while True:
            loader = torch.utils.data.DataLoader(
                train_data, batch_size=batch_size, shuffle=False
            )
            remember_iter += 1

            train_x_parts = []
            train_y_parts = []
            change_train_x_parts = []
            dip = 0
            total_rewards_list = []
            total_change_list = torch.tensor([], device=device)

            for bt, (s, lbl) in enumerate(loader):
                s = s.to(device)
                lbl = lbl.to(device)

                action_means, action_stds = agent(s)
                actions, log_probs = sample_actions(action_means, action_stds, n_pixels, device)

                if n_pixels != 1:
                    actions = actions.view(-1, 5)
                    log_probs = log_probs.sum(axis=0)

                ori_prob_slice = None
                if not init_step:
                    start = bt * batch_size
                    end = start + len(lbl)
                    ori_prob_slice = env.ori_prob[start:end]

                rewards, change_list, changed_imgs, change_labels = env.step(
                    s.float(), actions, lbl, init=init_step
                )
                query_count += len(lbl)

                if init_step:
                    if isinstance(init_ori_prob_t, list):
                        init_ori_prob_t.append(env.ori_prob.clone())

                train_x_parts.append(s[change_list == 0])
                train_y_parts.append(lbl[change_list == 0])
                succes_list.append(changed_imgs[change_list == 1])
                succes_list_idx.append(
                    idx_list[batch_size * bt : batch_size * bt + batch_size][change_list == 1]
                )
                succes_labels.append(change_labels[change_list == 1])
                all_iterations.extend([remember_iter] * int(change_list.sum().item()))

                change_train_x_parts.append(changed_imgs)
                total_change_list = torch.cat(
                    (total_change_list, change_list.float()), dim=0
                ).long()
                total_rewards_list.extend(rewards.tolist())

                dip += change_list.sum().item()

                agent.train_step(log_probs, rewards + change_list.float())

            if init_step and isinstance(init_ori_prob_t, list) and init_ori_prob_t:
                env.ori_prob = torch.cat(init_ori_prob_t, 0)

            if not train_x_parts or all(t.shape[0] == 0 for t in train_x_parts):
                break

            change_train_x = torch.cat(change_train_x_parts, 0)
            train_x = torch.cat([t for t in train_x_parts if t.shape[0] > 0], 0)
            train_y = torch.cat([t for t in train_y_parts if t.shape[0] > 0], 0)

            deceived_count += dip

            if total_change_list.sum() > 0:
                for idx_i, val in enumerate(total_change_list):
                    if val == 1:
                        all_l0.append(_l0_norm(metric_images[idx_i], change_train_x[idx_i]))
                        all_l2.append(_l2_norm(metric_images[idx_i], change_train_x[idx_i]))
                        delta_list.append(abs(metric_images[idx_i] - change_train_x[idx_i]))

            metric_images = metric_images[total_change_list == 0]
            idx_list = idx_list[total_change_list == 0]
            env.ori_prob = env.ori_prob[total_change_list == 0]
            update_rewards = update_rewards[total_change_list == 0]
            update_images = update_images[total_change_list == 0]

            standard = update_rewards.mean()
            total_rewards_t = torch.tensor(total_rewards_list, device=device)[
                total_change_list == 0
            ]

            stacked_rewards = torch.stack(
                [update_rewards, total_rewards_t], dim=0
            )
            best_idx = torch.max(stacked_rewards, axis=0).indices
            update_rewards = stacked_rewards[best_idx, torch.arange(total_rewards_t.shape[0])]

            change_train_x_remaining = change_train_x[total_change_list == 0]
            stacked_imgs = torch.stack([update_images, change_train_x_remaining], dim=0)
            update_images = stacked_imgs[best_idx, torch.arange(total_rewards_t.shape[0])]

            update_sum = update_rewards.mean() + total_change_list.sum()
            delta = (
                ((update_sum - standard) / (standard + 1e-8)).item()
                if standard.abs() > 1e-8
                else update_sum.item()
            )
            stop_count, flag = _early_stopping(delta, stop_count, limit, patience)

            if flag:
                train_data = MyDataset(update_images, train_y)
                break
            else:
                train_data = MyDataset(train_x, train_y)
                init_step = False

        total_deceived += deceived_count

        if succes_list:
            all_succ = torch.cat(succes_list, 0)
            all_succ_idx = torch.cat(succes_list_idx, 0)
            all_succ_labels = torch.cat(succes_labels, 0)

            for n, changed_image in enumerate(all_succ):
                img = Image.fromarray(
                    (changed_image.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                )
                img_idx = int(all_succ_idx[n])
                img.save(adv_dir / f"adv_{img_idx:04d}.png")
                save_labels[img_idx] = all_succ_labels[n].item()

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
        mean_l0=float(np.mean(all_l0)) if all_l0 else 0.0,
        mean_l2=float(np.mean(all_l2)) if all_l2 else 0.0,
        mean_queries=query_count / N if N > 0 else 0.0,
        forget_iterations=p + 1,
        elapsed_sec=elapsed,
    )

    torch.save(save_labels, output_dir / "adv_labels.pt")
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(asdict(metrics), f, indent=2)

    logger.info(f"[RFPAR-CLS] Done: {metrics.success_rate:.1%} success, {elapsed:.1f}s")
    return AttackResult(metrics=metrics, agent_state=agent.state_dict())


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
    """Run RFPAR object detection attack (YOLO).

    Args:
        model: YOLO model instance
        images: list of (H, W, 3) uint8 numpy arrays
        hw_array: (N, 2) actual heights/widths
        device: CUDA device
        output_dir: save adversarial images here
        max_iterations: max Forget iterations
        alpha: pixel ratio (paper: 0.05 for detection)
        patience: convergence duration T (paper: 20)
    """
    import math
    import torchvision.transforms as transforms

    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    N = len(images)
    h_max, w_max = hw_array.max(axis=0)
    shape_unity = (hw_array[:, 0] == h_max).all() and (hw_array[:, 1] == w_max).all()
    n_pixels = max(1, int((h_max + w_max) / 2 * alpha))

    adv_dir = output_dir / "adv_images"
    delta_dir = output_dir / "delta_images"
    adv_dir.mkdir(parents=True, exist_ok=True)
    delta_dir.mkdir(parents=True, exist_ok=True)

    env = DetectionEnv(model, device, n_pixels, conf)

    if not shape_unity:
        for i in range(N):
            images[i] = np.pad(
                images[i],
                ((0, h_max - hw_array[i, 0]), (0, w_max - hw_array[i, 1]), (0, 0)),
                "constant",
            )

    logger.info(
        f"[RFPAR-DET] N={N}, pixels={n_pixels}, alpha={alpha}, "
        f"shape_unity={shape_unity}, patience={patience}"
    )

    # Get initial detection results
    for n, img in enumerate(images):
        if shape_unity:
            results = model(img, conf=conf, verbose=False)
        else:
            results = model(img[: hw_array[n, 0], : hw_array[n, 1]], conf=conf, verbose=False)
        env.ori_prob.append(results[0].boxes.conf)
        env.ori_cls.append(results[0].boxes.cls)
        env.ori_box_num.append(results[0].boxes.shape[0])

    env.ori_box_num_t = torch.tensor(env.ori_box_num).long()
    avg_boxes = env.ori_box_num_t.float().mean().item()
    logger.info(f"Average detected objects: {avg_boxes:.1f}")

    update_images = torch.tensor(np.array(images), device=device).clone()
    metric_images = torch.tensor(np.array(images)).clone()
    yolo_list = torch.tensor(np.array(images)).clone()

    torchvision_transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.CenterCrop(224)]
    )
    trick_element = torch.zeros(N)
    train_data = MyDataset(torch.tensor(np.array(images)), trick_element)

    it = 0
    iteration = torch.zeros(N, device=device)
    box_count = torch.zeros(N)
    all_l0: list[float] = []
    all_l2: list[float] = []
    query_count = 0

    for p in range(max_iterations):
        agent = REINFORCEAgent(
            img_h=224, img_w=224, channels=3, lr=lr, detector_mode=True
        ).to(device)

        flag = False
        stop_count = 0
        prev_change_list = torch.zeros(update_images.shape[0], device=device)
        update_rewards = torch.zeros(update_images.shape[0], device=device)
        need_init = len(env.ori_cls) == 0

        while True:
            bts = math.ceil(train_data.x_data.shape[0] / batch_size)
            change_train_x = []
            it += 1
            total_rewards_list = []
            total_change_list = torch.tensor([], device=device)

            for bt in range(bts):
                if bt != bts - 1:
                    s = train_data.x_data[bt * batch_size : (bt + 1) * batch_size]
                else:
                    s = train_data.x_data[bt * batch_size :]

                if not need_init:
                    labels = env.ori_cls[bt * batch_size : bt * batch_size + len(s)]
                    probs = env.ori_prob[bt * batch_size : bt * batch_size + len(s)]
                else:
                    labels = None
                    probs = None

                s_perm = s.permute(0, 3, 1, 2).float()
                s_norm = torchvision_transform(s_perm.to(device)) / 255
                action_means, action_stds = agent(s_norm)
                actions, log_probs = sample_actions(
                    action_means, action_stds, n_pixels, device
                )

                if n_pixels != 1:
                    actions = actions.view(-1, 5)
                    log_probs = log_probs.sum(axis=0)

                rewards, dif_list, changed_np = env.step(
                    s_perm, actions, bt, init=need_init, labels=labels, probs=probs
                )
                query_count += len(s)

                for img_np in changed_np:
                    change_train_x.append(img_np)

                total_change_list = torch.cat(
                    (total_change_list, dif_list.to(device).float()), dim=0
                ).long()
                total_rewards_list.extend(rewards.tolist())

                agent.train_step(log_probs, rewards)

            need_init = False

            # Memory update
            standard = update_rewards.mean()
            total_rewards_t = torch.tensor(total_rewards_list, device=device)

            stacked = torch.stack([update_rewards, total_rewards_t], dim=0)
            best_idx = torch.max(stacked, axis=0).indices
            update_rewards = stacked[best_idx, torch.arange(len(total_rewards_t))]

            change_t = torch.tensor(np.array(change_train_x), device=device)
            stacked_imgs = torch.stack([update_images, change_t], dim=0)
            update_images = stacked_imgs[best_idx, torch.arange(len(total_rewards_t))]

            temp = torch.max(
                torch.stack([prev_change_list, total_change_list.float()], dim=0), axis=0
            ).values
            delta_box = (temp - prev_change_list).cpu()
            prev_change_list = temp.clone()

            if delta_box.sum() > 0:
                change_idx = [i for i in range(len(delta_box)) if delta_box[i] > 0]
                for ci in change_idx:
                    yolo_list[ci] = torch.tensor(change_train_x[ci])
                    iteration[ci] = it

            box_count += delta_box

            update_sum = update_rewards.mean()
            delta_val = (
                ((update_sum - standard) / (standard + 1e-8)).item() + delta_box.sum().item()
            )
            stop_count, flag = _early_stopping(delta_val, stop_count, limit, patience)

            if flag:
                env.ori_cls = []
                env.ori_prob = []
                train_data = MyDataset(update_images.cpu(), torch.zeros(len(update_images)))
                it += 1
                break
            else:
                env.ori_cls = []
                env.ori_prob = []
                # Re-detect for next iteration
                for n_i in range(len(change_train_x)):
                    img_i = change_train_x[n_i]
                    res = model(img_i, conf=conf, verbose=False)
                    env.ori_prob.append(res[0].boxes.conf)
                    env.ori_cls.append(res[0].boxes.cls)

        total_elim = box_count.mean().item()
        logger.info(f"Forget:{p + 1}, eliminated_boxes_avg={total_elim:.1f}")

        if avg_boxes > 0 and avg_boxes - total_elim <= 0:
            logger.info("All objects eliminated")
            break

    # Compute metrics
    for i in range(N):
        all_l0.append(_l0_norm(metric_images[i], yolo_list[i]))
        all_l2.append(_l2_norm(metric_images[i].float(), yolo_list[i].float()))

    # Save adversarial images
    for i in range(N):
        if shape_unity:
            img = Image.fromarray(yolo_list[i].numpy().astype("uint8"))
        else:
            img = Image.fromarray(
                yolo_list[i, : hw_array[i, 0], : hw_array[i, 1]].numpy().astype("uint8")
            )
        img.save(adv_dir / f"adv_{i + 1:04d}.png")

    elapsed = time.time() - t0
    metrics = AttackMetrics(
        mode="detection",
        total_images=N,
        total_deceived=int(box_count.sum().item()),
        success_rate=total_elim / avg_boxes if avg_boxes > 0 else 0.0,
        mean_l0=float(np.mean(all_l0)) if all_l0 else 0.0,
        mean_l2=float(np.mean(all_l2)) if all_l2 else 0.0,
        mean_queries=query_count / N if N > 0 else 0.0,
        forget_iterations=p + 1,
        elapsed_sec=elapsed,
    )

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(asdict(metrics), f, indent=2)

    logger.info(f"[RFPAR-DET] Done: {metrics.success_rate:.1%} box elimination, {elapsed:.1f}s")
    return AttackResult(metrics=metrics, agent_state=agent.state_dict())
