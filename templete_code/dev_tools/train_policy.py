#!/usr/bin/env python3
"""Train a compact behavior-cloning policy and export ONNX.

Dev-only tool. This script may use torch locally, but torch must not appear in
the final submission runtime.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import yaml


def load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def expand_paths(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(item) for item in matches)
        else:
            paths.append(Path(pattern))
    return paths


def load_npz_many(paths: Sequence[Path]) -> Dict[str, np.ndarray]:
    chunks: Dict[str, List[np.ndarray]] = {}
    for path in paths:
        data = np.load(path, allow_pickle=True)
        for key in ("height_map", "buffer_features", "action_mask", "action"):
            chunks.setdefault(key, []).append(data[key])
    return {key: np.concatenate(value, axis=0) for key, value in chunks.items()}


def actions_to_flat(actions: np.ndarray, mask_shape: Tuple[int, int, int, int]) -> np.ndarray:
    b_count, rotations, h_grid, w_grid = mask_shape
    flat: List[int] = []
    for b_idx, r_idx, x_idx, y_idx in actions:
        flat.append(
            int(b_idx) * rotations * h_grid * w_grid
            + int(r_idx) * h_grid * w_grid
            + int(y_idx) * w_grid
            + int(x_idx)
        )
    return np.asarray(flat, dtype=np.int64)


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required only for dev training. Install it locally, "
            "train/export ONNX, and keep torch out of submission runtime."
        ) from exc
    return torch, nn, F, DataLoader, TensorDataset


def print_device_info(torch, device) -> None:
    cuda_available = bool(torch.cuda.is_available())
    print(f"device={device} cuda_available={cuda_available} torch_cuda={torch.version.cuda}")
    if cuda_available:
        index = int(torch.cuda.current_device())
        print(f"gpu={index} name={torch.cuda.get_device_name(index)}")


def build_model(torch, nn, hidden_channels: int, buffer_hidden: int):
    class PolicyNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c = int(hidden_channels)
            self.height_branch = nn.Sequential(
                nn.Conv2d(1, c, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(c, c, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(c, c, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.buffer_branch = nn.Sequential(
                nn.Linear(6, int(buffer_hidden)),
                nn.ReLU(),
                nn.Linear(int(buffer_hidden), c),
                nn.ReLU(),
            )
            self.head = nn.Conv2d(c, 2, kernel_size=1)

        def forward(self, height_map_input, buffer_input, action_mask_input):
            height_feat = self.height_branch(height_map_input)
            buffer_feat = self.buffer_branch(buffer_input)
            n_batch, b_count, channels = buffer_feat.shape
            h_grid, w_grid = height_feat.shape[-2:]
            fused = height_feat[:, None, :, :, :] + buffer_feat[:, :, :, None, None]
            fused = fused.reshape(n_batch * b_count, channels, h_grid, w_grid)
            logits = self.head(fused)
            logits = logits.reshape(n_batch, b_count, 2, h_grid, w_grid)
            return logits

    return PolicyNet()


def evaluate_model(torch, model, loader, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    top1 = 0
    top5 = 0
    feasible = 0
    ce = torch.nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for height, buffer, mask, target in loader:
            height = height.to(device)
            buffer = buffer.to(device)
            mask = mask.to(device)
            target = target.to(device)
            logits = model(height, buffer, mask)
            masked_logits = logits.masked_fill(mask <= 0.0, -1.0e9)
            flat_logits = masked_logits.reshape(masked_logits.shape[0], -1)
            loss = ce(flat_logits, target)
            total_loss += float(loss.item())
            total += int(target.numel())
            top = torch.topk(flat_logits, k=min(5, flat_logits.shape[1]), dim=1).indices
            top1 += int((top[:, 0] == target).sum().item())
            top5 += int((top == target[:, None]).any(dim=1).sum().item())
            feasible += int((flat_logits.gather(1, target[:, None]) > -1.0e8).sum().item())

    denom = max(total, 1)
    return {
        "loss": total_loss / denom,
        "top1": top1 / denom,
        "top5": top5 / denom,
        "feasible": feasible / denom,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train policy network and export ONNX.")
    parser.add_argument("--config", default="config/policy_train_config.yaml")
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--valid", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    torch, nn, _F, DataLoader, TensorDataset = require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_device_info(torch, device)

    train_np = load_npz_many(expand_paths(args.train))
    valid_np = load_npz_many(expand_paths(args.valid))
    mask_shape = tuple(int(v) for v in train_np["action_mask"].shape[1:])
    train_target = actions_to_flat(train_np["action"], mask_shape)
    valid_target = actions_to_flat(valid_np["action"], mask_shape)

    train_ds = TensorDataset(
        torch.from_numpy(train_np["height_map"]).float(),
        torch.from_numpy(train_np["buffer_features"]).float(),
        torch.from_numpy(train_np["action_mask"]).float(),
        torch.from_numpy(train_target).long(),
    )
    valid_ds = TensorDataset(
        torch.from_numpy(valid_np["height_map"]).float(),
        torch.from_numpy(valid_np["buffer_features"]).float(),
        torch.from_numpy(valid_np["action_mask"]).float(),
        torch.from_numpy(valid_target).long(),
    )

    training = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    batch_size = int(training.get("batch_size", 64))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    model = build_model(
        torch=torch,
        nn=nn,
        hidden_channels=int(model_cfg.get("hidden_channels", 32)),
        buffer_hidden=int(model_cfg.get("buffer_hidden", 64)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 0.001)),
        weight_decay=float(training.get("weight_decay", 0.00001)),
    )
    ce = nn.CrossEntropyLoss()
    best_state = None
    best_loss = float("inf")
    stale = 0
    patience = int(training.get("early_stopping_patience", 3))

    for epoch in range(1, int(training.get("epochs", 10)) + 1):
        model.train()
        for height, buffer, mask, target in train_loader:
            height = height.to(device)
            buffer = buffer.to(device)
            mask = mask.to(device)
            target = target.to(device)
            logits = model(height, buffer, mask)
            masked_logits = logits.masked_fill(mask <= 0.0, -1.0e9)
            loss = ce(masked_logits.reshape(masked_logits.shape[0], -1), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        metrics = evaluate_model(torch, model, valid_loader, device)
        print(
            f"epoch={epoch} valid_ce={metrics['loss']:.4f} "
            f"top1={metrics['top1']:.3f} top5={metrics['top5']:.3f} "
            f"feasible={metrics['feasible']:.3f}"
        )
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    n_batch = 1
    b_count, rotations, h_grid, w_grid = mask_shape
    feature_count = int(train_np["buffer_features"].shape[-1])
    dummy_height = torch.zeros(n_batch, 1, h_grid, w_grid, dtype=torch.float32, device=device)
    dummy_buffer = torch.zeros(n_batch, b_count, feature_count, dtype=torch.float32, device=device)
    dummy_mask = torch.ones(n_batch, b_count, rotations, h_grid, w_grid, dtype=torch.float32, device=device)
    torch.onnx.export(
        model,
        (dummy_height, dummy_buffer, dummy_mask),
        output,
        input_names=["height_map_input", "buffer_input", "action_mask_input"],
        output_names=["logits"],
        opset_version=int(cfg.get("export", {}).get("opset_version", 17)),
        dynamic_axes={
            "height_map_input": {0: "N"},
            "buffer_input": {0: "N"},
            "action_mask_input": {0: "N"},
            "logits": {0: "N"},
        },
    )
    print(f"exported {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
