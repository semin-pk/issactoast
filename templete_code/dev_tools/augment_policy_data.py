#!/usr/bin/env python3
"""Augment policy datasets with rectangle-safe 2D symmetries.

Dev-only tool. 90/270 degree rotations are intentionally excluded because the
pallet grid is rectangular.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def transform_sample(
    height_map: np.ndarray,
    action_mask: np.ndarray,
    action: np.ndarray,
    footprint: np.ndarray,
    footprint_all: np.ndarray,
    transform: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    b_idx, r_idx, x_idx, y_idx = [int(value) for value in action]
    box_w, box_h = [int(value) for value in footprint]
    h_grid = int(height_map.shape[-2])
    w_grid = int(height_map.shape[-1])

    new_height = height_map.copy()
    new_mask = remap_action_mask(action_mask, footprint_all, transform)
    new_action = np.asarray([b_idx, r_idx, x_idx, y_idx], dtype=np.int64)

    if transform in {"flip_x", "rot180"}:
        new_height = np.flip(new_height, axis=-1)
        new_action[2] = w_grid - x_idx - box_w

    if transform in {"flip_y", "rot180"}:
        new_height = np.flip(new_height, axis=-2)
        new_action[3] = h_grid - y_idx - box_h

    nb, nr, nx, ny = [int(value) for value in new_action]
    if nx < 0 or ny < 0 or nx >= w_grid or ny >= h_grid:
        raise AssertionError(f"transformed action out of grid: {new_action.tolist()}")
    if new_mask[nb, nr, ny, nx] <= 0.0:
        raise AssertionError(
            f"transformed label is masked out: transform={transform} action={new_action.tolist()}"
        )

    return (
        np.ascontiguousarray(new_height),
        np.ascontiguousarray(new_mask),
        new_action,
    )


def remap_action_mask(
    action_mask: np.ndarray,
    footprint_all: np.ndarray,
    transform: str,
) -> np.ndarray:
    return remap_action_tensor(action_mask, footprint_all, transform, binary=True)


def remap_action_tensor(
    action_tensor: np.ndarray,
    footprint_all: np.ndarray,
    transform: str,
    *,
    binary: bool = False,
) -> np.ndarray:
    b_count, rotations, h_grid, w_grid = action_tensor.shape
    remapped = np.zeros_like(action_tensor)

    for b_idx in range(b_count):
        for r_idx in range(rotations):
            box_w, box_h = [int(value) for value in footprint_all[b_idx, r_idx]]
            if box_w <= 0 or box_h <= 0:
                continue
            if binary:
                ys, xs = np.where(action_tensor[b_idx, r_idx] > 0.0)
            else:
                ys, xs = np.where(action_tensor[b_idx, r_idx] != 0.0)
            for y_idx, x_idx in zip(ys, xs):
                new_x = int(x_idx)
                new_y = int(y_idx)
                if transform in {"flip_x", "rot180"}:
                    new_x = w_grid - int(x_idx) - box_w
                if transform in {"flip_y", "rot180"}:
                    new_y = h_grid - int(y_idx) - box_h
                if 0 <= new_x < w_grid and 0 <= new_y < h_grid:
                    remapped[b_idx, r_idx, new_y, new_x] = (
                        1.0 if binary else action_tensor[b_idx, r_idx, y_idx, x_idx]
                    )

    return remapped


def augment_npz(input_path: Path, output_path: Path, transforms: List[str]) -> None:
    data = np.load(input_path, allow_pickle=True)
    arrays: Dict[str, np.ndarray] = {key: data[key] for key in data.files}

    height_maps = [arrays["height_map"]]
    action_masks = [arrays["action_mask"]]
    actions = [arrays["action"]]
    action_map_keys = [
        key
        for key in ("mcts_policy", "visit_counts", "q_values")
        if key in arrays
    ]
    action_map_augments: Dict[str, List[np.ndarray]] = {
        key: [arrays[key]] for key in action_map_keys
    }

    for transform in transforms:
        aug_heights: List[np.ndarray] = []
        aug_masks: List[np.ndarray] = []
        aug_actions: List[np.ndarray] = []
        aug_action_maps: Dict[str, List[np.ndarray]] = {key: [] for key in action_map_keys}
        for idx in range(len(arrays["action"])):
            height, mask, action = transform_sample(
                arrays["height_map"][idx],
                arrays["action_mask"][idx],
                arrays["action"][idx],
                arrays["action_footprint"][idx],
                arrays["action_footprint_all"][idx],
                transform,
            )
            aug_heights.append(height)
            aug_masks.append(mask)
            aug_actions.append(action)
            for key in action_map_keys:
                aug_action_maps[key].append(remap_action_tensor(
                    arrays[key][idx],
                    arrays["action_footprint_all"][idx],
                    transform,
                    binary=False,
                ))
        height_maps.append(np.asarray(aug_heights, dtype=np.float32))
        action_masks.append(np.asarray(aug_masks, dtype=np.float32))
        actions.append(np.asarray(aug_actions, dtype=np.int64))
        for key in action_map_keys:
            action_map_augments[key].append(np.asarray(aug_action_maps[key], dtype=np.float32))

    output: Dict[str, np.ndarray] = dict(arrays)
    output["height_map"] = np.concatenate(height_maps, axis=0)
    output["action_mask"] = np.concatenate(action_masks, axis=0)
    output["action"] = np.concatenate(actions, axis=0)
    for key in action_map_keys:
        output[key] = np.concatenate(action_map_augments[key], axis=0)

    repeat_count = 1 + len(transforms)
    n_original = len(arrays["action"])
    for key, value in arrays.items():
        if key in {"height_map", "action_mask", "action"} | set(action_map_keys):
            continue
        if len(value) == n_original:
            output[key] = np.concatenate([value.copy() for _ in range(repeat_count)], axis=0)

    if "sequence_id" in output:
        suffixes = ["orig"] + transforms
        output["sequence_id"] = np.concatenate([
            np.char.add(output["sequence_id"][:n_original].astype(str), f"::{suffix}")
            for suffix in suffixes
        ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output)
    print(f"saved {output_path} samples={len(output['action'])}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Augment policy .npz data.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--transforms",
        default="flip_x,flip_y,rot180",
        help="Comma list among flip_x,flip_y,rot180",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    transforms = [item.strip() for item in args.transforms.split(",") if item.strip()]
    for transform in transforms:
        if transform not in {"flip_x", "flip_y", "rot180"}:
            raise ValueError(f"unsupported transform: {transform}")
    augment_npz(Path(args.input), Path(args.output), transforms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
