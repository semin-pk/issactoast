#!/usr/bin/env python3
"""Collect behavior-cloning data from the safe palletizing teacher.

Dev-only tool. Do not include this file in the final submission package.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm import AlgorithmConfig, BoxInput, PalletConfig, Palletizer  # noqa: E402
from buffer_manager import BufferManager  # noqa: E402
from evaluate import buffer_bonus_from_capacity  # noqa: E402


def load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_boxes(path: Path) -> List[BoxInput]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def iter_input_files(path: Path, max_sequences: int) -> List[Path]:
    if path.is_file():
        return [path]
    files = sorted(item for item in path.glob("*.json") if item.is_file())
    if max_sequences > 0:
        files = files[:max_sequences]
    return files


def utilization_pct(sequence: Sequence[Dict[str, Any]], pallet: PalletConfig) -> float:
    volume = sum(float(np.prod(item["size"])) for item in sequence)
    pallet_volume = pallet.length * pallet.width * pallet.height
    return volume / pallet_volume * 100.0 if pallet_volume > 0.0 else 0.0


def collect_one(
    input_path: Path,
    cfg: Dict[str, Any],
    teacher: str,
    baseline_score: float | None,
) -> Dict[str, Any]:
    pallet_cfg = PalletConfig(
        length=float(cfg["pallet"]["length"]),
        width=float(cfg["pallet"]["width"]),
        height=float(cfg["pallet"]["height"]),
    )
    algo_cfg = AlgorithmConfig(
        allow_rotation=bool(cfg["algorithm"]["allow_rotation"]),
        buffer_size=int(cfg["buffer"]["size"]),
    )
    palletizer = Palletizer(pallet_cfg, algo_cfg)
    boxes = load_boxes(input_path)
    buf = BufferManager(algo_cfg.buffer_size)
    buf.reset(boxes)

    height_maps: List[np.ndarray] = []
    buffer_features: List[np.ndarray] = []
    action_masks: List[np.ndarray] = []
    actions: List[Tuple[int, int, int, int]] = []
    action_footprints: List[Tuple[int, int]] = []
    action_footprint_all: List[np.ndarray] = []
    teacher_scores: List[float] = []
    steps: List[int] = []
    box_ids: List[int] = []
    mask_violations = 0

    while buf.has_pending():
        current = [buf.peek_next()] if algo_cfg.buffer_size == 0 else buf.get_buffer()
        if palletizer.should_finish(current):
            palletizer.finished_by_user = True
            break

        indexed = list(enumerate(current))
        state = palletizer.policy_state(indexed)
        footprint_all = np.zeros((state["action_mask"].shape[0], 2, 2), dtype=np.int64)
        for buffer_index, box in indexed:
            for dims, rotation in palletizer._candidate_orientations(box["size"]):
                rotation_index = palletizer._rotation_index_for_degrees(rotation)
                footprint_all[buffer_index, rotation_index] = [
                    int(np.ceil(dims[0] / palletizer.grid_resolution)),
                    int(np.ceil(dims[1] / palletizer.grid_resolution)),
                ]
        selected, action = palletizer._best_grid_candidate(indexed)

        if selected is None or action is None:
            palletizer.finished_by_user = True
            break

        b_idx, r_idx, x_idx, y_idx = action
        if state["action_mask"][b_idx, r_idx, y_idx, x_idx] <= 0.0:
            mask_violations += 1
            break

        height_maps.append(state["height_map"])
        buffer_features.append(state["buffer_features"])
        action_masks.append(state["action_mask"])
        actions.append((b_idx, r_idx, x_idx, y_idx))
        action_footprints.append((
            int(np.ceil(selected.dims[0] / palletizer.grid_resolution)),
            int(np.ceil(selected.dims[1] / palletizer.grid_resolution)),
        ))
        action_footprint_all.append(footprint_all)
        teacher_scores.append(float(selected.score))
        steps.append(int(selected.box["step"]))
        box_ids.append(int(selected.box["id"]))

        palletizer._place_candidate(selected)
        if algo_cfg.buffer_size == 0:
            buf.pop_next()
        else:
            buf.pop_selected(selected.buffer_index)

    palletizer._assert_valid_result()
    util = utilization_pct(palletizer.sequence, pallet_cfg)
    final_score = util + buffer_bonus_from_capacity(algo_cfg.buffer_size)
    fail_flag = bool(mask_violations)
    baseline = final_score if baseline_score is None else float(baseline_score)

    return {
        "input_path": str(input_path),
        "height_maps": height_maps,
        "buffer_features": buffer_features,
        "action_masks": action_masks,
        "actions": actions,
        "action_footprints": action_footprints,
        "action_footprint_all": action_footprint_all,
        "teacher_scores": teacher_scores,
        "steps": steps,
        "box_ids": box_ids,
        "sequence_id": input_path.stem,
        "final_score": float(final_score),
        "baseline_score": float(baseline),
        "placed_volume_ratio": float(util / 100.0),
        "fail_flag": fail_flag,
        "mask_violations": int(mask_violations),
        "teacher": teacher,
    }


def episode_accepted(episode: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    policy_data = cfg.get("policy_data", {})
    min_util = float(policy_data.get("min_utilization_threshold", 0.45))
    baseline_margin = float(policy_data.get("baseline_margin", 0.0))
    save_failed = bool(policy_data.get("save_failed_episodes", False))

    if episode["fail_flag"] and not save_failed:
        return False
    if episode["final_score"] + 1e-9 < episode["baseline_score"] + baseline_margin:
        return False
    if episode["placed_volume_ratio"] + 1e-9 < min_util:
        return False
    if episode["mask_violations"] > 0:
        return False
    return True


def save_dataset(path: Path, episodes: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    height_maps: List[np.ndarray] = []
    buffer_features: List[np.ndarray] = []
    action_masks: List[np.ndarray] = []
    actions: List[Tuple[int, int, int, int]] = []
    action_footprints: List[Tuple[int, int]] = []
    action_footprint_all: List[np.ndarray] = []
    teacher_scores: List[float] = []
    sequence_ids: List[str] = []
    steps: List[int] = []
    box_ids: List[int] = []
    final_scores: List[float] = []
    fail_flags: List[bool] = []

    for episode in episodes:
        n = len(episode["actions"])
        height_maps.extend(episode["height_maps"])
        buffer_features.extend(episode["buffer_features"])
        action_masks.extend(episode["action_masks"])
        actions.extend(episode["actions"])
        action_footprints.extend(episode["action_footprints"])
        action_footprint_all.extend(episode["action_footprint_all"])
        teacher_scores.extend(episode["teacher_scores"])
        steps.extend(episode["steps"])
        box_ids.extend(episode["box_ids"])
        sequence_ids.extend([episode["sequence_id"]] * n)
        final_scores.extend([episode["final_score"]] * n)
        fail_flags.extend([episode["fail_flag"]] * n)

    if not actions:
        raise RuntimeError("no accepted policy samples to save")

    np.savez_compressed(
        path,
        height_map=np.asarray(height_maps, dtype=np.float32),
        buffer_features=np.asarray(buffer_features, dtype=np.float32),
        action_mask=np.asarray(action_masks, dtype=np.float32),
        action=np.asarray(actions, dtype=np.int64),
        action_footprint=np.asarray(action_footprints, dtype=np.int64),
        action_footprint_all=np.asarray(action_footprint_all, dtype=np.int64),
        teacher_score=np.asarray(teacher_scores, dtype=np.float32),
        sequence_id=np.asarray(sequence_ids),
        step=np.asarray(steps, dtype=np.int64),
        box_id=np.asarray(box_ids, dtype=np.int64),
        final_episode_score=np.asarray(final_scores, dtype=np.float32),
        fail_flag=np.asarray(fail_flags, dtype=np.bool_),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect policy imitation data.")
    parser.add_argument("--config", default="config/algorithm_config.yaml")
    parser.add_argument("--input", required=True, help="Input JSON/JSONL file or directory")
    parser.add_argument("--output", required=True, help="Output .npz file or directory")
    parser.add_argument("--teacher", choices=("beam", "best_fit", "heuristic"), default="beam")
    parser.add_argument("--max_sequences", type=int, default=1000)
    parser.add_argument("--baseline-score", type=float)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config))
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    input_files = iter_input_files(input_path, args.max_sequences)
    accepted: List[Dict[str, Any]] = []
    rejected = 0
    for file_path in input_files:
        episode = collect_one(
            input_path=file_path,
            cfg=cfg,
            teacher=args.teacher,
            baseline_score=args.baseline_score,
        )
        if episode_accepted(episode, cfg):
            accepted.append(episode)
        else:
            rejected += 1

    if output_path.suffix != ".npz":
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / f"{input_path.stem}_policy_dataset.npz"

    save_dataset(output_path, accepted)
    print(
        f"saved {output_path} accepted={len(accepted)} rejected={rejected} "
        f"samples={sum(len(ep['actions']) for ep in accepted)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
