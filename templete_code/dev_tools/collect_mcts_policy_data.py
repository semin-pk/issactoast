#!/usr/bin/env python3
"""Collect policy data using the full MCTS offline teacher.

Dev-only tool. Full MCTS is never used by submission runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm import AlgorithmConfig, BoxInput, PalletConfig, Palletizer  # noqa: E402
from evaluate import buffer_bonus_from_capacity  # noqa: E402
from dev_tools.mcts_teacher import (  # noqa: E402
    MCTSAction,
    MCTSConfig,
    MCTSState,
    config_from_dict,
    decide,
    final_score,
    generate_actions,
    initial_state,
)


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


def build_palletizer(cfg: Dict[str, Any]) -> Palletizer:
    pallet_cfg = PalletConfig(
        length=float(cfg["pallet"]["length"]),
        width=float(cfg["pallet"]["width"]),
        height=float(cfg["pallet"]["height"]),
    )
    algo_cfg = AlgorithmConfig(
        allow_rotation=bool(cfg["algorithm"]["allow_rotation"]),
        buffer_size=int(cfg["buffer"]["size"]),
    )
    return Palletizer(pallet_cfg, algo_cfg)


def heuristic_baseline_score(cfg: Dict[str, Any], boxes: Sequence[BoxInput]) -> float:
    palletizer = build_palletizer(cfg)
    result = palletizer.run([dict(box) for box in boxes])
    volume = sum(float(np.prod(item["size"])) for item in result["sequence"])
    pallet = palletizer.pallet
    pallet_volume = pallet.length * pallet.width * pallet.height
    util = volume / pallet_volume * 100.0 if pallet_volume > 0.0 else 0.0
    return util + buffer_bonus_from_capacity(palletizer.algo.buffer_size)


def footprint_all_for_state(state: MCTSState, action_mask: np.ndarray) -> np.ndarray:
    footprint_all = np.zeros((action_mask.shape[0], 2, 2), dtype=np.int64)
    p = state.palletizer
    for buffer_index, box in state.indexed_buffer():
        for dims, rotation in p._candidate_orientations(box["size"]):
            rotation_index = p._rotation_index_for_degrees(rotation)
            footprint_all[buffer_index, rotation_index] = [
                int(np.ceil(dims[0] / p.grid_resolution)),
                int(np.ceil(dims[1] / p.grid_resolution)),
            ]
    return footprint_all


def find_action(state: MCTSState, key: Tuple[int, int, int, int], cfg: MCTSConfig) -> Optional[MCTSAction]:
    for action in generate_actions(state, cfg):
        if action.key == key:
            return action
    return None


def collect_one(
    input_path: Path,
    cfg: Dict[str, Any],
    mcts_cfg: MCTSConfig,
    label: str,
) -> Dict[str, Any]:
    boxes = load_boxes(input_path)
    baseline_score = heuristic_baseline_score(cfg, boxes)
    base_palletizer = build_palletizer(cfg)
    state = initial_state(base_palletizer, boxes)

    height_maps: List[np.ndarray] = []
    buffer_features: List[np.ndarray] = []
    action_masks: List[np.ndarray] = []
    actions: List[Tuple[int, int, int, int]] = []
    action_footprints: List[Tuple[int, int]] = []
    action_footprint_all: List[np.ndarray] = []
    mcts_policies: List[np.ndarray] = []
    visit_counts: List[np.ndarray] = []
    q_values: List[np.ndarray] = []
    teacher_scores: List[float] = []
    baseline_scores: List[float] = []
    mcts_scores: List[float] = []
    delta_scores: List[float] = []
    steps: List[int] = []
    box_ids: List[int] = []
    timed_out: List[bool] = []
    mask_violations = 0

    while not state.done:
        if state.palletizer.should_finish(state.buffer):
            state.done = True
            break

        indexed = state.indexed_buffer()
        policy_state = state.palletizer.policy_state(indexed)
        action_mask = policy_state["action_mask"]
        if float(np.sum(action_mask)) <= 0.0:
            state.done = True
            break

        decision = decide(state, mcts_cfg)
        if decision.best_action is None:
            state.done = True
            break

        key = decision.best_action.key
        b_idx, r_idx, x_idx, y_idx = key
        if action_mask[b_idx, r_idx, y_idx, x_idx] <= 0.0:
            mask_violations += 1
            break

        actual_action = find_action(state, key, mcts_cfg)
        if actual_action is None:
            mask_violations += 1
            break

        height_maps.append(policy_state["height_map"])
        buffer_features.append(policy_state["buffer_features"])
        action_masks.append(action_mask)
        actions.append(key)
        action_footprints.append((
            int(np.ceil(actual_action.candidate.dims[0] / state.palletizer.grid_resolution)),
            int(np.ceil(actual_action.candidate.dims[1] / state.palletizer.grid_resolution)),
        ))
        action_footprint_all.append(footprint_all_for_state(state, action_mask))
        mcts_policies.append(decision.mcts_policy)
        visit_counts.append(decision.visit_counts)
        q_values.append(decision.q_values)
        teacher_scores.append(float(actual_action.heuristic_score))
        baseline_scores.append(float(decision.baseline_return))
        mcts_scores.append(float(decision.rollout_return))
        delta_scores.append(float(decision.delta_return))
        steps.append(int(actual_action.candidate.box["step"]))
        box_ids.append(int(actual_action.candidate.box["id"]))
        timed_out.append(bool(decision.timed_out))

        state.apply(actual_action)

    state.palletizer._assert_valid_result()
    mcts_score = final_score(state)
    fail_flag = bool(mask_violations)

    return {
        "input_path": str(input_path),
        "sequence_id": input_path.stem,
        "label": label,
        "height_maps": height_maps,
        "buffer_features": buffer_features,
        "action_masks": action_masks,
        "actions": actions,
        "action_footprints": action_footprints,
        "action_footprint_all": action_footprint_all,
        "mcts_policies": mcts_policies,
        "visit_counts": visit_counts,
        "q_values": q_values,
        "teacher_scores": teacher_scores,
        "baseline_scores": baseline_scores,
        "mcts_scores": mcts_scores,
        "delta_scores": delta_scores,
        "steps": steps,
        "box_ids": box_ids,
        "timed_out": timed_out,
        "baseline_score": float(baseline_score),
        "final_score": float(mcts_score),
        "placed_volume_ratio": float(
            (mcts_score - buffer_bonus_from_capacity(state.palletizer.algo.buffer_size)) / 100.0
        ),
        "fail_flag": fail_flag,
        "mask_violations": int(mask_violations),
    }


def episode_accepted(
    episode: Dict[str, Any],
    cfg: Dict[str, Any],
    min_utilization_threshold: Optional[float] = None,
) -> bool:
    policy_data = cfg.get("policy_data", {})
    min_util = (
        float(policy_data.get("min_utilization_threshold", 0.45))
        if min_utilization_threshold is None
        else float(min_utilization_threshold)
    )
    save_failed = bool(policy_data.get("save_failed_episodes", False))
    if episode["fail_flag"] and not save_failed:
        return False
    if episode["final_score"] + 1e-9 < episode["baseline_score"]:
        return False
    if episode["placed_volume_ratio"] + 1e-9 < min_util:
        return False
    if episode["mask_violations"] > 0:
        return False
    return True


def save_dataset(path: Path, episodes: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not episodes:
        raise RuntimeError("no accepted MCTS episodes to save")

    arrays: Dict[str, List[Any]] = {
        "height_map": [],
        "buffer_features": [],
        "action_mask": [],
        "action": [],
        "action_footprint": [],
        "action_footprint_all": [],
        "mcts_policy": [],
        "visit_counts": [],
        "q_values": [],
        "teacher_score": [],
        "baseline_score": [],
        "mcts_score": [],
        "delta_score": [],
        "sequence_id": [],
        "step": [],
        "box_id": [],
        "final_episode_score": [],
        "fail_flag": [],
        "timed_out": [],
    }

    for episode in episodes:
        n = len(episode["actions"])
        arrays["height_map"].extend(episode["height_maps"])
        arrays["buffer_features"].extend(episode["buffer_features"])
        arrays["action_mask"].extend(episode["action_masks"])
        arrays["action"].extend(episode["actions"])
        arrays["action_footprint"].extend(episode["action_footprints"])
        arrays["action_footprint_all"].extend(episode["action_footprint_all"])
        arrays["mcts_policy"].extend(episode["mcts_policies"])
        arrays["visit_counts"].extend(episode["visit_counts"])
        arrays["q_values"].extend(episode["q_values"])
        arrays["teacher_score"].extend(episode["teacher_scores"])
        arrays["baseline_score"].extend(episode["baseline_scores"])
        arrays["mcts_score"].extend(episode["mcts_scores"])
        arrays["delta_score"].extend(episode["delta_scores"])
        arrays["sequence_id"].extend([episode["sequence_id"]] * n)
        arrays["step"].extend(episode["steps"])
        arrays["box_id"].extend(episode["box_ids"])
        arrays["final_episode_score"].extend([episode["final_score"]] * n)
        arrays["fail_flag"].extend([episode["fail_flag"]] * n)
        arrays["timed_out"].extend(episode["timed_out"])

    np.savez_compressed(
        path,
        height_map=np.asarray(arrays["height_map"], dtype=np.float32),
        buffer_features=np.asarray(arrays["buffer_features"], dtype=np.float32),
        action_mask=np.asarray(arrays["action_mask"], dtype=np.float32),
        action=np.asarray(arrays["action"], dtype=np.int64),
        action_footprint=np.asarray(arrays["action_footprint"], dtype=np.int64),
        action_footprint_all=np.asarray(arrays["action_footprint_all"], dtype=np.int64),
        mcts_policy=np.asarray(arrays["mcts_policy"], dtype=np.float32),
        visit_counts=np.asarray(arrays["visit_counts"], dtype=np.float32),
        q_values=np.asarray(arrays["q_values"], dtype=np.float32),
        teacher_score=np.asarray(arrays["teacher_score"], dtype=np.float32),
        baseline_score=np.asarray(arrays["baseline_score"], dtype=np.float32),
        mcts_score=np.asarray(arrays["mcts_score"], dtype=np.float32),
        delta_score=np.asarray(arrays["delta_score"], dtype=np.float32),
        sequence_id=np.asarray(arrays["sequence_id"]),
        step=np.asarray(arrays["step"], dtype=np.int64),
        box_id=np.asarray(arrays["box_id"], dtype=np.int64),
        final_episode_score=np.asarray(arrays["final_episode_score"], dtype=np.float32),
        fail_flag=np.asarray(arrays["fail_flag"], dtype=np.bool_),
        timed_out=np.asarray(arrays["timed_out"], dtype=np.bool_),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect MCTS teacher policy data.")
    parser.add_argument("--config", default="config/algorithm_config.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-simulations", type=int)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--max-sequences", type=int, default=1000)
    parser.add_argument("--label", default="mcts_teacher_v1")
    parser.add_argument(
        "--min-utilization-threshold",
        type=float,
        help="Override policy_data.min_utilization_threshold for this collection run",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = load_config(config_path)
    mcts_cfg = config_from_dict(cfg)
    if args.num_simulations is not None:
        mcts_cfg.num_simulations = int(args.num_simulations)
    if args.max_depth is not None:
        mcts_cfg.max_depth = int(args.max_depth)

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    episodes: List[Dict[str, Any]] = []
    rejected = 0
    for file_path in iter_input_files(input_path, args.max_sequences):
        episode = collect_one(file_path, cfg, mcts_cfg, args.label)
        if episode_accepted(episode, cfg, args.min_utilization_threshold):
            episodes.append(episode)
        else:
            rejected += 1

    if output_path.suffix != ".npz":
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / f"{input_path.stem}_mcts_dataset.npz"
    save_dataset(output_path, episodes)
    samples = sum(len(episode["actions"]) for episode in episodes)
    print(
        f"saved {output_path} accepted={len(episodes)} rejected={rejected} "
        f"samples={samples}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
