#!/usr/bin/env python3
"""Small regression checks for the v2 palletizing implementation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from algorithm import AlgorithmConfig, PalletConfig, Palletizer  # noqa: E402
from box_generator import generate_boxes  # noqa: E402
from evaluate import buffer_bonus_from_capacity, verify_score_contract  # noqa: E402
from src.policy_inference import top_k_from_logits  # noqa: E402


def check_score_contract() -> None:
    verify_score_contract()
    assert buffer_bonus_from_capacity(0) == 20.0
    assert buffer_bonus_from_capacity(4) == 16.0
    assert buffer_bonus_from_capacity(20) == 0.0
    assert buffer_bonus_from_capacity(25) == 0.0


def check_generator_reproducibility() -> None:
    a = generate_boxes(seed=123, count=10, mode="uniform")
    b = generate_boxes(seed=123, count=10, mode="uniform")
    c = generate_boxes(seed=123, count=10, mode="sku")
    d = generate_boxes(seed=123, count=10, mode="sku")
    assert a == b
    assert c == d
    assert a != c


def check_rotation_footprint() -> None:
    pallet = Palletizer(
        PalletConfig(length=1.2, width=1.0, height=1.25),
        AlgorithmConfig(allow_rotation=True, buffer_size=1),
    )
    orientations = pallet._candidate_orientations([0.3, 0.2, 0.1])
    assert ((0.3, 0.2, 0.1), 0) in orientations
    assert ((0.2, 0.3, 0.1), 90) in orientations

    candidate = pallet._evaluate_candidate(
        box={"step": 0, "id": 1, "size": [0.3, 0.2, 0.1], "mass": 1.0},
        buffer_index=0,
        dims=(0.2, 0.3, 0.1),
        rotation=90,
        x=1.0,
        y=0.7,
    )
    assert candidate is not None
    assert candidate.x + candidate.dims[0] <= pallet.pallet.length + 1e-9
    assert candidate.y + candidate.dims[1] <= pallet.pallet.width + 1e-9


def check_no_candidate_stops() -> None:
    pallet = Palletizer(
        PalletConfig(length=1.2, width=1.0, height=1.25),
        AlgorithmConfig(allow_rotation=True, buffer_size=2),
    )
    pallet.heuristic.support_threshold = 1.1
    result = pallet.run([
        {"step": 0, "id": 1, "size": [0.3, 0.2, 0.1], "mass": 1.0},
        {"step": 1, "id": 2, "size": [0.3, 0.2, 0.1], "mass": 1.0},
    ])
    assert result["finished_by_user"] is True
    assert result["sequence"] == []


def check_incremental_load_debug() -> None:
    pallet = Palletizer(
        PalletConfig(length=1.2, width=1.0, height=1.25),
        AlgorithmConfig(allow_rotation=True, buffer_size=1),
    )
    pallet.physics_mask.debug_recompute_loads = True
    result = pallet.run([
        {"step": 0, "id": 1, "size": [0.3, 0.3, 0.1], "mass": 2.0},
        {"step": 1, "id": 2, "size": [0.2, 0.2, 0.1], "mass": 1.0},
    ])
    assert len(result["sequence"]) >= 1


def check_no_forbidden_submission_imports() -> None:
    forbidden = {"pybullet", "torch", "optuna", "cma"}
    submission_files = [ROOT / "algorithm.py", ROOT / "main.py", ROOT / "buffer_manager.py"]
    for path in submission_files:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None
        text = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert f"import {name}" not in text
            assert f"from {name}" not in text


def check_policy_top_k_masking() -> None:
    logits = np.zeros((2, 2, 3, 4), dtype=np.float32)
    mask = np.zeros_like(logits)
    logits[1, 0, 2, 3] = 10.0
    logits[0, 1, 1, 2] = 9.0
    mask[0, 1, 1, 2] = 1.0
    actions = top_k_from_logits(logits, mask, top_k=2)
    assert actions[0][:4] == (0, 1, 1, 2)


def main() -> int:
    check_score_contract()
    check_generator_reproducibility()
    check_rotation_footprint()
    check_no_candidate_stops()
    check_incremental_load_debug()
    check_no_forbidden_submission_imports()
    check_policy_top_k_masking()
    print("regression checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
