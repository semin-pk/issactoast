#!/usr/bin/env python3
"""PyBullet-based local physics checker for palletizing results.

This module is for local development only. It is not part of the submission
runtime, should not be included in the final submission ZIP, and is never
imported by algorithm.py or main.py.

PyBullet is intentionally treated as an optional local dependency:

    pip install pybullet

That install is unrelated to the challenge submission requirements. The
submitted algorithm still uses only the allowed dependencies.

Limitations:
PyBullet is an approximation of the official Isaac Sim / PhysX evaluation.
Friction, contact, solver, and settling behavior can differ, especially for
borderline cases. Use this checker as a conservative tool for catching likely
collapse/drop/out-of-bounds failures before submitting; the official server is
the final authority.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_SIM_CONFIG = "../palletizing_simulator/config/sim_config.yaml"
DEFAULT_TIME_STEP = 1.0 / 60.0
DEFAULT_SOLVER_ITERATIONS = 80
DEFAULT_GUI_STEP_DELAY = 0.0


@dataclass
class PhysicsFailure:
    index: int
    box_id: Optional[int]
    reason: str
    drift: float
    final_position: List[float]
    intended_position: List[float]


@dataclass
class PhysicsResult:
    pallet_pass: bool
    success_rate: float
    failed_boxes: List[PhysicsFailure]
    reasons: List[str]
    max_final_drift: float
    max_final_top_z: float
    n_boxes: int
    time_step: float
    solver_iterations: int
    sim_config_path: str


def parse_scalar(value: str) -> Any:
    text = value.strip()
    if not text:
        return ""
    if text.startswith("[") and text.endswith("]"):
        return [parse_scalar(part) for part in text[1:-1].split(",") if part.strip()]
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        if any(ch in text for ch in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        return text.strip('"').strip("'")


def read_simple_yaml(path: Path) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue

        indent = len(line) - len(line.lstrip(" "))
        key, value = line.strip().split(":", 1)

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"{path}:{line_no}: invalid indentation")

        parent = stack[-1][1]
        if value.strip() == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(value)

    return root


def require_key(data: Dict[str, Any], key: str, source: Path) -> Any:
    if key not in data:
        raise ValueError(f"{source}: missing required key '{key}'")
    return data[key]


def load_result(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: result JSON must be an object")

    require_key(data, "buffer_size", path)
    sequence = require_key(data, "sequence", path)
    if not isinstance(sequence, list):
        raise ValueError(f"{path}: 'sequence' must be a list")

    for idx, box in enumerate(sequence):
        if not isinstance(box, dict):
            raise ValueError(f"{path}: sequence[{idx}] must be an object")
        for key in ("size", "position", "rotation", "mass"):
            require_key(box, key, path)
        if len(box["size"]) != 3:
            raise ValueError(f"{path}: sequence[{idx}].size must have 3 values")
        if len(box["position"]) != 3:
            raise ValueError(f"{path}: sequence[{idx}].position must have 3 values")
        if int(box["rotation"]) not in (0, 90):
            raise ValueError(f"{path}: sequence[{idx}].rotation must be 0 or 90")

    return data


def rotation_quat_xyzw(degrees: float) -> Tuple[float, float, float, float]:
    rad = math.radians(degrees)
    return (0.0, 0.0, math.sin(rad / 2.0), math.cos(rad / 2.0))


def local_size_for_output_size(size: Sequence[float], rotation: int) -> List[float]:
    """Return PyBullet local box size for the output contract.

    The result JSON stores size after rotation has already been applied in
    world axes. PyBullet, however, applies orientation to local half-extents.
    For rotation=90 we swap X/Y back before applying the quaternion so the
    final world footprint matches evaluate.py's AABB reconstruction.
    """

    sx, sy, sz = [float(v) for v in size]
    if int(rotation) == 90:
        return [sy, sx, sz]
    return [sx, sy, sz]


def import_pybullet():
    try:
        import pybullet as p  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyBullet is not installed. Install it locally with: pip install pybullet "
            "or .venv/bin/pip install pybullet. This dependency is for local "
            "validation only and must not be added to submission requirements.txt."
        ) from exc
    return p


def create_box_body(
    p: Any,
    size: Sequence[float],
    rotation: int,
    mass: float,
    position: Sequence[float],
    box_cfg: Dict[str, Any],
) -> int:
    local_size = local_size_for_output_size(size, rotation)
    half_extents = [value / 2.0 for value in local_size]
    quat = rotation_quat_xyzw(float(rotation))

    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=[0.2, 0.7, 0.9, 0.65],
    )
    body = p.createMultiBody(
        baseMass=max(float(mass), 1e-6),
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=list(position),
        baseOrientation=quat,
    )

    # PyBullet uses lateralFriction rather than separate static/dynamic
    # friction. The closest practical mapping is to set lateralFriction to the
    # static friction value from sim_config. PhysX contactOffset/restOffset do
    # not have an exact PyBullet equivalent, so they are intentionally not set.
    p.changeDynamics(
        body,
        -1,
        lateralFriction=float(box_cfg["static_friction"]),
        restitution=float(box_cfg["restitution"]),
        linearDamping=float(box_cfg["linear_damping"]),
        angularDamping=float(box_cfg["angular_damping"]),
    )
    return body


def create_static_box(
    p: Any,
    half_extents: Sequence[float],
    position: Sequence[float],
    color: Sequence[float],
    lateral_friction: float,
    restitution: float,
) -> int:
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(half_extents))
    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=list(half_extents),
        rgbaColor=list(color),
    )
    body = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=list(position),
    )
    p.changeDynamics(
        body,
        -1,
        lateralFriction=float(lateral_friction),
        restitution=float(restitution),
    )
    return body


def create_buffer_platform(
    p: Any,
    cfg: Dict[str, Any],
    pallet_lx: float,
    n_slots: int,
) -> None:
    """Create the buffer platform to the right of the pallet for GUI viewing."""

    buffer_cfg = cfg["buffer"]
    pallet_cfg = cfg["physics"]["pallet"]
    slot_w, slot_d = [float(v) for v in buffer_cfg["slot_size"]]
    slots_per_row = int(buffer_cfg["slots_per_row"])
    n_cols = max(1, slots_per_row)
    n_rows = max(1, math.ceil(max(n_slots, 1) / n_cols))
    gap = float(buffer_cfg["gap_from_pallet"])
    thickness = float(buffer_cfg["platform_thickness"])
    lx = n_cols * slot_w
    ly = n_rows * slot_d

    create_static_box(
        p=p,
        half_extents=[lx / 2.0, ly / 2.0, thickness / 2.0],
        position=[pallet_lx + gap + lx / 2.0, ly / 2.0, -thickness / 2.0],
        color=[0.4, 0.4, 0.65, 1.0],
        lateral_friction=float(pallet_cfg["static_friction"]),
        restitution=float(pallet_cfg["restitution"]),
    )


def buffer_slot_position(
    cfg: Dict[str, Any],
    slot_idx: int,
    pallet_lx: float,
) -> List[float]:
    """Return a buffer slot center with platform top at z=0 in this PyBullet scene."""

    buffer_cfg = cfg["buffer"]
    gap = float(buffer_cfg["gap_from_pallet"])
    slot_w, slot_d = [float(v) for v in buffer_cfg["slot_size"]]
    slots_per_row = int(buffer_cfg["slots_per_row"])
    col = slot_idx % slots_per_row
    row = slot_idx // slots_per_row
    return [
        pallet_lx + gap + slot_w / 2.0 + col * slot_w,
        slot_d / 2.0 + row * slot_d,
        0.0,
    ]


def step_simulation(
    p: Any,
    *,
    gui: bool,
    time_step: float,
    gui_step_delay: float,
) -> None:
    p.stepSimulation()
    if gui and gui_step_delay > 0.0:
        time.sleep(gui_step_delay)


def body_velocity_norms(p: Any, body_id: int) -> Tuple[float, float]:
    lin_vel, ang_vel = p.getBaseVelocity(body_id)
    return float(np.linalg.norm(lin_vel)), float(np.linalg.norm(ang_vel))


def settle_body(
    p: Any,
    body_id: int,
    max_steps: int,
    min_frames: int,
    velocity_threshold: float,
    *,
    gui: bool = False,
    time_step: float = DEFAULT_TIME_STEP,
    gui_step_delay: float = DEFAULT_GUI_STEP_DELAY,
) -> None:
    steps = max(int(max_steps), int(min_frames) + 1)
    for step in range(steps):
        step_simulation(
            p,
            gui=gui,
            time_step=time_step,
            gui_step_delay=gui_step_delay,
        )
        lin_norm, ang_norm = body_velocity_norms(p, body_id)
        if step >= min_frames and lin_norm < velocity_threshold and ang_norm < velocity_threshold:
            break


def move_body_to_pallet(
    p: Any,
    body_id: int,
    box: Dict[str, Any],
    drop_offset: float,
) -> None:
    target = np.asarray(box["position"], dtype=float)
    target[2] += drop_offset
    p.resetBasePositionAndOrientation(
        body_id,
        target.tolist(),
        rotation_quat_xyzw(float(box["rotation"])),
    )
    p.resetBaseVelocity(body_id, linearVelocity=[0.0, 0.0, 0.0], angularVelocity=[0.0, 0.0, 0.0])


def final_aabb_from_body(p: Any, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
    aabb_min, aabb_max = p.getAABB(body_id)
    return np.asarray(aabb_min, dtype=float), np.asarray(aabb_max, dtype=float)


def check_body_failure(
    p: Any,
    body_id: int,
    idx: int,
    box: Dict[str, Any],
    intended: np.ndarray,
    pallet_size: Sequence[float],
    drift_threshold: float,
    bounds_tol: float,
) -> Tuple[Optional[PhysicsFailure], float, float]:
    final_pos, _ = p.getBasePositionAndOrientation(body_id)
    final = np.asarray(final_pos, dtype=float)
    drift = float(np.linalg.norm(final - intended))
    aabb_min, aabb_max = final_aabb_from_body(p, body_id)
    max_top = float(aabb_max[2])

    reasons: List[str] = []
    if drift >= drift_threshold:
        reasons.append(f"DROP/COLLAPSE drift={drift:.4f} >= {drift_threshold:.4f}")

    if (
        aabb_min[0] < -bounds_tol
        or aabb_min[1] < -bounds_tol
        or aabb_max[0] > float(pallet_size[0]) + bounds_tol
        or aabb_max[1] > float(pallet_size[1]) + bounds_tol
    ):
        reasons.append(
            "OUT_OF_BOUNDS "
            f"aabb_min={aabb_min.round(4).tolist()} "
            f"aabb_max={aabb_max.round(4).tolist()}"
        )

    if max_top > float(pallet_size[2]) + bounds_tol:
        reasons.append(
            f"HEIGHT_OVERFLOW top={max_top:.4f} > {float(pallet_size[2]) + bounds_tol:.4f}"
        )

    if not reasons:
        return None, drift, max_top

    return (
        PhysicsFailure(
            index=idx,
            box_id=int(box["id"]) if "id" in box else None,
            reason="; ".join(reasons),
            drift=drift,
            final_position=final.round(6).tolist(),
            intended_position=intended.round(6).tolist(),
        ),
        drift,
        max_top,
    )


def run(
    result_path: str | Path,
    *,
    sim_config_path: str | Path = DEFAULT_SIM_CONFIG,
    gui: bool = False,
    time_step: float = DEFAULT_TIME_STEP,
    solver_iterations: int = DEFAULT_SOLVER_ITERATIONS,
    show_buffer: bool = False,
    gui_step_delay: float = DEFAULT_GUI_STEP_DELAY,
    strict: bool = False,
    strict_friction_scale: float = 0.6,
    strict_drift_threshold_m: float = 0.01,
) -> PhysicsResult:
    """Run PyBullet simulation for one result JSON file."""

    p = import_pybullet()
    result_path = Path(result_path)
    sim_config_path = Path(sim_config_path)
    result = load_result(result_path)
    cfg = read_simple_yaml(sim_config_path)

    pallet_size = [float(v) for v in cfg["pallet"]["size"]]
    settling = dict(cfg["settling"])
    evaluation = dict(cfg["evaluation"])
    box_cfg = dict(cfg["physics"]["box"])
    pallet_cfg = dict(cfg["physics"]["pallet"])
    ground_cfg = dict(cfg["physics"]["ground"])

    if strict:
        solver_iterations = max(int(solver_iterations), 160)
        evaluation["drift_threshold_m"] = min(
            float(evaluation["drift_threshold_m"]),
            float(strict_drift_threshold_m),
        )
        settling["final_steps"] = int(float(settling["final_steps"]) * 2.0)
        settling["max_steps"] = int(float(settling["max_steps"]) * 1.5)
        for dynamics_cfg in (box_cfg, pallet_cfg, ground_cfg):
            dynamics_cfg["static_friction"] = (
                float(dynamics_cfg["static_friction"]) * float(strict_friction_scale)
            )

    client_id = p.connect(p.GUI if gui else p.DIRECT)
    body_ids: List[int] = []
    failures: List[PhysicsFailure] = []
    max_drift = 0.0
    max_top = 0.0

    try:
        p.resetSimulation()
        p.setGravity(0.0, 0.0, -9.81)
        p.setTimeStep(float(time_step))
        p.setPhysicsEngineParameter(
            fixedTimeStep=float(time_step),
            numSolverIterations=int(solver_iterations),
            deterministicOverlappingPairs=1,
        )

        plane_collision = p.createCollisionShape(p.GEOM_PLANE)
        plane = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=plane_collision)
        p.changeDynamics(
            plane,
            -1,
            lateralFriction=float(ground_cfg["static_friction"]),
            restitution=float(ground_cfg["restitution"]),
        )

        pallet_lx, pallet_ly, _ = pallet_size
        pallet_thickness = float(pallet_cfg["thickness"])
        pallet_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[pallet_lx / 2.0, pallet_ly / 2.0, pallet_thickness / 2.0],
        )
        pallet_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[pallet_lx / 2.0, pallet_ly / 2.0, pallet_thickness / 2.0],
            rgbaColor=[0.55, 0.35, 0.15, 1.0],
        )
        pallet = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=pallet_collision,
            baseVisualShapeIndex=pallet_visual,
            basePosition=[pallet_lx / 2.0, pallet_ly / 2.0, -pallet_thickness / 2.0],
        )
        p.changeDynamics(
            pallet,
            -1,
            lateralFriction=float(pallet_cfg["static_friction"]),
            restitution=float(pallet_cfg["restitution"]),
        )

        drop_offset = float(settling["drop_offset"])
        max_steps = int(settling["max_steps"])
        min_frames = int(settling["min_frames"])
        velocity_threshold = float(settling["velocity_threshold"])

        sequence = result["sequence"]
        buffer_size = int(result.get("buffer_size", 0))

        if show_buffer and buffer_size > 0:
            create_buffer_platform(p, cfg, pallet_lx, buffer_size)
            n_initial = min(buffer_size, len(sequence))
            slot_bodies: List[Optional[int]] = [None] * buffer_size
            next_to_spawn = n_initial

            for slot_idx in range(n_initial):
                box = sequence[slot_idx]
                slot = buffer_slot_position(cfg, slot_idx, pallet_lx)
                spawn_z = slot[2] + float(box["size"][2]) / 2.0 + drop_offset
                slot_bodies[slot_idx] = create_box_body(
                    p=p,
                    size=box["size"],
                    rotation=int(box["rotation"]),
                    mass=float(box["mass"]),
                    position=[slot[0], slot[1], spawn_z],
                    box_cfg=box_cfg,
                )
                settle_body(
                    p,
                    slot_bodies[slot_idx],
                    max_steps,
                    min_frames,
                    velocity_threshold,
                    gui=gui,
                    time_step=time_step,
                    gui_step_delay=gui_step_delay,
                )

            for idx, box in enumerate(sequence):
                slot_idx = idx % buffer_size
                body = slot_bodies[slot_idx]
                if body is None:
                    slot = buffer_slot_position(cfg, slot_idx, pallet_lx)
                    spawn_z = slot[2] + float(box["size"][2]) / 2.0 + drop_offset
                    body = create_box_body(
                        p=p,
                        size=box["size"],
                        rotation=int(box["rotation"]),
                        mass=float(box["mass"]),
                        position=[slot[0], slot[1], spawn_z],
                        box_cfg=box_cfg,
                    )

                body_ids.append(body)
                move_body_to_pallet(p, body, box, drop_offset)
                settle_body(
                    p,
                    body,
                    max_steps,
                    min_frames,
                    velocity_threshold,
                    gui=gui,
                    time_step=time_step,
                    gui_step_delay=gui_step_delay,
                )

                if next_to_spawn < len(sequence):
                    refill = sequence[next_to_spawn]
                    slot = buffer_slot_position(cfg, slot_idx, pallet_lx)
                    spawn_z = slot[2] + float(refill["size"][2]) / 2.0 + drop_offset
                    slot_bodies[slot_idx] = create_box_body(
                        p=p,
                        size=refill["size"],
                        rotation=int(refill["rotation"]),
                        mass=float(refill["mass"]),
                        position=[slot[0], slot[1], spawn_z],
                        box_cfg=box_cfg,
                    )
                    settle_body(
                        p,
                        slot_bodies[slot_idx],
                        max_steps,
                        min_frames,
                        velocity_threshold,
                        gui=gui,
                        time_step=time_step,
                        gui_step_delay=gui_step_delay,
                    )
                    next_to_spawn += 1
                else:
                    slot_bodies[slot_idx] = None
        else:
            for idx, box in enumerate(sequence):
                intended = np.asarray(box["position"], dtype=float)
                spawn_position = intended.copy()
                spawn_position[2] += drop_offset
                body = create_box_body(
                    p=p,
                    size=box["size"],
                    rotation=int(box["rotation"]),
                    mass=float(box["mass"]),
                    position=spawn_position.tolist(),
                    box_cfg=box_cfg,
                )
                body_ids.append(body)
                settle_body(
                    p,
                    body,
                    max_steps,
                    min_frames,
                    velocity_threshold,
                    gui=gui,
                    time_step=time_step,
                    gui_step_delay=gui_step_delay,
                )

        for _ in range(int(settling["final_steps"])):
            step_simulation(
                p,
                gui=gui,
                time_step=time_step,
                gui_step_delay=gui_step_delay,
            )

        for idx, (box, body) in enumerate(zip(result["sequence"], body_ids)):
            intended = np.asarray(box["position"], dtype=float)
            failure, drift, top = check_body_failure(
                p=p,
                body_id=body,
                idx=idx,
                box=box,
                intended=intended,
                pallet_size=pallet_size,
                drift_threshold=float(evaluation["drift_threshold_m"]),
                bounds_tol=float(evaluation["bounds_tolerance_m"]),
            )
            max_drift = max(max_drift, drift)
            max_top = max(max_top, top)
            if failure is not None:
                failures.append(failure)

        n_boxes = len(result["sequence"])
        success_count = n_boxes - len(failures)
        success_rate = 1.0 if n_boxes == 0 else success_count / n_boxes
        pallet_pass = success_rate >= float(evaluation["episode_success_min_rate"])

        reasons = [failure.reason for failure in failures[:12]]
        if len(failures) > 12:
            reasons.append(f"... {len(failures) - 12} additional physics failures")

        return PhysicsResult(
            pallet_pass=pallet_pass,
            success_rate=success_rate,
            failed_boxes=failures,
            reasons=reasons,
            max_final_drift=max_drift,
            max_final_top_z=max_top,
            n_boxes=n_boxes,
            time_step=float(time_step),
            solver_iterations=int(solver_iterations),
            sim_config_path=str(sim_config_path),
        )
    finally:
        p.disconnect(client_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PyBullet physics validation for palletizing JSON results.")
    parser.add_argument("--result", help="Single result JSON file")
    parser.add_argument("--results", help="Directory of result JSON files")
    parser.add_argument("--sim-config", default=DEFAULT_SIM_CONFIG)
    parser.add_argument("--gui", action="store_true", help="Use pybullet.GUI instead of DIRECT")
    parser.add_argument(
        "--show-buffer",
        action="store_true",
        help="Show a sliding-window staging buffer before boxes move to the pallet",
    )
    parser.add_argument(
        "--gui-step-delay",
        type=float,
        default=DEFAULT_GUI_STEP_DELAY,
        help="Sleep seconds after each PyBullet step when --gui is used",
    )
    parser.add_argument("--time-step", type=float, default=DEFAULT_TIME_STEP)
    parser.add_argument("--solver-iterations", type=int, default=DEFAULT_SOLVER_ITERATIONS)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use conservative local settings: lower friction, more settling, tighter drift",
    )
    parser.add_argument("--strict-friction-scale", type=float, default=0.6)
    parser.add_argument("--strict-drift-threshold-m", type=float, default=0.01)
    return parser


def iter_result_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    files = sorted(item for item in path.glob("*.json") if item.is_file())
    if not files:
        raise FileNotFoundError(f"no result .json files found in {path}")
    return files


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.result and not args.results:
        parser.error("one of --result or --results is required")

    files = iter_result_files(Path(args.result or args.results))
    print("PyBullet Physics Checker")
    print("NOTE: PyBullet is an Isaac Sim / PhysX approximation for local risk screening.")
    print(
        f"time_step={args.time_step} solver_iterations={args.solver_iterations} "
        f"gui={args.gui} show_buffer={args.show_buffer} strict={args.strict}"
    )

    for path in files:
        result = run(
            path,
            sim_config_path=args.sim_config,
            gui=args.gui,
            time_step=args.time_step,
            solver_iterations=args.solver_iterations,
            show_buffer=args.show_buffer,
            gui_step_delay=args.gui_step_delay,
            strict=args.strict,
            strict_friction_scale=args.strict_friction_scale,
            strict_drift_threshold_m=args.strict_drift_threshold_m,
        )
        status = "PASS" if result.pallet_pass else "FAIL"
        reason_text = " | ".join(result.reasons[:3]) if result.reasons else "-"
        print(
            f"{path.name}: {status} "
            f"success_rate={result.success_rate:.2%} "
            f"max_drift={result.max_final_drift:.4f} "
            f"max_top={result.max_final_top_z:.4f} "
            f"reasons={reason_text}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
