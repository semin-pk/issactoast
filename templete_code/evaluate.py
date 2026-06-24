#!/usr/bin/env python3
"""Geometric evaluator for palletizing result JSON files.

This tool approximates the official palletizing score:

    final_score = utilization_points + buffer_bonus

with an all-or-nothing HARD FAIL rule. If any box leaves the pallet bounds,
exceeds pallet height, or penetrates another box, that result file receives
0 points. Stability checks are reported as warnings by default because Isaac
Sim physics can differ from this geometric approximation. Use
--strict-stability to promote stability warnings to HARD FAIL.

This is an independent development tool. By default it only reads algorithm
outputs and configuration files; with --refresh-results it runs main.py first
so algorithm_results reflect the current algorithm_config.yaml.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_BOUNDS_TOL = 0.0
DEFAULT_SUPPORT_THRESHOLD = 0.6
DEFAULT_Z_TOL = 0.01
DEFAULT_EPSILON = 1e-6
DEFAULT_PALLET_SIZE = (1.2, 1.0, 1.25)


@dataclass
class EvalThresholds:
    bounds_tol: float = DEFAULT_BOUNDS_TOL
    support_threshold: float = DEFAULT_SUPPORT_THRESHOLD
    z_tol: float = DEFAULT_Z_TOL
    epsilon: float = DEFAULT_EPSILON
    strict_stability: bool = False


@dataclass
class FileScore:
    file: str
    placed_count: int
    utilization_pct: float
    util_points: float
    buffer_size: int
    buffer_bonus: float
    hard_fail: bool
    stability_risk: bool
    final_score: float
    reasons: List[str]
    stability_reasons: List[str]
    physics_mode: str = "geom"
    physics_pass: Optional[bool] = None
    max_final_drift: Optional[float] = None
    max_final_top_z: Optional[float] = None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def require_key(data: Dict[str, Any], key: str, source: Path) -> Any:
    if key not in data:
        raise ValueError(f"{source}: missing required key '{key}'")
    return data[key]


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
    """Read the simple nested key/value YAML used by this project.

    The evaluator is intentionally limited to the standard library + numpy, so
    it does not import PyYAML. This parser supports the subset needed by the
    current config files: indentation-based dictionaries, scalar values, and
    inline lists such as [1.2, 1.0, 1.25].
    """

    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if ":" not in line:
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


def load_pallet_size(config_path: Path) -> Tuple[float, float, float]:
    cfg = read_simple_yaml(config_path)
    pallet = cfg.get("pallet", {})
    try:
        return (
            float(pallet["length"]),
            float(pallet["width"]),
            float(pallet["height"]),
        )
    except KeyError as exc:
        raise ValueError(f"{config_path}: missing pallet.{exc.args[0]}") from exc


def load_config_buffer_size(config_path: Path) -> int:
    cfg = read_simple_yaml(config_path)
    buffer_cfg = cfg.get("buffer", {})
    try:
        return int(buffer_cfg["size"])
    except KeyError as exc:
        raise ValueError(f"{config_path}: missing buffer.{exc.args[0]}") from exc


def load_sim_bounds_tolerance(sim_config_path: Path) -> Optional[float]:
    if not sim_config_path.exists():
        return None
    cfg = read_simple_yaml(sim_config_path)
    evaluation = cfg.get("evaluation", {})
    value = evaluation.get("bounds_tolerance_m")
    return None if value is None else float(value)


def load_result(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"{path}: result JSON must be an object")

    buffer_size = require_key(data, "buffer_size", path)
    sequence = require_key(data, "sequence", path)

    if not isinstance(buffer_size, int):
        raise ValueError(f"{path}: 'buffer_size' must be an integer")
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


def result_to_arrays(result: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Return sizes and centroid positions as float arrays.

    The current output contract stores size after rotation has already been
    applied. Therefore AABB is reconstructed directly from position +/- size/2.
    """

    sequence = result["sequence"]
    if not sequence:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)

    sizes = np.asarray([box["size"] for box in sequence], dtype=float)
    positions = np.asarray([box["position"] for box in sequence], dtype=float)
    return sizes, positions


def aabbs_from_centroids(
    sizes: np.ndarray,
    positions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    half = sizes / 2.0
    return positions - half, positions + half


def check_bounds_and_height(
    mins: np.ndarray,
    maxs: np.ndarray,
    pallet_size: Tuple[float, float, float],
    bounds_tol: float,
) -> List[str]:
    reasons: List[str] = []
    if mins.size == 0:
        return reasons

    low_limit = np.array([0.0, 0.0, 0.0]) - bounds_tol
    high_limit = np.array(pallet_size) + bounds_tol

    below = mins < low_limit
    above_xy = maxs[:, :2] > high_limit[:2]
    above_z = maxs[:, 2] > high_limit[2]

    bad_low = np.where(np.any(below, axis=1))[0]
    bad_xy = np.where(np.any(above_xy, axis=1))[0]
    bad_z = np.where(above_z)[0]

    for idx in bad_low[:5]:
        reasons.append(f"box[{idx}] below lower bounds min={mins[idx].round(4).tolist()}")
    for idx in bad_xy[:5]:
        reasons.append(f"box[{idx}] outside XY bounds max={maxs[idx, :2].round(4).tolist()}")
    for idx in bad_z[:5]:
        reasons.append(f"box[{idx}] exceeds height top={maxs[idx, 2]:.4f}")

    extra = len(bad_low) + len(bad_xy) + len(bad_z) - len(reasons)
    if extra > 0:
        reasons.append(f"... {extra} additional bounds/height failures")

    return reasons


def check_overlaps(mins: np.ndarray, maxs: np.ndarray, epsilon: float) -> List[str]:
    reasons: List[str] = []
    n_boxes = len(mins)

    for i in range(n_boxes):
        if i + 1 >= n_boxes:
            break
        overlap_min = np.maximum(mins[i], mins[i + 1:])
        overlap_max = np.minimum(maxs[i], maxs[i + 1:])
        overlap = overlap_max - overlap_min
        bad_offsets = np.where(np.all(overlap > epsilon, axis=1))[0]

        for offset in bad_offsets[: max(0, 5 - len(reasons))]:
            j = i + 1 + int(offset)
            reasons.append(
                f"box[{i}] intersects box[{j}] "
                f"overlap={overlap[int(offset)].round(6).tolist()}"
            )
        if len(reasons) >= 5:
            reasons.append("... additional overlap failures omitted")
            break

    return reasons


def support_rectangles_for_box(
    idx: int,
    mins: np.ndarray,
    maxs: np.ndarray,
    z_tol: float,
) -> List[Tuple[float, float, float, float]]:
    bottom_z = mins[idx, 2]
    if bottom_z <= z_tol:
        return [(mins[idx, 0], mins[idx, 1], maxs[idx, 0], maxs[idx, 1])]

    z_contact = np.abs(maxs[:, 2] - bottom_z) <= z_tol
    z_contact[idx] = False
    candidates = np.where(z_contact)[0]

    rects: List[Tuple[float, float, float, float]] = []
    for below_idx in candidates:
        x0 = max(mins[idx, 0], mins[below_idx, 0])
        y0 = max(mins[idx, 1], mins[below_idx, 1])
        x1 = min(maxs[idx, 0], maxs[below_idx, 0])
        y1 = min(maxs[idx, 1], maxs[below_idx, 1])
        if x1 > x0 and y1 > y0:
            rects.append((float(x0), float(y0), float(x1), float(y1)))

    return rects


def union_area_rectangles(rects: Sequence[Tuple[float, float, float, float]]) -> float:
    """Compute exact union area of axis-aligned support rectangles."""

    if not rects:
        return 0.0

    xs = sorted({rect[0] for rect in rects} | {rect[2] for rect in rects})
    area = 0.0

    for x0, x1 in zip(xs, xs[1:]):
        if x1 <= x0:
            continue

        intervals: List[Tuple[float, float]] = []
        for rx0, ry0, rx1, ry1 in rects:
            if rx0 < x1 and rx1 > x0:
                intervals.append((ry0, ry1))

        if not intervals:
            continue

        intervals.sort()
        covered_y = 0.0
        cur0, cur1 = intervals[0]
        for y0, y1 in intervals[1:]:
            if y0 <= cur1:
                cur1 = max(cur1, y1)
            else:
                covered_y += cur1 - cur0
                cur0, cur1 = y0, y1
        covered_y += cur1 - cur0
        area += (x1 - x0) * covered_y

    return area


def check_stability(
    mins: np.ndarray,
    maxs: np.ndarray,
    positions: np.ndarray,
    support_threshold: float,
    z_tol: float,
) -> List[str]:
    warnings: List[str] = []

    for idx in range(len(mins)):
        footprint_area = float((maxs[idx, 0] - mins[idx, 0]) * (maxs[idx, 1] - mins[idx, 1]))
        if footprint_area <= 0.0:
            warnings.append(f"box[{idx}] has non-positive footprint area")
            continue

        rects = support_rectangles_for_box(idx, mins, maxs, z_tol)
        support_area = union_area_rectangles(rects)
        support_ratio = support_area / footprint_area

        if support_ratio + 1e-12 < support_threshold:
            warnings.append(
                f"box[{idx}] low support ratio={support_ratio:.3f} "
                f"< {support_threshold:.3f}"
            )

        if not rects:
            warnings.append(f"box[{idx}] center of mass unsupported")
            continue

        sx0 = min(rect[0] for rect in rects)
        sy0 = min(rect[1] for rect in rects)
        sx1 = max(rect[2] for rect in rects)
        sy1 = max(rect[3] for rect in rects)
        cx, cy = float(positions[idx, 0]), float(positions[idx, 1])
        if not (sx0 - 1e-12 <= cx <= sx1 + 1e-12 and sy0 - 1e-12 <= cy <= sy1 + 1e-12):
            warnings.append(
                f"box[{idx}] center of mass outside support range "
                f"center=({cx:.3f},{cy:.3f}) support=({sx0:.3f},{sy0:.3f})-({sx1:.3f},{sy1:.3f})"
            )

    if len(warnings) > 12:
        return warnings[:12] + [f"... {len(warnings) - 12} additional stability warnings"]
    return warnings


def evaluate_result_file(
    path: Path,
    pallet_size: Tuple[float, float, float],
    thresholds: EvalThresholds,
    physics_options: Optional[Dict[str, Any]] = None,
) -> FileScore:
    result = load_result(path)
    sizes, positions = result_to_arrays(result)
    mins, maxs = aabbs_from_centroids(sizes, positions)

    placed_count = int(len(sizes))
    placed_volume = float(np.sum(np.prod(sizes, axis=1))) if placed_count else 0.0
    pallet_volume = float(np.prod(np.asarray(pallet_size, dtype=float)))
    utilization_pct = placed_volume / pallet_volume * 100.0 if pallet_volume > 0 else 0.0
    util_points = min(utilization_pct, 100.0)

    buffer_size = int(result["buffer_size"])
    buffer_bonus = clamp(20.0 - float(buffer_size), 0.0, 20.0)

    hard_reasons: List[str] = []
    hard_reasons.extend(
        check_bounds_and_height(mins, maxs, pallet_size, thresholds.bounds_tol)
    )
    hard_reasons.extend(check_overlaps(mins, maxs, thresholds.epsilon))

    physics_mode = "pybullet" if physics_options is not None else "geom"
    physics_pass: Optional[bool] = None
    max_final_drift: Optional[float] = None
    max_final_top_z: Optional[float] = None

    if physics_options is None:
        stability_reasons = check_stability(
            mins=mins,
            maxs=maxs,
            positions=positions,
            support_threshold=thresholds.support_threshold,
            z_tol=thresholds.z_tol,
        )
    else:
        import physics_check

        physics_result = physics_check.run(
            path,
            sim_config_path=physics_options["sim_config_path"],
            gui=bool(physics_options["gui"]),
            time_step=float(physics_options["time_step"]),
            solver_iterations=int(physics_options["solver_iterations"]),
            show_buffer=bool(physics_options["show_buffer"]),
            gui_step_delay=float(physics_options["gui_step_delay"]),
        )
        physics_pass = bool(physics_result.pallet_pass)
        max_final_drift = float(physics_result.max_final_drift)
        max_final_top_z = float(physics_result.max_final_top_z)
        stability_reasons = list(physics_result.reasons)

        if not physics_result.pallet_pass:
            hard_reasons.extend(f"physics: {reason}" for reason in physics_result.reasons[:5])
            if not physics_result.reasons:
                hard_reasons.append("physics: pallet failed")

    if thresholds.strict_stability and stability_reasons:
        hard_reasons.extend(f"strict stability: {reason}" for reason in stability_reasons)

    hard_fail = bool(hard_reasons)
    final_score = 0.0 if hard_fail else util_points + buffer_bonus

    return FileScore(
        file=path.name,
        placed_count=placed_count,
        utilization_pct=utilization_pct,
        util_points=util_points,
        buffer_size=buffer_size,
        buffer_bonus=buffer_bonus,
        hard_fail=hard_fail,
        stability_risk=bool(stability_reasons),
        final_score=final_score,
        reasons=hard_reasons,
        stability_reasons=stability_reasons,
        physics_mode=physics_mode,
        physics_pass=physics_pass,
        max_final_drift=max_final_drift,
        max_final_top_z=max_final_top_z,
    )


def iter_result_files(results_dir: Path) -> List[Path]:
    if not results_dir.exists():
        raise FileNotFoundError(f"results path not found: {results_dir}")
    if results_dir.is_file():
        return [results_dir]

    files = sorted(path for path in results_dir.glob("*.json") if path.is_file())
    if not files:
        raise FileNotFoundError(f"no result .json files found in: {results_dir}")
    return files


def result_buffer_sizes(result_files: Sequence[Path]) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    for path in result_files:
        result = load_result(path)
        sizes[path.name] = int(result["buffer_size"])
    return sizes


def warn_if_results_stale(result_files: Sequence[Path], config_buffer_size: int) -> None:
    mismatches = {
        name: size
        for name, size in result_buffer_sizes(result_files).items()
        if size != config_buffer_size
    }
    if not mismatches:
        return

    mismatch_text = ", ".join(
        f"{name}: result={size}" for name, size in sorted(mismatches.items())
    )
    print(
        "[WARN] result JSON buffer_size does not match config "
        f"buffer.size={config_buffer_size}. Stale results? {mismatch_text}"
    )
    print("[WARN] Run evaluate.py with --refresh-results to execute main.py first.")


def refresh_results_with_main() -> None:
    here = Path(__file__).resolve().parent
    main_path = here / "main.py"
    if not main_path.exists():
        raise FileNotFoundError(f"main.py not found next to evaluate.py: {main_path}")

    print("[INFO] --refresh-results enabled: running main.py before evaluation", flush=True)
    subprocess.run([sys.executable, str(main_path)], cwd=here, check=True)


def format_reasons(score: FileScore) -> str:
    parts: List[str] = []
    if score.reasons:
        parts.extend(score.reasons[:3])
    if score.stability_reasons:
        parts.extend(f"RISK: {reason}" for reason in score.stability_reasons[:2])
    return " | ".join(parts) if parts else "-"


def print_report(scores: Sequence[FileScore], thresholds: EvalThresholds) -> None:
    print()
    print("Geometric Palletizing Evaluator")
    print("NOTE: This approximates Isaac Sim physics with AABB/support geometry.")
    print("      HARD FAIL => 0 points. Stability risk is warning unless --strict-stability is set.")
    print("      With --physics, PyBullet replaces geometric stability risk and can fail the pallet.")
    print(
        "thresholds: "
        f"bounds_tol={thresholds.bounds_tol}, "
        f"support_threshold={thresholds.support_threshold}, "
        f"z_tol={thresholds.z_tol}, "
        f"epsilon={thresholds.epsilon}, "
        f"strict_stability={thresholds.strict_stability}"
    )
    print()

    headers = [
        "file",
        "placed",
        "util%",
        "util_pts",
        "buffer",
        "bonus",
        "pass/fail",
        "score",
        "phys",
        "reasons",
    ]
    widths = [24, 7, 8, 9, 7, 7, 10, 8, 8, 60]
    print(" ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("-" * (sum(widths) + len(widths) - 1))

    for score in scores:
        status = "FAIL" if score.hard_fail else "PASS"
        if score.stability_risk and not score.hard_fail:
            status = "PASS/RISK"
        row = [
            score.file,
            str(score.placed_count),
            f"{score.utilization_pct:.2f}",
            f"{score.util_points:.2f}",
            str(score.buffer_size),
            f"{score.buffer_bonus:.2f}",
            status,
            f"{score.final_score:.2f}",
            score.physics_mode,
            format_reasons(score),
        ]
        print(" ".join(value[:width].ljust(width) for value, width in zip(row, widths)))

    summary = summarize_scores(scores)
    print("-" * (sum(widths) + len(widths) - 1))
    print(
        "SUMMARY "
        f"n_files={summary['n_files']} "
        f"mean_final_score={summary['mean_final_score']:.2f} "
        f"pass_rate={summary['pass_rate']:.2%} "
        f"mean_utilization={summary['mean_utilization_pct']:.2f}% "
        f"mean_buffer_bonus={summary['mean_buffer_bonus']:.2f} "
        f"physics_mode={scores[0].physics_mode if scores else 'geom'}"
    )
    print("대표 지표: mean_final_score (공식 5세트 평균 평가를 흉내낸 산술평균)")


def summarize_scores(scores: Sequence[FileScore]) -> Dict[str, float]:
    if not scores:
        return {
            "n_files": 0,
            "mean_final_score": 0.0,
            "pass_rate": 0.0,
            "mean_utilization_pct": 0.0,
            "mean_buffer_bonus": 0.0,
        }

    return {
        "n_files": float(len(scores)),
        "mean_final_score": float(np.mean([score.final_score for score in scores])),
        "pass_rate": float(np.mean([not score.hard_fail for score in scores])),
        "mean_utilization_pct": float(np.mean([score.utilization_pct for score in scores])),
        "mean_buffer_bonus": float(np.mean([score.buffer_bonus for score in scores])),
    }


def append_benchmark_log(
    log_path: Path,
    label: str,
    scores: Sequence[FileScore],
) -> None:
    summary = summarize_scores(scores)
    fieldnames = [
        "timestamp",
        "label",
        "physics_mode",
        "n_files",
        "mean_final_score",
        "pass_rate",
        "mean_utilization_pct",
        "mean_buffer_bonus",
    ]
    ensure_benchmark_log_schema(log_path, fieldnames)
    exists = log_path.exists()

    with log_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "label": label,
            "physics_mode": scores[0].physics_mode if scores else "geom",
            "n_files": int(summary["n_files"]),
            "mean_final_score": f"{summary['mean_final_score']:.6f}",
            "pass_rate": f"{summary['pass_rate']:.6f}",
            "mean_utilization_pct": f"{summary['mean_utilization_pct']:.6f}",
            "mean_buffer_bonus": f"{summary['mean_buffer_bonus']:.6f}",
        })


def ensure_benchmark_log_schema(log_path: Path, fieldnames: Sequence[str]) -> None:
    """Migrate older benchmark logs that did not have physics_mode."""

    if not log_path.exists():
        return

    with log_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == list(fieldnames):
            return
        rows = list(reader)

    migrated_rows: List[Dict[str, Any]] = []
    for row in rows:
        migrated = {field: row.get(field, "") for field in fieldnames}
        if not migrated.get("physics_mode"):
            migrated["physics_mode"] = "geom"
        migrated_rows.append(migrated)

    with log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(migrated_rows)


def write_eval_report(report_path: Path, label: str, scores: Sequence[FileScore]) -> None:
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "summary": summarize_scores(scores),
        "files": [asdict(score) for score in scores],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score palletizing algorithm_results JSON files with a geometric official-score approximation."
    )
    parser.add_argument("--results", default="algorithm_results", help="Result JSON file or directory")
    parser.add_argument("--label", default="unlabeled", help="Version label appended to benchmark_log.csv")
    parser.add_argument("--config", default="config/algorithm_config.yaml", help="Algorithm config path")
    parser.add_argument(
        "--sim-config",
        default="../palletizing_simulator/config/sim_config.yaml",
        help="Simulator config path, used only with --use-sim-bounds-tol",
    )
    parser.add_argument("--bounds-tol", type=float, default=DEFAULT_BOUNDS_TOL)
    parser.add_argument(
        "--use-sim-bounds-tol",
        action="store_true",
        help="Use evaluation.bounds_tolerance_m from sim config instead of --bounds-tol",
    )
    parser.add_argument("--support-threshold", type=float, default=DEFAULT_SUPPORT_THRESHOLD)
    parser.add_argument("--z-tol", type=float, default=DEFAULT_Z_TOL)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--strict-stability", action="store_true")
    parser.add_argument(
        "--physics",
        action="store_true",
        help="Use PyBullet physics_check.py result instead of geometric stability warnings",
    )
    parser.add_argument("--gui", action="store_true", help="Use pybullet.GUI when --physics is enabled")
    parser.add_argument(
        "--physics-show-buffer",
        action="store_true",
        help="With --physics, show a sliding-window staging buffer before pallet placement",
    )
    parser.add_argument(
        "--gui-step-delay",
        type=float,
        default=0.0,
        help="Sleep seconds after each PyBullet step when --gui is used",
    )
    parser.add_argument("--time-step", type=float, default=1.0 / 60.0)
    parser.add_argument("--solver-iterations", type=int, default=80)
    parser.add_argument("--benchmark-log", default="benchmark_log.csv")
    parser.add_argument("--report-json", default="eval_report.json")
    parser.add_argument("--no-report-json", action="store_true")
    parser.add_argument(
        "--refresh-results",
        action="store_true",
        help="Run main.py first so algorithm_results reflect the current algorithm_config.yaml",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.refresh_results:
        try:
            refresh_results_with_main()
        except subprocess.CalledProcessError as exc:
            parser.exit(status=exc.returncode, message=f"[ERROR] main.py failed with exit code {exc.returncode}\n")

    results_path = Path(args.results)
    config_path = Path(args.config)
    sim_config_path = Path(args.sim_config)
    pallet_size = load_pallet_size(config_path)
    config_buffer_size = load_config_buffer_size(config_path)
    result_files = iter_result_files(results_path)

    if not args.refresh_results:
        warn_if_results_stale(result_files, config_buffer_size)

    bounds_tol = float(args.bounds_tol)
    if args.use_sim_bounds_tol:
        sim_tol = load_sim_bounds_tolerance(sim_config_path)
        if sim_tol is None:
            raise ValueError(f"{sim_config_path}: missing evaluation.bounds_tolerance_m")
        bounds_tol = sim_tol

    thresholds = EvalThresholds(
        bounds_tol=bounds_tol,
        support_threshold=float(args.support_threshold),
        z_tol=float(args.z_tol),
        epsilon=float(args.epsilon),
        strict_stability=bool(args.strict_stability),
    )

    physics_options = None
    if args.physics:
        physics_options = {
            "sim_config_path": sim_config_path,
            "gui": bool(args.gui),
            "time_step": float(args.time_step),
            "solver_iterations": int(args.solver_iterations),
            "show_buffer": bool(args.physics_show_buffer),
            "gui_step_delay": float(args.gui_step_delay),
        }

    try:
        scores = [
            evaluate_result_file(path, pallet_size, thresholds, physics_options=physics_options)
            for path in result_files
        ]
    except RuntimeError as exc:
        parser.exit(status=1, message=f"[ERROR] {exc}\n")

    print_report(scores, thresholds)
    append_benchmark_log(Path(args.benchmark_log), args.label, scores)

    if not args.no_report_json:
        write_eval_report(Path(args.report_json), args.label, scores)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
