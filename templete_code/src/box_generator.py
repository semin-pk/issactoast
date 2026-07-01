#!/usr/bin/env python3
"""Generate reproducible palletizing box sequences for tuning and holdout.

The hidden evaluation distribution is not known. Treat the provided uniform
and small-SKU modes as stress distributions and measure both instead of tuning
to only one of them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
from typing import Dict, Iterable, List, Sequence


LENGTH_WIDTH_RANGE = (0.17, 0.32)
HEIGHT_RANGE = (0.13, 0.26)
MASS_RANGE = (0.5, 6.0)

DEFAULT_TUNING_SEEDS = tuple(range(1000, 1050))
DEFAULT_HOLDOUT_SEEDS = tuple(range(9000, 9020))


Box = Dict[str, object]


def rounded(value: float) -> float:
    return round(float(value), 3)


def build_sku_catalog(rng: Random, n_skus: int = 12) -> List[Dict[str, float]]:
    catalog: List[Dict[str, float]] = []
    for _ in range(n_skus):
        length = rounded(rng.uniform(*LENGTH_WIDTH_RANGE))
        width = rounded(rng.uniform(*LENGTH_WIDTH_RANGE))
        height = rounded(rng.uniform(*HEIGHT_RANGE))
        mass = rounded(rng.uniform(*MASS_RANGE))
        catalog.append({
            "length": length,
            "width": width,
            "height": height,
            "mass": mass,
        })
    return catalog


def generate_boxes(seed: int, count: int, mode: str) -> List[Box]:
    """Generate a content-reproducible sequence for one seed."""

    rng = Random(int(seed))
    boxes: List[Box] = []
    catalog = build_sku_catalog(rng) if mode == "sku" else []

    for step in range(int(count)):
        if mode == "uniform":
            length = rounded(rng.uniform(*LENGTH_WIDTH_RANGE))
            width = rounded(rng.uniform(*LENGTH_WIDTH_RANGE))
            height = rounded(rng.uniform(*HEIGHT_RANGE))
            mass = rounded(rng.uniform(*MASS_RANGE))
        elif mode == "sku":
            sku = catalog[rng.randrange(len(catalog))]
            length = float(sku["length"])
            width = float(sku["width"])
            height = float(sku["height"])
            mass = float(sku["mass"])
        else:
            raise ValueError(f"unsupported mode: {mode}")

        boxes.append({
            "step": step,
            "id": step + 1,
            "size": [length, width, height],
            "mass": mass,
        })

    return boxes


def write_sequence(path: Path, boxes: Sequence[Box]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(list(boxes), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_seed_list(text: str) -> List[int]:
    seeds: List[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            seeds.extend(range(start, end + step, step))
        else:
            seeds.append(int(chunk))
    return seeds


def generate_seed_set(
    output_dir: Path,
    seeds: Iterable[int],
    count: int,
    mode: str,
    prefix: str,
) -> None:
    for seed in seeds:
        file_name = f"{prefix}_{mode}_seed_{int(seed)}.json"
        write_sequence(output_dir / file_name, generate_boxes(int(seed), count, mode))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate random palletizing box sequences.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for single-file generation")
    parser.add_argument("--count", type=int, default=120, help="Number of boxes per sequence")
    parser.add_argument("--mode", choices=("uniform", "sku"), default="uniform")
    parser.add_argument("--output", required=True, help="Output JSON file or directory")
    parser.add_argument(
        "--seed-set",
        choices=("single", "tuning", "holdout", "both"),
        default="single",
        help="Generate one file or a named seed set directory",
    )
    parser.add_argument(
        "--tuning-seeds",
        default=",".join(str(seed) for seed in DEFAULT_TUNING_SEEDS),
        help="Comma/range list such as 1000-1049,2001",
    )
    parser.add_argument(
        "--holdout-seeds",
        default=",".join(str(seed) for seed in DEFAULT_HOLDOUT_SEEDS),
        help="Comma/range list such as 9000-9019,9901",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = Path(args.output)

    if args.seed_set == "single":
        write_sequence(output, generate_boxes(args.seed, args.count, args.mode))
        return 0

    if args.seed_set in {"tuning", "both"}:
        generate_seed_set(
            output / "tuning",
            parse_seed_list(args.tuning_seeds),
            args.count,
            args.mode,
            "tuning",
        )

    if args.seed_set in {"holdout", "both"}:
        generate_seed_set(
            output / "holdout",
            parse_seed_list(args.holdout_seeds),
            args.count,
            args.mode,
            "holdout",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
