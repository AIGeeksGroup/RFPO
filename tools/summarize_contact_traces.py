#!/usr/bin/env python3
"""Summarize object-hand contact trace JSON files."""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Contact JSON files or glob patterns.")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _angle_deg(unit: list[float] | None) -> float | None:
    if unit is None:
        return None
    if len(unit) < 2:
        return None
    return float(math.degrees(math.atan2(float(unit[1]), float(unit[0]))))


def _expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return sorted(set(paths))


def _summarize_file(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_group: dict[str, list[dict]] = defaultdict(list)
    by_link: dict[str, list[dict]] = defaultdict(list)
    contact_steps = 0
    for row in data:
        details = row.get("contact_details", [])
        if details:
            contact_steps += 1
        for detail in details:
            by_group[str(detail.get("group"))].append(detail)
            by_link[str(detail.get("link"))].append(detail)

    def summarize_items(items: list[dict]) -> dict:
        impulse = np.asarray([float(item.get("impulse_abs_sum", 0.0)) for item in items], dtype=np.float32)
        radial_z = np.asarray(
            [
                float(item["object_radial_z"])
                for item in items
                if item.get("object_radial_z") is not None
            ],
            dtype=np.float32,
        )
        angles = np.asarray(
            [
                _angle_deg(item.get("object_radial_xy_unit"))
                for item in items
                if _angle_deg(item.get("object_radial_xy_unit")) is not None
            ],
            dtype=np.float32,
        )
        return {
            "num_contact_events": len(items),
            "impulse_abs_sum_total": float(np.sum(impulse)) if impulse.size else 0.0,
            "impulse_abs_sum_mean": float(np.mean(impulse)) if impulse.size else 0.0,
            "radial_z_min_mean_max": [
                float(np.min(radial_z)) if radial_z.size else 0.0,
                float(np.mean(radial_z)) if radial_z.size else 0.0,
                float(np.max(radial_z)) if radial_z.size else 0.0,
            ],
            "radial_angle_deg_min_mean_max": [
                float(np.min(angles)) if angles.size else 0.0,
                float(np.mean(angles)) if angles.size else 0.0,
                float(np.max(angles)) if angles.size else 0.0,
            ],
        }

    return {
        "path": str(path),
        "steps_with_any_contact": contact_steps,
        "groups": {group: summarize_items(items) for group, items in sorted(by_group.items())},
        "links": {link: summarize_items(items) for link, items in sorted(by_link.items())},
    }


def main() -> None:
    args = _parse_args()
    paths = _expand_paths(args.paths)
    summaries = [_summarize_file(path) for path in paths if path.exists()]
    text = json.dumps({"files": summaries}, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
