#!/usr/bin/env python3
"""Compare relocate robot contact geometry against the ViViDex benchmark robot."""

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _repo_root()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "hand_imitation"))

from hand_imitation.env.rl_env.relocate_env import get_relocate_robot_link_config
from hand_imitation.utils.common_robot_utils import generate_robot_hand_info
from hand_imitation.utils.ycb_object_utils import YCB_SIZE


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-name", default="mustard_bottle")
    parser.add_argument("--object-scale", type=float, default=float(os.environ.get("VIVIDEX_YCB_OBJECT_SCALE", 1.0)))
    parser.add_argument(
        "--allegro-urdf",
        type=Path,
        default=REPO_ROOT / "assets/robot/ur5_description/ur5_allegro.urdf",
    )
    parser.add_argument(
        "--rm75-urdf",
        type=Path,
        default=REPO_ROOT / "assets/robot/rm75_inspire_right/urdf/rm75_inspire_hand_right_nocyl.urdf",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _geometry_desc(collision: ET.Element) -> dict:
    origin = collision.find("origin")
    geometry = collision.find("geometry")
    desc = {
        "origin_xyz": (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"),
        "origin_rpy": (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"),
        "kind": "unknown",
    }
    if geometry is None:
        return desc
    for kind in ("sphere", "box", "cylinder", "mesh"):
        elem = geometry.find(kind)
        if elem is None:
            continue
        desc["kind"] = kind
        desc.update(elem.attrib)
        return desc
    return desc


def _urdf_link_summaries(urdf_path: Path) -> dict[str, dict]:
    root = ET.parse(urdf_path).getroot()
    summaries = {}
    for link in root.findall("link"):
        collisions = [_geometry_desc(collision) for collision in link.findall("collision")]
        summaries[link.get("name", "")] = {
            "collision_count": len(collisions),
            "collisions": collisions,
        }
    return summaries


def _summarize_robot(robot_name: str, urdf_path: Path) -> dict:
    finger_tip_names, contact_link_names, contact_ids, _ = get_relocate_robot_link_config(robot_name)
    palm_name = generate_robot_hand_info()[robot_name].palm_name
    contact_link_names_with_palm = contact_link_names + [palm_name]
    link_summaries = _urdf_link_summaries(urdf_path)
    contact_groups: dict[str, list[str]] = {}
    for link_name, group_id in zip(contact_link_names_with_palm, contact_ids.tolist()):
        contact_groups.setdefault(str(group_id), []).append(link_name)

    missing_links = [name for name in contact_link_names_with_palm if name not in link_summaries]
    contact_without_collision = [
        name
        for name in contact_link_names_with_palm
        if name in link_summaries and link_summaries[name]["collision_count"] == 0
    ]
    group_collision_counts = {
        group_id: sum(link_summaries.get(name, {}).get("collision_count", 0) for name in names)
        for group_id, names in contact_groups.items()
    }
    tip_collisions = {
        name: link_summaries.get(name, {"collision_count": 0, "collisions": []})
        for name in finger_tip_names
    }
    sphere_tip_radii = {
        name: [
            collision.get("radius")
            for collision in summary["collisions"]
            if collision.get("kind") == "sphere"
        ]
        for name, summary in tip_collisions.items()
    }
    return {
        "robot_name": robot_name,
        "urdf": str(urdf_path),
        "finger_tip_count": len(finger_tip_names),
        "finger_tip_names": finger_tip_names,
        "contact_link_count_including_palm": len(contact_link_names_with_palm),
        "contact_group_count": len(contact_groups),
        "contact_groups": contact_groups,
        "group_collision_counts": group_collision_counts,
        "missing_contact_links": missing_links,
        "contact_links_without_collision": contact_without_collision,
        "tip_collisions": tip_collisions,
        "sphere_tip_radii": sphere_tip_radii,
    }


def main() -> None:
    args = _parse_args()
    object_size = tuple(float(x) * float(args.object_scale) for x in YCB_SIZE[args.object_name])
    summary = {
        "object": {
            "name": args.object_name,
            "scale": float(args.object_scale),
            "size_xyz": object_size,
            "half_height": object_size[2] / 2,
        },
        "friction_defaults": {
            "allegro_robot_static_dynamic": [1.5, 1.0],
            "rm75_robot_static_dynamic": [
                float(os.environ.get("VIVIDEX_RM75_STATIC_FRICTION", 1.5)),
                float(os.environ.get("VIVIDEX_RM75_DYNAMIC_FRICTION", 1.0)),
            ],
            "ycb_static_dynamic": [
                float(os.environ.get("VIVIDEX_YCB_STATIC_FRICTION", 1.5)),
                float(os.environ.get("VIVIDEX_YCB_DYNAMIC_FRICTION", 1.0)),
            ],
            "patch_radius": 0.04,
            "min_patch_radius": 0.02,
        },
        "robots": {
            "allegro_hand_ur5": _summarize_robot("allegro_hand_ur5", args.allegro_urdf),
            "rm75_inspire_right": _summarize_robot("rm75_inspire_right", args.rm75_urdf),
        },
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
