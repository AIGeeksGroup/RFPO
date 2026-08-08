#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.policy_interpolation import create_interpolated_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a linear interpolation of two compatible policy checkpoints."
    )
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--finetuned-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()

    manifest = create_interpolated_checkpoint(
        args.anchor_checkpoint,
        args.finetuned_checkpoint,
        args.output_checkpoint,
        args.alpha,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
