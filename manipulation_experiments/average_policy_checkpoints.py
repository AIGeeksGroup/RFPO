#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.policy_averaging import create_averaged_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Uniformly average compatible policy checkpoints."
    )
    parser.add_argument(
        "--source-checkpoint", type=Path, action="append", required=True
    )
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    args = parser.parse_args()

    manifest = create_averaged_checkpoint(
        args.source_checkpoint, args.output_checkpoint
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
