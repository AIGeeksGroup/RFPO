"""Create the preregistered H65 fixed-tail actor-average checkpoint."""

import argparse
import json
from pathlib import Path

from isaaclab_fpo.checkpoint_averaging import create_actor_average_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = create_actor_average_checkpoint(
        args.source_checkpoint, args.output, args.manifest
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
