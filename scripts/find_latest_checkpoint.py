from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl4plc.checkpoints import find_checkpoints, find_latest_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find latest Isaac Lab training checkpoint.")
    parser.add_argument("--logs-root", default="~/isaac_ws/IsaacLab/logs/rsl_rl", help="Isaac Lab RL log root.")
    parser.add_argument("--task-hint", default="Franka", help="Substring filter for checkpoint path.")
    parser.add_argument("--list", action="store_true", help="List all matching checkpoints.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        checkpoints = find_checkpoints(args.logs_root, task_hint=args.task_hint)
        for item in checkpoints:
            print(f"{item.path} step={item.step}")
        if not checkpoints:
            print("No checkpoints found.")
        return

    checkpoint = find_latest_checkpoint(args.logs_root, task_hint=args.task_hint)
    print(checkpoint.path)


if __name__ == "__main__":
    main()

