from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl4plc.isaaclab_workflows import DEFAULT_IK_TASK, DEFAULT_TASK, build_play_command, find_isaaclab_root, run_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a trained Isaac Lab Franka grasping policy.")
    parser.add_argument("--isaaclab-root", default=None, help="IsaacLab root directory containing isaaclab.sh.")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Isaac Lab task name.")
    parser.add_argument("--ik", action="store_true", help=f"Use relative IK task: {DEFAULT_IK_TASK}.")
    parser.add_argument("--library", default="rsl_rl", choices=["rsl_rl", "skrl"], help="RL workflow library.")
    parser.add_argument("--num-envs", type=int, default=1, help="Parallel environments for playback.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. If omitted, Isaac Lab uses workflow default.")
    parser.add_argument("--headless", action="store_true", help="Run without GUI.")
    parser.add_argument("--video", action="store_true", help="Record playback video if workflow supports it.")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing it.")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Arguments forwarded after '--'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = find_isaaclab_root(args.isaaclab_root, require_exists=not args.dry_run)
    task = DEFAULT_IK_TASK if args.ik else args.task
    extra_args = args.extra_args[1:] if args.extra_args[:1] == ["--"] else args.extra_args
    command = build_play_command(
        isaaclab_root=root,
        task=task,
        library=args.library,
        num_envs=args.num_envs,
        checkpoint=args.checkpoint,
        headless=args.headless,
        video=args.video,
        validate_paths=not args.dry_run,
        extra_args=extra_args,
    )
    raise SystemExit(run_command(command, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
