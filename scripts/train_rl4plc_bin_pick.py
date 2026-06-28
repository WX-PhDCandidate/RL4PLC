from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl4plc.isaaclab_workflows import find_isaaclab_root, workflow_script
from rl4plc_isaaclab.tasks import DEFAULT_BASE_TASK, RL4PLC_BIN_PICK_TASK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL4PLC Franka bin-pick PPO task in Isaac Lab.")
    parser.add_argument("--isaaclab-root", default=None, help="IsaacLab root directory containing isaaclab.sh.")
    parser.add_argument("--task", default=RL4PLC_BIN_PICK_TASK, help="Registered RL4PLC task name.")
    parser.add_argument("--base-task", default=DEFAULT_BASE_TASK, help="Existing Isaac Lab task used for stage-1 env.")
    parser.add_argument("--library", default="rsl_rl", choices=["rsl_rl"], help="RL workflow library.")
    parser.add_argument("--num-envs", type=int, default=64, help="Parallel training environments.")
    parser.add_argument("--max-iterations", type=int, default=500, help="Training iterations.")
    parser.add_argument("--headless", action="store_true", help="Run without GUI.")
    parser.add_argument("--video", action="store_true", help="Record training video if workflow supports it.")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing it.")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Arguments forwarded after '--'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    root = find_isaaclab_root(args.isaaclab_root, require_exists=not args.dry_run)
    train_script = workflow_script(root, args.library, "train", require_exists=not args.dry_run)
    runner = repo_root / "tools" / "isaaclab_rl4plc_runner.py"
    extra_args = args.extra_args[1:] if args.extra_args[:1] == ["--"] else args.extra_args
    command = [
        str(root / "isaaclab.sh"),
        "-p",
        str(runner),
        "--workflow-script",
        str(train_script),
        "--base-task",
        args.base_task,
        "--",
        "--task",
        args.task,
        "--num_envs",
        str(args.num_envs),
    ]
    if args.headless:
        command.append("--headless")
    if args.video:
        command.append("--video")
    if args.max_iterations is not None:
        command.extend(["--max_iterations", str(args.max_iterations)])
    command.extend(extra_args)
    print(f"Command: {' '.join(command)}")
    if args.dry_run:
        return
    raise SystemExit(subprocess.run(command, cwd=repo_root, check=False).returncode)


if __name__ == "__main__":
    main()
