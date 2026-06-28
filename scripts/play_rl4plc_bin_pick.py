from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl4plc.checkpoints import CheckpointInfo, find_checkpoints, find_latest_checkpoint
from rl4plc.isaaclab_workflows import find_isaaclab_root, workflow_script
from rl4plc_isaaclab.tasks import DEFAULT_BASE_TASK, RL4PLC_BIN_PICK_TASK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a trained RL4PLC Franka bin-pick policy.")
    parser.add_argument("--isaaclab-root", default=None, help="IsaacLab root directory containing isaaclab.sh.")
    parser.add_argument("--task", default=RL4PLC_BIN_PICK_TASK, help="Registered RL4PLC task name.")
    parser.add_argument("--base-task", default=DEFAULT_BASE_TASK, help="Existing Isaac Lab task used for registration.")
    parser.add_argument("--library", default="rsl_rl", choices=["rsl_rl"], help="RL workflow library.")
    parser.add_argument("--num-envs", type=int, default=1, help="Parallel environments for playback.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. If omitted, workflow default is used.")
    parser.add_argument("--latest", action="store_true", help="Use latest checkpoint from --logs-root.")
    parser.add_argument("--logs-root", default="~/isaac_ws/IsaacLab/logs/rsl_rl", help="Isaac Lab rsl_rl logs root.")
    parser.add_argument(
        "--task-hint",
        default=None,
        help="Substring used when searching checkpoint. Defaults to several RL4PLC/Franka fallbacks.",
    )
    parser.add_argument("--headless", action="store_true", help="Run without GUI.")
    parser.add_argument("--video", action="store_true", help="Record playback video if workflow supports it.")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing it.")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Arguments forwarded after '--'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    root = find_isaaclab_root(args.isaaclab_root, require_exists=not args.dry_run)
    play_script = workflow_script(root, args.library, "play", require_exists=not args.dry_run)
    runner = repo_root / "tools" / "isaaclab_rl4plc_runner.py"
    checkpoint = args.checkpoint
    if args.latest:
        checkpoint_info = _find_latest_with_fallbacks(
            logs_root=args.logs_root,
            hints=[args.task_hint, args.task, args.base_task, "RL4PLC", "Franka", "lift", None],
        )
        checkpoint = str(checkpoint_info.path)
        print(f"Using latest checkpoint: {checkpoint}")
    extra_args = args.extra_args[1:] if args.extra_args[:1] == ["--"] else args.extra_args
    command = [
        str(root / "isaaclab.sh"),
        "-p",
        str(runner),
        "--workflow-script",
        str(play_script),
        "--base-task",
        args.base_task,
        "--",
        "--task",
        args.task,
        "--num_envs",
        str(args.num_envs),
    ]
    if checkpoint:
        command.extend(["--checkpoint", checkpoint])
    if args.headless:
        command.append("--headless")
    if args.video:
        command.append("--video")
    command.extend(extra_args)
    print(f"Command: {' '.join(command)}")
    if args.dry_run:
        return
    raise SystemExit(subprocess.run(command, cwd=repo_root, check=False).returncode)


def _find_latest_with_fallbacks(logs_root: str, hints: list[str | None]) -> CheckpointInfo:
    seen: set[str] = set()
    for hint in hints:
        key = hint or ""
        if key.lower() in seen:
            continue
        seen.add(key.lower())
        try:
            checkpoint = find_latest_checkpoint(logs_root, task_hint=hint)
            print(f"Checkpoint search hint: {hint or '(none)'}")
            return checkpoint
        except FileNotFoundError:
            continue

    available = find_checkpoints(logs_root, task_hint=None)
    preview = "\n".join(f"  - {item.path}" for item in available[:10])
    raise FileNotFoundError(
        f"No checkpoints found under {Path(logs_root).expanduser()}.\n"
        f"First candidates seen:\n{preview if preview else '  (none)'}"
    )


if __name__ == "__main__":
    main()
