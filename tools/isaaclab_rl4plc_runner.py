from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register RL4PLC Isaac Lab tasks, then run an Isaac Lab RL workflow script."
    )
    parser.add_argument("--workflow-script", required=True, help="Path to Isaac Lab train.py or play.py.")
    parser.add_argument("--base-task", default="Isaac-Lift-Cube-Franka-v0", help="Existing Isaac Lab task to reuse.")
    parser.add_argument(
        "workflow_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the Isaac Lab workflow after '--'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from rl4plc_isaaclab import register_rl4plc_tasks

    task_id = register_rl4plc_tasks(base_task=args.base_task)
    workflow_args = args.workflow_args[1:] if args.workflow_args[:1] == ["--"] else args.workflow_args
    script = Path(args.workflow_script).expanduser().resolve()
    if not script.exists():
        raise FileNotFoundError(f"Isaac Lab workflow script not found: {script}")
    workflow_dir = str(script.parent)
    if workflow_dir not in sys.path:
        sys.path.insert(0, workflow_dir)

    print(f"Registered Isaac Lab task: {task_id} (base: {args.base_task})")
    print(f"Running workflow: {script}")
    sys.argv = [str(script), *workflow_args]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
