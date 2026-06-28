from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl4plc.isaaclab_workflows import build_python_command, find_isaaclab_root, run_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List Isaac Lab tasks and available agent config entry points.")
    parser.add_argument("--isaaclab-root", default=None, help="IsaacLab root directory containing isaaclab.sh.")
    parser.add_argument("--library", default="rsl_rl", help="Filter by library config key, e.g. rsl_rl, skrl, robomimic.")
    parser.add_argument("--contains", default="Franka", help="Filter task ids by substring.")
    parser.add_argument("--show-all", action="store_true", help="Show tasks even if the selected library is missing.")
    parser.add_argument("--gui", action="store_true", help="Open Isaac Sim GUI while probing. Default is headless.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = find_isaaclab_root(args.isaaclab_root)
    helper = Path(__file__).resolve().parents[1] / "tools" / "isaaclab_task_probe.py"
    command = build_python_command(
        isaaclab_root=root,
        script=helper,
        script_args=[
            "--library",
            args.library,
            "--contains",
            args.contains,
            *(["--show-all"] if args.show_all else []),
            *([] if args.gui else ["--headless"]),
        ],
    )
    raise SystemExit(run_command(command))


if __name__ == "__main__":
    main()
