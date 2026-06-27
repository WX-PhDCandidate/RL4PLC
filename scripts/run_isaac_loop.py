from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl4plc.config import DEFAULT_CONFIG
from rl4plc.isaac_compat import import_simulation_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RL4PLC baseline loop in Isaac Sim.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to task JSON config.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run.")
    parser.add_argument("--run-dir", default="runs/isaac", help="Output directory.")
    parser.add_argument("--headless", action="store_true", help="Run without GUI.")
    parser.add_argument("--keep-open", action="store_true", help="Keep Isaac Sim open after the episodes finish.")
    parser.add_argument(
        "--post-run-seconds",
        type=float,
        default=0.0,
        help="Keep rendering for this many seconds after the episodes finish.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    SimulationApp = import_simulation_app()
    simulation_app = SimulationApp({"headless": bool(args.headless)})
    try:
        from rl4plc.isaac_scene import IsaacBinPickingLoop

        loop = IsaacBinPickingLoop(config_path=args.config, run_dir=args.run_dir, episodes=args.episodes)
        summary = loop.run()
        print(summary)
        if args.keep_open:
            print("Episodes finished. Isaac Sim will stay open; press Ctrl+C in the terminal to exit.")
            while simulation_app.is_running():
                loop.world.step(render=True)
        elif args.post_run_seconds > 0:
            print(f"Episodes finished. Keeping Isaac Sim open for {args.post_run_seconds:.1f} seconds.")
            end_time = time.monotonic() + args.post_run_seconds
            while time.monotonic() < end_time and simulation_app.is_running():
                loop.world.step(render=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
