from __future__ import annotations

import argparse
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
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
