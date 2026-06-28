from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe RL4PLC Isaac Lab task registration.")
    parser.add_argument("--base-task", default="Isaac-Lift-Cube-Franka-v0")
    try:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    except Exception:
        parser.add_argument("--headless", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from isaaclab.app import AppLauncher

    if not hasattr(args, "headless"):
        setattr(args, "headless", True)
    simulation_app = AppLauncher(args).app
    try:
        import gymnasium as gym

        from rl4plc_isaaclab import register_rl4plc_tasks

        task_id = register_rl4plc_tasks(base_task=args.base_task)
        spec = gym.spec(task_id)
        print(f"Registered: {task_id}")
        print(f"Entry point: {spec.entry_point}")
        for key, value in sorted((spec.kwargs or {}).items()):
            if key.endswith("_cfg_entry_point"):
                print(f"{key}: {value}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
