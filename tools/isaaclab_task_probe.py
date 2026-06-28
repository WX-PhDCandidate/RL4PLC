from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Isaac Lab gym task registry.")
    parser.add_argument("--library", default="rsl_rl")
    parser.add_argument("--contains", default="Franka")
    parser.add_argument("--show-all", action="store_true")
    try:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    except Exception:
        parser.add_argument("--headless", action="store_true", default=True)
    return parser.parse_args()


def _start_app(args: argparse.Namespace):
    try:
        from isaaclab.app import AppLauncher
    except Exception as exc:
        raise RuntimeError(f"isaaclab.app.AppLauncher import failed: {exc}") from exc
    if not hasattr(args, "headless"):
        setattr(args, "headless", True)
    app_launcher = AppLauncher(args)
    return app_launcher.app


def _load_registry():
    try:
        import gymnasium as gym
    except Exception as exc:
        raise RuntimeError(f"gymnasium import failed: {exc}") from exc

    try:
        import isaaclab_tasks  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"isaaclab_tasks import failed: {exc}") from exc
    return gym.registry


def main() -> None:
    args = parse_args()
    simulation_app = _start_app(args)
    try:
        registry = _load_registry()
        cfg_key = f"{args.library}_cfg_entry_point"
        contains = args.contains.lower()
        rows = []

        for task_id, spec in sorted(registry.items()):
            if contains and contains not in task_id.lower():
                continue
            kwargs = spec.kwargs or {}
            keys = sorted(key for key in kwargs.keys() if key.endswith("_cfg_entry_point"))
            has_selected = cfg_key in kwargs
            if has_selected or args.show_all:
                rows.append((task_id, has_selected, keys))

        if not rows:
            print(f"No tasks matched contains={args.contains!r}, library={args.library!r}.")
            print("Try: --show-all or --library skrl")
            return

        print(f"Task registry probe: contains={args.contains!r}, library={args.library!r}")
        print(f"Required key: {cfg_key}")
        for task_id, has_selected, keys in rows:
            mark = "OK" if has_selected else "--"
            print(f"{mark} {task_id}")
            print(f"   cfg keys: {', '.join(keys) if keys else '(none)'}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
