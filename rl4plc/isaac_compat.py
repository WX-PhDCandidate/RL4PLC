from __future__ import annotations

import importlib


def import_simulation_app():
    try:
        from isaacsim import SimulationApp  # type: ignore

        return SimulationApp
    except Exception:
        from omni.isaac.kit import SimulationApp  # type: ignore

        return SimulationApp


def import_isaac_core():
    Franka = None
    try:
        from isaacsim.core.api import World  # type: ignore
        from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder, FixedCuboid, VisualCuboid  # type: ignore
        from isaacsim.core.api.robots import Robot  # type: ignore
        from isaacsim.core.utils.types import ArticulationAction  # type: ignore
        from isaacsim.core.utils.stage import add_reference_to_stage  # type: ignore
        from isaacsim.core.utils.nucleus import get_assets_root_path  # type: ignore
        try:
            from isaacsim.robot.manipulators.examples.franka import Franka  # type: ignore
        except Exception:
            Franka = None

        return {
            "World": World,
            "DynamicCuboid": DynamicCuboid,
            "DynamicCylinder": DynamicCylinder,
            "FixedCuboid": FixedCuboid,
            "VisualCuboid": VisualCuboid,
            "Robot": Robot,
            "Franka": Franka,
            "ArticulationAction": ArticulationAction,
            "add_reference_to_stage": add_reference_to_stage,
            "get_assets_root_path": get_assets_root_path,
        }
    except Exception:
        from omni.isaac.core import World  # type: ignore
        from omni.isaac.core.objects import DynamicCuboid, DynamicCylinder, FixedCuboid, VisualCuboid  # type: ignore
        from omni.isaac.core.robots import Robot  # type: ignore
        from omni.isaac.core.utils.types import ArticulationAction  # type: ignore
        from omni.isaac.core.utils.nucleus import get_assets_root_path  # type: ignore
        from omni.isaac.core.utils.stage import add_reference_to_stage  # type: ignore
        try:
            from omni.isaac.franka import Franka  # type: ignore
        except Exception:
            Franka = None

        return {
            "World": World,
            "DynamicCuboid": DynamicCuboid,
            "DynamicCylinder": DynamicCylinder,
            "FixedCuboid": FixedCuboid,
            "VisualCuboid": VisualCuboid,
            "Robot": Robot,
            "Franka": Franka,
            "ArticulationAction": ArticulationAction,
            "add_reference_to_stage": add_reference_to_stage,
            "get_assets_root_path": get_assets_root_path,
        }


def import_camera():
    try:
        from isaacsim.sensors.camera import Camera  # type: ignore

        return Camera
    except Exception:
        from omni.isaac.sensor import Camera  # type: ignore

        return Camera


def import_pick_place_controller():
    candidates = [
        "isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller",
        "isaacsim.robot.manipulators.controllers.pick_place_controller",
        "omni.isaac.franka.controllers.pick_place_controller",
        "omni.isaac.franka.controllers",
    ]
    errors: list[str] = []
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, "PickPlaceController")
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
    raise ImportError("PickPlaceController import failed:\n" + "\n".join(errors))
