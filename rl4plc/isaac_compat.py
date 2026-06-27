from __future__ import annotations


def import_simulation_app():
    try:
        from isaacsim import SimulationApp  # type: ignore

        return SimulationApp
    except Exception:
        from omni.isaac.kit import SimulationApp  # type: ignore

        return SimulationApp


def import_isaac_core():
    try:
        from isaacsim.core.api import World  # type: ignore
        from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder, FixedCuboid, VisualCuboid  # type: ignore
        from isaacsim.core.utils.stage import add_reference_to_stage  # type: ignore
        from isaacsim.core.utils.nucleus import get_assets_root_path  # type: ignore

        return {
            "World": World,
            "DynamicCuboid": DynamicCuboid,
            "DynamicCylinder": DynamicCylinder,
            "FixedCuboid": FixedCuboid,
            "VisualCuboid": VisualCuboid,
            "add_reference_to_stage": add_reference_to_stage,
            "get_assets_root_path": get_assets_root_path,
        }
    except Exception:
        from omni.isaac.core import World  # type: ignore
        from omni.isaac.core.objects import DynamicCuboid, DynamicCylinder, FixedCuboid, VisualCuboid  # type: ignore
        from omni.isaac.core.utils.nucleus import get_assets_root_path  # type: ignore
        from omni.isaac.core.utils.stage import add_reference_to_stage  # type: ignore

        return {
            "World": World,
            "DynamicCuboid": DynamicCuboid,
            "DynamicCylinder": DynamicCylinder,
            "FixedCuboid": FixedCuboid,
            "VisualCuboid": VisualCuboid,
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

