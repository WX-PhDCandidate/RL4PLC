from __future__ import annotations

from typing import Any

import numpy as np


class FrankaWaypointController:
    """Thin wrapper around Isaac Sim's built-in Franka articulation.

    Franka Panda is used here because it is the default manipulator in many Isaac
    Sim / Isaac Lab manipulation examples and RL environments. This class does
    not define a custom robot model; it only references NVIDIA's packaged USD.
    """

    def __init__(self, world, core: dict[str, Any], robot_config: dict[str, Any]):
        self.world = world
        self.core = core
        self.config = robot_config
        self.robot = None
        self.controller = None
        self.articulation_action = core.get("ArticulationAction")
        self.current_joints = np.array(robot_config["home_joints"], dtype=float)
        self.loaded_usd_path = ""

    def add_to_stage(self) -> None:
        Franka = self.core.get("Franka")
        if Franka is not None:
            self._add_official_franka(Franka)
            return

        usd_path = self._resolve_usd_path()
        if not usd_path:
            print("Franka USD path not found; continuing with TCP marker only.")
            return

        prim_path = self.config.get("prim_path", "/World/Robot")
        try:
            # Fallback only: use the USD as a visual reference. We deliberately do not
            # wrap it with Robot(...) here, because Isaac Sim versions differ in where
            # the Franka USD places its ArticulationRootAPI. Binding the wrong root
            # causes PhysX tensor errors such as "Pattern did not match any rigid bodies".
            self.core["add_reference_to_stage"](usd_path=usd_path, prim_path=prim_path)
            self.loaded_usd_path = usd_path
            self.robot = None
            print(f"Loaded Franka USD as visual reference only: {usd_path}")
        except Exception as exc:
            self.robot = None
            print(f"Franka USD not loaded, continuing with TCP marker only: {exc}")

    def _add_official_franka(self, Franka) -> None:
        prim_path = self.config.get("prim_path", "/World/Franka")
        try:
            self.robot = self.world.scene.add(
                Franka(
                    prim_path=prim_path,
                    name=self.config.get("name", "franka_panda"),
                    position=np.array(self.config.get("base_position", [0.0, 0.0, 0.0]), dtype=float),
                )
            )
            self.loaded_usd_path = "isaacsim.robot.manipulators.examples.franka.Franka"
            print(f"Loaded mainstream RL manipulator with official Franka class at {prim_path}")
        except Exception as exc:
            self.robot = None
            print(f"Official Franka class failed, continuing with TCP marker only: {exc}")

    def initialize_after_reset(self) -> None:
        if self.robot is None:
            return
        try:
            if hasattr(self.robot, "initialize"):
                self.robot.initialize()
            if hasattr(self.robot, "get_articulation_controller"):
                self.controller = self.robot.get_articulation_controller()
            self.apply_joint_positions(self.current_joints)
        except Exception as exc:
            print(f"Franka articulation initialization skipped: {exc}")

    def waypoint(self, phase: str, target_bin: str | None = None) -> np.ndarray:
        waypoints = self.config.get("waypoints", {})
        aliases = {
            "approach": "grasp",
            "attach": "grasp",
            "release": target_bin or "home",
            "transfer": target_bin or "home",
        }
        key = aliases.get(phase, phase)
        values = waypoints.get(key, self.config["home_joints"])
        return np.array(values, dtype=float)

    def apply_joint_positions(self, joints: np.ndarray | list[float]) -> None:
        joints = np.array(joints, dtype=float)
        self.current_joints = joints
        if self.robot is None:
            return
        try:
            if self.controller is not None and self.articulation_action is not None:
                action = self.articulation_action(joint_positions=joints)
                self.controller.apply_action(action)
            elif hasattr(self.robot, "set_joint_positions"):
                self.robot.set_joint_positions(joints)
        except Exception as exc:
            print(f"Franka joint command skipped: {exc}")

    def get_joint_positions(self) -> np.ndarray:
        if self.robot is not None and hasattr(self.robot, "get_joint_positions"):
            try:
                joints = self.robot.get_joint_positions()
                if joints is not None:
                    self.current_joints = np.array(joints, dtype=float)
            except Exception as exc:
                print(f"Franka joint read skipped: {exc}")
        return np.array(self.current_joints, dtype=float)

    def apply_action(self, action) -> None:
        if self.robot is None:
            return
        try:
            if hasattr(self.robot, "apply_action"):
                self.robot.apply_action(action)
            elif self.controller is not None:
                self.controller.apply_action(action)
        except Exception as exc:
            print(f"Franka action command skipped: {exc}")

    @property
    def gripper(self):
        return getattr(self.robot, "gripper", None) if self.robot is not None else None

    def _resolve_usd_path(self) -> str:
        configured = self.config.get("usd_path", "")
        if configured and configured != "AUTO_FRANKA":
            return configured
        assets_root = self.core["get_assets_root_path"]()
        if not assets_root:
            return ""
        return f"{assets_root}/Isaac/Robots/Franka/franka.usd"
