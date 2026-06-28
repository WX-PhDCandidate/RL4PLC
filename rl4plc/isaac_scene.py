from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any

import numpy as np

from rl4plc.config import load_config
from rl4plc.franka_controller import FrankaWaypointController
from rl4plc.grasp_planner import RuleBasedGraspPlanner
from rl4plc.isaac_compat import import_camera, import_isaac_core
from rl4plc.logging_utils import append_episode_csv, ensure_run_dir, summarize, write_jsonl
from rl4plc.perception import GroundTruthPerception
from rl4plc.randomization import WorkpieceRandomizer
from rl4plc.state_machine import PickPlaceStateMachine
from rl4plc.types import GraspPlan, Pose6D, WorkpieceState


class IsaacBinPickingLoop:
    def __init__(self, config_path: str | Path, run_dir: str | Path, episodes: int = 5):
        self.config = load_config(config_path)
        self.run_dir = ensure_run_dir(run_dir)
        self.episodes = int(episodes)
        self.core = import_isaac_core()
        self.camera_cls = import_camera()
        self.world = self.core["World"](stage_units_in_meters=1.0)
        self.randomizer = WorkpieceRandomizer(self.config)
        self.perception = GroundTruthPerception()
        self.planner = RuleBasedGraspPlanner(self.config)
        self.task = PickPlaceStateMachine()
        self.tcp_marker = None
        self.camera = None
        self.robot_controller: FrankaWaypointController | None = None

    def setup(self) -> None:
        self.world.scene.add_default_ground_plane()
        self._add_bins()
        self._add_camera()
        self._add_robot()
        self._add_tcp_marker()
        self.world.reset()
        if self.robot_controller is not None:
            self.robot_controller.initialize_after_reset()

    def run(self) -> dict:
        self.setup()
        detections = []
        plans = []
        results = []
        trajectories = []

        for episode_id in range(self.episodes):
            workpiece = self.randomizer.sample_one(episode_id)
            prim = self._spawn_workpiece(workpiece)
            self._step(self.config["episode"]["settle_steps"])

            detection = self.perception.detect(workpiece)
            plan = self.planner.plan(workpiece, detection)
            placed, trajectory = self._execute_visual_pick_place(prim, plan, workpiece, episode_id)
            result = self.task.run_baseline(episode_id, plan, placed_in_target=placed)

            detections.append(detection)
            plans.append(plan.to_dict())
            results.append(result)
            trajectories.extend(trajectory)
            append_episode_csv(self.run_dir / "episodes.csv", result)
            print(
                f"[Isaac episode {episode_id:03d}] {workpiece.spec.type} -> {plan.target_bin} "
                f"success={result.success}"
            )
            self._park_prim(prim, episode_id)

        write_jsonl(self.run_dir / "detections.jsonl", detections)
        write_jsonl(self.run_dir / "grasp_plans.jsonl", plans)
        write_jsonl(self.run_dir / "trajectory.jsonl", trajectories)
        summary = summarize(results)
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _add_bins(self) -> None:
        source = self.config["scene"]["source_bin"]
        self._add_open_bin("/World/SourceBin", "source_bin", source, [0.18, 0.20, 0.22], [0.32, 0.34, 0.36])

        colors = {"red": [0.65, 0.15, 0.12], "blue": [0.12, 0.22, 0.70], "green": [0.14, 0.50, 0.25]}
        for bin_name, raw in self.config["scene"]["target_bins"].items():
            color = colors.get(bin_name, [0.5, 0.5, 0.5])
            wall_color = [min(1.0, value + 0.12) for value in color]
            self._add_open_bin(f"/World/TargetBins/{bin_name}", f"target_bin_{bin_name}", raw, color, wall_color)

    def _add_open_bin(
        self,
        prim_root: str,
        name_root: str,
        raw: dict,
        floor_color: list[float],
        wall_color: list[float],
    ) -> None:
        FixedCuboid = self.core["FixedCuboid"]
        sx, sy, sz = raw["size"]
        px, py, pz = raw["position"]
        wall = float(raw.get("wall_thickness", 0.018))
        self.world.scene.add(
            FixedCuboid(
                prim_path=f"{prim_root}/Floor",
                name=f"{name_root}_floor",
                position=np.array([px, py, pz - wall * 0.5]),
                scale=np.array([sx, sy, wall]),
                color=np.array(floor_color),
            )
        )
        walls = [
            ("front", [px, py - sy / 2, pz + sz / 2], [sx, wall, sz]),
            ("back", [px, py + sy / 2, pz + sz / 2], [sx, wall, sz]),
            ("left", [px - sx / 2, py, pz + sz / 2], [wall, sy, sz]),
            ("right", [px + sx / 2, py, pz + sz / 2], [wall, sy, sz]),
        ]
        for wall_name, pos, scale in walls:
            self.world.scene.add(
                FixedCuboid(
                    prim_path=f"{prim_root}/{wall_name}",
                    name=f"{name_root}_{wall_name}",
                    position=np.array(pos),
                    scale=np.array(scale),
                    color=np.array(wall_color),
                )
            )

    def _add_camera(self) -> None:
        camera_cfg = self.config["scene"]["camera"]
        self.camera = self.camera_cls(
            prim_path="/World/FixedRGBCamera",
            position=np.array(camera_cfg["position"]),
            frequency=20,
            resolution=tuple(camera_cfg["resolution"]),
        )
        self.world.scene.add(self.camera)
        self.camera.initialize()
        if hasattr(self.camera, "set_focal_length"):
            self.camera.set_focal_length(float(camera_cfg.get("focal_length", 24.0)))

    def _add_robot(self) -> None:
        robot_cfg = self.config["robot"]
        if robot_cfg.get("model") == "franka_panda" and robot_cfg.get("use_articulation", True):
            self.robot_controller = FrankaWaypointController(self.world, self.core, robot_cfg)
            self.robot_controller.add_to_stage()
            return
        print("No mainstream robot articulation configured; continuing with TCP marker baseline.")

    def _add_tcp_marker(self) -> None:
        if not self.config["robot"].get("use_visual_tcp_marker", True):
            return
        VisualCuboid = self.core["VisualCuboid"]
        self.tcp_marker = self.world.scene.add(
            VisualCuboid(
                prim_path="/World/TCPMarker",
                name="tcp_marker",
                position=np.array([0.0, 0.0, 0.45]),
                scale=np.array([0.035, 0.035, 0.035]),
                color=np.array([1.0, 0.82, 0.18]),
            )
        )

    def _spawn_workpiece(self, workpiece: WorkpieceState):
        if workpiece.spec.shape == "cylinder":
            DynamicCylinder = self.core["DynamicCylinder"]
            prim = DynamicCylinder(
                prim_path=f"/World/Workpieces/{workpiece.id}",
                name=workpiece.id,
                position=np.array(workpiece.pose.position),
                radius=workpiece.spec.radius or workpiece.spec.size[0] / 2.0,
                height=workpiece.spec.height or workpiece.spec.size[2],
                mass=workpiece.spec.mass,
                color=np.array(workpiece.spec.color),
            )
        else:
            DynamicCuboid = self.core["DynamicCuboid"]
            prim = DynamicCuboid(
                prim_path=f"/World/Workpieces/{workpiece.id}",
                name=workpiece.id,
                position=np.array(workpiece.pose.position),
                scale=np.array(workpiece.spec.size),
                mass=workpiece.spec.mass,
                color=np.array(workpiece.spec.color),
            )
        return self.world.scene.add(prim)

    def _execute_visual_pick_place(
        self,
        prim,
        plan: GraspPlan,
        workpiece: WorkpieceState,
        episode_id: int,
    ) -> tuple[bool, list[dict]]:
        move_steps = int(self.config["episode"].get("move_steps", 45))
        attach_steps = int(self.config["episode"].get("attach_steps", 18))
        release_steps = int(self.config["episode"].get("release_steps", 45))
        grasp_gap = self._attached_center_gap(workpiece)
        release_pose = self._release_pose(plan, workpiece)
        trajectory: list[dict] = []

        current = Pose6D(0.0, 0.0, 0.45)
        current = self._move_segment(current, plan.pregrasp_pose, move_steps, "pregrasp", episode_id, trajectory)
        current = self._move_segment(current, plan.grasp_pose, move_steps, "approach", episode_id, trajectory)

        for step_index in range(attach_steps):
            object_pose = self._object_pose_from_tcp(current, grasp_gap)
            self._set_prim_pose(prim, object_pose)
            self._record_trajectory(trajectory, episode_id, "attach", step_index, current, object_pose)
            self._step(1)

        current = self._move_segment(
            current,
            plan.lift_pose,
            move_steps,
            "lift",
            episode_id,
            trajectory,
            target_bin=plan.target_bin,
            prim=prim,
            grasp_gap=grasp_gap,
        )
        current = self._move_segment(
            current,
            plan.place_pose,
            move_steps,
            "transfer",
            episode_id,
            trajectory,
            target_bin=plan.target_bin,
            prim=prim,
            grasp_gap=grasp_gap,
        )

        self._apply_robot_waypoint("release", plan.target_bin)
        self._set_prim_pose(prim, release_pose)
        self._move_marker(plan.place_pose)
        for step_index in range(release_steps):
            self._record_trajectory(trajectory, episode_id, "release", step_index, plan.place_pose, release_pose)
            self._step(1)

        return self._is_inside_target(release_pose, plan.target_bin), trajectory

    def _move_segment(
        self,
        start: Pose6D,
        end: Pose6D,
        steps: int,
        phase: str,
        episode_id: int,
        trajectory: list[dict],
        target_bin: str | None = None,
        prim=None,
        grasp_gap: float | None = None,
    ) -> Pose6D:
        steps = max(1, int(steps))
        joint_start = self._current_robot_joints()
        joint_end = self._robot_waypoint(phase, target_bin)
        for step_index in range(steps):
            t = (step_index + 1) / steps
            eased = self._smoothstep(t)
            pose = self._interpolate_pose(start, end, eased)
            self._move_marker(pose)
            self._apply_interpolated_robot_joints(joint_start, joint_end, eased)
            object_pose = None
            if prim is not None and grasp_gap is not None:
                object_pose = self._object_pose_from_tcp(pose, grasp_gap)
                self._set_prim_pose(prim, object_pose)
            self._record_trajectory(trajectory, episode_id, phase, step_index, pose, object_pose)
            self._step(1)
        return end

    def _current_robot_joints(self) -> np.ndarray | None:
        if self.robot_controller is None:
            return None
        return np.array(self.robot_controller.current_joints, dtype=float)

    def _robot_waypoint(self, phase: str, target_bin: str | None) -> np.ndarray | None:
        if self.robot_controller is None:
            return None
        return self.robot_controller.waypoint(phase, target_bin)

    def _apply_interpolated_robot_joints(
        self,
        start: np.ndarray | None,
        end: np.ndarray | None,
        t: float,
    ) -> None:
        if self.robot_controller is None or start is None or end is None:
            return
        self.robot_controller.apply_joint_positions(start + (end - start) * t)

    def _apply_robot_waypoint(self, phase: str, target_bin: str | None = None) -> None:
        if self.robot_controller is None:
            return
        self.robot_controller.apply_joint_positions(self.robot_controller.waypoint(phase, target_bin))

    @staticmethod
    def _interpolate_pose(start: Pose6D, end: Pose6D, t: float) -> Pose6D:
        return Pose6D(
            x=start.x + (end.x - start.x) * t,
            y=start.y + (end.y - start.y) * t,
            z=start.z + (end.z - start.z) * t,
            rx=start.rx + (end.rx - start.rx) * t,
            ry=start.ry + (end.ry - start.ry) * t,
            rz=start.rz + (end.rz - start.rz) * t,
        )

    @staticmethod
    def _smoothstep(t: float) -> float:
        t = min(1.0, max(0.0, t))
        return t * t * (3.0 - 2.0 * t)

    def _attached_center_gap(self, workpiece: WorkpieceState) -> float:
        half_height = workpiece.spec.height * 0.5 if workpiece.spec.height else workpiece.spec.size[2] * 0.5
        return half_height + float(self.config["grasp"].get("grasp_clearance", 0.018))

    @staticmethod
    def _object_pose_from_tcp(tcp_pose: Pose6D, grasp_gap: float) -> Pose6D:
        return Pose6D(
            x=tcp_pose.x,
            y=tcp_pose.y,
            z=tcp_pose.z - grasp_gap,
            rx=tcp_pose.rx,
            ry=tcp_pose.ry,
            rz=tcp_pose.rz,
        )

    def _release_pose(self, plan: GraspPlan, workpiece: WorkpieceState) -> Pose6D:
        raw = self.config["scene"]["target_bins"][plan.target_bin]
        bin_z = float(raw["position"][2])
        bin_height = float(raw["size"][2])
        half_height = workpiece.spec.height * 0.5 if workpiece.spec.height else workpiece.spec.size[2] * 0.5
        return Pose6D(
            x=float(raw["position"][0]),
            y=float(raw["position"][1]),
            z=bin_z + bin_height * 0.5 + half_height + 0.01,
            rx=0.0,
            ry=0.0,
            rz=plan.place_pose.rz,
        )

    @staticmethod
    def _record_trajectory(
        trajectory: list[dict],
        episode_id: int,
        phase: str,
        step_index: int,
        tcp_pose: Pose6D,
        object_pose: Pose6D | None,
    ) -> None:
        if step_index % 5 != 0:
            return
        trajectory.append(
            {
                "episode_id": episode_id,
                "phase": phase,
                "step": step_index,
                "tcp_pose": tcp_pose.to_list(),
                "object_pose": object_pose.to_list() if object_pose else None,
            }
        )

    def _move_marker(self, pose: Pose6D) -> None:
        if self.tcp_marker is not None and hasattr(self.tcp_marker, "set_world_pose"):
            self.tcp_marker.set_world_pose(position=np.array(pose.position))

    def _set_prim_pose(self, prim, pose: Pose6D) -> None:
        if hasattr(prim, "set_world_pose"):
            prim.set_world_pose(position=np.array(pose.position))

    def _is_inside_target(self, pose: Pose6D, target_bin: str) -> bool:
        raw = self.config["scene"]["target_bins"][target_bin]
        bx, by, _ = raw["position"]
        sx, sy, _ = raw["size"]
        return math.fabs(pose.x - bx) <= sx / 2 and math.fabs(pose.y - by) <= sy / 2

    def _park_prim(self, prim, episode_id: int) -> None:
        # Do not remove dynamic objects while the simulator is running. Some Isaac Sim /
        # PhysX versions invalidate tensor simulation views after scene removal, which can
        # crash later calls such as getVelocities. Parking keeps the baseline stable.
        self._set_prim_pose(prim, Pose6D(x=-4.0, y=-4.0 - episode_id * 0.08, z=0.35))
        self._step(2)

    def _step(self, steps: int) -> None:
        for _ in range(int(steps)):
            self.world.step(render=True)
