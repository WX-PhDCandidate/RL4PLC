from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any

import numpy as np

from rl4plc.config import load_config
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

    def setup(self) -> None:
        self.world.scene.add_default_ground_plane()
        self._add_bins()
        self._add_camera()
        self._add_robot_reference_if_configured()
        self._add_tcp_marker()
        self.world.reset()

    def run(self) -> dict:
        self.setup()
        detections = []
        plans = []
        results = []

        for episode_id in range(self.episodes):
            workpiece = self.randomizer.sample_one(episode_id)
            prim = self._spawn_workpiece(workpiece)
            self._step(self.config["episode"]["settle_steps"])

            detection = self.perception.detect(workpiece)
            plan = self.planner.plan(workpiece, detection)
            placed = self._execute_visual_pick_place(prim, plan)
            result = self.task.run_baseline(episode_id, plan, placed_in_target=placed)

            detections.append(detection)
            plans.append(plan.to_dict())
            results.append(result)
            append_episode_csv(self.run_dir / "episodes.csv", result)
            print(
                f"[Isaac episode {episode_id:03d}] {workpiece.spec.type} -> {plan.target_bin} "
                f"success={result.success}"
            )
            self._remove_prim(prim)

        write_jsonl(self.run_dir / "detections.jsonl", detections)
        write_jsonl(self.run_dir / "grasp_plans.jsonl", plans)
        summary = summarize(results)
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _add_bins(self) -> None:
        FixedCuboid = self.core["FixedCuboid"]
        source = self.config["scene"]["source_bin"]
        self.world.scene.add(
            FixedCuboid(
                prim_path="/World/SourceBin/Floor",
                name="source_bin_floor",
                position=np.array(source["position"]) + np.array([0.0, 0.0, -0.01]),
                scale=np.array([source["size"][0], source["size"][1], 0.02]),
                color=np.array([0.18, 0.20, 0.22]),
            )
        )
        sx, sy, sz = source["size"]
        px, py, pz = source["position"]
        walls = [
            ("front", [px, py - sy / 2, pz + sz / 2], [sx, 0.018, sz]),
            ("back", [px, py + sy / 2, pz + sz / 2], [sx, 0.018, sz]),
            ("left", [px - sx / 2, py, pz + sz / 2], [0.018, sy, sz]),
            ("right", [px + sx / 2, py, pz + sz / 2], [0.018, sy, sz]),
        ]
        for name, pos, scale in walls:
            self.world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/SourceBin/{name}",
                    name=f"source_bin_{name}",
                    position=np.array(pos),
                    scale=np.array(scale),
                    color=np.array([0.32, 0.34, 0.36]),
                )
            )

        colors = {"red": [0.65, 0.15, 0.12], "blue": [0.12, 0.22, 0.70], "green": [0.14, 0.50, 0.25]}
        for bin_name, raw in self.config["scene"]["target_bins"].items():
            self.world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/TargetBins/{bin_name}",
                    name=f"target_bin_{bin_name}",
                    position=np.array(raw["position"]),
                    scale=np.array(raw["size"]),
                    color=np.array(colors.get(bin_name, [0.5, 0.5, 0.5])),
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

    def _add_robot_reference_if_configured(self) -> None:
        robot_cfg = self.config["robot"]
        usd_path = robot_cfg.get("usd_path", "")
        if not usd_path:
            assets_root = self.core["get_assets_root_path"]()
            if assets_root:
                usd_path = f"{assets_root}/Isaac/Robots/Franka/franka.usd"
        if usd_path:
            try:
                self.core["add_reference_to_stage"](usd_path=usd_path, prim_path="/World/Robot")
                print(f"Loaded robot reference: {usd_path}")
            except Exception as exc:
                print(f"Robot USD not loaded, continuing with TCP marker baseline: {exc}")

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

    def _execute_visual_pick_place(self, prim, plan: GraspPlan) -> bool:
        for pose in (plan.pregrasp_pose, plan.grasp_pose, plan.lift_pose, plan.place_pose):
            self._move_marker(pose)
            self._step(35)
        self._set_prim_pose(prim, plan.place_pose.lifted(-0.18))
        self._step(45)
        return self._is_inside_target(plan.place_pose, plan.target_bin)

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

    def _remove_prim(self, prim) -> None:
        if hasattr(self.world.scene, "remove_object") and hasattr(prim, "name"):
            try:
                self.world.scene.remove_object(prim.name)
            except Exception:
                pass

    def _step(self, steps: int) -> None:
        for _ in range(int(steps)):
            self.world.step(render=True)
