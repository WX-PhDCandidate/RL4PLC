from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rl4plc.isaac_compat import import_pick_place_controller
from rl4plc.isaac_scene import IsaacBinPickingLoop
from rl4plc.logging_utils import append_episode_csv, summarize, write_jsonl
from rl4plc.types import GraspPlan, Pose6D, WorkpieceState


class FrankaControllerBinPickLoop(IsaacBinPickingLoop):
    """Isaac Sim Franka pick-and-place loop for the first physical bin-pick demo.

    The class keeps the existing perception/planning/state-machine path, but
    executes each grasp with Isaac Sim's official Franka PickPlaceController
    whenever that controller is available in the installed Isaac Sim version.
    """

    def __init__(
        self,
        config_path: str | Path,
        run_dir: str | Path,
        episodes: int = 3,
        fallback_visual: bool = True,
    ):
        super().__init__(config_path=config_path, run_dir=run_dir, episodes=episodes)
        self.fallback_visual = bool(fallback_visual)
        self.pick_place_controller_cls = None

    def run(self) -> dict:
        self.setup()
        if not self._controller_ready():
            if not self.fallback_visual:
                raise RuntimeError("Official Franka PickPlaceController is unavailable.")
            print("Falling back to visual attached-object baseline.")
            return self._run_visual_episodes()

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
            placed, trajectory = self._execute_franka_pick_place(prim, plan, workpiece, episode_id)
            result = self.task.run_baseline(episode_id, plan, placed_in_target=placed)

            detections.append(detection)
            plans.append(plan.to_dict())
            results.append(result)
            trajectories.extend(trajectory)
            append_episode_csv(self.run_dir / "episodes.csv", result)
            print(
                f"[Franka bin-pick {episode_id:03d}] {workpiece.spec.type} -> {plan.target_bin} "
                f"success={result.success}"
            )
            self._park_prim(prim, episode_id)

        write_jsonl(self.run_dir / "detections.jsonl", detections)
        write_jsonl(self.run_dir / "grasp_plans.jsonl", plans)
        write_jsonl(self.run_dir / "franka_pickplace_trajectory.jsonl", trajectories)
        summary = summarize(results)
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _controller_ready(self) -> bool:
        try:
            self.pick_place_controller_cls = import_pick_place_controller()
        except ImportError as exc:
            print(str(exc))
            return False
        if self.robot_controller is None or self.robot_controller.robot is None:
            print("Official Franka articulation was not loaded.")
            return False
        if self.robot_controller.gripper is None:
            print("Official Franka gripper handle was not found.")
            return False
        return True

    def _run_visual_episodes(self) -> dict:
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
                f"[visual fallback {episode_id:03d}] {workpiece.spec.type} -> {plan.target_bin} "
                f"success={result.success}"
            )
            self._park_prim(prim, episode_id)
        write_jsonl(self.run_dir / "detections.jsonl", detections)
        write_jsonl(self.run_dir / "grasp_plans.jsonl", plans)
        write_jsonl(self.run_dir / "trajectory.jsonl", trajectories)
        summary = summarize(results)
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _execute_franka_pick_place(
        self,
        prim,
        plan: GraspPlan,
        workpiece: WorkpieceState,
        episode_id: int,
    ) -> tuple[bool, list[dict[str, Any]]]:
        controller = self._new_pick_place_controller(episode_id)
        self._reset_pick_place_controller(controller)

        picking_position = np.array(workpiece.pose.position, dtype=float)
        placing_position = np.array(plan.place_pose.position, dtype=float)
        max_steps = int(self.config["episode"].get("max_motion_steps", 240))
        settle_steps = int(self.config["episode"].get("release_steps", 45))
        trajectory: list[dict[str, Any]] = []

        done = False
        for step_index in range(max_steps):
            action = self._forward_pick_place(controller, picking_position, placing_position)
            self.robot_controller.apply_action(action)
            self._record_controller_step(trajectory, episode_id, step_index, prim, plan)
            self._step(1)
            done = self._is_controller_done(controller)
            if done:
                break

        for _ in range(settle_steps):
            self._step(1)

        object_pose = self._read_prim_pose(prim)
        placed = self._is_inside_target(object_pose, plan.target_bin)
        used_assisted_fallback = False
        if not placed and self.config["episode"].get("assisted_fallback_on_failure", True):
            print(
                f"Official PickPlace did not place {workpiece.id} in {plan.target_bin}; "
                "switching to assisted attachment fallback."
            )
            used_assisted_fallback = True
            assisted_placed, assisted_trajectory = self._execute_visual_pick_place(
                prim=prim,
                plan=plan,
                workpiece=workpiece,
                episode_id=episode_id,
            )
            for item in assisted_trajectory:
                item["assisted_fallback"] = True
            trajectory.extend(assisted_trajectory)
            object_pose = self._read_prim_pose(prim)
            placed = assisted_placed

        trajectory.append(
            {
                "episode_id": episode_id,
                "phase": "settled",
                "step": max_steps if not done else step_index,
                "controller_done": done,
                "object_pose": object_pose.to_list(),
                "target_bin": plan.target_bin,
                "placed_in_target": placed,
                "assisted_fallback": used_assisted_fallback,
            }
        )
        return placed, trajectory

    def _new_pick_place_controller(self, episode_id: int):
        assert self.pick_place_controller_cls is not None
        assert self.robot_controller is not None
        base_kwargs = {
            "name": f"franka_pick_place_controller_{episode_id}",
            "gripper": self.robot_controller.gripper,
            "robot_articulation": self.robot_controller.robot,
        }
        attempts = [
            base_kwargs,
            {
                **base_kwargs,
                "events_dt": [0.008, 0.005, 1.0, 0.1, 0.05, 0.05, 0.0025, 1.0, 0.008, 0.08],
            },
            {**base_kwargs, "end_effector_initial_height": 0.6},
        ]
        errors = []
        for kwargs in attempts:
            try:
                return self.pick_place_controller_cls(**kwargs)
            except TypeError as exc:
                errors.append(str(exc))
        raise RuntimeError("PickPlaceController could not be constructed: " + " | ".join(errors))

    @staticmethod
    def _reset_pick_place_controller(controller) -> None:
        if hasattr(controller, "reset"):
            controller.reset()

    def _forward_pick_place(self, controller, picking_position: np.ndarray, placing_position: np.ndarray):
        assert self.robot_controller is not None
        current_joint_positions = self.robot_controller.get_joint_positions()
        try:
            return controller.forward(
                picking_position=picking_position,
                placing_position=placing_position,
                current_joint_positions=current_joint_positions,
            )
        except TypeError:
            return controller.forward(
                picking_position,
                placing_position,
                current_joint_positions,
            )

    @staticmethod
    def _is_controller_done(controller) -> bool:
        return bool(controller.is_done()) if hasattr(controller, "is_done") else False

    def _record_controller_step(
        self,
        trajectory: list[dict[str, Any]],
        episode_id: int,
        step_index: int,
        prim,
        plan: GraspPlan,
    ) -> None:
        if step_index % 10 != 0:
            return
        trajectory.append(
            {
                "episode_id": episode_id,
                "phase": "franka_pick_place",
                "step": step_index,
                "object_pose": self._read_prim_pose(prim).to_list(),
                "place_pose": plan.place_pose.to_list(),
                "target_bin": plan.target_bin,
            }
        )

    @staticmethod
    def _read_prim_pose(prim) -> Pose6D:
        if hasattr(prim, "get_world_pose"):
            position, _orientation = prim.get_world_pose()
            return Pose6D(float(position[0]), float(position[1]), float(position[2]))
        return Pose6D(0.0, 0.0, 0.0)
