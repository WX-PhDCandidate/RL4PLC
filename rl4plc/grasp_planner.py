from __future__ import annotations

from typing import Any

from rl4plc.types import GraspPlan, Pose6D, WorkpieceState


class RuleBasedGraspPlanner:
    def __init__(self, config: dict[str, Any]):
        self.grasp_cfg = config["grasp"]
        self.target_bins = config["scene"]["target_bins"]

    def plan(self, workpiece: WorkpieceState, detection: dict) -> GraspPlan:
        confidence = float(detection.get("Confidence", 0.0))
        threshold = float(self.grasp_cfg.get("confidence_threshold", 0.65))
        if confidence < threshold:
            raise ValueError(f"low confidence: {confidence:.2f} < {threshold:.2f}")

        top_z = workpiece.pose.z + self._half_height(workpiece)
        grasp = Pose6D(
            x=workpiece.pose.x,
            y=workpiece.pose.y,
            z=top_z + float(self.grasp_cfg["grasp_clearance"]),
            rx=3.1415926,
            ry=0.0,
            rz=workpiece.pose.rz,
        )
        pregrasp = grasp.lifted(float(self.grasp_cfg["pregrasp_height"]))
        lift = grasp.lifted(float(self.grasp_cfg["lift_height"]))
        bin_pose = self.target_bins[workpiece.spec.target_bin]["position"]
        place = Pose6D(
            x=float(bin_pose[0]),
            y=float(bin_pose[1]),
            z=float(bin_pose[2]) + float(self.grasp_cfg["place_height"]),
            rx=3.1415926,
            ry=0.0,
            rz=0.0,
        )
        return GraspPlan(
            workpiece_id=workpiece.id,
            workpiece_type=workpiece.spec.type,
            target_bin=workpiece.spec.target_bin,
            pregrasp_pose=pregrasp,
            grasp_pose=grasp,
            lift_pose=lift,
            place_pose=place,
            confidence=confidence,
        )

    @staticmethod
    def _half_height(workpiece: WorkpieceState) -> float:
        if workpiece.spec.shape == "cylinder" and workpiece.spec.height:
            return workpiece.spec.height * 0.5
        return workpiece.spec.size[2] * 0.5

