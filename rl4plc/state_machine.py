from __future__ import annotations

from dataclasses import dataclass

from rl4plc.types import EpisodeResult, GraspPlan


@dataclass(frozen=True)
class TaskStep:
    name: str
    description: str


PICK_PLACE_STEPS = [
    TaskStep("idle", "robot waits at home pose"),
    TaskStep("detect", "read RGB-D / point-cloud or ground-truth detection"),
    TaskStep("plan", "generate grasp and place poses"),
    TaskStep("pregrasp", "move above the selected workpiece"),
    TaskStep("approach", "approach the grasp pose"),
    TaskStep("attach", "enable suction or gripper constraint"),
    TaskStep("lift", "lift the workpiece"),
    TaskStep("transfer", "move above the target bin"),
    TaskStep("release", "release into target bin"),
    TaskStep("verify", "check final placement"),
]


class PickPlaceStateMachine:
    def __init__(self):
        self.steps = PICK_PLACE_STEPS

    def run_baseline(
        self,
        episode_id: int,
        plan: GraspPlan,
        placed_in_target: bool = True,
        fail_reason: str = "",
    ) -> EpisodeResult:
        return EpisodeResult(
            episode_id=episode_id,
            workpiece_id=plan.workpiece_id,
            workpiece_type=plan.workpiece_type,
            target_bin=plan.target_bin,
            success=bool(placed_in_target),
            fail_reason="" if placed_in_target else fail_reason or "placement_check_failed",
            confidence=plan.confidence,
            cycle_steps=len(self.steps),
        )
