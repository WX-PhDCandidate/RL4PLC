from __future__ import annotations

import unittest

from rl4plc.config import load_config
from rl4plc.grasp_planner import RuleBasedGraspPlanner
from rl4plc.perception import GroundTruthPerception
from rl4plc.randomization import WorkpieceRandomizer
from rl4plc.state_machine import PickPlaceStateMachine


class BaselineLoopTest(unittest.TestCase):
    def test_single_episode_generates_successful_plan(self) -> None:
        cfg = load_config()
        workpiece = WorkpieceRandomizer(cfg, seed=7).sample_one(0)
        detection = GroundTruthPerception().detect(workpiece)
        plan = RuleBasedGraspPlanner(cfg).plan(workpiece, detection)
        result = PickPlaceStateMachine().run_baseline(0, plan)

        self.assertEqual(detection["DetectState"], "OK")
        self.assertEqual(plan.target_bin, workpiece.spec.target_bin)
        self.assertGreater(plan.pregrasp_pose.z, plan.grasp_pose.z)
        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()

