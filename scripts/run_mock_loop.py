from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl4plc.config import DEFAULT_CONFIG, load_config
from rl4plc.grasp_planner import RuleBasedGraspPlanner
from rl4plc.logging_utils import append_episode_csv, ensure_run_dir, summarize, write_jsonl
from rl4plc.perception import GroundTruthPerception
from rl4plc.randomization import WorkpieceRandomizer
from rl4plc.state_machine import PickPlaceStateMachine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RL4PLC baseline loop without Isaac Sim.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to task JSON config.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run.")
    parser.add_argument("--run-dir", default="runs/mock", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_dir = ensure_run_dir(args.run_dir)
    randomizer = WorkpieceRandomizer(cfg)
    perception = GroundTruthPerception()
    planner = RuleBasedGraspPlanner(cfg)
    task = PickPlaceStateMachine()
    detections = []
    plans = []
    results = []

    for episode_id in range(args.episodes):
        workpiece = randomizer.sample_one(episode_id)
        detection = perception.detect(workpiece)
        plan = planner.plan(workpiece, detection)
        result = task.run_baseline(episode_id, plan, placed_in_target=True)
        detections.append(detection)
        plans.append(plan.to_dict())
        results.append(result)
        append_episode_csv(run_dir / "episodes.csv", result)
        print(
            f"[episode {episode_id:03d}] {workpiece.spec.type} -> {plan.target_bin} "
            f"grasp={plan.grasp_pose.to_list()[:3]} success={result.success}"
        )

    write_jsonl(run_dir / "detections.jsonl", detections)
    write_jsonl(run_dir / "grasp_plans.jsonl", plans)
    summary = summarize(results)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
