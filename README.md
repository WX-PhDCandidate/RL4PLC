# RL4PLC

Isaac Sim based bin-picking research scaffold for industrial robot digital twins and reinforcement-learning grasping.

The first milestone is a small closed loop:

```text
random workpiece in source bin
-> ground-truth perception baseline
-> rule-based grasp pose generation
-> pick/place state sequence
-> success/failure logging
```

This repository intentionally drops the old ROS2/Gazebo path. The current target runtime is Ubuntu + NVIDIA Isaac Sim, with Isaac Lab / RL training planned after the baseline loop is stable.

The baseline robot is **Franka Emika Panda**, loaded from Isaac Sim's packaged USD asset. Franka is used because it is a mainstream manipulation baseline in Isaac Sim / Isaac Lab examples and RL environments, including Isaac Lab lift-cube Franka variants and Isaac Sim's official Franka pick-and-place example.
See [docs/robot_choice.md](docs/robot_choice.md) for the selection rationale and primary references.

## Project Layout

```text
configs/bin_picking_task.json       Task, scene and robot configuration
rl4plc/                            Pure Python task logic
scripts/run_mock_loop.py            Runs the loop without Isaac Sim
scripts/run_isaac_loop.py           Runs the loop inside Isaac Sim Python
docs/isaac_usage_ubuntu.md          Ubuntu Isaac Sim usage notes
tests/                              Lightweight tests for the baseline logic
```

## Quick Local Check

On any machine with Python 3.10+:

```bash
python scripts/run_mock_loop.py --episodes 5
python -m unittest discover -s tests
```

This checks the task logic before running Isaac Sim.

## Isaac Sim Usage

On the Ubuntu server, run with Isaac Sim's bundled Python:

```bash
cd ~/RL4PLC
/path/to/isaac-sim/python.sh scripts/run_isaac_loop.py --episodes 10 --headless
```

For GUI:

```bash
/path/to/isaac-sim/python.sh scripts/run_isaac_loop.py --episodes 3
```

Outputs are written to `runs/`.
The Isaac loop additionally writes `trajectory.jsonl`, which records baseline TCP and object poses by phase.

See [docs/isaac_usage_ubuntu.md](docs/isaac_usage_ubuntu.md) for detailed steps.

## Research Roadmap

1. Build a deterministic Isaac Sim baseline scene.
2. Add random workpiece spawning in a bin.
3. Use ground-truth pose as the first perception baseline.
4. Generate grasp and place poses with rules.
5. Record success metrics over many episodes.
6. Replace ground-truth perception with RGB-D / point-cloud perception.
7. Replace rule-based grasping with Isaac Lab reinforcement learning.
8. Add domain randomization for sim-to-real transfer.
