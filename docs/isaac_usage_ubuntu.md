# Ubuntu Isaac Sim Usage

## 1. Clone

```bash
cd ~
git clone https://github.com/WX-PhDCandidate/RL4PLC.git
cd RL4PLC
```

If the repository already exists:

```bash
cd ~/RL4PLC
git pull --ff-only
```

## 2. Find Isaac Sim Python

Common locations:

```bash
~/isaacsim/python.sh
~/.local/share/ov/pkg/isaac-sim-*/python.sh
/opt/isaac-sim/python.sh
```

Check one of them:

```bash
~/isaacsim/python.sh -c "from isaacsim import SimulationApp; print('isaac ok')"
```

For older Isaac Sim builds, this may also work:

```bash
~/isaacsim/python.sh -c "from omni.isaac.kit import SimulationApp; print('isaac ok')"
```

## 3. Run Logic Without Isaac

This verifies the closed-loop task code:

```bash
python3 scripts/run_mock_loop.py --episodes 5
```

## 4. Run in Isaac Sim

Headless server run:

```bash
~/isaacsim/python.sh scripts/run_isaac_loop.py --episodes 10 --headless
```

GUI run on Ubuntu desktop:

```bash
~/isaacsim/python.sh scripts/run_isaac_loop.py --episodes 3
```

Keep the Isaac Sim window open after the baseline episodes finish:

```bash
~/isaacsim/python.sh scripts/run_isaac_loop.py --episodes 3 --keep-open
```

Or keep it open for a fixed time:

```bash
~/isaacsim/python.sh scripts/run_isaac_loop.py --episodes 3 --post-run-seconds 60
```

If your Isaac Sim path is different, replace `~/isaacsim/python.sh`.

## 5. Run the First Franka Bin-Pick Demo

After the PPO lift task can train and play back, run the project-level
grasp-and-drop demo:

```bash
~/isaacsim/python.sh scripts/run_franka_bin_pick.py --episodes 3 --keep-open
```

Headless run:

```bash
~/isaacsim/python.sh scripts/run_franka_bin_pick.py --episodes 10 --headless
```

This script creates the source bin, three target bins, random workpieces and the
official Franka Panda. It then uses ground-truth pose detection plus a rule-based
grasp plan, and tries to execute the action with Isaac Sim's official Franka
`PickPlaceController`.

If Isaac Sim cannot import or construct `PickPlaceController`, the script prints
the reason and falls back to the visual attached-object baseline unless
`--no-fallback-visual` is passed.

## 6. Outputs

The scripts write:

```text
runs/isaac/episodes.csv
runs/isaac/detections.jsonl
runs/isaac/grasp_plans.jsonl
runs/isaac/trajectory.jsonl
runs/isaac/summary.json
runs/franka_bin_pick/franka_pickplace_trajectory.jsonl
```

## 7. Robot USD

By default, the script uses Isaac Sim's official Franka Panda manipulator class when available. This avoids guessing the articulation root path inside the USD. If that class is unavailable, it falls back to loading the Franka USD as a visual reference only. To override the visual USD fallback, edit:

```text
configs/bin_picking_task.json
robot.usd_path
```

Example:

```json
  "model": "franka_panda",
  "usd_path": "/home/user/assets/robots/my_robot.usd"
```

The basic Isaac loop drives the Franka articulation through preset joint waypoints while keeping the TCP marker and workpiece trajectory explicit. The Franka bin-pick demo uses Isaac Sim's official PickPlace controller when available. The next development step is to replace the rule-based target pose with an Isaac Lab policy trained for correct-bin placement.

## 8. Known Isaac Sim Runtime Note

If Isaac Sim reports an error similar to:

```text
Simulation view object is invalidated and cannot be used again to call getVelocities
```

update to the latest repository version:

```bash
cd ~/isaac_ws/RL4PLC
git pull --ff-only
```

The baseline avoids deleting dynamic objects during simulation and parks completed workpieces outside the workcell instead. This is intentional because runtime scene deletion can invalidate PhysX tensor views in some Isaac Sim builds.
