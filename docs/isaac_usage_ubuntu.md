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

## 5. Outputs

The scripts write:

```text
runs/isaac/episodes.csv
runs/isaac/detections.jsonl
runs/isaac/grasp_plans.jsonl
runs/isaac/summary.json
```

## 6. Robot USD

By default, the script tries to load Isaac's Franka USD as a visual robot reference. For your own robot, edit:

```text
configs/bin_picking_task.json
robot.usd_path
```

Example:

```json
"usd_path": "/home/user/assets/robots/my_robot.usd"
```

The current v0.1 baseline uses a TCP visual marker and ground-truth perception. The next development step is to connect this interface to a real articulation controller or Isaac Lab RL environment.

## 7. Known Isaac Sim Runtime Note

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
