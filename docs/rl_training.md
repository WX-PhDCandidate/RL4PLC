# RL Training Plan

This project uses open-source Isaac Lab workflows for the first RL grasping milestone instead of building a new RL framework from scratch.

## Baseline Open-Source Task

Start with Isaac Lab's Franka lift-cube environments:

- `Isaac-Lift-Cube-Franka-v0`: Franka joint-position control.
- `Isaac-Lift-Cube-Franka-IK-Rel-v0`: Franka relative IK control.

These tasks are suitable as the first grasping baseline because the goal is already close to our small closed loop: pick an object with Franka and lift or move it to a target.

References:

- Isaac Lab environment list: https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html
- Isaac Lab RL workflow scripts: https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html

## Train

From this repository on the server:

```bash
cd ~/isaac_ws/RL4PLC
python scripts/train_franka_lift.py \
  --isaaclab-root ~/isaac_ws/IsaacLab \
  --task Isaac-Lift-Cube-Franka-v0 \
  --library rsl_rl \
  --num-envs 64 \
  --max-iterations 500 \
  --headless
```

If your Isaac Lab version uses a different internal layout, the wrapper auto-detects common paths such as:

```text
scripts/reinforcement_learning/rsl_rl/train.py
source/standalone/workflows/rsl_rl/train.py
```

If detection still fails, locate the script with:

```bash
find ~/isaac_ws/IsaacLab -path '*rsl_rl/train.py'
```

Relative IK variant:

```bash
python scripts/train_franka_lift.py \
  --isaaclab-root ~/isaac_ws/IsaacLab \
  --ik \
  --library rsl_rl \
  --num-envs 64 \
  --max-iterations 500 \
  --headless
```

Preview command without running:

```bash
python scripts/train_franka_lift.py --isaaclab-root ~/isaac_ws/IsaacLab --ik --dry-run
```

## Play

After training, use the checkpoint path printed by Isaac Lab:

```bash
python scripts/play_franka_lift.py \
  --isaaclab-root ~/isaac_ws/IsaacLab \
  --task Isaac-Lift-Cube-Franka-v0 \
  --library rsl_rl \
  --checkpoint /path/to/model.pt \
  --num-envs 1
```

If you want GUI playback, do not pass `--headless`.

## How This Connects to RL4PLC

Development sequence:

1. Train and play Isaac Lab's open-source Franka lift task.
2. Record what observation/action/reward terms are required for reliable grasping.
3. Replace the single cube with our three workpiece types.
4. Add target-bin reward and wrong-bin penalty.
5. Add fixed RGB-D camera observation or point-cloud-derived object pose.
6. Register the custom bin-picking task as an Isaac Lab extension.

The current `scripts/run_isaac_loop.py` remains the deterministic baseline and visualization/debug tool. The Isaac Lab workflow scripts provide the actual RL training path.
