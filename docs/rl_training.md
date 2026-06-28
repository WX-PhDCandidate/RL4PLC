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

### RL4PLC task

Train the project task id:

```bash
cd ~/isaac_ws/RL4PLC
python scripts/train_rl4plc_bin_pick.py \
  --isaaclab-root ~/isaac_ws/IsaacLab \
  --num-envs 64 \
  --max-iterations 500 \
  --headless
```

Short smoke test:

```bash
python scripts/train_rl4plc_bin_pick.py \
  --isaaclab-root ~/isaac_ws/IsaacLab \
  --num-envs 8 \
  --max-iterations 20
```

The wrapper registers `RL4PLC-Franka-BinPick-v0` in the same Python process
before launching Isaac Lab's official rsl_rl workflow. Stage 2 uses a project
env config with the RL4PLC source tray, red target bin, randomized object start
inside the tray, target command above the bin, and an object-in-bin reward. The
PPO runner config is still reused from Isaac Lab's Franka lift task to keep the
training workflow compatible with your installation.

Preview command without running:

```bash
python scripts/train_rl4plc_bin_pick.py \
  --isaaclab-root ~/isaac_ws/IsaacLab \
  --num-envs 8 \
  --max-iterations 20 \
  --dry-run
```

Probe project task registration:

```bash
~/isaac_ws/IsaacLab/isaaclab.sh -p tools/rl4plc_task_probe.py --headless
```

### Built-in task check

First inspect which Franka tasks in your Isaac Lab installation actually provide an RL config for the selected library:

```bash
python scripts/list_isaaclab_tasks.py \
  --isaaclab-root ~/isaac_ws/IsaacLab \
  --contains Franka \
  --library rsl_rl \
  --show-all
```

The probe starts Isaac Lab headlessly before importing `isaaclab_tasks`, because modules such as `pxr` are provided by the Isaac/Kit runtime in some installations.

Only tasks marked `OK` have `rsl_rl_cfg_entry_point` and can be trained through `scripts/reinforcement_learning/rsl_rl/train.py`.

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

If the relative IK task reports only `robomimic: BC` and no `rsl_rl_cfg_entry_point`, do not use `--ik` for PPO training in that Isaac Lab version. Use a task shown as `OK` by `list_isaaclab_tasks.py`, for example the joint-position Franka lift task if available:

```bash
python scripts/train_franka_lift.py \
  --isaaclab-root ~/isaac_ws/IsaacLab \
  --task Isaac-Lift-Cube-Franka-v0 \
  --library rsl_rl \
  --num-envs 8 \
  --max-iterations 20
```

Preview command without running:

```bash
python scripts/train_franka_lift.py --isaaclab-root ~/isaac_ws/IsaacLab --ik --dry-run
```

## Play

Replay the latest RL4PLC checkpoint:

```bash
cd ~/isaac_ws/RL4PLC
python scripts/play_rl4plc_bin_pick.py \
  --isaaclab-root ~/isaac_ws/IsaacLab \
  --latest \
  --num-envs 1
```

Replay a specific checkpoint:

```bash
python scripts/play_rl4plc_bin_pick.py \
  --isaaclab-root ~/isaac_ws/IsaacLab \
  --checkpoint /path/to/model.pt \
  --num-envs 1
```

If you want GUI playback, do not pass `--headless`.

### Built-in task playback

After training, use the checkpoint path printed by Isaac Lab:

```bash
python scripts/play_franka_lift.py \
  --isaaclab-root ~/isaac_ws/IsaacLab \
  --task Isaac-Lift-Cube-Franka-v0 \
  --library rsl_rl \
  --checkpoint /path/to/model.pt \
  --num-envs 1
```

## How This Connects to RL4PLC

Development sequence:

1. Register and train `RL4PLC-Franka-BinPick-v0`.
2. Train the Stage-2 single-object source-tray to target-bin task.
3. Record what observation/action/reward terms are required for reliable placement.
4. Replace the single cube with our three workpiece types.
5. Add multi-target-bin mapping and wrong-bin penalty.
6. Add fixed RGB-D camera observation or point-cloud-derived object pose.
7. Add domain randomization for sim-to-real transfer.

The current `scripts/run_isaac_loop.py` remains the deterministic baseline and visualization/debug tool. The Isaac Lab workflow scripts provide the actual RL training path.
