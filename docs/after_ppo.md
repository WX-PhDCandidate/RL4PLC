# After PPO Runs

Once PPO training completes, do three things:

## 1. Find the Latest Checkpoint

```bash
cd ~/isaac_ws/RL4PLC
python scripts/find_latest_checkpoint.py \
  --logs-root ~/isaac_ws/IsaacLab/logs/rsl_rl \
  --task-hint Isaac-Lift-Cube-Franka-v0
```

List candidates:

```bash
python scripts/find_latest_checkpoint.py \
  --logs-root ~/isaac_ws/IsaacLab/logs/rsl_rl \
  --task-hint Franka \
  --list
```

## 2. Play the Latest Policy

```bash
python scripts/play_franka_lift.py \
  --isaaclab-root ~/isaac_ws/IsaacLab \
  --task Isaac-Lift-Cube-Franka-v0 \
  --latest \
  --task-hint Isaac-Lift-Cube-Franka-v0 \
  --num-envs 1
```

If the task hint does not match your log folder name, use a broader hint:

```bash
python scripts/play_franka_lift.py \
  --isaaclab-root ~/isaac_ws/IsaacLab \
  --task Isaac-Lift-Cube-Franka-v0 \
  --latest \
  --task-hint Franka \
  --num-envs 1
```

## 3. Move Toward Bin Picking

The first custom-task specification is in:

```text
configs/bin_picking_rl_spec.json
```

The intended migration is:

1. Keep Franka and PPO workflow unchanged.
2. Replace cube object with one selected workpiece.
3. Replace lift reward with correct-bin placement reward.
4. Add three workpiece types and target-bin mapping.
5. Add RGB-D / point-cloud observation after pose-based RL works.

