# After PPO Runs

Once PPO training completes, do three things:

## 1. Find the Latest Checkpoint

```bash
cd ~/isaac_ws/RL4PLC
python scripts/find_latest_checkpoint.py \
  --logs-root ~/isaac_ws/IsaacLab/logs/rsl_rl \
  --task-hint RL4PLC-Franka-BinPick-v0
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
python scripts/play_rl4plc_bin_pick.py \
  --isaaclab-root ~/isaac_ws/IsaacLab \
  --latest \
  --num-envs 1
```

If the task hint does not match your log folder name, use a broader hint:

```bash
python scripts/play_rl4plc_bin_pick.py \
  --isaaclab-root ~/isaac_ws/IsaacLab \
  --latest \
  --task-hint Franka \
  --num-envs 1
```

## 3. Move Toward Bin Picking

The first custom-task specification is in:

```text
configs/bin_picking_rl_spec.json
```

`RL4PLC-Franka-BinPick-v0` now registers as the Stage-2 single-object source-tray
to target-bin task. The intended migration after this is:

1. Keep Franka, PPO workflow and the project task id unchanged.
2. Tune source-tray placement reward until the object reliably reaches the red bin.
3. Replace cube object with one selected workpiece.
4. Add three workpiece types and target-bin mapping.
5. Add wrong-bin penalty and multi-bin curriculum.
6. Add RGB-D / point-cloud observation after pose-based RL works.
