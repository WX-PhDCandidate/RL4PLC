"""Project-level Isaac Lab tasks."""

from __future__ import annotations

from typing import Any


RL4PLC_BIN_PICK_TASK = "RL4PLC-Franka-BinPick-v0"
DEFAULT_BASE_TASK = "Isaac-Lift-Cube-Franka-v0"


def register_rl4plc_tasks(base_task: str = DEFAULT_BASE_TASK) -> str:
    import gymnasium as gym

    if _is_registered(gym, RL4PLC_BIN_PICK_TASK):
        return RL4PLC_BIN_PICK_TASK

    entry_point, kwargs = _stage2_registration(gym, base_task)
    kwargs["rl4plc_task_note"] = (
        "Stage-2 bin-picking task. It uses the RL4PLC source tray, target bin "
        "and placement reward while reusing the official Franka PPO runner config."
    )
    gym.register(
        id=RL4PLC_BIN_PICK_TASK,
        entry_point=entry_point,
        kwargs=kwargs,
        disable_env_checker=True,
    )
    return RL4PLC_BIN_PICK_TASK


def _stage2_registration(gym_module, base_task: str) -> tuple[str, dict[str, Any]]:
    kwargs = _base_agent_kwargs(gym_module, base_task)
    kwargs["env_cfg_entry_point"] = (
        "rl4plc_isaaclab.tasks.bin_picking.joint_pos_env_cfg:RL4PLCFrankaBinPickEnvCfg"
    )
    return "isaaclab.envs:ManagerBasedRLEnv", kwargs


def _base_agent_kwargs(gym_module, base_task: str) -> dict[str, Any]:
    try:
        base_spec = gym_module.spec(base_task)
        kwargs = dict(base_spec.kwargs or {})
        return {key: value for key, value in kwargs.items() if key.endswith("_cfg_entry_point")}
    except Exception:
        pass

    if base_task != DEFAULT_BASE_TASK:
        raise RuntimeError(
            f"Base task {base_task!r} is not already registered. Use {DEFAULT_BASE_TASK!r} "
            "or import/register the base task before calling register_rl4plc_tasks()."
        )

    module = "isaaclab_tasks.manager_based.manipulation.lift.config.franka"
    agents = f"{module}.agents"
    return {
        "rsl_rl_cfg_entry_point": f"{agents}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents}:sb3_ppo_cfg.yaml",
    }


def _is_registered(gym_module, task_id: str) -> bool:
    try:
        gym_module.spec(task_id)
        return True
    except Exception:
        return False
