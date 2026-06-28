from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    FrankaCubeLiftEnvCfg,
    FrankaCubeLiftEnvCfg_PLAY,
)

import rl4plc_isaaclab.tasks.bin_picking.mdp as bin_mdp


SOURCE_BIN_CENTER = (0.52, -0.18, 0.0)
TARGET_BIN_CENTER = (0.78, 0.34, 0.0)


def _cuboid_asset(
    prim_path: str,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
    )


def _add_tray(scene, env_root: str, name: str, center: tuple[float, float, float], color: tuple[float, float, float]) -> None:
    cx, cy, cz = center
    sx, sy = 0.72, 0.54
    wall = 0.012
    scene.source_bin_floor = _cuboid_asset(
        f"{env_root}/{name}_Floor",
        (cx, cy, cz + 0.006),
        (sx, sy, wall),
        (0.18, 0.20, 0.22),
    )
    scene.source_bin_front = _cuboid_asset(
        f"{env_root}/{name}_Front",
        (cx, cy - sy * 0.5, cz + 0.0125),
        (sx, wall, 0.025),
        color,
    )
    scene.source_bin_back = _cuboid_asset(
        f"{env_root}/{name}_Back",
        (cx, cy + sy * 0.5, cz + 0.0375),
        (sx, wall, 0.075),
        color,
    )
    scene.source_bin_left = _cuboid_asset(
        f"{env_root}/{name}_Left",
        (cx - sx * 0.5, cy, cz + 0.0225),
        (wall, sy, 0.045),
        color,
    )
    scene.source_bin_right = _cuboid_asset(
        f"{env_root}/{name}_Right",
        (cx + sx * 0.5, cy, cz + 0.0375),
        (wall, sy, 0.075),
        color,
    )


def _add_target_bin(
    scene,
    env_root: str,
    name: str,
    center: tuple[float, float, float],
    color: tuple[float, float, float],
) -> None:
    cx, cy, cz = center
    sx, sy, sz = 0.34, 0.34, 0.14
    wall = 0.012
    scene.target_bin_floor = _cuboid_asset(
        f"{env_root}/{name}_Floor",
        (cx, cy, cz + 0.006),
        (sx, sy, wall),
        color,
    )
    scene.target_bin_front = _cuboid_asset(
        f"{env_root}/{name}_Front",
        (cx, cy - sy * 0.5, cz + sz * 0.5),
        (sx, wall, sz),
        color,
    )
    scene.target_bin_back = _cuboid_asset(
        f"{env_root}/{name}_Back",
        (cx, cy + sy * 0.5, cz + sz * 0.5),
        (sx, wall, sz),
        color,
    )
    scene.target_bin_left = _cuboid_asset(
        f"{env_root}/{name}_Left",
        (cx - sx * 0.5, cy, cz + sz * 0.5),
        (wall, sy, sz),
        color,
    )
    scene.target_bin_right = _cuboid_asset(
        f"{env_root}/{name}_Right",
        (cx + sx * 0.5, cy, cz + sz * 0.5),
        (wall, sy, sz),
        color,
    )


@configclass
class RL4PLCFrankaBinPickEnvCfg(FrankaCubeLiftEnvCfg):
    """Stage-2 RL4PLC bin-picking task.

    This task keeps Isaac Lab's Franka joint-position action space and lift MDP
    plumbing, then changes the scene and target command to train placement into
    a visible target bin.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4096
        self.scene.env_spacing = 2.75

        _add_tray(
            self.scene,
            "{ENV_REGEX_NS}",
            "SourceTray",
            SOURCE_BIN_CENTER,
            (0.32, 0.34, 0.36),
        )
        _add_target_bin(
            self.scene,
            "{ENV_REGEX_NS}",
            "TargetBin_Red",
            TARGET_BIN_CENTER,
            (0.65, 0.15, 0.12),
        )

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[SOURCE_BIN_CENTER[0], SOURCE_BIN_CENTER[1], 0.055]),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.55, 0.55, 0.55),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.20, 0.20),
            "y": (-0.16, 0.16),
            "z": (0.0, 0.0),
        }

        self.commands.object_pose.ranges.pos_x = (TARGET_BIN_CENTER[0], TARGET_BIN_CENTER[0])
        self.commands.object_pose.ranges.pos_y = (TARGET_BIN_CENTER[1], TARGET_BIN_CENTER[1])
        self.commands.object_pose.ranges.pos_z = (0.20, 0.22)
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)
        self.commands.object_pose.ranges.yaw = (0.0, 0.0)

        self.rewards.object_goal_tracking.weight = 24.0
        self.rewards.object_goal_tracking.params["std"] = 0.22
        self.rewards.object_goal_tracking_fine_grained.weight = 8.0
        self.rewards.object_goal_tracking_fine_grained.params["std"] = 0.06
        self.rewards.object_in_target_bin = RewTerm(
            func=bin_mdp.object_in_target_bin,
            weight=20.0,
            params={
                "target_position": (TARGET_BIN_CENTER[0], TARGET_BIN_CENTER[1], TARGET_BIN_CENTER[2]),
                "xy_tolerance": 0.13,
                "z_min": 0.02,
                "asset_cfg": SceneEntityCfg("object"),
            },
        )

        self.episode_length_s = 7.0


@configclass
class RL4PLCFrankaBinPickEnvCfg_PLAY(RL4PLCFrankaBinPickEnvCfg, FrankaCubeLiftEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.75
        self.observations.policy.enable_corruption = False
