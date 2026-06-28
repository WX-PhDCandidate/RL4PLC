from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg


def object_in_target_bin(
    env,
    target_position: tuple[float, float, float] = (0.78, 0.34, 0.08),
    xy_tolerance: float = 0.12,
    z_min: float = 0.03,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward object placement inside the target bin footprint."""

    obj: RigidObject = env.scene[asset_cfg.name]
    pos = obj.data.root_pos_w[:, :3]
    target = torch.tensor(target_position, device=pos.device, dtype=pos.dtype)
    xy_distance = torch.linalg.norm(pos[:, :2] - target[:2], dim=1)
    inside_xy = xy_distance < xy_tolerance
    above_floor = pos[:, 2] > target[2] + z_min
    return torch.where(inside_xy & above_floor, torch.ones_like(xy_distance), torch.zeros_like(xy_distance))
