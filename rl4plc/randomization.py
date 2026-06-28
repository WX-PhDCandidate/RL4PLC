from __future__ import annotations

import math
import random
from typing import Any

from rl4plc.config import workpiece_specs
from rl4plc.types import Pose6D, WorkpieceState


class WorkpieceRandomizer:
    def __init__(self, config: dict[str, Any], seed: int | None = None):
        self.config = config
        self.specs = workpiece_specs(config)
        self.rng = random.Random(config.get("seed", 0) if seed is None else seed)
        self.source_bin = config["scene"]["source_bin"]
        self.randomization = config.get("randomization", {})

    def sample_one(self, episode_id: int) -> WorkpieceState:
        spec = self.specs[episode_id % len(self.specs)]
        bx, by, bz = self.source_bin["position"]
        sx, sy, sz = self.source_bin["size"]
        x = bx + self.rng.uniform(-sx * 0.32, sx * 0.32)
        y = by + self.rng.uniform(-sy * 0.28, sy * 0.28)
        half_height = max(spec.size[2], spec.height or spec.size[2]) * 0.5
        stack_lift = self.rng.uniform(0.005, min(0.045, sz * 0.25))
        z = bz + half_height + stack_lift
        tilt_range = float(self.randomization.get("tilt_range", 0.0))
        yaw_range = float(self.randomization.get("yaw_range", math.pi))
        pose = Pose6D(
            x=x,
            y=y,
            z=z,
            rx=self.rng.uniform(-tilt_range, tilt_range),
            ry=self.rng.uniform(-tilt_range, tilt_range),
            rz=self.rng.uniform(-yaw_range, yaw_range),
        )
        return WorkpieceState(id=f"episode_{episode_id:05d}_{spec.type}", spec=spec, pose=pose)
