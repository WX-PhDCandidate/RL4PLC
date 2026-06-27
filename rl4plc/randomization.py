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

    def sample_one(self, episode_id: int) -> WorkpieceState:
        spec = self.specs[episode_id % len(self.specs)]
        bx, by, bz = self.source_bin["position"]
        sx, sy, sz = self.source_bin["size"]
        x = bx + self.rng.uniform(-sx * 0.32, sx * 0.32)
        y = by + self.rng.uniform(-sy * 0.28, sy * 0.28)
        z = bz + sz + max(spec.size[2], spec.height or spec.size[2]) * 0.5 + self.rng.uniform(0.005, 0.025)
        pose = Pose6D(
            x=x,
            y=y,
            z=z,
            rx=self.rng.uniform(-0.18, 0.18),
            ry=self.rng.uniform(-0.18, 0.18),
            rz=self.rng.uniform(-math.pi, math.pi),
        )
        return WorkpieceState(id=f"episode_{episode_id:05d}_{spec.type}", spec=spec, pose=pose)

