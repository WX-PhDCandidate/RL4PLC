from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rl4plc.types import WorkpieceSpec


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "bin_picking_task.json"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _vector3(values: list[float], fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    if len(values) != 3:
        return fallback
    return (float(values[0]), float(values[1]), float(values[2]))


def workpiece_specs(config: dict[str, Any]) -> list[WorkpieceSpec]:
    specs: list[WorkpieceSpec] = []
    for raw in config["workpieces"]:
        size = raw.get("size", [raw.get("radius", 0.03) * 2, raw.get("radius", 0.03) * 2, raw.get("height", 0.05)])
        specs.append(
            WorkpieceSpec(
                type=raw["type"],
                target_bin=raw["target_bin"],
                shape=raw["shape"],
                size=_vector3(size, (0.06, 0.06, 0.05)),
                mass=float(raw.get("mass", 0.12)),
                color=_vector3(raw.get("color", [0.7, 0.7, 0.7]), (0.7, 0.7, 0.7)),
                radius=float(raw["radius"]) if "radius" in raw else None,
                height=float(raw["height"]) if "height" in raw else None,
            )
        )
    return specs

