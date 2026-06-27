from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


Vector3 = tuple[float, float, float]
RPY = tuple[float, float, float]


@dataclass(frozen=True)
class Pose6D:
    x: float
    y: float
    z: float
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0

    @property
    def position(self) -> Vector3:
        return (self.x, self.y, self.z)

    @property
    def rpy(self) -> RPY:
        return (self.rx, self.ry, self.rz)

    def lifted(self, dz: float) -> "Pose6D":
        return Pose6D(self.x, self.y, self.z + dz, self.rx, self.ry, self.rz)

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z, self.rx, self.ry, self.rz]


@dataclass(frozen=True)
class WorkpieceSpec:
    type: str
    target_bin: str
    shape: str
    size: Vector3
    mass: float
    color: Vector3
    radius: float | None = None
    height: float | None = None


@dataclass(frozen=True)
class WorkpieceState:
    id: str
    spec: WorkpieceSpec
    pose: Pose6D

    def to_detection(self, confidence: float = 1.0, frame: str = "world") -> dict[str, Any]:
        return {
            "WorkpieceID": self.id,
            "WorkpieceType": self.spec.type,
            "TargetBin": self.spec.target_bin,
            "WorkpiecePose": self.pose.to_list(),
            "Confidence": confidence,
            "DetectState": "OK",
            "Frame": frame,
        }


@dataclass(frozen=True)
class GraspPlan:
    workpiece_id: str
    workpiece_type: str
    target_bin: str
    pregrasp_pose: Pose6D
    grasp_pose: Pose6D
    lift_pose: Pose6D
    place_pose: Pose6D
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("pregrasp_pose", "grasp_pose", "lift_pose", "place_pose"):
            data[key] = getattr(self, key).to_list()
        return data


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: int
    workpiece_id: str
    workpiece_type: str
    target_bin: str
    success: bool
    fail_reason: str
    confidence: float
    cycle_steps: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

