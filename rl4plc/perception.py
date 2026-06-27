from __future__ import annotations

from rl4plc.types import WorkpieceState


class GroundTruthPerception:
    """First perception baseline: use simulator ground truth as if it were detector output."""

    def __init__(self, confidence: float = 1.0):
        self.confidence = float(confidence)

    def detect(self, workpiece: WorkpieceState) -> dict:
        return workpiece.to_detection(confidence=self.confidence, frame="world")

