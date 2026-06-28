from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


MODEL_RE = re.compile(r"model_(\d+)\.pt$")


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    step: int
    modified_time: float


def _checkpoint_step(path: Path) -> int:
    match = MODEL_RE.search(path.name)
    if match:
        return int(match.group(1))
    if path.name == "model.pt":
        return -1
    return -2


def find_checkpoints(log_root: str | Path, task_hint: str | None = None) -> list[CheckpointInfo]:
    root = Path(log_root).expanduser().resolve()
    if not root.exists():
        return []

    candidates = []
    for pattern in ("**/model_*.pt", "**/model.pt", "**/*.pth"):
        candidates.extend(root.glob(pattern))

    rows: list[CheckpointInfo] = []
    hint = task_hint.lower() if task_hint else ""
    for path in candidates:
        if not path.is_file():
            continue
        if hint and hint not in str(path).lower():
            continue
        rows.append(CheckpointInfo(path=path, step=_checkpoint_step(path), modified_time=path.stat().st_mtime))

    rows.sort(key=lambda item: (item.step, item.modified_time), reverse=True)
    return rows


def find_latest_checkpoint(log_root: str | Path, task_hint: str | None = None) -> CheckpointInfo:
    checkpoints = find_checkpoints(log_root, task_hint=task_hint)
    if not checkpoints:
        hint = f" matching {task_hint!r}" if task_hint else ""
        raise FileNotFoundError(f"No checkpoints found under {Path(log_root).expanduser()}{hint}.")
    return checkpoints[0]

