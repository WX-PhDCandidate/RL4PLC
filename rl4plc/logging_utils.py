from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from rl4plc.types import EpisodeResult


def ensure_run_dir(run_dir: str | Path) -> Path:
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_episode_csv(path: str | Path, result: EpisodeResult) -> None:
    path = Path(path)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.to_dict().keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(result.to_dict())


def summarize(results: list[EpisodeResult]) -> dict:
    total = len(results)
    success = sum(1 for item in results if item.success)
    return {
        "episodes": total,
        "success": success,
        "failed": total - success,
        "success_rate": success / total if total else 0.0,
    }

