from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TASK = "Isaac-Lift-Cube-Franka-v0"
DEFAULT_IK_TASK = "Isaac-Lift-Cube-Franka-IK-Rel-v0"


@dataclass(frozen=True)
class IsaacLabCommand:
    isaaclab_root: Path
    command: list[str]

    def as_shell_text(self) -> str:
        return " ".join(self.command)


def find_isaaclab_root(value: str | None = None, require_exists: bool = True) -> Path:
    candidates = [
        value,
        os.environ.get("ISAACLAB_PATH"),
        os.environ.get("ISAAC_LAB_PATH"),
        "~/isaac_ws/IsaacLab",
        "~/IsaacLab",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if not require_exists:
            return path
        if (path / "isaaclab.sh").exists():
            return path
    raise FileNotFoundError(
        "IsaacLab root not found. Pass --isaaclab-root or set ISAACLAB_PATH. "
        "Expected an isaaclab.sh file, for example ~/isaac_ws/IsaacLab/isaaclab.sh."
    )


def workflow_script(root: Path, library: str, mode: str, require_exists: bool = True) -> Path:
    candidates = [
        root / "scripts" / "reinforcement_learning" / library / f"{mode}.py",
        root / "source" / "standalone" / "workflows" / library / f"{mode}.py",
        root / "scripts" / "rl_games" / library / f"{mode}.py",
    ]
    if not require_exists:
        return candidates[0]

    for path in candidates:
        if path.exists():
            return path

    discovered = sorted(root.glob(f"**/{library}/{mode}.py"))
    if discovered:
        return discovered[0]

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"Isaac Lab {library} {mode}.py script not found.\n"
        f"Searched:\n{searched}\n"
        f"Tip: run `find {root} -path '*{library}/{mode}.py'` on the server to locate your Isaac Lab layout."
    )


def build_train_command(
    isaaclab_root: Path,
    task: str,
    library: str,
    num_envs: int,
    max_iterations: int | None,
    headless: bool,
    video: bool,
    validate_paths: bool = True,
    extra_args: list[str] | None = None,
) -> IsaacLabCommand:
    script = workflow_script(isaaclab_root, library, "train", require_exists=validate_paths)
    command = [
        str(isaaclab_root / "isaaclab.sh"),
        "-p",
        str(script.relative_to(isaaclab_root)),
        "--task",
        task,
        "--num_envs",
        str(num_envs),
    ]
    if headless:
        command.append("--headless")
    if video:
        command.append("--video")
    if max_iterations is not None:
        command.extend(["--max_iterations", str(max_iterations)])
    if extra_args:
        command.extend(extra_args)
    return IsaacLabCommand(isaaclab_root=isaaclab_root, command=command)


def build_play_command(
    isaaclab_root: Path,
    task: str,
    library: str,
    num_envs: int,
    checkpoint: str | None,
    headless: bool,
    video: bool,
    validate_paths: bool = True,
    extra_args: list[str] | None = None,
) -> IsaacLabCommand:
    script = workflow_script(isaaclab_root, library, "play", require_exists=validate_paths)
    command = [
        str(isaaclab_root / "isaaclab.sh"),
        "-p",
        str(script.relative_to(isaaclab_root)),
        "--task",
        task,
        "--num_envs",
        str(num_envs),
    ]
    if checkpoint:
        command.extend(["--checkpoint", checkpoint])
    if headless:
        command.append("--headless")
    if video:
        command.append("--video")
    if extra_args:
        command.extend(extra_args)
    return IsaacLabCommand(isaaclab_root=isaaclab_root, command=command)


def build_python_command(isaaclab_root: Path, script: Path, script_args: list[str] | None = None) -> IsaacLabCommand:
    command = [
        str(isaaclab_root / "isaaclab.sh"),
        "-p",
        str(script),
    ]
    if script_args:
        command.extend(script_args)
    return IsaacLabCommand(isaaclab_root=isaaclab_root, command=command)


def run_command(command: IsaacLabCommand, dry_run: bool = False) -> int:
    print(f"IsaacLab root: {command.isaaclab_root}")
    print(f"Command: {command.as_shell_text()}")
    if dry_run:
        return 0
    completed = subprocess.run(command.command, cwd=command.isaaclab_root, check=False)
    return int(completed.returncode)
