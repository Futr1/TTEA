from __future__ import annotations

from pathlib import Path
from typing import Any

from ..exceptions import PersistenceError
from ..utils import write_json_file


def list_checkpoint_directories(root: str | Path) -> list[str]:
    base = Path(root)
    if not base.exists():
        return []
    checkpoints = [
        item
        for item in base.iterdir()
        if item.is_dir() and (item.name.startswith("checkpoint-") or item.name == "final_model")
    ]
    return [str(path) for path in sorted(checkpoints)]


def persist_checkpoint_index(root: str | Path, metadata: dict[str, Any]) -> str:
    try:
        path = write_json_file(Path(root) / "checkpoints.json", metadata)
        return str(path)
    except Exception as exc:
        raise PersistenceError(f"Failed to write checkpoint index under {root}: {exc}") from exc
