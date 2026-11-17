"""Small IO helpers shared by the downloaders: atomic writes and failure markers."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write to <name>.tmp-<pid> then rename, so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


def write_marker(path: Path, info: Any) -> None:
    """A JSON marker for a request that will never succeed (e.g. symbol not covered), so it is not retried."""
    atomic_write_bytes(path, json.dumps(info, default=str).encode())


def read_marker(path: Path) -> Optional[Any]:
    return json.loads(path.read_text()) if path.is_file() else None
