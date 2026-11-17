"""Credentials and data-root lookup.

Secrets come from the environment, else from a .env file: $KIT_ENV, then ~/.config/kit/.env,
then <repo>/.env. Values are never printed.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

REPO = Path(__file__).resolve().parents[1]
_ENV_CANDIDATES = (os.environ.get("KIT_ENV", ""), str(Path.home() / ".config/kit/.env"), str(REPO / ".env"))


def _read_env_file(path: Path) -> Dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def load_env() -> Dict[str, str]:
    """os.environ overrides the first .env file found."""
    for cand in _ENV_CANDIDATES:
        if cand and Path(cand).is_file():
            return {**_read_env_file(Path(cand)), **os.environ}
    return dict(os.environ)


def secret(name: str) -> str:
    val = load_env().get(name, "")
    if not val:
        raise SystemExit(f"{name} is not set (environment, $KIT_ENV, ~/.config/kit/.env or {REPO / '.env'})")
    return val


def data_root() -> Path:
    """Root of the KiT-2 data tree; $KIT2_DATA, default /workspace/kit2/data."""
    return Path(os.environ.get("KIT2_DATA", "/workspace/kit2/data"))
