"""Safe path resolution for experiment runner CLI scripts."""

from __future__ import annotations

from pathlib import Path

KEY_JSON = "key.json"


def safe_path_under_base(raw: Path | str, base: Path) -> Path:
    """Resolve raw under base; reject path traversal via ``..`` components."""
    candidate = raw if isinstance(raw, Path) else Path(raw)
    if ".." in candidate.parts:
        msg = f"Path traversal is not allowed: {raw}"
        raise ValueError(msg)

    base_resolved = base.expanduser().resolve()
    if candidate.is_absolute():
        return candidate.expanduser().resolve()

    resolved = (base_resolved / candidate).resolve()
    if not resolved.is_relative_to(base_resolved):
        msg = f"Path escapes allowed directory {base_resolved}: {raw}"
        raise ValueError(msg)
    return resolved


def safe_output_dir(raw: Path | str, *, default_base: Path) -> Path:
    """Resolve an output directory under default_base."""
    return safe_path_under_base(raw, default_base)


def safe_config_path(raw: Path | str, *, base: Path) -> Path:
    """Resolve a config file path under base."""
    return safe_path_under_base(raw, base)
