"""Path validation helpers for CLI entry points."""

from __future__ import annotations

from pathlib import Path


def safe_path_under_base(raw: Path, base: Path) -> Path:
    """Resolve a user-supplied path and reject directory traversal.

    Relative paths must resolve inside ``base``. Absolute paths are resolved as-is
    but reject ``..`` components in the user input.
    """
    if ".." in raw.parts:
        msg = f"Path traversal is not allowed: {raw}"
        raise ValueError(msg)

    base_resolved = base.expanduser().resolve()
    if raw.is_absolute():
        return raw.expanduser().resolve()

    resolved = (base_resolved / raw).resolve()
    if not resolved.is_relative_to(base_resolved):
        msg = f"Path escapes allowed directory {base_resolved}: {raw}"
        raise ValueError(msg)
    return resolved
