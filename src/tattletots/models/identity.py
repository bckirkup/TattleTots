"""Deterministic identifiers and digests used by the simulation engine."""

from __future__ import annotations

import hashlib
import uuid

import numpy as np


def seeded_id(rng: np.random.Generator) -> str:
    """Create a UUID-compatible identifier from a seeded generator."""
    raw = bytearray(rng.bytes(16))
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(raw)))


def stable_id_digest(identifier: str) -> int:
    """Return a process-independent integer digest for an identifier."""
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def is_uuid_identifier(identifier: str) -> bool:
    """Return whether an identifier has the UUID-compatible default format."""
    try:
        uuid.UUID(identifier)
    except (ValueError, AttributeError):
        return False
    return True
