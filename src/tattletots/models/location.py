"""Spatial location types for event reporting and verification."""

from __future__ import annotations

# Grid cell (row, col) or ocean zone (zone_x, zone_y).
EventLocation = tuple[int, int]

# Inclusive minimum and maximum coordinates for a domain's public grading frame.
LocationFrame = tuple[EventLocation, EventLocation]
