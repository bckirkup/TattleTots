"""GPU utilities: transparent CuPy / NumPy dispatch.

Call ``get_array_module(use_gpu=True)`` to obtain ``cupy`` when available and
``numpy`` otherwise.  All engine modules import ``xp`` from here so that the
same math code runs on CPU or GPU without branching.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np

_cupy: ModuleType | None = None
_cupy_import_attempted: bool = False


def _try_import_cupy() -> ModuleType | None:
    global _cupy, _cupy_import_attempted
    if _cupy_import_attempted:
        return _cupy
    _cupy_import_attempted = True
    try:
        import cupy as cp  # type: ignore[import-untyped]

        _cupy = cp
    except Exception:
        _cupy = None
    return _cupy


def get_array_module(use_gpu: bool = False) -> ModuleType:
    """Return cupy if ``use_gpu`` is True and CuPy is available, else numpy."""
    if use_gpu:
        cp = _try_import_cupy()
        if cp is not None:
            return cp
    return np


def to_numpy(arr: object) -> np.ndarray:
    """Convert a cupy or numpy array to a plain numpy array."""
    cp = _try_import_cupy()
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)  # type: ignore[return-value]
    return np.asarray(arr)


def gpu_available() -> bool:
    """Return True if CuPy is importable and at least one CUDA device exists."""
    cp = _try_import_cupy()
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False
