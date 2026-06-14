"""Pluggable compression models that agents use to extract structure from streams."""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import ModuleType

import numpy as np
from numpy.typing import NDArray

from tattletots.engine.gpu_utils import get_array_module, to_numpy
from tattletots.models.genome import CompressionType


class CompressionModel(ABC):
    """Abstract base for compression models."""

    def set_array_module(self, xp: ModuleType) -> None:
        """Switch the array backend (numpy or cupy) used by this model."""
        self._xp = xp

    @abstractmethod
    def fit_transform(self, data: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
        """Compress input data. Returns (residual, yield).

        - residual: the unmodeled remainder (same dimensionality as input)
        - yield: information extracted (non-negative scalar)
        """

    @abstractmethod
    def anomaly_score(self, data: NDArray[np.float64]) -> float:
        """Compute anomaly score for current data (higher = more anomalous)."""

    @abstractmethod
    def get_signal_vector(self) -> NDArray[np.float64]:
        """Return the compressed signal representation for reporting."""


class PCACompression(CompressionModel):
    """PCA-based compression: extracts top-k principal components.

    Maintains a sliding window of recent samples so that PCA can operate
    even when the engine feeds one sample per step (the common case).
    """

    _WINDOW_SIZE: int = 20

    def __init__(self, n_components: int, efficiency: float = 1.0) -> None:
        self.n_components = n_components
        self.efficiency = efficiency
        self._xp: ModuleType = np
        self._mean: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._components: NDArray[np.float64] | None = None
        self._signal: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._explained_var: float = 0.0
        self._history: list[NDArray[np.float64]] = []

    def fit_transform(self, data: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
        xp = self._xp
        flat = xp.asarray(data.flatten(), dtype=xp.float64)

        # Maintain sliding window; reset on dimensionality change
        if self._history and len(self._history[0]) != len(flat):
            self._history.clear()
        self._history.append(flat.copy())
        if len(self._history) > self._WINDOW_SIZE:
            self._history = self._history[-self._WINDOW_SIZE :]

        window = xp.array(self._history, dtype=xp.float64)
        mean = window.mean(axis=0)
        self._mean = mean
        centered = window - mean

        if centered.shape[0] < 2:
            magnitude = float(xp.linalg.norm(flat))
            self._signal = to_numpy(flat[: self.n_components])
            self._explained_var = magnitude * self.efficiency
            self._components = None
            return to_numpy(flat), self._explained_var

        # SVD for PCA
        n_comp = min(self.n_components, min(centered.shape))
        _u, s, vt = xp.linalg.svd(centered, full_matrices=False)
        self._components = vt[:n_comp]

        # Project and reconstruct the *current* sample only
        current_centered = (flat - mean).reshape(1, -1)
        projected = current_centered @ self._components.T
        reconstructed = projected @ self._components
        residual = (current_centered - reconstructed).flatten()

        total_var = float(xp.sum(s**2))
        explained_var = float(xp.sum(s[:n_comp] ** 2))
        info_yield = (explained_var / max(total_var, 1e-10)) * self.efficiency

        self._signal = to_numpy(projected.flatten()[:n_comp])
        self._explained_var = info_yield
        return to_numpy(residual), info_yield

    def anomaly_score(self, data: NDArray[np.float64]) -> float:
        if self._mean.size == 0 or self._components is None:
            return 0.0
        xp = self._xp
        flat = xp.asarray(data.flatten(), dtype=xp.float64)
        if flat.shape[0] != self._mean.shape[0]:
            return 0.0
        centered = (flat - self._mean).reshape(1, -1)
        projected = centered @ self._components.T
        reconstructed = projected @ self._components
        reconstruction_error = float(xp.mean((centered - reconstructed) ** 2))
        return reconstruction_error

    def get_signal_vector(self) -> NDArray[np.float64]:
        return self._signal


class AR1Compression(CompressionModel):
    """AR(1) autoregressive compression: models temporal dependence."""

    def __init__(self, n_components: int, efficiency: float = 1.0) -> None:
        self.n_components = n_components
        self.efficiency = efficiency
        self._xp: ModuleType = np
        self._prev: NDArray[np.float64] | None = None
        self._coeffs: NDArray[np.float64] | None = None
        self._signal: NDArray[np.float64] = np.array([], dtype=np.float64)

    def fit_transform(self, data: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
        xp = self._xp
        flat = xp.asarray(data.flatten(), dtype=xp.float64)
        if self._prev is None or len(flat) != len(self._prev):
            self._prev = flat
            self._signal = to_numpy(flat[: self.n_components])
            return to_numpy(flat), 0.0

        # Simple AR(1): predict current from previous
        if len(flat) == len(self._prev) and len(flat) > 0:
            # Least-squares coefficient
            denom = float(xp.dot(self._prev, self._prev))
            if denom > 1e-10:
                coeff = float(xp.dot(flat, self._prev)) / denom
            else:
                coeff = 0.0
            predicted = coeff * self._prev
            residual = flat - predicted
            var_residual = float(xp.var(residual))
            var_flat = float(xp.var(flat))
            info_yield = float(1.0 - var_residual / max(var_flat, 1e-10))
            info_yield = max(0.0, info_yield) * self.efficiency
        else:
            residual = flat
            info_yield = 0.0

        self._signal = to_numpy(flat[: self.n_components])
        self._prev = flat
        return to_numpy(residual), info_yield

    def anomaly_score(self, data: NDArray[np.float64]) -> float:
        xp = self._xp
        flat = xp.asarray(data.flatten(), dtype=xp.float64)
        if self._prev is None or len(flat) != len(self._prev):
            return 0.0
        denom = float(xp.dot(self._prev, self._prev))
        if denom > 1e-10:
            coeff = float(xp.dot(flat, self._prev)) / denom
        else:
            coeff = 0.0
        predicted = coeff * self._prev
        return float(xp.mean((flat - predicted) ** 2))

    def get_signal_vector(self) -> NDArray[np.float64]:
        return self._signal


class ThresholdCompression(CompressionModel):
    """Simple threshold detector: flags dimensions exceeding running statistics."""

    def __init__(self, n_components: int, efficiency: float = 1.0) -> None:
        self.n_components = n_components
        self.efficiency = efficiency
        self._xp: ModuleType = np
        self._running_mean: NDArray[np.float64] | None = None
        self._running_var: NDArray[np.float64] | None = None
        self._count: int = 0
        self._signal: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._alpha: float = 0.1  # EMA smoothing

    def fit_transform(self, data: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
        xp = self._xp
        flat = xp.asarray(data.flatten(), dtype=xp.float64)

        if self._running_mean is None or len(flat) != len(self._running_mean):
            self._running_mean = flat.copy()
            self._running_var = xp.ones_like(flat)
            self._count = 1
            self._signal = to_numpy(flat[: self.n_components])
            return to_numpy(flat), 0.0

        # Update running statistics
        self._count += 1
        self._running_mean = (1 - self._alpha) * self._running_mean + self._alpha * flat
        diff = flat - self._running_mean
        assert self._running_var is not None
        self._running_var = (1 - self._alpha) * self._running_var + self._alpha * diff**2

        # Residual is the z-scored deviation
        std = xp.sqrt(xp.maximum(self._running_var, 1e-10))
        z_scores = diff / std
        # "Compression" = identifying which dimensions are anomalous
        residual = flat - self._running_mean
        info_yield = float(xp.mean(xp.abs(z_scores) > 1.0)) * self.efficiency

        self._signal = to_numpy(z_scores[: self.n_components])
        return to_numpy(residual), info_yield

    def anomaly_score(self, data: NDArray[np.float64]) -> float:
        xp = self._xp
        flat = xp.asarray(data.flatten(), dtype=xp.float64)
        if self._running_mean is None or len(flat) != len(self._running_mean):
            return 0.0
        diff = flat - self._running_mean
        assert self._running_var is not None
        std = xp.sqrt(xp.maximum(self._running_var, 1e-10))
        z_scores = diff / std
        return float(xp.max(xp.abs(z_scores)))

    def get_signal_vector(self) -> NDArray[np.float64]:
        return self._signal


def create_compression_model(
    compression_type: CompressionType,
    n_components: int,
    efficiency: float = 1.0,
    use_gpu: bool = False,
) -> CompressionModel:
    """Factory for compression models based on genome specification."""
    xp = get_array_module(use_gpu)
    match compression_type:
        case CompressionType.PCA:
            model: CompressionModel = PCACompression(n_components, efficiency)
        case CompressionType.AR1:
            model = AR1Compression(n_components, efficiency)
        case CompressionType.THRESHOLD:
            model = ThresholdCompression(n_components, efficiency)
        case CompressionType.WAVELET:
            # Fallback to PCA for now; wavelet is a future extension
            model = PCACompression(n_components, efficiency)
    model.set_array_module(xp)
    return model
