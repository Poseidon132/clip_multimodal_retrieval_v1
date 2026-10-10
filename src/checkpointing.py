"""Validation-based checkpoint selection and early stopping helpers."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass
class BestScoreTracker:
    """Maximize a validation metric without ever looking at the test split."""

    patience: int = 3
    min_delta: float = 0.0
    best_score: float = float("-inf")
    best_epoch: int = 0
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        if self.patience < 0:
            raise ValueError("patience must be >= 0 (0 disables early stopping)")
        if self.min_delta < 0:
            raise ValueError("min_delta must be >= 0")

    def step(self, value: float, epoch: int) -> tuple[bool, bool]:
        """Return (improved, should_stop) after observing a validation score."""
        score = float(value)
        if not isfinite(score):
            raise ValueError(f"Validation metric is not finite: {score}")
        improved = score > self.best_score + self.min_delta
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, self.patience > 0 and self.bad_epochs >= self.patience
