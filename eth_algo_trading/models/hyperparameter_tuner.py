"""
Dynamic hyperparameter tuner for AI/ML models.

Tracks rolling prediction accuracy and confidence, then recommends
adjusted hyperparameter values so each model can self-calibrate based
on recent decisions it got right (or wrong).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, NamedTuple, Optional


class _Outcome(NamedTuple):
    predicted: int   # model's predicted class
    actual: int      # ground-truth class
    confidence: float  # probability of the predicted class in [0, 1]


@dataclass
class AdaptiveConfig:
    """
    Controls how aggressively the tuner reacts to performance changes.

    Parameters
    ----------
    window:
        Number of recent outcomes used to compute rolling metrics.
    accuracy_target:
        Rolling accuracy below this value triggers a hyperparameter increase.
    confidence_target:
        Mean confidence below this value is treated as a signal of low
        certainty and can widen the adaptation range.
    estimators_step:
        Amount by which ``n_estimators``-style parameters are incremented /
        decremented when accuracy is outside the target band.
    estimators_min:
        Hard lower bound on ``n_estimators``-style parameters.
    estimators_max:
        Hard upper bound on ``n_estimators``-style parameters.
    contamination_step:
        Amount by which ``contamination`` is adjusted when the observed
        anomaly rate drifts.
    contamination_min:
        Hard lower bound on ``contamination``.
    contamination_max:
        Hard upper bound on ``contamination``.
    """

    window: int = 50
    accuracy_target: float = 0.55
    confidence_target: float = 0.50
    estimators_step: int = 25
    estimators_min: int = 50
    estimators_max: int = 600
    contamination_step: float = 0.01
    contamination_min: float = 0.01
    contamination_max: float = 0.20


class PerformanceTracker:
    """
    Lightweight online tracker that records prediction outcomes and
    exposes rolling accuracy / confidence metrics.

    Usage
    -----
    >>> tracker = PerformanceTracker()
    >>> tracker.record(predicted=2, actual=2, confidence=0.75)
    >>> tracker.rolling_accuracy()
    1.0
    >>> tracker.rolling_confidence()
    0.75
    """

    def __init__(self, config: Optional[AdaptiveConfig] = None) -> None:
        self._cfg = config or AdaptiveConfig()
        self._history: Deque[_Outcome] = deque(maxlen=self._cfg.window)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, predicted: int, actual: int, confidence: float) -> None:
        """Append a single outcome to the rolling history."""
        self._history.append(_Outcome(predicted=predicted, actual=actual, confidence=confidence))

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def rolling_accuracy(self) -> float:
        """Fraction of correct predictions in the current window."""
        if not self._history:
            return 0.0
        correct = sum(1 for o in self._history if o.predicted == o.actual)
        return correct / len(self._history)

    def rolling_confidence(self) -> float:
        """Mean confidence of predictions in the current window."""
        if not self._history:
            return 0.0
        return sum(o.confidence for o in self._history) / len(self._history)

    @property
    def n_recorded(self) -> int:
        """Number of outcomes recorded so far."""
        return len(self._history)

    def get_actuals(self) -> list[int]:
        """Return the list of ground-truth labels in the current window."""
        return [o.actual for o in self._history]

    # ------------------------------------------------------------------
    # Adaptation suggestions
    # ------------------------------------------------------------------

    def suggest_n_estimators(self, current: int) -> int:
        """
        Return an adjusted ``n_estimators`` value based on rolling accuracy.

        * Accuracy < ``accuracy_target``  → increase (up to ``estimators_max``)
        * Accuracy ≥ ``accuracy_target`` and confidence ≥ ``confidence_target``
          → cautiously decrease toward baseline (down to ``estimators_min``)

        No adjustment is made until at least half the tracking window is
        filled, to avoid premature tuning on a handful of samples.
        """
        cfg = self._cfg
        if self.n_recorded < cfg.window // 2:
            return current

        acc = self.rolling_accuracy()
        conf = self.rolling_confidence()

        if acc < cfg.accuracy_target:
            return min(current + cfg.estimators_step, cfg.estimators_max)
        if acc >= cfg.accuracy_target and conf >= cfg.confidence_target:
            return max(current - cfg.estimators_step, cfg.estimators_min)
        return current

    def suggest_contamination(self, current: float, observed_anomaly_rate: float) -> float:
        """
        Return an adjusted ``contamination`` value based on the observed
        fraction of anomalies in recent data.

        If the model is flagging far more (or fewer) anomalies than
        ``current`` implies, nudge contamination toward the observed rate.
        """
        cfg = self._cfg
        if self.n_recorded < cfg.window // 2:
            return current

        delta = observed_anomaly_rate - current
        if abs(delta) < cfg.contamination_step:
            return current
        adjusted = current + cfg.contamination_step * (1 if delta > 0 else -1)
        return float(max(cfg.contamination_min, min(cfg.contamination_max, adjusted)))
