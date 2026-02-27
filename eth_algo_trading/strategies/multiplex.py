"""
Multiplex strategy.

Aggregates signals from multiple child strategies using a golden-ratio
weighted voting scheme and optionally blends in a learned price-direction
probability from a fitted :class:`~eth_algo_trading.models.PriceForecaster`.

The golden ratio φ = (1 + √5) / 2 ≈ 1.618 underpins the weight series:
the highest-confidence child signal receives weight φ⁰ = 1, the next
φ⁻¹ ≈ 0.618, then φ⁻² ≈ 0.382, and so on.  The series is normalised so
that all weights sum to 1.0.  Blending with a fitted forecaster then
adjusts the aggregate toward the model's learned probability of the next
price move.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd

from eth_algo_trading.strategies.base import BaseStrategy, Signal

if TYPE_CHECKING:  # avoid hard runtime dependency on models package
    from eth_algo_trading.models.forecasting import PriceForecaster

_PHI: float = (1.0 + 5.0 ** 0.5) / 2.0  # golden ratio ≈ 1.618


class MultiplexStrategy(BaseStrategy):
    """
    Golden-ratio weighted ensemble of child strategies with optional AI blend.

    Signals from child strategies are sorted by descending strength so that
    the most confident signal receives the largest (φ⁰) weight.  A
    :class:`~eth_algo_trading.models.PriceForecaster` can be supplied to
    further adjust the final signal using data-driven direction probabilities.

    Parameters
    ----------
    strategies:
        Ordered list of child :class:`BaseStrategy` instances.  At least one
        strategy is required.
    forecaster:
        Optional fitted :class:`~eth_algo_trading.models.PriceForecaster`.
        When provided its ``predict_proba`` output is blended into the
        composite vote.
    forecaster_weight:
        Fraction (0–1) of the final vote attributed to the forecaster.
        The remainder comes from the golden-ratio weighted rule-based vote.
    min_consensus:
        Minimum absolute weighted vote required to emit a directional signal.
        Votes below this threshold produce a ``flat`` signal.
    """

    def __init__(
        self,
        strategies: List[BaseStrategy],
        forecaster: Optional["PriceForecaster"] = None,
        forecaster_weight: float = 0.3,
        min_consensus: float = 0.1,
    ) -> None:
        super().__init__(name="multiplex")
        if not strategies:
            raise ValueError("At least one child strategy is required.")
        self.strategies = strategies
        self.forecaster = forecaster
        self.forecaster_weight = float(max(0.0, min(1.0, forecaster_weight)))
        self.min_consensus = float(max(0.0, min(1.0, min_consensus)))

    # ------------------------------------------------------------------
    # Golden-ratio weight helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _golden_weights(n: int) -> np.ndarray:
        """Return *n* normalised golden-ratio weights [φ⁰, φ⁻¹, φ⁻², …].

        Weights decay geometrically by factor 1/φ and are normalised to
        sum to 1.0.
        """
        if n == 0:
            return np.array([], dtype=float)
        weights = np.fromiter((_PHI ** (-i) for i in range(n)), dtype=float, count=n)
        return weights / weights.sum()

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        ohlcv: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> Signal:
        # Collect signals from every child strategy.
        raw_signals: List[Signal] = [
            s.generate_signal(ohlcv, features) for s in self.strategies
        ]

        # Sort by descending strength so the most confident signal gets the
        # largest golden-ratio weight.
        sorted_signals = sorted(raw_signals, key=lambda sig: sig.strength, reverse=True)
        weights = self._golden_weights(len(sorted_signals))

        # Compute weighted directional vote: long → +1, short → −1, flat → 0.
        _dir_map = {"long": 1.0, "short": -1.0, "flat": 0.0}
        rule_vote = float(
            sum(
                w * _dir_map[sig.direction] * sig.strength
                for w, sig in zip(weights, sorted_signals)
            )
        )  # in (−1, +1)

        vote = rule_vote

        # Blend in the forecaster's learned probability when available.
        if self.forecaster is not None and getattr(self.forecaster, "is_fitted", False):
            try:
                prob_down, _prob_flat, prob_up = self.forecaster.predict_proba(
                    ohlcv, features
                )
                learned_vote = float(prob_up - prob_down)  # in (−1, +1)
                vote = (
                    (1.0 - self.forecaster_weight) * rule_vote
                    + self.forecaster_weight * learned_vote
                )
            except Exception:
                pass  # fall back to pure rule-based vote on inference failure

        abs_vote = abs(vote)

        if abs_vote < self.min_consensus:
            return Signal(
                direction="flat",
                strength=0.0,
                strategy=self.name,
                metadata={
                    "vote": round(vote, 4),
                    "child_signals": [
                        {
                            "strategy": sig.strategy,
                            "direction": sig.direction,
                            "strength": sig.strength,
                        }
                        for sig in raw_signals
                    ],
                },
            )

        direction = "long" if vote > 0 else "short"
        strength = round(min(1.0, abs_vote), 4)
        return Signal(
            direction=direction,
            strength=strength,
            strategy=self.name,
            metadata={
                "vote": round(vote, 4),
                "child_signals": [
                    {
                        "strategy": sig.strategy,
                        "direction": sig.direction,
                        "strength": sig.strength,
                    }
                    for sig in raw_signals
                ],
            },
        )
