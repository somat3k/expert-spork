"""
Cross-exchange arbitrage strategy.

Detects price dislocations between two or more venues for the same ETH pair
and generates a signal to buy on the cheaper venue while selling on the
more expensive one (basis arbitrage).
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from eth_algo_trading.strategies.base import BaseStrategy, Signal


class ArbitrageStrategy(BaseStrategy):
    """
    Spot / perpetual basis and cross-exchange spread arbitrage.

    Parameters
    ----------
    min_spread_pct:
        Minimum spread (as a fraction of price) required to trigger a signal.
    max_spread_pct:
        Spread above which the opportunity is assumed to be stale / erroneous.
    """

    def __init__(
        self,
        min_spread_pct: float = 0.001,
        max_spread_pct: float = 0.05,
    ) -> None:
        super().__init__(name="arbitrage")
        self.min_spread_pct = min_spread_pct
        self.max_spread_pct = max_spread_pct

    # ------------------------------------------------------------------
    # Primary interface: compare two OHLCV streams
    # ------------------------------------------------------------------

    def compute_spread(self, price_a: float, price_b: float) -> float:
        """Return the relative spread between two venue prices."""
        mid = (price_a + price_b) / 2.0
        if mid == 0:
            return 0.0
        return abs(price_a - price_b) / mid

    def generate_signal(
        self,
        ohlcv: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> Signal:
        """
        Generate an arbitrage signal.

        Expects *features* to contain columns ``price_venue_a`` and
        ``price_venue_b`` representing the latest mid-prices on two venues.
        Falls back to a ``flat`` signal when these columns are absent.
        """
        if features is None:
            return Signal(direction="flat", strength=0.0, strategy=self.name)

        required = {"price_venue_a", "price_venue_b"}
        if not required.issubset(features.columns):
            return Signal(direction="flat", strength=0.0, strategy=self.name)

        price_a = float(features["price_venue_a"].iloc[-1])
        price_b = float(features["price_venue_b"].iloc[-1])
        spread = self.compute_spread(price_a, price_b)

        if spread < self.min_spread_pct or spread > self.max_spread_pct:
            return Signal(
                direction="flat",
                strength=0.0,
                strategy=self.name,
                metadata={"spread_pct": spread},
            )

        # Buy on the cheaper venue
        direction = "long" if price_a < price_b else "short"
        strength = min(1.0, (spread - self.min_spread_pct) / (self.max_spread_pct - self.min_spread_pct))
        return Signal(
            direction=direction,
            strength=round(strength, 4),
            strategy=self.name,
            metadata={"spread_pct": spread, "price_a": price_a, "price_b": price_b},
        )
