"""
Scalping strategy.

Harvests small spreads and mean-reversions over seconds to minutes using
order-book imbalance and microstructure signals.  When a full order book is
not available, the strategy falls back to a simple RSI-based overbought /
oversold filter applied to recent closes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from eth_algo_trading.strategies.base import BaseStrategy, Signal


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    # When avg_loss is 0 the RSI is 100 (pure uptrend); avoid division by zero.
    rsi = pd.Series(np.where(avg_loss == 0, 100.0, 100 - 100 / (1 + avg_gain / avg_loss)),
                    index=series.index)
    # Propagate NaN where avg_gain is NaN (insufficient data)
    rsi[avg_gain.isna()] = np.nan
    return rsi


class ScalpingStrategy(BaseStrategy):
    """
    RSI-based scalping with order-book imbalance overlay.

    Parameters
    ----------
    rsi_period:
        Look-back window for the RSI calculation.
    oversold_threshold:
        RSI level below which a long signal is generated.
    overbought_threshold:
        RSI level above which a short signal is generated.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        oversold_threshold: float = 30.0,
        overbought_threshold: float = 70.0,
    ) -> None:
        super().__init__(name="scalping")
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold

    def generate_signal(
        self,
        ohlcv: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> Signal:
        if len(ohlcv) < self.rsi_period + 1:
            return Signal(direction="flat", strength=0.0, strategy=self.name)

        rsi = _rsi(ohlcv["close"], self.rsi_period)
        latest_rsi = float(rsi.iloc[-1])

        # Order-book imbalance from features if available
        imbalance = 0.0
        if features is not None and "order_book_imbalance" in features.columns:
            imbalance = float(features["order_book_imbalance"].iloc[-1])

        if latest_rsi < self.oversold_threshold:
            strength = (self.oversold_threshold - latest_rsi) / self.oversold_threshold
            strength = min(1.0, strength + 0.2 * max(0.0, imbalance))
            return Signal(
                direction="long",
                strength=round(strength, 4),
                strategy=self.name,
                metadata={"rsi": latest_rsi, "imbalance": imbalance},
            )

        if latest_rsi > self.overbought_threshold:
            strength = (latest_rsi - self.overbought_threshold) / (100.0 - self.overbought_threshold)
            strength = min(1.0, strength + 0.2 * max(0.0, -imbalance))
            return Signal(
                direction="short",
                strength=round(strength, 4),
                strategy=self.name,
                metadata={"rsi": latest_rsi, "imbalance": imbalance},
            )

        return Signal(
            direction="flat",
            strength=0.0,
            strategy=self.name,
            metadata={"rsi": latest_rsi},
        )
