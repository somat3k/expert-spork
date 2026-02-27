"""
Trend-following and regime-rotation strategy.

Uses moving-average crossovers, ADX, and volatility-adjusted breakouts to
ride medium-term trends.  An optional AI regime classifier (see
:mod:`eth_algo_trading.models.regime_detection`) can be plugged in via the
*features* DataFrame to override the simple rule-based regime filter.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from eth_algo_trading.strategies.base import BaseStrategy, Signal


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _adx(ohlcv: pd.DataFrame, period: int = 14) -> pd.Series:
    high = ohlcv["high"]
    low = ohlcv["low"]
    close = ohlcv["close"]

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    dm_plus = (high.diff()).clip(lower=0)
    dm_minus = (-low.diff()).clip(lower=0)
    mask = dm_plus < dm_minus
    dm_plus[mask] = 0.0
    mask2 = dm_minus < dm_plus
    dm_minus[mask2] = 0.0

    atr = tr.ewm(com=period - 1, min_periods=period).mean()
    di_plus = 100 * dm_plus.ewm(com=period - 1, min_periods=period).mean() / atr.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(com=period - 1, min_periods=period).mean() / atr.replace(0, np.nan)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    return dx.ewm(com=period - 1, min_periods=period).mean()


class TrendFollowingStrategy(BaseStrategy):
    """
    Dual-EMA crossover with ADX trend-strength filter.

    Parameters
    ----------
    fast_period:
        Period for the fast exponential moving average.
    slow_period:
        Period for the slow exponential moving average.
    adx_threshold:
        Minimum ADX value required to confirm a trending market.
    """

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        adx_threshold: float = 25.0,
    ) -> None:
        super().__init__(name="trend_following")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.adx_threshold = adx_threshold

    def generate_signal(
        self,
        ohlcv: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> Signal:
        min_bars = self.slow_period + 1
        if len(ohlcv) < min_bars:
            return Signal(direction="flat", strength=0.0, strategy=self.name)

        fast = _ema(ohlcv["close"], self.fast_period)
        slow = _ema(ohlcv["close"], self.slow_period)
        adx = _adx(ohlcv)

        latest_fast = float(fast.iloc[-1])
        latest_slow = float(slow.iloc[-1])
        latest_adx = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0.0

        # Optional AI regime override: expect 'regime' column with values
        # 'bull', 'bear', or 'sideways'.
        regime = None
        if features is not None and "regime" in features.columns:
            regime = features["regime"].iloc[-1]

        trending = latest_adx >= self.adx_threshold and regime != "sideways"

        if not trending:
            return Signal(direction="flat", strength=0.0, strategy=self.name,
                          metadata={"adx": latest_adx})

        cross_strength = abs(latest_fast - latest_slow) / latest_slow if latest_slow else 0.0
        strength = min(1.0, cross_strength * 10)

        if latest_fast > latest_slow:
            return Signal(direction="long", strength=round(strength, 4),
                          strategy=self.name,
                          metadata={"adx": latest_adx, "fast": latest_fast, "slow": latest_slow})

        return Signal(direction="short", strength=round(strength, 4),
                      strategy=self.name,
                      metadata={"adx": latest_adx, "fast": latest_fast, "slow": latest_slow})
