"""
Sentiment and on-chain signal fusion strategy.

Converts social-media sentiment scores, whale-transfer anomalies, and DEX
flow indicators into a tradeable signal.  Sentiment scores are expected to be
pre-computed by :mod:`eth_algo_trading.models.sentiment` and passed in via the
*features* DataFrame.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from eth_algo_trading.strategies.base import BaseStrategy, Signal


class SentimentStrategy(BaseStrategy):
    """
    Signal fusion from NLP sentiment and on-chain metrics.

    Parameters
    ----------
    sentiment_threshold:
        Absolute sentiment score (range –1 to +1) required to act.
    whale_alert_weight:
        Multiplier applied to the whale-alert anomaly score when blending.
    """

    def __init__(
        self,
        sentiment_threshold: float = 0.3,
        whale_alert_weight: float = 0.4,
    ) -> None:
        super().__init__(name="sentiment")
        self.sentiment_threshold = sentiment_threshold
        self.whale_alert_weight = whale_alert_weight

    def generate_signal(
        self,
        ohlcv: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> Signal:
        if features is None:
            return Signal(direction="flat", strength=0.0, strategy=self.name)

        # Sentiment score in range [-1, +1] (positive = bullish)
        sentiment = float(features.get("sentiment_score", pd.Series([0.0])).iloc[-1])

        # On-chain: net exchange inflow (negative = coins leaving exchange → bullish)
        net_inflow = float(features.get("net_exchange_inflow", pd.Series([0.0])).iloc[-1])

        # Whale anomaly score in range [0, 1] (higher = more unusual activity)
        whale_score = float(features.get("whale_anomaly_score", pd.Series([0.0])).iloc[-1])

        # Blend signals
        onchain_signal = -net_inflow  # positive when coins leave exchange (bullish)
        blended = sentiment + self.whale_alert_weight * whale_score * (1 if onchain_signal > 0 else -1)
        blended = max(-1.0, min(1.0, blended))

        if abs(blended) < self.sentiment_threshold:
            return Signal(direction="flat", strength=0.0, strategy=self.name,
                          metadata={"blended": blended})

        direction = "long" if blended > 0 else "short"
        strength = round(min(1.0, abs(blended)), 4)
        return Signal(
            direction=direction,
            strength=strength,
            strategy=self.name,
            metadata={
                "sentiment": sentiment,
                "net_inflow": net_inflow,
                "whale_score": whale_score,
                "blended": blended,
            },
        )
