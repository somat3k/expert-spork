"""Base strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd


@dataclass
class Signal:
    """Represents a trading signal produced by a strategy."""

    direction: str          # 'long', 'short', or 'flat'
    strength: float         # 0.0 – 1.0 confidence score
    strategy: str           # name of the originating strategy
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction not in {"long", "short", "flat"}:
            raise ValueError(f"Invalid direction: {self.direction!r}")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be in [0, 1], got {self.strength}")


class BaseStrategy(ABC):
    """
    Abstract base class for all Ethereum trading strategies.

    Subclasses must implement :meth:`generate_signal`, which receives a
    DataFrame of OHLCV bars and optional supplementary features (on-chain
    metrics, sentiment scores, etc.) and returns a :class:`Signal`.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def generate_signal(
        self,
        ohlcv: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> Signal:
        """
        Analyse market data and return a trading signal.

        Parameters
        ----------
        ohlcv:
            DataFrame with columns ``open``, ``high``, ``low``, ``close``,
            ``volume`` indexed by UTC timestamp.
        features:
            Optional DataFrame of supplementary features (on-chain metrics,
            sentiment scores, funding rates, etc.) aligned to the same index.

        Returns
        -------
        Signal
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
