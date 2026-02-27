"""
Risk manager.

Enforces position sizing, stop-loss / take-profit rules, and circuit
breakers for the Ethereum algo trading system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PositionSizeResult:
    """Output of :meth:`RiskManager.compute_position_size`."""

    size_usd: float
    size_eth: float
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    rationale: str


class RiskManager:
    """
    Volatility-scaled position sizer with circuit-breaker logic.

    Parameters
    ----------
    max_position_pct:
        Maximum fraction of capital allocated to a single trade.
    stop_loss_pct:
        Stop-loss distance as a fraction of entry price.
    take_profit_pct:
        Take-profit distance as a fraction of entry price.
    max_drawdown_pct:
        Portfolio-level drawdown that triggers a trading halt.
    volatility_lookback:
        Number of recent bars used to estimate realised volatility.
    """

    def __init__(
        self,
        max_position_pct: float = 0.10,
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.06,
        max_drawdown_pct: float = 0.15,
        volatility_lookback: int = 24,
    ) -> None:
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.volatility_lookback = volatility_lookback
        self._peak_capital: Optional[float] = None

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def compute_position_size(
        self,
        capital_usd: float,
        entry_price: float,
        signal_strength: float,
        ohlcv: Optional[pd.DataFrame] = None,
    ) -> PositionSizeResult:
        """
        Calculate a volatility-scaled position size.

        The base allocation is ``capital * max_position_pct * signal_strength``.
        When OHLCV data is provided, the allocation is further scaled
        inversely to realised volatility so that higher-volatility
        regimes receive smaller exposures.

        Parameters
        ----------
        capital_usd:
            Available capital in USD.
        entry_price:
            Current ETH price in USD.
        signal_strength:
            Confidence score in [0, 1] from the strategy signal.
        ohlcv:
            Optional OHLCV DataFrame used to estimate realised vol.

        Returns
        -------
        PositionSizeResult
        """
        if capital_usd <= 0 or entry_price <= 0:
            return PositionSizeResult(0.0, 0.0, None, None, "invalid inputs")

        base_alloc = capital_usd * self.max_position_pct * signal_strength

        # Volatility scaling
        vol_scalar = 1.0
        if ohlcv is not None and len(ohlcv) >= self.volatility_lookback:
            returns = ohlcv["close"].pct_change().dropna()
            realised_vol = float(returns.iloc[-self.volatility_lookback:].std())
            # Scale down when vol is high relative to 1 % daily target
            target_vol = 0.01
            if realised_vol > 0:
                vol_scalar = min(1.0, target_vol / realised_vol)

        size_usd = base_alloc * vol_scalar
        size_eth = size_usd / entry_price if entry_price else 0.0

        stop = entry_price * (1 - self.stop_loss_pct)
        take = entry_price * (1 + self.take_profit_pct)

        return PositionSizeResult(
            size_usd=round(size_usd, 2),
            size_eth=round(size_eth, 6),
            stop_loss_price=round(stop, 2),
            take_profit_price=round(take, 2),
            rationale=f"vol_scalar={vol_scalar:.3f}, signal={signal_strength:.3f}",
        )

    # ------------------------------------------------------------------
    # Circuit-breaker
    # ------------------------------------------------------------------

    def check_circuit_breaker(self, current_capital: float) -> bool:
        """
        Return True if the circuit-breaker has been triggered.

        The breaker fires when the portfolio has declined more than
        ``max_drawdown_pct`` from its peak.
        """
        if self._peak_capital is None or current_capital > self._peak_capital:
            self._peak_capital = current_capital
            return False

        drawdown = (self._peak_capital - current_capital) / self._peak_capital
        return drawdown >= self.max_drawdown_pct
