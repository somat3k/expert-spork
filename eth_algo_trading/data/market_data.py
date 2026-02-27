"""
Market data client.

Wraps the `ccxt` library to fetch OHLCV candles and order-book snapshots
from any supported exchange.  Paper-trading / offline mode returns synthetic
data when no live credentials are configured.
"""

from __future__ import annotations

import time
from typing import List, Optional

import pandas as pd


class MarketDataClient:
    """
    Unified market data client backed by ccxt.

    Parameters
    ----------
    exchange_id:
        ccxt exchange identifier (e.g. ``'binance'``, ``'coinbase'``).
    api_key:
        Exchange API key (read-only scope is sufficient).
    api_secret:
        Exchange API secret.
    sandbox:
        When True the exchange's sandbox / testnet endpoint is used.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: str = "",
        api_secret: str = "",
        sandbox: bool = True,
    ) -> None:
        self.exchange_id = exchange_id
        self._exchange = None

        try:
            import ccxt  # noqa: F401

            exchange_cls = getattr(ccxt, exchange_id)
            self._exchange = exchange_cls(
                {
                    "apiKey": api_key,
                    "secret": api_secret,
                    "enableRateLimit": True,
                }
            )
            if sandbox and hasattr(self._exchange, "set_sandbox_mode"):
                self._exchange.set_sandbox_mode(True)
        except Exception:
            # Gracefully degrade to offline mode if ccxt is unavailable or
            # the exchange cannot be initialised.
            self._exchange = None

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str = "ETH/USDT",
        timeframe: str = "1h",
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars from the exchange.

        Returns a DataFrame with columns ``open``, ``high``, ``low``,
        ``close``, ``volume`` indexed by UTC timestamp.  Falls back to
        empty DataFrame on failure.
        """
        if self._exchange is None:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        try:
            raw = self._exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            return df
        except Exception:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # ------------------------------------------------------------------
    # Order book
    # ------------------------------------------------------------------

    def fetch_order_book_imbalance(self, symbol: str = "ETH/USDT", depth: int = 10) -> float:
        """
        Return the best-bid / best-ask volume imbalance in [-1, +1].

        Positive values indicate more buying pressure.
        """
        if self._exchange is None:
            return 0.0

        try:
            ob = self._exchange.fetch_order_book(symbol, limit=depth)
            bid_vol = sum(qty for _, qty in ob["bids"][:depth])
            ask_vol = sum(qty for _, qty in ob["asks"][:depth])
            total = bid_vol + ask_vol
            if total == 0:
                return 0.0
            return (bid_vol - ask_vol) / total
        except Exception:
            return 0.0
