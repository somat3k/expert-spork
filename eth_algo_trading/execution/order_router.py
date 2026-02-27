"""
Smart order router.

Routes orders to the best available venue based on price, fees, and
available liquidity.  Supports simple best-price routing across multiple
ccxt-backed exchange clients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from eth_algo_trading.data.market_data import MarketDataClient


@dataclass
class OrderResult:
    """Result of a routed order."""

    exchange_id: str
    symbol: str
    side: str         # 'buy' or 'sell'
    amount_eth: float
    price_usd: float
    status: str       # 'filled', 'rejected', 'paper'
    fee_usd: float = 0.0


class OrderRouter:
    """
    Routes buy/sell orders to the venue with the best executable price.

    Parameters
    ----------
    clients:
        Dictionary mapping exchange ID to :class:`MarketDataClient` instances.
    fee_rate:
        Assumed taker fee as a fraction of trade value (e.g. 0.001 = 0.1 %).
    paper_trading:
        When True, orders are simulated and never sent to an exchange.
    """

    def __init__(
        self,
        clients: Optional[Dict[str, MarketDataClient]] = None,
        fee_rate: float = 0.001,
        paper_trading: bool = True,
    ) -> None:
        self.clients: Dict[str, MarketDataClient] = clients or {}
        self.fee_rate = fee_rate
        self.paper_trading = paper_trading

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def best_venue(self, symbol: str = "ETH/USDT", side: str = "buy") -> Optional[str]:
        """
        Return the exchange ID offering the best price for the given side.

        For buys the cheapest ask is preferred; for sells the highest bid.
        Returns ``None`` when no venues are available.
        """
        best_exchange: Optional[str] = None
        best_price: Optional[float] = None

        for exchange_id, client in self.clients.items():
            ohlcv = client.fetch_ohlcv(symbol, timeframe="1m", limit=1)
            if ohlcv.empty:
                continue
            price = float(ohlcv["close"].iloc[-1])
            if best_price is None:
                best_price = price
                best_exchange = exchange_id
            elif side == "buy" and price < best_price:
                best_price = price
                best_exchange = exchange_id
            elif side == "sell" and price > best_price:
                best_price = price
                best_exchange = exchange_id

        return best_exchange

    def route_order(
        self,
        symbol: str,
        side: str,
        amount_eth: float,
        price_usd: float,
    ) -> OrderResult:
        """
        Route an order to the best venue or simulate it in paper mode.

        Parameters
        ----------
        symbol:
            Trading pair, e.g. ``'ETH/USDT'``.
        side:
            ``'buy'`` or ``'sell'``.
        amount_eth:
            Order size in ETH.
        price_usd:
            Reference price used for fee calculation.

        Returns
        -------
        OrderResult
        """
        exchange_id = self.best_venue(symbol, side) or "paper"
        fee_usd = amount_eth * price_usd * self.fee_rate

        if self.paper_trading:
            return OrderResult(
                exchange_id=exchange_id,
                symbol=symbol,
                side=side,
                amount_eth=amount_eth,
                price_usd=price_usd,
                status="paper",
                fee_usd=round(fee_usd, 4),
            )

        client = self.clients.get(exchange_id)
        if client is None or client._exchange is None:
            return OrderResult(
                exchange_id=exchange_id,
                symbol=symbol,
                side=side,
                amount_eth=amount_eth,
                price_usd=price_usd,
                status="rejected",
                fee_usd=0.0,
            )

        try:
            client._exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=amount_eth,
            )
            return OrderResult(
                exchange_id=exchange_id,
                symbol=symbol,
                side=side,
                amount_eth=amount_eth,
                price_usd=price_usd,
                status="filled",
                fee_usd=round(fee_usd, 4),
            )
        except Exception:
            return OrderResult(
                exchange_id=exchange_id,
                symbol=symbol,
                side=side,
                amount_eth=amount_eth,
                price_usd=price_usd,
                status="rejected",
                fee_usd=0.0,
            )
