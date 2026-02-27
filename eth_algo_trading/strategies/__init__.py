"""Trading strategies package."""

from eth_algo_trading.strategies.base import BaseStrategy, Signal
from eth_algo_trading.strategies.scalping import ScalpingStrategy
from eth_algo_trading.strategies.arbitrage import ArbitrageStrategy
from eth_algo_trading.strategies.trend_following import TrendFollowingStrategy
from eth_algo_trading.strategies.sentiment import SentimentStrategy
from eth_algo_trading.strategies.multiplex import MultiplexStrategy

__all__ = [
    "BaseStrategy",
    "Signal",
    "ScalpingStrategy",
    "ArbitrageStrategy",
    "TrendFollowingStrategy",
    "SentimentStrategy",
    "MultiplexStrategy",
]
