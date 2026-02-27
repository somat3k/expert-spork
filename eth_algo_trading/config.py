"""
Configuration for the Ethereum algo trading system.

Exchange credentials and sensitive values are loaded from environment
variables (see .env.example).  All other tuneable parameters have
sensible defaults that can be overridden at runtime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()


@dataclass
class HyperliquidConfig:
    """Credentials and endpoint settings for Hyperliquid."""

    account_address: str = field(default_factory=lambda: os.getenv("HL_ACCOUNT_ADDRESS", ""))
    private_key: str = field(default_factory=lambda: os.getenv("HL_PRIVATE_KEY", ""))
    testnet: bool = True


@dataclass
class ExchangeConfig:
    """Credentials and settings for a single exchange connection."""

    exchange_id: str
    api_key: str = field(default_factory=lambda: os.getenv("EXCHANGE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("EXCHANGE_API_SECRET", ""))
    sandbox: bool = True


@dataclass
class RiskConfig:
    """Global risk-management parameters."""

    max_position_pct: float = 0.10       # max 10 % of capital per position
    max_drawdown_pct: float = 0.15       # circuit-breaker at 15 % drawdown
    volatility_lookback: int = 24        # hours used to estimate realised vol
    stop_loss_pct: float = 0.03          # 3 % stop-loss per trade
    take_profit_pct: float = 0.06        # 6 % take-profit per trade


@dataclass
class ModelConfig:
    """Hyper-parameters shared across AI/ML models."""

    forecast_horizon: int = 12           # bars ahead to forecast
    regime_n_states: int = 3             # bull / sideways / bear
    anomaly_contamination: float = 0.05  # expected fraction of anomalies
    sentiment_model: str = "ProsusAI/finbert"


@dataclass
class RLConfig:
    """Hyperparameters for the reinforcement learning trading agent."""

    timeframes: List[str] = field(default_factory=lambda: ["1h", "4h", "1d"])
    lookback_bars: int = 50           # bars of history per timeframe for state
    gamma: float = 0.99               # discount factor
    learning_rate: float = 1e-3       # Adam optimizer learning rate
    epsilon_start: float = 1.0        # initial exploration rate (epsilon-greedy)
    epsilon_end: float = 0.05         # minimum exploration rate
    epsilon_decay: int = 1000         # steps over which epsilon decays linearly
    batch_size: int = 64              # mini-batch size for replay training
    replay_capacity: int = 10_000     # maximum transitions in replay buffer
    target_update_freq: int = 100     # steps between target-network syncs
    hidden_size: int = 128            # neurons per hidden layer in Q-network


@dataclass
class DatabaseConfig:
    """Configuration for the hyperparameter persistence database."""

    db_path: str = ":memory:"         # ':memory:' for in-process, or a file path


@dataclass
class TradingConfig:
    """Top-level trading configuration."""

    symbol: str = "ETH/USDT"
    timeframe: str = "1h"
    exchanges: List[ExchangeConfig] = field(default_factory=list)
    risk: RiskConfig = field(default_factory=RiskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    hyperliquid: HyperliquidConfig = field(default_factory=HyperliquidConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    paper_trading: bool = True

    @classmethod
    def default(cls) -> "TradingConfig":
        return cls(
            exchanges=[
                ExchangeConfig(exchange_id="binance"),
                ExchangeConfig(exchange_id="coinbase"),
            ]
        )
